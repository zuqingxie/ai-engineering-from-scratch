# Native Sparse Attention（DeepSeek NSA）

> 在 64k tokens 下，attention 占据了 70-80% 的解码延迟。每个开源模型实验室都有计划解决它。DeepSeek 的 NSA（ACL 2025 最佳论文）是最终方案：三个并行的 attention 分支——压缩粗粒度 tokens、有选择地保留细粒度 tokens，以及用于本地上下文的滑动窗口——通过学习的门控结合。它是硬件对齐的（内核友好），原生可训练的（在预训练中起作用，而非推理时附加），在 64k 解码时运行速度比 FlashAttention 更快，同时匹配或优于全 attention 质量。本课程构建这三个分支的端到端实现，并展示为何稀疏模式是端到端可微的。

**类型：** 构建  
**语言：** Python（stdlib）  
**先决条件：** Phase 7 · 12（KV cache，flash-attention）、Phase 7 · 15（attention 变体）、Phase 10 · 16（差分 attention）  
**时间：** 约 60 分钟

## 学习目标

- 阐述 NSA 的三条 attention 分支及其各自捕获的信息内容。
- 解释为何 NSA 是“原生可训练的”，而之前的稀疏 attention 方法仅限于推理。
- 计算 NSA 相对于全 attention 在 64k 上下文时的计算节省，作为压缩区块大小和选择 top-k 的函数。
- 用标准库 Python 实现三分支组合，在简短的合成序列上测试门控权重的表现。

## 问题背景

全 attention 在序列长度 N 时的时间复杂度为 `O(N^2)`，每层 KV cache 的空间复杂度为 `O(N)`。在 64k token 规模下，计算和内存带宽消耗极其庞大。NSA 论文测量的理论估计：在 64k 时 attention 占解码总延迟的 70-80%。下游任务——TTFT、tokens/秒、每百万 token 费用——均被 attention 成本主导。

稀疏 attention 是显而易见的答案。以往尝试分为两种。固定模式稀疏（滑动窗口、步幅、块局部）丢弃信息，在长距离回忆任务上失败。仅推理时稀疏（KV cache 剪枝、H2O、StreamingLLM）应用于预训练的密集 attention 模型，只恢复了部分潜在加速，因为模型未被训练来通过稀疏模式路由信息。

Native Sparse Attention（袁等，DeepSeek + PKU + UW，ACL 2025 最佳论文，arXiv:2502.11089）两者兼备：在预训练期间学习的稀疏模式，实现为硬件友好的内核级算法，推理时能真正带来计算节省。两年后，NSA 或其直接后代将成为所有前沿长上下文模型的默认 attention。

## 核心概念

### 三条并行分支

对每个 query，NSA 对三种不同视角的 KV cache 运行三次 attention：

1. **压缩分支。** tokens 被分成大小为 `l`（通常为 32 或 64）的区块。每个区块通过小型学习型 MLP 压缩成一个摘要 token。query 对这些压缩 tokens 进行 attention，获得整个序列的粗粒度视角。

2. **选择分支。** 利用压缩分支的 attention 分数，选出对当前 query 最相关的 top-k 区块。从这些区块读取未压缩的细粒度 tokens，query 对它们进行 attention。可将压缩分支的 attention 理解为选择的路由信号。

3. **滑动窗口分支。** query 对最近的 `W` 个 tokens（通常为 512）进行 attention，为本地上下文提供支撑。该分支捕获结构密集的短程模式（语法、本地共指），其他两者可能忽略。

三个分支的输出通过一个位置学习门控组合：

```text
out = g_cmp * out_cmp + g_sel * out_sel + g_win * out_win
```

`g_cmp, g_sel, g_win` 是 query 输入到小型 MLP 得到的门控权重，不必和为 1，可以独立加权各分支。

### “原生可训练” 的原因

选择步骤（top-k 区块）是离散的，离散操作会断开梯度流。之前的稀疏 attention 工作要么放弃了选择的反向传播（限制训练），要么使用连续松弛方法，导致推理时不产生真正的稀疏。

NSA 绕开这一点：压缩分支 attention 是针对全序列的可微粗粒度 attention。top-k 操作仅重用压缩分支的 top attention 分数，挑选要加载的细粒度区块。梯度从压缩分支分数流过，它们既影响压缩输出，也影响选择逻辑，被选区块对最终输出的贡献也是可微的。不可微的 `top_k` 操作在正向计算图中无动作，仅控制从内存加载哪些区块。

这就是 NSA 可以端到端用于预训练的原因。模型学会通过三条分支联合路由信息，产生推理时真正带来加速的稀疏模式。

### 硬件对齐的内核

NSA 的内核设计针对现代 GPU 内存层次结构。内核按 GQA（grouped query attention）组加载查询（外层循环），针对每组获取对应的稀疏 KV 区块（内层循环），在 SRAM 上运行 attention。因为每个查询组看到相同的被选区块（选择是按查询组，而非查询头），KV 加载被组内均摊，算力密度保持高。

论文报道 Triton 内核在 64k 解码上比 FlashAttention 快 9 倍，且随序列长度增长加速比增加。正向和反向内核均已提供。

### 计算预算

设 `N` 为序列长度，`l` 为压缩区块大小，`k` 为 top-k 选择数量，`w` 为滑动窗口大小，`b` 为被选区块大小（通常等于 `l`）。

- 压缩分支：每个 query 访问 `O(N/l)` 个 key，总计 `O(N * N/l)`。
- 选择分支：每个 query 访问 `O(k * b)` 个 key，总计 `O(N * k * b)`。
- 滑动窗口分支：每个 query 访问 `O(w)` 个 key，总计 `O(N * w)`。

总计：`O(N * (N/l + k*b + w))`。

例如，`N=64k, l=64, k=16, b=64, w=512`，每 query 访问 `1000 + 1024 + 512 = 2536` keys；全 attention 则是 `64000` keys，计算量减少 25 倍。

`N=128k, l=64, k=16, b=64, w=512`，每 query 访问 `2000 + 1024 + 512 = 3536` keys；全 attention 是 `128000` keys，减少 36 倍。效益随序列长度增加，正是设计初衷。

### 与其他方法比较

| 方法                | 可微分性         | 实际推理加速      | 长距离回忆           |
|--------------------|-----------------|-----------------|---------------------|
| 仅滑动窗口          | 是              | 是              | 失败                |
| 步幅 / 块稀疏       | 是              | 是              | 部分                 |
| KV 剪枝（H2O, StreamingLLM）| 不适用（推理时） | 是              | 部分                 |
| MoBA (Moonshot)     | 部分            | 是              | 良好                 |
| NSA                 | 是（原生）       | 是（64k 9 倍）  | 匹配全 attention     |

MoBA（Moonshot，arXiv:2502.13189）同期发布，采用类似的“三个比一个好”策略，应用 MoE 原则于 attention 区块。NSA 和 MoBA 是 2026 年长上下文预训练要关注的两种架构。

## 构建它

`code/main.py` 实现三条分支，测试简短合成序列：

- 压缩 MLP（为教学简洁起见实际用均值池化；真实 NSA 使用学习型 MLP）。
- 基于压缩分支分数的 top-k 区块选择。
- 最后 `w` tokens 的滑动窗口 attention。
- 门控组合。
- 对比全 attention 的计算量统计打印。

### 步骤 1：将 tokens 压缩成区块

```python
def compress(K, l):
    n = len(K)
    n_blocks = (n + l - 1) // l
    out = []
    for b in range(n_blocks):
        start, end = b * l, min((b + 1) * l, n)
        block = K[start:end]
        summary = [sum(row[d] for row in block) / len(block) for d in range(len(K[0]))]
        out.append(summary)
    return out
```

### 步骤 2：压缩分支 attention

运行 query 对压缩后的 keys 的 softmax attention。压缩分支分数同时作为 top-k 选择信号。

### 步骤 3：top-k 区块选择

选择 `k` 个得分最高的压缩区块索引。加载这些区块的原始未压缩 tokens，对其进行 attention。

### 步骤 4：滑动窗口 attention

取最后的 `w` tokens，运行标准 attention。

### 步骤 5：门控 + 组合

query 输入小型 MLP，输出三个门控权重，最终输出是三分支输出的加权和。

### 步骤 6：计算统计

打印每个 query 各分支处理的 keys 数和总数，比较全 attention 的 `N`。在 1024 token 合成序列，`l=32, k=4, w=128` 时，NSA 每 query 访问 `32 + 128 + 128 = 288` keys，全 attention 是 1024，计算减少约 3.5 倍。

## 使用它

NSA 已在 DeepSeek 自身长上下文预训练管道中投入使用。截止 2026 年 4 月，公开推理栈整合状态：

- **DeepSeek 内部**：原生，发布模型权重用 NSA 或其继任者 DSA（Deepseek Sparse Attention）。
- **vLLM**：正在开发实验中，支持 DeepSeek-V3.x 权重。
- **SGLang**：发布 NSA 基准测试，生产路径跟随 vLLM。
- **llama.cpp / CPU**：不支持；内核分解的开销对 CPU 吞吐不划算。

适用 NSA 情形：

- 目标 64k+ 上下文预训练或继续训练，且计算预算充足。
- 推理 DeepSeek 自有长上下文检查点，权重原生 NSA。

不适用 NSA 情形：

- 服务现有密集 attention 预训练模型，NSA 无继续训练难以改造。
- 上下文长度小于 16k，三分支开销超过节省。
- 单条交互式聊天，低延迟推理受益有限，仅长上下文有效。

## 发布它

本课程生成 `outputs/skill-nsa-integrator.md`。给定长上下文预训练运行规格，输出 NSA 集成方案：压缩区块大小、top-k、滑窗、门控 MLP 宽度、内核选择，以及特定长上下文评测，支持架构变更的合理性。

## 练习

1. 在 1024 token 合成序列上运行 `code/main.py`。遍历 `(l, k, w)` 三组预设，打印计算计数。找出在保持对全 attention 针对稻草堆中抽针式测试 95% 回忆率的同时，键数量最少的预设。

2. 用一个小型学习型 MLP（2 层，隐藏维度 32）替换均值池压缩器。它在合成任务上训练，使信号为区块均值。在留出数据集上测量相对于均值池压缩的困惑度差异。

3. 实现门控 MLP。输入 query，输出三个标量。展示门控合理行为：随机 query 权重近似均匀，当 query 命中远端区块时，选择分支权重大。

4. 计算启用 NSA 的 70B 模型在 128k 上下文的 KV cache 内存预算。KV head 数 8，head 维度 128，BF16。与全 attention 和 MLA（Phase 10 · 14 显示 MLA 数据）对比。找出 NSA 细粒度分支 KV cache 等于全 attention 的序列长度。

5. 阅读NSA论文（arXiv:2502.11089）的第4节，并用三句话解释为什么压缩分支（compressed branch）的注意力分数被重用于top-k选择而不是计算单独的路由分数。请将答案与梯度流（gradient flow）联系起来。

## 术语表

| 术语 | 普通说法 | 实际含义 |
|------|----------|----------|
| Compressed branch（压缩分支） | “粗略视角” | 对块平均键（block-averaged keys）的注意力，为每个查询提供全局上下文，键数量为O(N/l) |
| Selected branch（选择分支） | “top-k块” | 对压缩分支分数最高的k个块进行细粒度注意力 |
| Sliding window（滑动窗口） | “局部上下文” | 对最近的W个token的注意力，用于捕捉短距模式 |
| Native trainability（原生可训练性） | “带稀疏性的预训练” | 稀疏模式在预训练期间学习，而非推理时附加 |
| Compression block size l（压缩块大小 l） | “粗略视角的组大小” | 多少个token合并为一个摘要；典型值32-64 |
| Top-k（top-k） | “保留的块数” | 读取的压缩块中未压缩token的数量；典型值16 |
| Sliding window W（滑动窗口 W） | “局部注意范围” | 通常为512；太短影响局部连贯性，太长浪费计算 |
| Branch gate（分支门控） | “三者组合方式” | 每个位置的MLP输出，用以加权三条分支的贡献 |
| Hardware alignment（硬件对齐） | “硬核友好型稀疏” | 稀疏模式选择使实际GPU核达到理论加速 |
| DSA | “NSA的继任者” | Deepseek Sparse Attention，NSA在DeepSeek系列后的架构 |

## 延伸阅读

- [Yuan等 — Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention (arXiv:2502.11089, ACL 2025最佳论文)](https://arxiv.org/abs/2502.11089) — 论文原文
- [DeepSeek-V3技术报告 (arXiv:2412.19437)](https://arxiv.org/abs/2412.19437) — NSA针对的架构家族
- [Moonshot AI — MoBA: Mixture of Block Attention for Long-Context LLMs (arXiv:2502.13189)](https://arxiv.org/abs/2502.13189) — 同期工作，基于块的MoE风格注意力
- [Beltagy等 — Longformer: The Long-Document Transformer (arXiv:2004.05150)](https://arxiv.org/abs/2004.05150) — 滑动窗口的起源
- [Xiao等 — StreamingLLM: Efficient Streaming Language Models with Attention Sinks (arXiv:2309.17453)](https://arxiv.org/abs/2309.17453) — NSA改进的推理时稀疏基线
- [Dao等 — FlashAttention-2 (arXiv:2307.08691)](https://arxiv.org/abs/2307.08691) — NSA核性能超过的全注意力基线，适用于64k长度
