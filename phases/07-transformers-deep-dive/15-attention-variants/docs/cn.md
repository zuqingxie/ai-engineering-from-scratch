# 注意力变体 — 滑动窗口（Sliding Window）、稀疏（Sparse）、微分（Differential）

> 完整注意力是一个圈。每个 token 都能看见其他所有 token，代价是巨大的内存开销。四种变体改变了这个圈的形状，恢复了一半的开销。

**类型：** 构建  
**语言：** Python  
**先决条件：** 阶段 7 · 02（自注意力（Self-Attention）），阶段 7 · 03（多头（Multi-Head）），阶段 7 · 12（KV 缓存 / Flash Attention）  
**时间：** ~60 分钟

## 问题

完整注意力在序列长度上需要 `O(N²)` 的内存和计算开销。对于一个拥有 128K 上下文的 Llama 3 70B，每层有 160 亿条注意力条目，乘以 80 层。Flash Attention（课程 12）隐藏了 `O(N²)` 的激活内存，但并没有改变计算复杂度 —— 依然是每个 token 关注每个 token。

三类变体改变了注意力矩阵的拓扑结构：

1. **滑动窗口注意力（SWA）**。每个 token 仅关注固定窗口内的邻居，而非整个前缀。内存和计算降为 `O(N · W)`，其中 `W` 是窗口大小。Gemma 2/3、Mistral 7B 的前几层、Phi-3-Long 皆采用。
2. **稀疏 / 块注意力。** 只对选定的 `(i, j)` 对进行评分，其余强制为零权重。代表包括 Longformer、BigBird、OpenAI 稀疏 Transformer。
3. **微分注意力。** 使用独立的 Q/K 投影计算两个注意力图，互相相减。解决了令权重流入首个几个 token 的“注意力陷阱（attention sink）”问题。微软的 DIFF Transformer（2024）采用。

以上三种方法可以共存。2026 年的前沿模型多采用混合方式：大多数层为 SWA-1024，每隔五层有一层全局完整注意力，还有少量微分头用于清理检索噪声。Gemma 3 采用 5:1 的 SWA 与全局层比例，是当前教科书默认方案。

## 概念

### 滑动窗口注意力（SWA）

位置 `i` 的查询只关注 `[i - W, i]`（因果 SWA）或 `[i - W/2, i + W/2]`（双向）。窗口外的 token 在得分矩阵中赋为 `-inf`。

```text
完整因果:              滑动窗口（W=4）：
位置 0-7                位置 0-7，W=4
    0 1 2 3 4 5 6 7        0 1 2 3 4 5 6 7
0 | x                0 |  x
1 | x x              1 |  x x
2 | x x x            2 |  x x x
3 | x x x x          3 |  x x x x
4 | x x x x x        4 |    x x x x
5 | x x x x x x      5 |      x x x x
6 | x x x x x x x    6 |        x x x x
7 | x x x x x x x x  7 |          x x x x
```

对于 `N = 8192`，`W = 1024`，得分矩阵的期望非零条目为 1024 × 8192，减少约 8 倍。

**KV 缓存随着 SWA 缩小。** 每层仅需保留最后 `W` 个 token 的 K 和 V。以 Gemma-3 风格配置（1024 窗口，128K 上下文）为例，KV 缓存减少 128 倍。

**质量代价。** 纯 SWA Transformer 在长距离检索上表现较差。解决办法是在 SWA 层之间穿插全注意力层。Gemma 3 采用 5:1 的 SWA 到全局比例。Mistral 7B 使用因果 SWA 堆栈，信息通过重叠窗口“向前流动”——每层扩展有效感受野 `W`，经过 `L` 层后模型可回顾 `L × W` 个 token。

### 稀疏 / 块注意力

提前选择一个 `N × N` 的稀疏模式。三种典型形状：

- **本地 + 间隔采样（OpenAI 稀疏 Transformer）。** 关注最近的 `W` 个 token，外加每隔 `stride` 个采样一个 token。以 `O(N · sqrt(N))` 计算捕获局部和长距离信息。
- **Longformer / BigBird。** 本地窗口 + 一小部分全局 token（如 `[CLS]`），全局 token 关注所有人且被所有人关注 + 随机稀疏链接。实测可在相同质量下扩展 2 倍上下文。
- **原生稀疏注意力（DeepSeek，2025）。** 学习哪些 `(Q, K)` 块重要；在内核级别跳过全零块。兼容 FlashAttention。

稀疏注意力是内核工程问题。数学只需掩码得分矩阵；性能提升来自永远不加载零条目到 SRAM。FlashAttention-3 和 2026 FlexAttention API 在 PyTorch 中让自定义稀疏模式成为一等公民。

### 微分注意力（DIFF Transformer，2024）

常规注意力有“注意力陷阱”问题：softmax 强制每行和为 1，导致不想关注特定内容的 token 将权重“倾泻”到第一个 token（或头几个）。这抢占了本应用于真实内容的容量。

微分注意力通过计算**两个**注意力图并相减解决：

```text
A1 = softmax(Q1 K1^T / √d)
A2 = softmax(Q2 K2^T / √d)
DiffAttn = (A1 - λ · A2) V
```

其中 `λ` 是学习得到的标量（通常为 0.5–0.8）。A1 捕获真实内容权重；A2 捕获“陷阱”权重。相减抵消陷阱，将权重重新分配给相关 token。

微软 2024 报告结果：困惑度下降 5–10%，有效上下文长度增加 1.5–2 倍，实体检索（needle-in-haystack）更精准。

### 变体对比

| 变体 | 计算复杂度 | KV 缓存 | 质量 vs 全量 | 生产使用 |
|---------|---------|----------|-----------------|----------------|
| 完整注意力 | O(N²) | 每层 O(N) | 基准线 | 所有模型默认层 |
| SWA（窗口 1024） | O(N·W) | 每层 O(W) | 降低 0.1 ppl，结合全局层效果好 | Gemma 2/3，Phi-3-Long |
| 本地 + 间隔稀疏 | O(N·√N) | 混合 | 类似 SWA | OpenAI 稀疏 Transformer，Longformer |
| BigBird（本地 + 全局 + 随机） | 约 O(N) | 混合 | 在 2 倍上下文匹配全量质量 | 早期长上下文 BERT |
| 原生稀疏（DeepSeek-V3.2） | O(N · 活跃比例) | O(N) | 质量差异在 0.05 ppl 内 | DeepSeek-V3.2，2025 |
| 微分 | O(2·N²) | O(2N) | 降低 5–10% ppl | DIFF Transformer，2026 早期模型 |

## 构建实现

见 `code/main.py`。实现一个因果掩码对比器，能并列展示完整、SWA、本地+间隔稀疏、微分注意力在玩具序列上的区别。

### 第 1 步：完整因果掩码（基线）

```python
def causal_mask(n):
    return [[0.0 if j <= i else float("-inf") for j in range(n)] for i in range(n)]
```

课程 07 的基础。下三角，全零重权重，上三角为 `-inf`。

### 第 2 步：滑动窗口因果掩码

```python
def swa_mask(n, window):
    M = [[float("-inf")] * n for _ in range(n)]
    for i in range(n):
        lo = max(0, i - window + 1)
        for j in range(lo, i + 1):
            M[i][j] = 0.0
    return M
```

唯一参数是 `window`。当 `window >= n` 时，恢复完整因果注意力。`window = 1` 时，每个 token 只关注自己。

### 第 3 步：本地 + 间隔稀疏掩码

```python
def strided_mask(n, window, stride):
    M = [[float("-inf")] * n for _ in range(n)]
    for i in range(n):
        lo = max(0, i - window + 1)
        for j in range(lo, i + 1):
            M[i][j] = 0.0
        for j in range(0, i + 1, stride):
            M[i][j] = 0.0
    return M
```

稠密的本地窗口，外加每隔 `stride` 个采样一个 token，覆盖到序列开头。多层叠加使感受野呈对数增长。

### 第 4 步：微分注意力

```python
def diff_attention(Q1, K1, Q2, K2, V, lam):
    A1 = softmax_causal(Q1 @ K1.T / sqrt_d)
    A2 = softmax_causal(Q2 @ K2.T / sqrt_d)
    return (A1 - lam * A2) @ V
```

两次注意力计算，用一个学习出的混合系数做差。在代码里我们对比单一和微分注意力的“注意力陷阱”热力图，观察陷阱消失。

### 第 5 步：KV 缓存大小

打印每层 `N = 131072` 时各种变体的缓存大小。SWA 和稀疏版本减少 10–100 倍；微分注意力缓存翻倍。合理规划内存预算。

## 使用方法

2026 年生产模式示例：

```python
from transformers import AutoModelForCausalLM
# Gemma 3 以 5:1 比例混合 SWA（窗口1024）和全局层。
model = AutoModelForCausalLM.from_pretrained("google/gemma-3-27b-it")
# print(model.config.sliding_window, model.config.layer_types)
```

PyTorch 2.5+ 的 FlexAttention 支持掩码函数：

```python
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

def swa_pattern(b, h, q_idx, kv_idx):
    return (q_idx - kv_idx < 1024) & (q_idx >= kv_idx)

mask = create_block_mask(swa_pattern, B=batch, H=heads, Q_LEN=n, KV_LEN=n)
out = flex_attention(q, k, v, block_mask=mask)
```

此代码会编译成定制 Triton 内核。性能在常见模式下约为 FlashAttention-3 的 90%，掩码函数是 Python 可调用对象。

**选择建议：**

- **纯全量注意力** — 适用于最高约 16K 上下文，或当检索质量首要时。
- **SWA + 全局混合** — 适合超长上下文（>32K），训练和推理受内存限制。2026 年 32K 以上默认方案。
- **稀疏块注意力** — 自定义内核和稀疏模式，适合专门工作负载（检索、音频）。
- **微分注意力** — 针对“注意力陷阱”影响明显的场景（长上下文 RAG、needle-in-haystack 检索）。

## 交付实现

见 `outputs/skill-attention-variant-picker.md`。此技能根据目标上下文长度、检索需求、训练与推理计算资源，选择合适的注意力拓扑。

## 练习

1. **简单。** 运行 `code/main.py`。验证 `window=4` 时每行外的前几个 token 得分变为零。验证 `window=n` 时完整因果注意力的掩码完全一致。
2. **中等。** 基于课程 07 的结业作业，实现因果 SWA（`window=1024`）。在 tinyshakespeare 训练 1,000 步。验证验证集损失相比全量注意力是否退化，峰值内存下降多少。
3. **困难。** 实现 Gemma-3 风格的 5:1 层混合（5 层 SWA，1 层全局）。对比纯 SWA、纯全局模型在相同性能下的损失、内存和生成质量。
4. **困难。** 实现微分注意力，每头使用可学习的 `λ`。在合成检索任务（1 针，2000 干扰）上训练。对比单一注意力基线的检索准确率。

## 关键词

| 术语 | 大众说法 | 真实含义 |
|------|-----------------|-----------------------|
| 滑动窗口注意力（SWA） | “局部注意力” | 每个查询只关注最近 `W` 个 token；KV 缓存缩小到 `O(W)`。 |
| 有效感受野 | “模型能够看到多远” | `L` 层 SWA 堆栈，窗口大小为 `W`，最大可看 `L × W` 个 token。 |
| Longformer / BigBird | “局部 + 全局 + 随机” | 稀疏模式，含少量始终关注的全局 token；早期长上下文方案。 |
| 原生稀疏注意力 | “DeepSeek 的内核技巧” | 学习块级稀疏；在内核级跳过零块，保持质量。 |
| 微分注意力 | “两个图，一个相减” | DIFF Transformer：用学习系数乘第二个注意力图并从第一个中减去，抵消注意力陷阱。 |
| 注意力陷阱 | “权重流向 token 0” | softmax 正规化导致不具信息的查询权重落在位置 0。 |
| FlexAttention | “掩码即 Python 函数” | PyTorch 2.5+ 提供的 API，将任意掩码函数编译为 FlashAttention 式内核。 |
| 层类型混合 | “5:1 SWA 到全局” | 在堆栈中交替稀疏和全注意力层，以降低内存的同时保持质量。 |

## 延伸阅读

- [Beltagy, Peters, Cohan (2020). Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150) — 经典的滑动窗口 + 全局 token 论文。
- [Zaheer et al. (2020). Big Bird: Transformers for Longer Sequences](https://arxiv.org/abs/2007.14062) — 本地 + 全局 + 随机。
- [Child et al. (2019). Generating Long Sequences with Sparse Transformers](https://arxiv.org/abs/1904.10509) — OpenAI 的本地+间隔模式。
- [Gemma Team (2024). Gemma 2: Improving Open Language Models at a Practical Size](https://arxiv.org/abs/2408.00118) — 1:1 比例的 SWA:全局混合。
- [Gemma Team (2025). Gemma 3 technical report](https://arxiv.org/abs/2503.19786) — 5:1 比例、窗口 1024，现为教科书默认。
- [Ye et al. (2024). Differential Transformer](https://arxiv.org/abs/2410.05258) — DIFF Transformer 论文。
- [Yuan et al. (2025). Native Sparse Attention](https://arxiv.org/abs/2502.11089) — DeepSeek-V3.2 学习稀疏注意力。
- [PyTorch — FlexAttention 博客与文档](https://pytorch.org/blog/flexattention/) — 掩码可调用模式的 API 参考。
