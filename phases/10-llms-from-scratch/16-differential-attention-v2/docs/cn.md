# 差分注意力（V2）

> Softmax注意力（Softmax attention）会将少量概率分散到每个不匹配的标记上。在超过10万个标记时，这些噪声会累积并淹没信号。差分变换器（Differential Transformer）（Ye等，ICLR 2025）通过计算两个softmax的差值来修正，减去共享的噪声底层。DIFF V2（微软，2026年1月）是面向生产的重写版本：解码延迟匹配基线Transformer，无需自定义内核，兼容FlashAttention。本课程涵盖了从V1到V2的端到端内容，包含一个可在标准库Python中运行的差分操作玩具实现。

**类型：** 构建  
**语言：** Python（标准库）  
**前提条件：** 阶段7 · 02（自注意力），阶段7 · 15（注意力变体），阶段10 · 14（架构讲解）  
**时长：** 约60分钟

## 学习目标

- 准确说明为什么softmax注意力存在噪声底层以及它为何随上下文长度增长。  
- 推导差分注意力公式，解释为何相减能抵消共享的噪声成分同时保留信号。  
- 讲解从V1到V2的差分：哪些变得更快，哪些更简化，哪些更稳定，以及为何每个改动对生产预训练至关重要。  
- 用纯Python从零实现差分注意力，并在合成的信号加噪声查询上实证验证噪声抵消特性。

## 问题

标准softmax注意力在数学性质上存在一个在大规模应用中会变成操作难题的问题。对查询`q`，注意力权重是`softmax(qK^T / sqrt(d))`。softmax永远不会产生精确的零——每个不匹配的标记都会获得一些正概率。这残留的概率即噪声，且随上下文长度线性增长。在128k标记时，即使每个不匹配标记只获得0.001%的概率，127,999个标记累积起来仍贡献约12%的总概率。模型必须学会绕开这个随上下文增长的噪声底层。

实证上，这表现为注意力头干扰：长上下文检索增强生成（RAG）中出现的幻觉引用，100k标记检索任务中的“丢中间”失败，以及超过32k时的“针中寻找”基准测试中细微的准确度下降。差分变换器论文（arXiv:2410.05258, ICLR 2025）测量了差距：DIFF变换器比同尺寸基线达到更低的困惑度、更高的长上下文准确率和更少的幻觉。

DIFF V1有三个问题阻碍其进入前沿预训练流水线：其value缓存每步解码需要加载两次，依赖自定义CUDA内核导致FlashAttention不兼容，以及每头RMSNorm在70B级规模中训练不稳定。DIFF V2（微软 unilm 博客，2026年1月）解决了这三个问题。本课程讲解两个版本，构建差分操作，并在玩具查询上基准噪声抵消效果。

## 概念

### Softmax的噪声底层

给定查询`q`和键`K = [k_1, ..., k_N]`，注意力权重为：

```text
w_i = exp(q · k_i / sqrt(d)) / sum_j exp(q · k_j / sqrt(d))
```

没有任何`w_i`会是零。如果`k_i`与`q`完全无关，分数`q · k_i`不会是0——它以方差`||q||^2 / d`围绕零波动。经softmax归一化，每个无关标记仍贡献`O(1/N)`，无关标记总贡献为`O((N-1)/N) = O(1)`——数量不可忽视。

模型期望的是类似硬top-k的机制：匹配标记权重高，其他接近零。softmax太平滑，不能直接做到这一点。

### 差分思想

将每个头的Q和K投影拆成两部分：Q = (Q_1, Q_2)，K = (K_1, K_2)。计算两张注意力图：

```text
A_1 = softmax(Q_1 K_1^T / sqrt(d))
A_2 = softmax(Q_2 K_2^T / sqrt(d))
```

输出：

```text
DiffAttn = (A_1 - lambda * A_2) V
```

相减抵消两张图共同的噪声分布。如果两图在127k无关标记上权重大致均匀（初始随机时如此），它们会互相抵消。信号——在少量实际相关标记上的峰值权重——只有在两图中同幅度出现才会抵消，训练后显然不会。

`lambda`是每头学得的标量，参数化为`lambda = exp(lambda_q1 · lambda_k1) - exp(lambda_q2 · lambda_k2) + lambda_init`，可负。`lambda_init`默认小的正数，如0.8。

### 为何这匹配噪声消除机制

如两只含噪麦克风录同一声音，均拾取讲话声与相关背景噪声。二者相减，共享噪声被剔除。讲话声保留，因为两个信号在相位或幅度上足够不同，避免完全抵消。每头`lambda`学得正是这种平衡。

### V1与V2的差别

V1保持参数数量与基线相等，为实现每头两个查询，将头维度减半，降低表达能力，更痛苦的是value缓存减半。解码时需加载value缓存两次（每个softmax分支各一次），结果解码速度低于基线。

V2将查询头数量翻倍，KV头数不变（从上采样参数借用），头维保持基线不变。差分后将额外维度投影回匹配基线Transformer的O_W投影。三件事同时发生：

1. 解码速度匹配基线（KV缓存加载一次）。  
2. FlashAttention运行无改动（无自定义内核）。  
3. 解码时算术强度提升（每从HBM加载字节算力增多）。  

V2也移除了V1中用于稳定相减的每头RMSNorm。在70B级预训练中，该RMSNorm后期训练不稳定。V2用更简洁的初始化方案替代，保持训练稳定无需额外模块。

### 何时采用

| 工作负载 | 受益情况 |
|----------|---------|
| 长上下文RAG（64k+） | 注意力图更干净，幻觉引用更少 |
| 针中寻找基准测试 | 超过32k时准确率显著提升 |
| 多文档问答 | 跨文档干扰减少 |
| 8k代码补全 | 较小提升，不值得改架构 |
| 短对话（< 4k） | 与基线基本无差异 |

优势随上下文长度增长。4k时噪声底层足够小，标准注意力可用。128k则显著影响性能。

### 与2026年其他调优配合情况

| 特性 | 是否兼容DIFF V2？ |
|------|-------------------|
| GQA | 是（V2增加Q头，不增KV头） |
| MLA（DeepSeek） | 原则支持，尚无联合发表论文 |
| MoE | 是（注意力独立于MLP块） |
| RoPE | 是（未改变） |
| YaRN / 长上下文扩展 | 是（DIFF正是覆盖此处需求） |
| FlashAttention | 是（V2支持，V1不支持） |
| 预测解码（Speculative decoding） | 是（注意力更改对预测解码环节透明） |

## 构建实现

`code/main.py`用纯Python实现差分注意力。通过一个结构已知的信号加噪声玩具查询，可以直接测量噪声抵消比率。

### 步骤1：标准softmax注意力

使用标准库矩阵操作：列表嵌套列表，手写矩阵乘法，softmax时减去最大值保持数值稳定。

```python
def softmax(row):
    m = max(row)
    exps = [math.exp(x - m) for x in row]
    s = sum(exps)
    return [e / s for e in exps]
```

### 步骤2：将Q、K分成两半

V1风格：头维度减半。  
V2风格：头维度不变，头数翻倍。  
玩具实现采用V1以便教学清晰——数学等价，仅记账不同。

### 步骤3：两个softmax分支 + 相减

```python
A1 = [softmax([dot(q1, k) / scale for k in K1]) for q1 in Q1]
A2 = [softmax([dot(q2, k) / scale for k in K2]) for q2 in Q2]
diff_weights = [[a1 - lam * a2 for a1, a2 in zip(r1, r2)] for r1, r2 in zip(A1, A2)]
out = [[sum(w * v[j] for w, v in zip(row, V)) for j in range(d_v)] for row in diff_weights]
```

注意：输出权重可为负。没关系——value缓存仍可处理带符号贡献。后续的V投影吸收符号。

### 步骤4：噪声抵消测量

构建长度为1024的合成序列，在指定位置放置信号标记，其余填充噪声。计算：（a）标准softmax注意力在信号位置的权重；（b）差分注意力权重。测量每种注意力的信噪比。DIFF注意力在两分支差异训练充分后，信噪比可提升3倍到10倍。

### 步骤5：V1与V2参数对比

参数配置（hidden=4096, heads=32, d_head=128），输出：  

- 基线Transformer：Q、K、V均为`hidden * hidden`大小，MLP为4倍hidden。  
- DIFF V1：Q、K为`hidden * hidden`大小，V大小不变，头维减半，额外每头`lambda`参数（数量级为heads · d_head）。  
- DIFF V2：Q大小为`2 * hidden * hidden`，K、V大小不变，额外维度投影回，新增同样`lambda`参数。  

玩具示例会测算V2额外参数开销（大概是每注意力块多`hidden * hidden`参数），并打印。

## 使用建议

截至2026年4月，DIFF V2尚未部署到每个生产推理服务器，但正集成至vLLM和SGLang，同时已见于：

- 微软内部长上下文生产模型。  
- 多个目标256k+上下文的开放模型训练复制产物。  
- 结合DIFF注意力与滑动窗口注意力的混合架构。  

2026年采用场景：  
- 从零训练目标64k+有效上下文模型，从一开始加入差分注意力；后期重新训练代价高。  
- 微调长上下文模型，存在丢中间失败，可用LoRA调节Q投影近似DIFF结构。  

避免采用：  
- 服役已稳定长上下文表现的预训练稠密模型，重新训练成本往往难以回收。  
- 上下文长保持在16k以下，噪声底层微不足道。

## 交付方案

本课程生成`outputs/skill-diff-attention-integrator.md`。根据模型架构、目标上下文长度、幻觉表现及训练预算，制定集成差分注意力到新预训练或LoRA微调的方案。

## 练习

1. 运行`code/main.py`。验证差分注意力在合成查询上的信噪比高于标准softmax注意力。改变噪声幅度，展示标准注意力变得不可用的临界点。  

2. 计算7B级模型（hidden=4096, heads=32, d_head=128, 32层）从基线到DIFF V1以及DIFF V2的参数差异。展示哪些组件参数增加，哪些保持不变。

3. 阅读 DIFF V1 论文（arXiv:2410.05258）第3节和 DIFF V2 Hugging Face 博客第2节。用两句话解释为什么 V1 需要每头 RMSNorm（每头均方根归一化）以及为什么 V2 可以去掉它而不导致训练发散。

4. 实现一个消融实验：计算微分注意力（differential attention），令 `lambda = 0`（纯第一softmax）和 `lambda = 1`（完全相减）。在合成查询上，测量信噪比在不同 lambda 值下的变化。找出使信噪比最大的 `lambda`。

5. 将玩具模型扩展到 GQA + DIFF V2。选择 8 个 KV 头和 32 个 Q 头。证明 KV 缓存大小与相同（8，32）配置的基线 GQA 模型相匹配。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|------------|----------|
| Differential attention（微分注意力） | “两个 softmax 相减” | 将 Q、K 分成两半，计算两个 softmax 映射，用第二个（乘以缩放因子 lambda）从第一个中减去，再乘以 V |
| Noise floor（噪声底） | “softmax 的非零尾部” | softmax 在每个无关 token 上放的 O(1/N) 权重，在长上下文中加起来为 O(1) |
| lambda（lambda 缩放因子） | “相减的缩放比例” | 每头可学习的标量，参数化为 `exp(lq1.lk1) - exp(lq2.lk2) + lambda_init`；可以为负 |
| DIFF V1 | “ICLR 2025 版本” | 原始微分 Transformer；头维度减半以保持参数量，需要自定义内核，解码速度较慢 |
| DIFF V2 | “2026年1月修正” | Q 头数翻倍，KV 头保持不变；解码速度匹配基线，支持 FlashAttention |
| Per-head RMSNorm（每头 RMSNorm） | “V1 的稳定器” | V1 在差分后应用的额外归一化；V2 去除它以防训练后期不稳定 |
| Signal-to-noise ratio（信噪比） | “浪费的注意力多少” | 真实信号位置上的权重与无关位置平均权重的比值 |
| Lost in the middle（中间遗失现象） | “长上下文失败模式” | 经验现象，长上下文中文档中间部分的检索准确率下降——DIFF 注意力能减少这种情况 |
| Arithmetic intensity（算术强度） | “每字节加载的 FLOPs” | V2 在解码时通过每次 KV 加载双倍查询数提高的比率；对内存带宽受限的解码非常重要 |

## 延伸阅读

- [Ye 等 — Differential Transformer（arXiv:2410.05258，ICLR 2025）](https://arxiv.org/abs/2410.05258) — 原始论文，包含降噪理论和长上下文消融实验
- [Microsoft unilm — Differential Transformer V2（Hugging Face 博客，2026年1月）](https://huggingface.co/blog/microsoft/diff-attn-v2) — 生产堆栈重写，匹配基线解码，支持 FlashAttention
- [理解 Differential Transformer 打破预训练自注意力的束缚（arXiv:2505.16333）](https://arxiv.org/abs/2505.16333) — 理论分析为什么减法能恢复预训练注意力结构
- [共享 DIFF Transformer（arXiv:2501.17900）](https://arxiv.org/html/2501.17900) — 参数共享变体
- [Vaswani 等 — Attention Is All You Need（arXiv:1706.03762）](https://arxiv.org/abs/1706.03762) — DIFF 以此 Transformer 作为基线
- [Liu 等 — Lost in the Middle（arXiv:2307.03172）](https://arxiv.org/abs/2307.03172) — DIFF 注意力针对的长上下文基准
