# Jamba — 混合状态空间模型-Transformer（Hybrid SSM-Transformer）

> 状态空间模型（SSM）和Transformer（Transformer 架构）追求不同目标。Transformer通过注意力机制（attention）以二次时间复杂度换取质量，而SSM通过递归实现线性时间推理和常数内存，但质量稍逊。AI21的Jamba（2024年3月）和Jamba 1.5（2024年8月）将两者融合到同一模型中：每7层Mamba层搭配1层Transformer层，每隔一个块应用MoE，支持256k上下文窗口，能运行在单个80GB GPU上。Mamba-3（ICLR 2026）通过复数值状态空间和MIMO投影加强了SSM部分。此课程完整阅读两种架构，解释为什么这种混合方案在纯SSM和纯Transformer的长上下文探索未果时，已在扩展下存活三年。

**类型：** 学习  
**语言：** Python（标准库，层混合计算器）  
**先决条件：** 阶段10 · 14（开放模型架构），阶段10 · 17（本地稀疏注意力）  
**时长：** 约60分钟

## 学习目标

- 解释Jamba模块中的三大基本单元 — Transformer层、Mamba层、MoE — 及1:7:偶数间隔的交替配方。
- 概述SSM的递归形式及其为何支持常量内存推理。
- 计算256k上下文下Jamba模型的KV缓存占用，并与纯Transformer模型做对比。
- 列举Mamba-3的三项创新（指数-梯形离散化、复数值状态更新、MIMO），以及它们各自解决的问题。

## 问题陈述

注意力的计算复杂度是序列长度的平方，状态空间模型是线性的。这种差异被放大：在256k长度的序列中，Transformer每个注意力头的注意力矩阵大小达650亿条目，而SSM的递归状态尺寸固定，不随序列长度变化。

纯SSM模型（Mamba, Mamba-2）在小规模时能匹配Transformer困惑度（perplexity），但在状态跟踪任务和某些上下文检索类别表现不佳。直觉上讲，SSM将历史压缩到固定状态，历史越长信息泄漏越严重。注意力机制则能精确记忆全部信息，但付出二次时间和内存成本。

显而易见的解决方案是两者兼用：在需要精确回忆的地方使用Transformer层，其余使用SSM层，并调整比例。Jamba是首个大规模生产级别应用这一混合配方的模型（总参数52B，活跃参数12B，256k上下文，单80GB GPU）。Jamba 1.5将规模扩展至398B/94B。Mamba-3（ICLR 2026）是当前最佳纯SSM基线，未来混合模型可基于此重建。

本课程读取上述三篇论文，构建“如何选对比例”的思维模型。

## 核心概念

### 一页理解SSM

状态空间模型处理序列`x_1, ..., x_N`，通过固定大小的状态`h`：

```text
h_t = A h_{t-1} + B x_t
y_t = C h_t
```

每步状态基于线性动力学矩阵`A`演化，接收输入`B x_t`，输出`C h_t`。参数`A, B, C`可以学习。关键特性：计算`y_t`只需`h_{t-1}`和`x_t`，不需早期输入。内存恒定，推理每个token是O(1)。

保证建模质量的诀窍在于`A`的结构。S4（Gu 2021）使用高度结构化矩阵，在训练时可高效实现为长卷积。Mamba（Gu, Dao 2023）将固定`A, B, C`替换为数据相关的参数（“选择性”部分）。Mamba-2（2024）简化结构，Mamba-3（2026）在特定部分重新增加复杂度。

关键特性：对解码LLM，SSM层可作为注意力层的替代，拥有固定大小的层内状态而非随序列增长的KV缓存。

### Jamba模块

Jamba模块按照两个数字交错层：

- `l`：注意力层与Mamba层的比例。Jamba用`l = 8`，即每7个Mamba层配1个Transformer层（7 Mamba + 1 Attention，8层为一组）。
- `e`：MoE出现频率。Jamba用`e = 2`，即每隔一层应用MoE。

模块内层序列：

```text
M  M  M  M  M  M  M  A    （7个Mamba + 1个Attention）
|  M  |  M  |  M  |  M    （|表示应用MoE）
```

每个Jamba模块为8层。4个模块深时（共32层），包含28个Mamba层和4个Attention层，其中16层使用MoE。

### 为什么是1:7比例

AI21做了消融实验：什么比例的注意力层对其长上下文评估中的困惑度与上下文回忆表现最好？

- 注意力过多（1:1）：质量提升，但速度和内存下降。
- 注意力过少（1:15）：内存表现优异，但上下文检索失败。
- 最佳区间：1:7或1:8。

这是因为Transformer层负责精确回忆和状态跟踪，Mamba层处理大量廉价计算。

### 位置编码

Mamba层本身通过递归感知位置。原始基于Mamba的混合模型中，Transformer层不使用RoPE（旋转位置编码），SSM层提供位置信息。Jamba 1.5为Transformer层添加了RoPE，以提升长上下文泛化，这是基于经验长期评估的后期调整。

### 内存预算

以Jamba-1结构（32层：28 Mamba + 4 Attention，隐藏维4096，32注意力头）为例：

- KV缓存（仅Attention层）：`2 * 4 * 32 * 128 * 256k * 2 = 8.4 GB`，BF16格式，256k长度。仅4个Attention层贡献。
- SSM状态：每个token前缀为`28 * hidden * state_size`，但这是每层固定大小，不随序列增长。典型Mamba状态为每特征16个，隐藏4096：`28 * 4096 * 16 * 2 = 3.7 MB`总计。

相比之下，纯Transformer 32层、隐藏4096、32头全多头注意力（MHA）时，256k上下文下KV缓存为：`2 * 32 * 32 * 128 * 256k * 2 = 128 GB`。减少约8倍。和大多数2024模型采用的GQA(8)基线（`2 * 32 * 8 * 128 * 256k * 2 = 32 GB`）比，Jamba 1:7混合方案16GB仍小一半。

这就是AI21口中“256k上下文，单80GB GPU”的含义。纯全MHA Transformer的KV缓存不可能装进80GB，即使是GQA也无剩余给权重和激活，Jamba方案则可以。

### Mamba-3：2026年纯SSM基线

Mamba-3（ICLR 2026，arXiv:2603.15569）在纯SSM方面引入三项创新：

1. **指数-梯形离散化。** 用更具表现力的递归代替Mamba-2的Euler法离散化。在核心递归中，对状态输入应用类卷积操作，而非对`x_t`外部卷积。

2. **复数值状态更新。** 先前的Mamba版本将状态矩阵从复数（S4）降至实数对角（Mamba）再到缩放单位矩阵（Mamba-2）。Mamba-3重新引入复数值，相当于状态上的数据相关旋转嵌入（rotary embedding）。恢复了早期实数值简化牺牲的状态跟踪能力。

3. **多输入多输出（MIMO）投影。** 由特征标量投影变为矩阵值投影。提升建模能力和推理时硬件利用率，且不增加解码时延。

在15亿参数规模下，Mamba-3相较Gated DeltaNet提升下游平均准确度0.6点；MIMO变体在此基础上多提升1.2点，合计1.8点。相同状态大小时，Mamba-3以一半状态匹配Mamba-2。

Mamba-3尚未以混合形式大规模生产，但显然是未来Jamba级别模型中SSM部分的理想候选。

### 何时选用混合架构

混合方案胜出时：

- 上下文足够长，以致纯Transformer KV缓存难以承载（64k以上）。
- 任务混合短距离结构要求（适合SSM）与长距离回忆（需Transformer）。
- 希望在单GPU显存预算内部署，纯Transformer的KV缓存过大。

混合方案失利时：

- 上下文较短（不足16k）。SSM开销无效，纯Transformer足够。
- 任务需要全局attention（深度推理、多文档交叉引用）。混合中注意力层稀疏度限制效果。
- 扩展至万亿参数前沿模型。目前纯Transformer + MLA + MoE（如DeepSeek-V3）占优。

### 竞争格局

| 模型 | 系列 | 规模 | 独特主张 |
|-------|--------|------|-------------|
| Mamba-2 | 纯SSM | 3B | 线性时间，常数内存 |
| Jamba | 混合 | 52B/12B | 256k上下文，80GB GPU实装 |
| Jamba 1.5 大型 | 混合 | 398B/94B | 企业级长上下文 |
| Mamba-3 | 纯SSM | 1.5B（论文） | 恢复状态跟踪能力 |
| DeepSeek-V3 | 纯Transformer + MoE | 671B/37B | 前沿能力 |

2026年格局：纯Transformer MoE主导前沿，混合架构占据256k及更长上下文细分领域。Mamba-3的状态跟踪优势可能推动下一代混合比例向SSM倾斜（更少Attention）。

## 使用方法

`code/main.py` 是一个混合架构内存计算器。给定SSM-Transformer比例和隐藏尺寸/层数配置，它计算：

- 目标上下文的KV缓存。
- SSM状态内存。
- 不同模型结构在上下文N时的总内存。

计算器支持：

- 纯Transformer基线（KV缓存随N增长）。
- Jamba风格的1:7混合。
- 纯SSM（无KV缓存）。

数据直接来源Jamba-1和Jamba-1.5论文公开形态，假设变体外推。

实际部署集成注意：

- 多数生产推理服务器（vLLM, SGLang）支持Jamba和Mamba。请确认支持的版本。
- 在256k上下文下，Jamba的内存优势体现于并发请求吞吐量。同等显存下，Jamba能处理更多序列。
- Mamba-3单独模型尚未大规模生产，现为1.5B参数的研究预览。

## 部署建议

本课程生成`outputs/skill-hybrid-picker.md`。给定工作负载规格（上下文长度分布、任务类型、内存预算），推荐纯Transformer、Jamba风格混合或纯SSM模型，并明确内存与质量权衡原因。

## 练习

1. 运行`code/main.py`，计算256k上下文时32层纯Transformer（隐藏4096，32头）与相同配置Jamba-1混合模型的KV缓存。验证论文称的约8倍内存减少。

2. 修改计算器建模1:3混合（4 Mamba：1 Attention）和1:15混合（14 Mamba：1 Attention），绘制KV缓存随比例变化图。KV缓存与SSM状态内存相等时比例多少？

3. 阅读Jamba论文第3节（arXiv:2403.19887）。解释为什么AI21选用Mamba-1而非更快的Mamba-2。提示：混合消融部分有明确记录。

4. 计算Jamba 1.5大型（398B总参数，94B活跃参数）中每隔一层MoE的参数开销。将活跃比例与DeepSeek-V3（37B/671B）比较，解释Jamba架构为何推动更高活跃比例。

5. 阅读Mamba-3论文第3节（arXiv:2603.15569）。用三句话解释为何复数值状态更新等价于受数据驱动的旋转嵌入。结合阶段7 · 课程04的RoPE推导。

## 关键词

| 术语 | 通用说法 | 实际含义 |
|------|-----------|----------|
| 状态空间模型（SSM） | “固定状态的递归” | 具有学习递归 `h_t = A h_{t-1} + B x_t` 的层；每token常数内存 |
| 选择性SSM | “Mamba的技巧” | 数据相关A、B、C参数，实现线性时间的门控选择性 |
| 注意力与Mamba比例 | “多少注意力层” | Jamba中`l = 8`表示7个Mamba层配1个注意力层 |
| Jamba块 | “8层组” | 一个注意力层 + 七个Mamba层，隔层应用MoE |
| SSM状态 | “隐藏缓存” | 每层固定大小的状态，替代Mamba层的KV缓存 |
| 256k上下文 | “Jamba旗舰指标” | Jamba-1能在单80GB GPU运行的序列长度，纯Transformer做不到 |
| Mamba-3 | “2026纯SSM” | 当前最优纯SSM架构，带复数状态和MIMO；混合模型可基于此构建 |
| MIMO | “多输入多输出” | Mamba-3创新，使用矩阵值投影替代特征标量投影 |
| 指数-梯形离散化 | “Mamba-3的递归” | 表达力更强的递归，涵盖Mamba-2的Euler法离散化 |
| 混合架构 | “混合注意力和SSM” | 交错使用Transformer和SSM层的任意模型；Jamba是生产范例 |

## 深入阅读

- [Lieber 等人 — Jamba: A Hybrid Transformer-Mamba Language Model（arXiv:2403.19887）](https://arxiv.org/abs/2403.19887) — 原始 Jamba 论文，比例消融，256k 上下文长度宣称
- [AI21 — Jamba 1.5: Hybrid Transformer-Mamba at Scale（arXiv:2408.12570）](https://arxiv.org/abs/2408.12570) — 大规模版本，398B/94B 和 12B/52B 公开发布
- [Gu, Dao — Mamba: Linear-Time Sequence Modeling with Selective State Spaces（arXiv:2312.00752）](https://arxiv.org/abs/2312.00752) — 选择性 SSM 论文，Jamba 的基础
- [Dao, Gu — Mamba-2（arXiv:2405.21060）](https://arxiv.org/abs/2405.21060) — 简化的结构化状态空间继任者
- [Lahoti 等人 — Mamba-3（arXiv:2603.15569, ICLR 2026）](https://arxiv.org/abs/2603.15569) — 复值状态，MIMO，2026 年纯 SSM 最前沿
- [Gu 等人 — Efficiently Modeling Long Sequences with Structured State Spaces（arXiv:2111.00396）](https://arxiv.org/abs/2111.00396) — S4 论文，长序列语言模型（LLM）中的 SSM 系谱起点
