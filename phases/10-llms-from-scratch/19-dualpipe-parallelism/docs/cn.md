# DualPipe 并行

> DeepSeek-V3 在 2048 个 H800 GPU 上训练，MoE 专家分布在各节点上。跨节点专家 all-to-all 通信的开销是每 1 GPU 小时计算耗费 1 GPU 小时通信，GPU 有一半时间处于空闲状态。DualPipe（DeepSeek，2024 年 12 月）是一种双向流水线（bidirectional pipeline），它使前向和反向计算与它们触发的 all-to-all 通信重叠进行。流水线气泡减少，吞吐量上升，且由于专家并行（Expert Parallelism）已将专家分散到各个进程，保留两个模型参数副本（“双重”即名称来源）成本很低。本课是一个 Learn 类型的教程，讲解 DualPipe 实际做了什么以及为什么 Sea AI Lab 的 DualPipeV 优化以牺牲稍微紧凑一点的气泡，降低了 2 倍参数成本。

**类型：** Learn  
**语言：** Python（标准库，调度模拟器）  
**先决条件：** 第 10 阶段 · 05（分布式训练，FSDP，DeepSpeed），第 10 阶段 · 14（开源模型架构和 MoE）  
**时间：** 约 60 分钟

## 学习目标

- 说出 DualPipe 前向-反向块的四个组成部分及其各自独立重叠窗口的原因。
- 解释大规模流水线气泡问题，以及“无气泡”在实践中和营销中的区别。
- 手工描绘一个 8 个流水线并行（PP）进程和 16 个微批次的 DualPipe 调度，确认前向和反向流填充彼此的空闲槽。
- 说明 DualPipeV（Sea AI Lab，2025 年）权衡：当专家并行不活跃时，放弃 2 倍参数复制，但气泡稍大。

## 问题

在 2000 个 H800 GPU 上训练一个 671B MoE 模型会遇到三个叠加瓶颈：

1. **内存压力。** 每个 GPU 持有模型的一个切片。序列长度 8k、61 层、128 头的激活内存巨大。
2. **流水线气泡。** 传统流水线并行(GPipe，1F1B)在等待输入或梯度时使 GPU 空闲。8 个阶段时，1F1B 调度下大约 12% GPU 时间因气泡浪费。
3. **跨节点 all-to-all。** MoE 专家并行把专家分散到节点上。每次前向传递都会触发一次 all-to-all 把 token 分发给专家，再一次 all-to-all 汇总。2000 GPU 规模容易导致计算与通信比约 1:1。

每个问题有独立的解决方案：内存用梯度检查点，流水线气泡用 Zero Bubble（Sea AI Lab，2023），all-to-all 用专家并行通信内核。DualPipe 做的是让它们协同工作。调度将计算和通信在单个前后向块中重叠，双端同时注入微批，利用调度隐藏 all-to-all 通信于计算窗口中。

报告结果：DeepSeek-V3 14.8T token 训练中流水线气泡几乎消除，GPU 利用率超过 95%。

## 概念

### 流水线并行回顾

将 N 层模型划分到 P 个设备。设备 `i` 持有第 `i * N/P` 层到 `(i+1) * N/P - 1` 层。微批顺序通过 0 到 P-1 设备的前向，随后反向从 P-1 到 0。一个设备的前向阶段只能在前一设备发送输出后开始，反向阶段只能在后一设备发送梯度后开始。

GPipe（Huang 等，2019）一次调度一个微批，浪费大量 GPU 时间。1F1B（Narayanan 等，2021）交叉调度多个微批的前向和反向。Zero Bubble（Qi 等，2023）将反向拆成两部分——针对输入的反向（B）和针对权重的反向（W）——并调度填充气泡，流水线近乎紧凑。

DualPipe 是下一步，叠加了两个新想法：

### 想法 1：块分解

每个前向块分为四部分：

- **Attention（注意力）。** Q/K/V 投影，注意力计算，输出投影。
- **All-to-all dispatch（分发）。** 跨节点通信，将 token 发给专家。
- **MLP。** MoE 专家计算。
- **All-to-all combine（合并）。** 跨节点通信，收集专家输出。

一个反向块增加了各部分的梯度版本。DualPipe 调度，使得 all-to-all dispatch 与下一块的 attention 计算并行，all-to-all combine 与后续块的 MLP 计算并行。

### 想法 2：双向调度

大多数流水线调度从 0 号阶段注入微批，顺序流向 P-1。DualPipe 从两端同时注入。阶段 0 处理来源于那端的前向微批；阶段 P-1 处理来源于那端的前向微批。两路流中间交汇。

为此，设备 `i` 必须同时持有早期流水线层 `i` 和晚期流水线层 `P - 1 - i`。这就是 DualPipe 的“双重”：每台设备保存了需要服务的两个模型层副本（每个方向一个）。在 DeepSeek-V3 规模下这是 2 倍参数复制成本。由于专家并行已经极度分散专家，复制非专家层两次成本很小。

核心在于，单向调度中气泡本该出现的位置，前向流和反向流正好重叠。气泡消失。

### 手工描绘的调度示例

考虑 P=4 的节点，8 个微批，4 个前向 + 4 个反向。时间由左至右，行是设备编号。

```text
           Time →
rank 0:  F1 F2 F3 F4  F5R F6R F7R F8R  B1 B2 B3 B4  ...
rank 1:     F1 F2 F3  F4/F5R F6R F7R   B1 B2 ...
rank 2:        F1 F2  F3/F5R F4/F6R    B1 ...
rank 3:           F1  F2/F5R F3/F6R    ...
```

“F4/F5R” 表示第 1 号设备同一时间既做第 4 微批从左往右的前向，也做第 5 微批从右往左的前向。这就是“双向”调度的含义。

第 2 号设备交汇较早，到第 0 和第 3 号设备最晚。调度稳定中期，每个设备都运行与方向 X 的前向和方向 Y 的反向重叠，计算保持繁忙。前向的 all-to-all dispatch 被反向计算隐藏，all-to-all combine 被前向计算隐藏，气泡被挤出。

### 气泡计算

标准 1F1B 的流水线气泡（每个设备浪费时间）：

```text
bubble_1F1B = (P - 1) * forward_chunk_time
```

Zero Bubble 优化减少了但没到零。DualPipe 稳定阶段在微批数量是流水线深度两倍的倍数时气泡为零。稳定期外（预热和冷却期）会有气泡，但气泡不随微批数量增长——这是一大特性。

市场上称“无气泡”，技术上是指气泡不随微批数增长。Sea AI Lab 后续分析（DualPipeV／剪半方案）指出只有当专家并行不是瓶颈时，才实现完全零气泡；用户若启用专家并行全员 all-to-all，调度总会有折中。

### DualPipeV——改进方案

Sea AI Lab（2025 年）观察到 2 倍参数复制当专家通信重叠无意义时浪费严重。DualPipeV 将双向注入变为单参数副本上的“V 型”调度。气泡比 DualPipe 稍大，但节省了大量内存。DeepSeek 在开源 DualPipe 实现中采用 DualPipeV 作为关闭专家并行时的方案。

权衡如下：

| 特性                     | DualPipe        | DualPipeV      | 1F1B         | Zero Bubble   |
|--------------------------|-----------------|---------------|--------------|--------------|
| 每设备参数副本数         | 2               | 1             | 1            | 1            |
| 气泡随微批变化           | 常数            | 小幅增长      | 增长         | 增长         |
| 计算-通信重叠程度         | 完全            | 部分          | 最小         | 部分         |
| 适用场景                 | 专家并行占主导   | 密集或轻专家   | 基准         | 任意流水线   |

### 对 14.8T token 训练的意义

DeepSeek-V3 预训练在 2048 H800 GPU 总耗费约 280 万 GPU 小时。使用粗暴的 1F1B，会损失 12%-15% 时间给气泡，相当 34-42 万 GPU 小时，足够训练完整 70B 模型。DualPipe 挽回了大部分浪费。由于无内部日志，难以精确量化其贡献，但论文声称训练平均 GPU 利用率超过 95%。

小规模（小于 1000 GPU）训练，DualPipe 过于复杂——气泡相较总成本小，密集模型训练也很少遇到 all-to-all 瓶颈。对于数千 GPU 级别的前沿 MoE 训练，它几乎是必须。

### 在堆栈中的位置

- **FSDP（第 10 阶段 · 05）** 互补。FSDP 把模型参数切片到进程，DualPipe 安排计算分配。它们可以共用。
- 兼容 **ZeRO-3** 梯度切片。两个副本的管理需要与 ZeRO 梯度切片配合。
- 需要**定制的 all-to-all 内核**，针对特定集群拓扑优化。DeepSeek 开源内核为参考。

## 使用

`code/main.py` 是流水线调度模拟器。输入 `(P, n_micro_batches, schedule)`，打印 1F1B、Zero Bubble、DualPipe 和 DualPipeV 在稳定阶段的利用率。是教学工具——结果与论文定性结论吻合，不代表生产加速。

模拟价值：用不同 P 和微批数量运行，观察气泡比例 1F1B 增长，DualPipe 不增长。

实际训练集成考虑：

- 选择流水线深度能整除微批数量。
- 保证专家并行网络支持双向全员 all-to-all，DeepSeek 内核为参考。
- 第一次调试调度会消耗一周时间，管理细节复杂。
- 监测每个设备 GPU 利用率，而不只是总和。DualPipe 优势体现在降低最慢设备空闲。

## 发布

本课生成 `outputs/skill-dualpipe-planner.md`。根据训练集群规格（GPU 数，拓扑，互联，模型结构），推荐流水线并行策略、调度算法及目标规模的气泡比例。

## 练习

1. 运行 `code/main.py` ，参数 `(P=8, micro_batches=16, schedule=dualpipe)` 和 `(P=8, micro_batches=16, schedule=1f1b)`。计算 GPU 利用率差异，换算成每百万 token 训练节省的 GPU 小时。

2. 手工绘制 `(P=4, micro_batches=8, schedule=dualpipe)` 的调度表。标记每个时间槽的微批 ID 和方向。找出气泡第一次消失的时间槽。

3. 阅读 DeepSeek-V3 技术报告（arXiv:2412.19437）的图 5。指出 DualPipe 前向块中 all-to-all dispatch 的重叠窗口。解释计算调度如何隐藏它。

4. 分别计算 70B 密集模型（P=8）和 671B MoE 模型（P=16）DualPipe 的 2 倍参数开销。说明 MoE 情况为何开销比例更小（大部分参数为专家，分散在大 EP 组）。

5. 比较 DualPipe 与 Chimera（2021 年的一个竞争性双向调度器）。根据论文第 3.4 节作为参考，指出 DualPipe 添加而 Chimera 没有的两个具体特性。

## 关键词

| 术语 | 常见说法 | 实际含义 |
|------|-----------|-----------|
| Pipeline bubble（流水线气泡） | “每个 rank 的空闲时间” | 由于流水线阶段等待输入或梯度而浪费的 GPU 周期 |
| 1F1B | “默认流水线调度” | 一个前向 / 一个反向交叉调度；DualPipe 超过的基线方案 |
| Zero Bubble（零气泡） | “Sea AI 实验室 2023” | 将反向拆分为 B（输入梯度）和 W（权重梯度）；几乎完全收紧流水线 |
| DualPipe | “DeepSeek-V3 调度” | 双向流水线 + 计算与通信重叠；气泡不会随着微批次数量增加 |
| DualPipeV | “减半” | V 形优化，牺牲略大气泡以减少 2 倍参数复制 |
| Chunk（块） | “流水线工作单元” | 一次微批次通过一个流水线阶段的正向或反向传递 |
| All-to-all dispatch（全互发派发） | “发送令牌到专家” | 跨节点通信，将令牌路由到分配的 MoE 专家 |
| All-to-all combine（全互发合并） | “收集专家输出” | 跨节点通信，收集 MLP 后的专家输出 |
| Expert Parallelism (EP)（专家并行） | “专家分布在 GPU 上” | 在不同 rank 上分片 MoE 专家，不同 GPU 持有不同专家 |
| Pipeline Parallelism (PP)（流水线并行） | “层分布在 GPU 上” | 在不同 rank 上分片模型层；DualPipe 调度的维度 |
| Bubble fraction（气泡比例） | “浪费的 GPU 时间” | (bubble_time / total_time)；DualPipe 追赶到接近零的比例 |

## 进一步阅读

- [DeepSeek-AI — DeepSeek-V3 技术报告 (arXiv:2412.19437)，第 3.3.2 节和图 5](https://arxiv.org/abs/2412.19437) — 主要的 DualPipe 参考资料
- [DeepSeek — DualPipe GitHub 仓库](https://github.com/deepseek-ai/DualPipe) — 开源参考实现，包括 DualPipeV（减半）模式
- [Qi 等 — Zero Bubble Pipeline Parallelism (arXiv:2401.10241，Sea AI 实验室 2023)](https://arxiv.org/abs/2401.10241) — Zero Bubble 的前身
- [Sea AI 实验室 — DualPipe 没有 Dual 会更好](https://sail.sea.com/blog/articles/63) — 通俗解析 DualPipeV，启发 DeepSeek 的 EP 关闭模式
- [Narayanan 等 — PipeDream / 1F1B (arXiv:1806.03377，2018-2021)](https://arxiv.org/abs/1806.03377) — DualPipe 比较的 1F1B 调度器
- [Huang 等 — GPipe (arXiv:1811.06965，2018)](https://arxiv.org/abs/1811.06965) — 最初的流水线并行论文及气泡问题
