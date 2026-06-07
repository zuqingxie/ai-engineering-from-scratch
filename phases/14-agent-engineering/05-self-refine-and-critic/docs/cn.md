# Self-Refine 和 CRITIC：迭代输出改进

> Self-Refine（Madaan 等人，2023）使用一个大型语言模型（LLM）扮演三个角色——生成（generate）、反馈（feedback）、改进（refine）——形成一个循环。平均提升：7 个任务上绝对提升 +20。CRITIC（Gou 等人，2023）通过将验证步骤依赖于外部工具加强了反馈环节。在 2026 年，这种模式将作为“评估器-优化器”（Anthropic）或守护环路（OpenAI Agents SDK）内置于所有框架。

**类型：** 构建  
**语言：** Python（标准库）  
**先决条件：** 第 14 阶段 · 01（Agent 循环），第 14 阶段 · 03（Reflexion）  
**时间：** ~60 分钟

## 学习目标

- 陈述 Self-Refine 的三个提示词（生成、反馈、改进）并解释为何历史记录对改进提示词很重要。
- 解释 CRITIC 的关键洞见：没有外部基础，LLM 无法可靠地进行自我验证。
- 实现一个带历史记录及可选外部验证器的标准库 Self-Refine 循环。
- 将此模式映射到 Anthropic 的“评估器-优化器”工作流和 OpenAI Agents SDK 的输出守护机制。

## 问题描述

代理产生了一个几乎正确的答案。也许某行代码有语法错误；或摘要过长；或计划遗漏了某个边界情况。你想要的是：代理自行批判自己的输出，然后修正它。

Self-Refine 证明单一模型可以做到这一点，无需训练数据，无需强化学习（RL）。但有一个槽点：LLM 在困难事实的自我验证上表现不佳。CRITIC 提出了解决方案——将验证步骤路由至外部工具（搜索、代码解释器、计算器、测试运行器）。

这两篇论文一起定义了 2026 年迭代改进的默认模式：生成，验证（尽可能外部），改进，当验证通过时停止。

## 概念详解

### Self-Refine（Madaan 等人，NeurIPS 2023）

单一 LLM，三种角色：

```text
generate(task)            -> output_0
feedback(task, output_0)  -> critique_0
refine(task, output_0, critique_0, history) -> output_1
feedback(task, output_1)  -> critique_1
refine(task, output_1, critique_1, history) -> output_2
...
当 feedback 不再发现问题或预算耗尽时停止。
```

关键细节：`refine` 可以看到完整的历史——所有先前的输出和反馈——从而避免重复错误。论文中消融实验显示：删除历史会导致质量急剧下降。

亮点：在涵盖数学、代码、缩略语、对话的 7 个任务上平均提升 +20 绝对分（包括 GPT-4），无训练，无外部工具，单模型。

### CRITIC（Gou 等人，arXiv:2305.11738，2024 年 2 月第四版）

Self-Refine 的弱点：反馈步骤是 LLM 自评分。事实声明的可靠性差（产生幻觉时模型往往自我欺骗）。CRITIC 将 `feedback(task, output)` 替换为 `verify(task, output, tools)`，其中 `tools` 包括：

- 用于事实判断的搜索引擎。
- 用于代码正确性的代码解释器。
- 用于算术运算的计算器。
- 领域特定验证器（单元测试、类型检查器、代码风格检查器）。

验证器基于工具结果生成结构化批评，优化器据此改进输出。

亮点：CRITIC 在事实型任务上优于 Self-Refine，因为批评有真实依据。对于无外部验证器的任务（创作类、格式类），CRITIC 等价于 Self-Refine。

### 停止条件

两种常见形式：

1. **验证器通过。** 外部测试返回成功。有单元测试、类型检查器、守护断言时优先使用。
2. **无反馈（模型认为没有问题）。** 代价较低，但不可靠；应配合最大迭代次数限制。

2026 年的默认策略是结合它们：“验证通过 OR（模型认为没问题 AND 迭代次数≥2） OR 迭代次数≥最大值”。

### 评估器-优化器（Anthropic，2024）

Anthropic 2024 年 12 月的公开文章将此定义为五大工作流模式之一。职责划分：

- 评估器（Evaluator）：对输出打分并给出批评。
- 优化器（Optimizer）：根据批评修订输出。

循环直到评估器通过。这是 Anthropic 封装的 Self-Refine/CRITIC。关键工程细节是：评估器和优化器的提示词应有显著差异，避免模型只是简单“盖章认同”。

### OpenAI Agents SDK 的输出守护

OpenAI Agents SDK 将此模式作为“输出守护”（output guardrails）引入。守护是一个在代理最终输出上运行的验证器。如果守护触发（抛出 `OutputGuardrailTripwireTriggered` 异常），则拒绝输出，代理可重试。守护可调用工具（类似 CRITIC）或仅为纯函数（类似 Self-Refine）。

### 2026 年的陷阱

- **盖章循环（Rubber-stamp loops）。** 同一模型用同一提示进行生成和反馈，往往陷入“看起来不错”的陷阱。应使用结构迥异的提示词或用较小廉价模型来做批评。
- **过度改进。** 每次改进都会增加延迟和令牌消耗。预算 1-3 次，超出则转人工审核。
- **CRITIC 在简单任务的适用性。** 无外部验证器时，CRITIC 退化为 Self-Refine；无需为无用的验证器承担额外延迟。

## 构建示例

`code/main.py` 实现了 Self-Refine 和 CRITIC，在一个玩具任务上：给定主题生成一个简短的要点列表。验证器检查格式（三条，每条不超过 60 字符）。CRITIC 加了一个外部“事实验证器”，用于惩罚已知幻觉。

组件：

- `generate` — 脚本生成器。
- `feedback` — LLM 风格的自我批评。
- `verify_external` — CRITIC 风格的有根验证。
- `refine` — 根据历史重写输出。
- 停止条件——验证通过或最多 4 次迭代。

运行方式：

```text
python3 code/main.py
```

对比 Self-Refine 和 CRITIC。CRITIC 发现了 Self-Refine 没有发现的事实错误，因为外部验证器有真实依据，而模型自批评无此保障。

## 使用场景

Anthropic 的评估器-优化器是本模式在 Claude 系生态的表达。OpenAI Agents SDK 的输出守护是 CRITIC 形态（守护可调用工具）。LangGraph 发布了类似 Self-Refine 的反思节点。Google 的 Gemini 2.5 计算机使用增添了每步的安全评估器，是 CRITIC 的变体：每个动作提交前都经过验证。

## 部署方案

`outputs/skill-refine-loop.md` 根据任务形态、验证器可用性和迭代预算配置评估器-优化器循环，生成生成器、评估器/验证器、优化器提示词及停止策略。

## 练习

1. 将最大迭代次数改为 1，运行示例。CRITIC 还能发挥作用吗？  
2. 用一个噪声较大的外部验证器（30% 假阳性）替换当前验证器。循环会怎样？这是 2026 年大多数守护堆栈的现实。  
3. 实现“不同模型生成-批评”版本：大模型生成，小模型批评。它能打败同模型方案吗？  
4. 阅读 CRITIC 第 3 节（arXiv:2305.11738 v4）。列举三类验证工具并举例说明。  
5. 将 OpenAI Agents SDK 的 `output_guardrails` 映射到 CRITIC 的验证者角色。SDK 哪些做法有偏差？哪些正确？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| Self-Refine | “能自我修正的 LLM” | 在一个模型内的生成 -> 反馈 -> 改进循环，包含历史 |
| CRITIC | “基于工具的验证” | 将反馈替换为外部验证器（搜索、代码、计算、测试） |
| Evaluator-Optimizer | “Anthropic 工作流模式” | 两个角色——评估器打分，优化器修改——循环达到收敛 |
| Output guardrail | “事后检查” | OpenAI Agents SDK 上跑在代理输出后的验证器 |
| Verify step | “批评阶段” | 关键决策：基于外部工具或模型自评分 |
| Refine history | “模型尝试记录” | 先前输出及批评拼接到改进提示；去掉历史质量大降 |
| Rubber-stamp loop | “自洽失败” | 同提示批评返回“看起来不错”，用结构不同提示解决 |
| Stop condition | “收敛测试” | 验证通过 OR 无反馈且迭代次数达标；永不单条件停止 |

## 延伸阅读

- [Madaan 等人，Self-Refine (arXiv:2303.17651)](https://arxiv.org/abs/2303.17651) — 经典论文  
- [Gou 等人，CRITIC (arXiv:2305.11738)](https://arxiv.org/abs/2305.11738) — 基于工具的验证  
- [Anthropic，Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — 评估器-优化器工作流模式  
- [OpenAI Agents SDK 文档](https://openai.github.io/openai-agents-python/) — 输出守护机制如 CRITIC 验证器
