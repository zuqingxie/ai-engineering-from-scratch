# Anthropic 的工作流模式：简单胜于复杂

> Schluntz 和 Zhang（Anthropic，2024 年 12 月）区分了工作流（预定义路径）与代理（动态工具使用）。五种工作流模式涵盖了大多数情况。先从直接 API 调用开始，只有当步骤无法预测时才添加代理。

**类型：** 学习 + 构建  
**语言：** Python（stdlib）  
**先决条件：** 第 14 阶段 · 01（代理循环）  
**时间：** 约 60 分钟

## 学习目标

- 了解 Anthropic 的五种工作流模式：prompt chaining（提示链）、routing（路由）、parallelization（并行化）、orchestrator-workers（协调者-工作者）、evaluator-optimizer（评估者-优化器）。
- 解释代理与工作流的区别及各自的工程代价。
- 辨别何时选择工作流而非代理（反之亦然）。
- 在 stdlib 中针对脚本化 LLM 实现所有五种模式。

## 问题

团队为单一函数调用的问题却求助于多代理框架。其代价真实存在：框架增加层次，掩盖提示内容，隐藏控制流，引入过早复杂度。Schluntz 和 Zhang 2024 年 12 月的文章对此是业界引用最多的反驳：先从简单做起，复杂度仅在其价值明显时才增加。

## 概念

### 工作流 vs 代理

- **工作流（Workflow）**。通过预定义代码路径协调 LLM 和工具。工程师掌控图结构。
- **代理（Agent）**。LLM 动态指挥自身工具并自主执行步骤。模型掌控图结构。

两者各有所长。工作流更便宜、更快且更易调试。代理能处理开放式问题，但失败模式更难理解。

### 增强型 LLM

所有五种模式的基础：一个 LLM 集成了三项能力 — 搜索（检索）、工具（动作）、记忆（持久化）。任何 API 调用均可利用。

### 五种模式

1. **提示链（Prompt chaining）。** 调用 1 的输出作为调用 2 的输入。适用于任务有明确线性分解的情况。步骤之间可选编程门控。
   
2. **路由（Routing）。** 一个分类器 LLM 选择调用下游 LLM 或工具。适用于分类不同输入需要不同处理场景（一级支持、退款、Bug、销售等）。

3. **并行化（Parallelization）。** 并发运行 N 个 LLM 调用，汇总结果。两种形式：分段（不同部分）和平票（相同提示多次调用，多数/合成结果）。

4. **协调者-工作者（Orchestrator-workers）。** 一个协调者 LLM 动态决定调度哪些工作者（同样是 LLM），并综合其输出。类似代理循环，但协调者不无限循环。

5. **评估者-优化器（Evaluator-optimizer）。** 一个 LLM 提出答案，另一个 LLM 评估，迭代直至评估通过。这是自我优化（Self-Refine，课程 05）的通用形式。

### 工作流胜过代理的情况

- **可预测的任务。** 如果步骤可枚举，就应采用。
- **受成本限制的任务。** 工作流步骤数有限，代理可能无限循环。
- **受合规限制的任务。** 审计人员希望查看图结构，而非从轨迹中推测。

### 代理胜过工作流的情况

- **开放式研究。** 后续步骤依赖于上一步结果。
- **变长任务。** 持续数分钟至数小时，步骤数未知。
- **新颖领域。** 尚未确定正确工作流，先探索后固化。

### 上下文工程伴侣

“AI 代理的有效上下文工程”（Anthropic 2025）规范了相邻学科：200k 窗口是预算，不是容器。何时包含、何时压缩、何时让上下文增长。详见第 14 阶段关于上下文压缩（第 14 阶段课程 06）的内容。

## 构建它

`code/main.py` 针对 `ScriptedLLM` 实现了所有五种工作流模式：

- `prompt_chain(input, steps)` — 顺序调用。
- `route(input, classifier, handlers)` — 分类 + 分发。
- `parallel_vote(prompt, n, aggregator)` — 并行 N 次运行，结果聚合。
- `orchestrator_workers(task, workers)` — 协调者选择工作者。
- `evaluator_optimizer(task, proposer, evaluator, max_iter)` — 循环直到通过。

运行：

```bash
python3 code/main.py
```

每种模式打印其执行轨迹。每种模式代码行数约 10-15 行；框架成本通常以千计。

## 使用它

- 大多数任务直接调用 API。
- 仅当模式确实需要持久状态（LangGraph）、角色模型并发（AutoGen v0.4）或角色模板化（CrewAI）时使用框架。
- 需要 Claude Code 执行环境但不想重建时，选用 Claude Agent SDK。

## 发布它

`outputs/skill-workflow-picker.md` 针对给定任务描述选择正确模式，包含决策理由以及工作流不足时重构为代理的路径。

## 练习

1. 实现带置信度阈值的路由。低于阈值转人工升级。一级支持用例的阈值在哪？
2. 给 `parallel_vote` 增加超时。若一个调用挂起，结果如何？缺失投票时如何聚合？
3. 将 `evaluator_optimizer` 改为多臂赌博机：保留各迭代中得分最高的两个结果，以免后期好结果被差结果覆盖。
4. 结合提示链和路由：路由器选三条链之一。比较代币消耗与单一大提示的差异。
5. 任选生产特性，绘制工作流图，统计步骤数。这里代理会更合适吗？

## 关键词

| 术语 | 说法 | 实际含义 |
|------|------|----------|
| Workflow（工作流） | “预定义流” | 工程师掌控的 LLM 和工具调用图 |
| Agent（代理） | “自治 AI” | 模型掌控图；动态指令工具 |
| Augmented LLM（增强型 LLM） | “带工具的 LLM” | LLM + 搜索 + 工具 + 记忆；基本单元 |
| Prompt chaining（提示链） | “顺序调用” | 第 N 次调用输出为第 N+1 次输入 |
| Routing（路由） | “分类器分发” | 选择哪个链/模型处理输入 |
| Parallelization（并行化） | “分发调用” | N 个并行调用；通过分段或投票聚合 |
| Orchestrator-workers（协调者-工作者） | “调度代理” | 协调者动态选用专业 LLM |
| Evaluator-optimizer（评估者-优化器） | “提出者+评判者” | 迭代直到评估者通过；自我优化通用化 |

## 拓展阅读

- [Anthropic，Building Effective Agents（2024 年 12 月）](https://www.anthropic.com/research/building-effective-agents) — 五种工作流模式  
- [Anthropic，Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) — 配套学科  
- [LangGraph 概述](https://docs.langchain.com/oss/python/langgraph/overview) — 何时有状态图值得其成本  
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) — 产品化的协调者-工作者模式
