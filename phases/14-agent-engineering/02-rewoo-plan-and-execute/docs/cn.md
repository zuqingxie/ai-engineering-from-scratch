# ReWOO 和 Plan-and-Execute：解耦规划（Decoupled Planning）

> ReAct 在一个流程中交织思考和行动。ReWOO 将两者分开：事先做一个完整的大计划，然后执行。令牌数减少 5 倍，HotpotQA 准确率提升 4%，且规划器可以蒸馏到一个 7B 模型中。Plan-and-Execute 对其进行了泛化；Plan-and-Act 将其扩展到网页导航。

**类型：** 构建  
**语言：** Python（标准库）  
**先决条件：** 第14阶段 · 01（智能体循环）  
**时长：** 约60分钟

## 学习目标

- 解释为什么 ReWOO 的 Planner（规划器）/ Worker（工作器）/ Solver（求解器）拆分，比 ReAct 的交织循环节省令牌并提升鲁棒性。
- 实现计划 DAG（有向无环图）、依赖顺序执行器和组合 Worker 输出的求解器——全部基于标准库。
- 使用 2026 年的 “五种工作流模式” 框架（Anthropic）决定何时使用先规划后执行 vs 交织式 ReAct。
- 识别何时需要 Plan-and-Act 的合成计划数据，适用于长时序网页或移动任务。

## 问题描述

ReAct 的交织思考-行动-观察循环简单灵活，但每次调用工具都要携带完整的前文上下文——包括所有之前的思考。令牌使用量随深度呈二次增长。更糟的是：当工具在循环中失败时，模型必须从错误观察中重新推导整个计划。

ReWOO（Xu 等，arXiv:2305.18323，2023年5月）发现了这一点，并做出一个赌注：预先计划好全部过程，证据并行获取，最后合成答案。一次 LLM 调用出计划，N 次工具调用获取证据（可并行），一次 LLM 调用求解。代价是灵活性降低（计划是静态的），但令牌效率大幅提升，失败模式更清晰。

## 概念介绍

### 三个角色

```text
Planner:  user_question -> [plan_dag]
Workers:  [plan_dag]     -> [evidence]        (工具调用，可并行)
Solver:   user_question, plan_dag, evidence -> final_answer
```

Planner 生成一个 DAG。每个节点表示一个工具、其参数及依赖的先前节点（引用如 `#E1`, `#E2`）。Workers 按拓扑顺序执行节点。Solver 负责将所有证据拼接生成最终答案。

### 为什么令牌减少5倍

ReAct 随步骤数线性增长提示长度。第10步时，提示包含思考1、行动1、观察1，思考2、行动2、观察2，如此往复。每个中间步骤也重复包含原始提示。

ReWOO 只需要一次大的规划提示，N 次小的 Worker 提示（只包含工具调用，无链上下文），和一次求解器提示。论文中 HotpotQA 评测显示令牌数减少约5倍，同时准确率绝对提升4%。

### 为什么更鲁棒

ReAct 中 Worker 3 失败时，循环必须从错误观察中重新推理。ReWOO 中 Worker 3 返回错误字符串，求解器可结合原始计划上下文优雅降级。失败定位变为按节点而非按步骤。

### 规划器蒸馏

论文的第二个结果是：由于规划器不看到观察，能用175B大模型生成的规划器输出微调7B小模型。小模型负责规划，推理时无须大模型。该方法已成2026年标准，许多生产智能体用小规划器配大执行器或反之。

### Plan-and-Execute（LangChain，2023）

LangChain 团队2023年8月将 ReWOO 泛化为模式名：Plan-and-Execute。规划器先给出步骤列表，执行器逐步执行，支持可选的重新规划器根据观察结果修正计划。其更接近 ReAct（重新规划器使观察回流规划）但保持令牌节省。

### Plan-and-Act（Erdogan 等，arXiv:2503.09572，ICML 2025）

Plan-and-Act 将该模式扩展到长时序网页和移动代理。关键贡献是合成计划数据：通过标签轨迹生成器构造计划明显的训练数据。用于微调规划器模型，使其在 WebArena 类似任务中超过30~50步仍保持连贯，而单一 ReAct 轨迹则会失效。

### 何时选用哪种方法

| 模式             | 适用场景                         |
|------------------|---------------------------------|
| ReAct            | 短任务，环境未知，需要响应式异常处理 |
| ReWOO            | 已知工具的结构化任务，令牌敏感，证据可并行 |
| Plan-and-Execute | 类似 ReWOO，但执行后可重新规划       |
| Plan-and-Act     | 长时序（>30步）、网页/移动/计算机使用 |
| Tree of Thoughts  | 搜索值得付出代价（第04课）          |

Anthropic 2024年12月建议：从最简单模式开始。如果任务仅一次工具调用加摘要，不要构建 ReWOO。如果是40步的调研任务，不宜只用 ReAct。

## 构建它

`code/main.py` 实现了一个演示版 ReWOO：

- `Planner` — 一个脚本策略，从提示生成计划 DAG。
- `Worker` — 通过注册表调度每个节点的工具调用。
- `Solver` — 脚本组合，读取证据并产出最终答案。
- 依赖解析 — 引用如 `#E1` 替换为先前 Worker 输出。

演示回答“法国首都的人口（千万为单位四舍五入）？”该计划包括两步：（1）查找首都，（2）查找人口，然后求解。

运行：

```bash
python3 code/main.py
```

跟踪输出显示先打印完整计划，再打印 Worker 结果，最后打印求解器输出。对比 ReAct 式交织执行的令牌数（我们打印粗略字符数），ReWOO 在这类结构化任务中更优。

## 使用它

LangGraph 提供 Plan-and-Execute 作为一个配方（`create_react_agent` 用于 ReAct，自定义图用于计划执行）。CrewAI 的 Flows 直接编码该模式：你事先定义任务，Flow DAG 执行它们。Plan-and-Act 的合成数据方法仍处于研究阶段；运行时模式（显式计划 DAG）已通过 LangGraph 和 CrewAI Flows 投产。

## 发布它

`outputs/skill-rewoo-planner.md` 根据用户请求和工具目录生成 ReWOO 计划 DAG。它会校验计划（无环，每个引用可解析，每个工具存在）后交给执行器。

## 练习

1. 为独立计划节点并行执行 Worker。在6节点DAG中有2组并行时，能带来什么收益？
2. 添加一个在任何 Worker 返回错误时触发的重新规划节点。对 ReWOO 做最小变动，使其变为 Plan-and-Execute？
3. 用小模型（7B 类）替换 `Planner`，保持 `Solver` 使用前沿模型。比较端到端效果，拆分机制在哪些情况下会失败？
4. 阅读 ReWOO 论文第4节关于规划器蒸馏。概念重现175B -> 7B结果：你需要什么训练数据，如何评估计划质量？
5. 将演示程序改为 Plan-and-Act 的轨迹形态：计划是序列而非 DAG。会带来哪些权衡变化？

## 关键词

| 术语             | 人们口语中的说法                  | 实际含义                                               |
|------------------|---------------------------------|--------------------------------------------------------|
| ReWOO            | “Reasoning without observations” | 先规划、证据并行获取、后求解——规划提示中无观察              |
| Plan-and-Execute | “LangChain 的 plan-execute 模式”  | 带执行后可选重新规划节点的 ReWOO                            |
| Plan-and-Act     | “扩展版 plan-execute”             | 显式规划器/执行器拆分，长时序任务用合成计划训练数据                |
| 证据引用         | “#E1, #E2, ...”                   | 计划节点占位符，调度时替换为之前 Worker 输出                        |
| 规划器蒸馏       | “小规划器，大执行器”               | 用大规模教师生成规划轨迹微调小模型                               |
| 令牌效率         | “减少往返次数”                    | 论文中相较 ReAct HotpotQA 减少5倍令牌                            |
| DAG 执行器       | “拓扑调度器”                      | 按依赖顺序执行计划节点，层级内可并行                                |

## 延伸阅读

- [Xu 等，ReWOO：Decoupling Reasoning from Observations (arXiv:2305.18323)](https://arxiv.org/abs/2305.18323) — 权威论文  
- [Erdogan 等，Plan-and-Act (arXiv:2503.09572)](https://arxiv.org/abs/2503.09572) — 带合成计划的规模化规划执行器  
- [LangGraph Plan-and-Execute 教程](https://docs.langchain.com/oss/python/langgraph/overview) — 框架配方  
- [Anthropic，Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — 选择最简单可行模式
