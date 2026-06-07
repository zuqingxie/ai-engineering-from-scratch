# 代理框架取舍 — LangGraph vs CrewAI vs AutoGen vs Agno

> 每个框架都炫耀同样的演示（研究型代理生成报告），却掩盖同样的缺陷（状态 schema 与编排层冲突）。选择与问题形态相匹配的框架抽象；其他都是你得写两次的胶水代码。

**类型:** 学习  
**语言:** Python  
**先决条件:** 第11阶段 · 09（函数调用 Function Calling），第11阶段 · 16（LangGraph）  
**时间:** ~45分钟

## 问题

你有一个任务需要多次调用 LLM。可能是一个研究工作流程（计划、搜索、总结、引用）。可能是代码审查管道（解析差异、批评、修补、验证）。也可能是一个多轮助手，预订机票、写邮件、报销。你选了一个框架。

三天后，你发现该框架的抽象会泄露。CrewAI 提供了角色，但当“研究员”需要将结构化计划交给“写手”时却不配合。AutoGen 可以让代理间聊天，但没有一流的状态管理，所以你的检查点仅是一个谈话日志的 pickle。LangGraph 提供了状态图，但要求你在知道代理要做什么之前就命名每个转移。Agno 提供了单代理抽象，但当你想扇出到三个并发工作者时会抱怨。

解决办法不是“选最好的框架”，而是将框架的核心抽象和你的问题形态匹配。本课将绘制这张地图。

## 概念

![代理框架矩阵：核心抽象 vs 问题形态](../assets/framework-matrix.svg)

Four个框架主导2026年格局，它们的核心抽象并不相同。

| 框架 | 核心抽象 | 最适合 | 最不适合 |
|-----------|------------------|----------|-----------|
| **LangGraph** | `StateGraph` — 类型化状态、节点、有条件的边、检查点。 | 明确状态且有人机交互打断的工作流；需要时间旅行调试的生产代理。 | 松散、基于角色的头脑风暴，拓扑未知的情况。 |
| **CrewAI** | `Crew` — 角色（目标、背景）、任务、流程（顺序或层级）。 | 角色扮演或以人物为驱动的工作流，短线性/层级计划。 | 除角色回合历史外的任何有状态流程；复杂分支。 |
| **AutoGen** | `ConversableAgent` 对 — 两个或多个代理轮流说话直到退出条件。 | 多代理*对话*（老师-学生、提议者-批评者、演员-审核者），思考通过聊天展开。 | 有已知 DAG 的确定性工作流；跨重启需要持久化状态。 |
| **Agno** | `Agent` — 单个 LLM + 工具 + 记忆，可组合成团队。 | 构建快速的单代理和轻量团队；强多模态和内建存储驱动。 | 深度显式分支图和自定义 Reducer。 |

### “抽象”究竟是什么意思

框架的核心抽象是你在白板上画架构时必画的东西。

- **LangGraph** → 你画图。节点是步骤，边是转移，状态对象在各点都有类型。心智模型是状态机。
- **CrewAI** → 你画组织图。每个角色有职位描述，经理分派任务。心智模型是小团队专家。
- **AutoGen** → 你画 Slack 私信。两个代理互发消息；需要主持时加第三个。心智模型是聊天。
- **Agno** → 你画一个带工具的盒子。多盒子并排组成团队。心智模型是“内置电池的代理”。

### 状态问题

状态是大多数框架在生产中崩溃的关键。

- **LangGraph.** 类型化状态（`TypedDict` 或 Pydantic 模型）、字段级 reducer、一流的检查点（SQLite/Postgres/Redis）。支持续跑、中断和时间旅行。*（见第11阶段 · 16）*
- **CrewAI.** 状态作为字符串在任务间通过 `context` 字段流动，或通过 `output_pydantic` 结构化。无开箱即用的持久化每队存储；若要存活重启必须自行搭建。
- **AutoGen.** 状态是聊天历史和任意用户定义的 `context`。会话记录持久化；任意工作流状态除非写适配器，否则不保留。
- **Agno.** 内建存储驱动（SQLite、Postgres、Mongo、Redis、DynamoDB），通过 `storage=` 附加给 `Agent`，会话和用户记忆自动持久化。不是完整的图检查点；是会话存储。

### 分支问题

每个非平凡的代理都会分支。谁决定分支很重要。

- **LangGraph** — 你通过条件边决定。路由是带命名分支的 Python 函数。分支在编译图中一等公民；检查点记录分支选择。
- **CrewAI** — 管理员在层级模式下决定；顺序模式下你在构建时决定。路由隐含于任务列表；管理员的提示之外无一流“if”条件。
- **AutoGen** — 代理通过聊天决定。分支由下一个发言者自然形成。`GroupChatManager` 选下一个发言者；可以手写 `speaker_selection_method`，默认由 LLM 驱动。
- **Agno** — 代理根据调用哪个工具决定。团队有协调员/路由员/协作者模式；其他分支由开发者负责。

### 可观察性问题

- **LangGraph** — 通过 LangSmith 或任意 OTel 导出器实现 OpenTelemetry。每个节点转移都是跟踪跨度；检查点即可回放的轨迹。LangSmith 是一方选择；Langfuse/Phoenix 亦有适配器。
- **CrewAI** — 自2025年底支持一流 OpenTelemetry；集成 Langfuse、Phoenix、Opik、AgentOps。
- **AutoGen** — 通过 `autogen-core` 支持 OpenTelemetry；AgentOps 和 Opik 有连接器。追踪粒度为每条代理消息，不是节点。
- **Agno** — 内置 `monitoring=True` 标记及 OTel 导出器；与 Langfuse 深度集成以追踪会话。

### 成本和延迟

四个框架都会增加调用成本（框架逻辑、验证、序列化）。大致开销顺序：Agno ≈ LangGraph < CrewAI ≈ AutoGen。差异主要源于框架的额外 LLM 路由。CrewAI 的层级管理者在决定下一个说话者时消耗 Token；AutoGen 的 `GroupChatManager` 亦然。LangGraph 仅在你写 `llm.invoke` 时消耗 Token。Agno 的单代理路径最轻。

当运行成本重要时，建议优先显式路由（LangGraph 边，AutoGen `speaker_selection_method`），而非 LLM 自动选择路由。

### 互操作性

- **LangGraph** ↔ **LangChain** 工具、检索器、LLM。内置 MCP 适配器（工具作为 MCP 服务器导入）。
- **CrewAI** ↔ 工具继承自 `BaseTool`；LangChain 工具、LlamaIndex 工具和 MCP 工具都可适配。支持队间委派 `allow_delegation=True`。
- **AutoGen** → `FunctionTool` 包装任意 Python 可调用对象；具备 MCP 适配器。紧密耦合 AG2 生态的代理间模式。
- **Agno** → `@tool` 装饰器或 `BaseTool` 子类；MCP 适配器；工具可在代理和团队间共享。

## 技能

> 你能用一句话说明为什么某框架适合某个代理问题。

预构建清单：

1. **画形态。** 是图（类型化状态、具名转移）？角色扮演（专家交接）？聊天（代理对话）？单代理带工具？
2. **决定谁分支。** 开发者决策 → LangGraph。管理员-代理决策 → CrewAI层级。聊天自然生成 → AutoGen。工具调用决策 → Agno。
3. **看状态预算。** 需要检查点续跑？时间旅行？人机中断？需要的话优先 LangGraph；Agno 会话涵盖对话范围状态。
4. **看成本预算。** LLM自动路由每步额外消耗 Token。高频运行推荐显式路由。
5. **预算框架开销。** 每个框架都是依赖。任务仅两次 LLM 调用外加工具时，写30行纯 Python 往往比用框架更廉价。

未能画出图、组织图、聊天、代理盒子之前，拒绝随便选框架。拒绝使用强迫你对抗其状态模型的框架。

## 决策矩阵

| 问题形态 | 首选框架 | 原因 |
|---------------|---------------------|-----|
| 具有类型化状态、人类批准、长时间运行的工作流 DAG | LangGraph | 一流状态、检查点、中断与时间旅行功能。 |
| 具有明确角色的研究/写作管道 | CrewAI（顺序流程）或 LangGraph 子图 | CrewAI 可简洁表达每任务角色；分支复杂时可升级 LangGraph。 |
| 提议者-批评者或师生对话 | AutoGen | 双代理聊天是其原生形态。 |
| 单代理带工具、会话与记忆 | Agno | 最轻量，内置存储和记忆。 |
| 数千并行扇出且带 Reducer | LangGraph + `Send` | 唯一一款支持一流并行调度 API 的框架。 |
| 快速原型，无框架束缚 | 纯 Python + 提供者 SDK | 没有框架才是最快框架。 |

## 练习

1. **简单。** 用 LangGraph（四节点：计划、搜索、写作、引用）和 CrewAI（三角色：研究员、写手、编辑）实现“研究 Anthropic 总部，写一份200字简报，引用来源”任务。报告每次运行的 Token 成本和代码行数。
2. **中等。** 分别用 AutoGen（研究员 ↔ 写手聊天，编辑通过 `GroupChat` 加入）和 Agno（单代理，带 `search_tools` 和 `write_tools`，及会话存储）实现同任务。对四种实现按 (a) 每次成本，(b) 崩溃续跑能力，(c) 写作前插入人工审批能力排序。
3. **困难。** 编写决策树脚本 `pick_framework.py`，接受简短问题描述（JSON: `{has_typed_state, has_roles, has_dialogue, has_parallel_fanout, needs_resume}`），返回推荐框架及一句话理由。自行设计六个测试案例验证。

## 关键词

| 术语 | 常说说法 | 实际含义 |
|------|-----------------|-----------------------|
| Orchestration（编排） | “代理如何协调” | 决定哪个节点/角色/代理接着运行的层。 |
| Durable state（持久状态） | “重启后续跑” | 生存进程终止的状态，附着于检查点或会话存储。 |
| LLM-selected routing（LLM选路） | “让模型决定” | 由规划 LLM 每步选择下一步；灵活但每次决策付费 Token。 |
| Explicit routing（显式路由） | “开发者决定” | 静态边或 Python 函数选择下一步；便宜且可审计。 |
| Crew（团队） | “CrewAI 团队” | 角色 + 任务 + 流程（顺序或层级）组成的单一可执行体。 |
| GroupChat（群聊） | “AutoGen 的多代理聊天” | N 个代理管理式对话和发言者选择器。 |
| Team (Agno)（团队） | “多代理 Agno” | 一组代理的路由/协调/协作模式。 |
| StateGraph（状态图） | “LangGraph 的图” | 类型化状态、节点、有条件边、检查点抽象。 |

## 深入阅读

- [LangGraph 文档](https://langchain-ai.github.io/langgraph/) — StateGraph（状态图）、checkpointers（检查点器）、interrupts（中断）、time-travel（时间旅行）。
- [CrewAI 文档](https://docs.crewai.com/) — Crews（团队）、Flows（流程）、Agents（代理）、Tasks（任务）、Processes（流程）。
- [AutoGen 文档](https://microsoft.github.io/autogen/) — ConversableAgent（可对话代理）、GroupChat（群聊）、teams（团队）、tools（工具）。
- [Agno 文档](https://docs.agno.com/) — Agent（代理）、Team（团队）、Workflow（工作流）、storage（存储）、memory（记忆）。
- [Anthropic — 构建有效代理（2024年12月）](https://www.anthropic.com/research/building-effective-agents) — 模式库（prompt chaining（提示链）、routing（路由）、parallelization（并行）、orchestrator-workers（协调器-工作者）、evaluator-optimizer（评估器-优化器）），与框架无关。
- [Yao 等人，“ReAct: Synergizing Reasoning and Acting”（ICLR 2023）](https://arxiv.org/abs/2210.03629) — 各框架中的标准循环模型。
- [Wu 等人，“AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation”（2023）](https://arxiv.org/abs/2308.08155) — AutoGen 的设计论文。
- [Park 等人，“Generative Agents: Interactive Simulacra of Human Behavior”（UIST 2023）](https://arxiv.org/abs/2304.03442) — CrewAI 风格角色栈构建的角色扮演基础。
- Phase 11 · 16 (LangGraph) — 本课程评测的框架。
- Phase 11 · 19 (Reflexion) — 与 LangGraph 匹配良好但与 CrewAI 不太契合的模式。
- Phase 11 · 22 (Production observability（生产环境可观察性）) — 如何为你选择的任何框架进行监控和度量。
