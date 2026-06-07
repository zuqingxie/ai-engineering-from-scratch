# CrewAI：基于角色的团队和流程

> CrewAI 是 2026 年的基于角色的多智能体框架。四个基本原语：Agent（智能体）、Task（任务）、Crew（团队）、Process（流程）。两个顶层形态：Crews（自治的基于角色的协作）和 Flows（事件驱动的确定性流水线）。文档直言不讳：“对于任何生产级应用，从 Flow 开始。”

**类型：** 学习 + 构建  
**语言：** Python（stdlib）  
**先决条件：** 第14阶段·12（工作流模式），第14阶段·14（Actor模型）  
**时间：** 约75分钟

## 学习目标

- 说出 CrewAI 的四个基本原语（Agent、Task、Crew、Process）及其职责。
- 区分 Sequential（顺序）、Hierarchical（分层）和计划中的 Consensus（共识）流程；根据工作负载选择合适流程。
- 区分基于角色自治的 Crews 与事件驱动确定性的 Flows，解释文档中生产推荐的原因。
- 使用 `@tool` 装饰器和 `BaseTool` 子类连接工具；理解结构化输出与自由文本的区别。
- 说出 CrewAI 的四种内存类型及各自适用场景。
- 实现一个标准库的三智能体团队（研究员、写手、编辑），产出一份简报。
- 识别三种 CrewAI 失败模式：prompt 膨胀、管理者 LLM 额外开销、脆弱的任务交接。

## 问题背景

采用多智能体框架的团队常会遇到同样的问题。“自治协作”听起来很棒，但实际中客户报 BUG，你就需要确定性的重放；财务问每次运行的成本是多少；或值班人员需要知道凌晨三点哪个智能体卡住了。

自由形式的 LLM 路由团队无法清晰回答这些问题。纯 DAGs 都能回答，但失去了脑力激荡智能体所需的探索性形态。

CrewAI 的划分坦诚地说明了权衡。Crews 用于协作、基于角色、探索性的工作。Flows 用于事件驱动、代码拥有、可审计的生产。相同框架，两种形态，根据场景选择。

## 概念详解

### 四大基本原语

CrewAI 的表层很小，牢记这些，剩下的都是配置。

- **Agent.** `role + goal + backstory + tools + (optional) llm`。backstory（背景故事）承重，塑造语气、判断以及智能体的终止条件。tools 是智能体能调用的函数（以下详述）。
- **Task.** `description + expected_output + agent + (optional) context + (optional) output_pydantic`。可复用的工作单元。`expected_output` 是合约。`context` 列出上游任务其输出传入本任务。`output_pydantic` 强制输出结构化。
- **Crew.** 容器。拥有 `agents` 列表，`tasks` 列表，`process`，以及可选的 `memory` + `verbose` + `manager_llm` 设置。
- **Process.** 执行策略。顺序（Sequential）、分层（Hierarchical）、共识（Consensus，计划中）。决定执行时的形态。

智能体不直接见到彼此，任务引用智能体，团队串联任务，流程决定谁挑下一个任务。这就是全部心智模型。

> **验证于** CrewAI 0.86（2026-05）。新版可能重命名或合并流程类型；依赖具体形态前务必查阅 [CrewAI 流程文档](https://docs.crewai.com/concepts/processes)。

### 顺序（Sequential） vs 分层（Hierarchical） vs 共识（Consensus）

- **顺序（Sequential）**。任务按声明顺序运行。任务 N 的输出作为 `context` 提供给任务 N+1。成本最低，最可预测。当顺序固定时使用。
- **分层（Hierarchical）**。一个管理者智能体（独立 LLM 调用）在专家间路由。CrewAI 从你的 `manager_llm` 配置或默认生成管理者。每轮管理者选下一个任务，可拒绝或重新路由。当有四个或更多专家且任务顺序真实依赖前置输出时使用。
- **共识（Consensus）**。计划中，目前公共 API 未实现。文档预留该名称作为未来基于投票的流程。当前勿依赖。

分层会在每个专家调用上叠加管理者 LLM 调用，令令牌成本在五步运行中可能翻三倍。仅当路由必需时才使用。

### Crews vs Flows

这是 2026 年文档开篇的框架视角。

- **Crew（团队）**。LLM 驱动的自治。框架运行时选形态。适合研究、头脑风暴、初稿等路径本身即答案的场景。难重放，难测试，原型开发成本低。
- **Flow（流程）**。自主拥有的事件驱动图。用 `@start` 标记入口。`@listen(topic)` 标记当步骤发出该主题时触发的下一步。每步纯 Python（内部可调用 Crew）。适合生产。可观测。可测试。确定性强。

2026 年文档生产建议：从 Flow 开始开发。当自治值回成本时，在 Flow 步骤里通过 `Crew.kickoff()` 调用嵌入 Crews。Flow 提供审计轨迹，Crew 提供探索性。组合而非选择。

### 工具集成

给智能体提供工具有三种方式，选最简单合适的。

1. **`@tool` 装饰器。** 纯函数变工具。签名定义 schema，文档字符串是 LLM 看到的描述。适合一次性助手。

   ```python
   from crewai.tools import tool

   @tool("Search the web")
   def search(query: str) -> str:
       """返回查询的顶部结果。"""
       return run_search(query)
   ```

2. **`BaseTool` 子类。** 类工具，显式声明参数 schema，支持异步和重试。当工具有状态（客户端、缓存）或需要结构化参数时使用。

   ```python
   from crewai.tools import BaseTool
   from pydantic import BaseModel

   class SearchArgs(BaseModel):
       query: str
       limit: int = 10

   class SearchTool(BaseTool):
       name = "web_search"
       description = "搜索网页并返回顶部结果。"
       args_schema = SearchArgs

       def _run(self, query: str, limit: int = 10) -> str:
           return self.client.search(query, limit=limit)
   ```

3. **内置工具包。** CrewAI 提供一方适配器：`SerperDevTool`、`FileReadTool`、`DirectoryReadTool`、`CodeInterpreterTool`、`RagTool`、`WebsiteSearchTool`。只需一次导入即可使用。

结构化输出用 Pydantic。任务中传递 `output_pydantic=MyModel`，CrewAI 校验 LLM 响应并执行类型强制或重试。与严格的 `expected_output` 字符串搭配。自由文本适合草稿，结构化输出是下游 Flow 可消费的格式。

### 内存挂钩

CrewAI 自带四种内存类型，支持组合：一个团队可同时启用四种内存。

> **验证于** CrewAI 0.86（2026-05）。新增版本通过统一的 `Memory` 系统封装四种存储。下面概念仍有效，但公共类接口可能简化成单一 `Memory` 出口；查看 [CrewAI 内存文档](https://docs.crewai.com/concepts/memory) 了解最新 API。

- **短期（Short-term）**。单次运行的对话缓冲。运行结束清空。
- **长期（Long-term）**。跨运行持久化。存储在向量数据库（默认 Chroma，可替换）。基于当前任务相似度检索。
- **实体（Entity）**。每实体事实。“客户 X 是企业版用户”。按实体键索引，不按相似度。跨运行持续。
- **上下文（Contextual）**。组装时检索。智能体需要时拉取关联记忆，非预加载。

通过 `memory=True` 或按类型配置启用。后端用你配置的向量嵌入提供者（默认 OpenAI，可替换成本地）。内存是 CrewAI 对比更轻量框架的明显优势；纯 LangGraph 需用户自己组装这些。

### 适用场景

- 三到六个有命名角色的智能体协作工作流。撰稿、审阅、计划、头脑风暴。
- LLM 判断下一步路由是价值关键（分层流程）。
- 团队更偏好阅读 `role + goal + backstory` 而非图形定义。

### 不适用场景

- 确定性的 DAG，严格排序。用 LangGraph（第13课）。图形形态是正确抽象，CrewAI 的角色框架反而增加摩擦。
- 亚秒级延迟预算。分层流程加多轮往返。顺序流程也串行含背景及前置输出的提示。
- 单智能体循环。跳过框架；一个 Agent 循环（第1课）加工具注册更简单。

第17课（智能体框架权衡）有完整矩阵。简言之：CrewAI 处于“协作基于角色”象限。

### 依赖形态

独立于 LangChain。支持 Python 3.10 到 3.13，用了 `uv`。Star 数见 [crewAIInc/crewAI](https://github.com/crewAIInc/crewAI)（2026-05 快照）。AWS Bedrock 集成有文档支持；厂商基准表示对比 LangGraph 在 QA 任务上显著加速，因缺少公开方法论（数据集、硬件、评估指标），框架厂商数字仅供参考。

### 常见失误

- **背景故事导致提示膨胀。** 五个智能体，每个两千字背景故事，首个工具调用前上下文就耗尽预算。背景故事控制在 200 字以内。跨智能体复用句式，避免五遍重复风格。
- **管理者 LLM 令牌税。** 分层流程在每个专家调用前额外一次管理者 LLM 调用。五任务团队变成六次调用，管理者调用需全任务列表和之前输出。若非必需路由，转顺序流程。
- **脆弱交接。** 任务 N 的 `expected_output` 是“大纲”，任务 N+1 作为 `context` 读取预期三部分，但 LLM 生成了四部分，后续智能体即席发挥。任务 N 使用 `output_pydantic` 强制类型，任务 N+1 读取结构化而非自由文本解决。
- **Crew 用作生产。** 没有 Flow 包装的自由形式团队投入生产，输出变异大，不能重放，值班无法对比好坏。务必用 Flow 包裹。

## 构建示例

`code/main.py` 实现两种形态的标准库版本和一个三智能体团队。

结构：

- `Agent`、`Task` 数据类，匹配 CrewAI 基础。
- `SequentialCrew.kickoff(inputs)` 按声明顺序运行任务，输出串联传递为 `context`。
- `HierarchicalCrew.kickoff(topic)` 增加管理者智能体轮选专家，遇“done”停止。
- 支持 `Flow`、`@start` 和 `@listen(topic)` 装饰、简易事件循环、跟踪追踪。
- `tool(name)` 装饰器仿照 CrewAI 的 `@tool`。
- `Memory` 包含短期、长期、实体存储；模拟相似度用 numpy。
- LLM 响应硬编码为角色加输入前缀键，离线且确定。

演示：研究员、写手、编辑团队，产出关于“agent engineering 2026”的简报。研究员拉取（模拟）来源，写手起草，编辑润色。用 Flow 运行同样团队，显现确定性形态。

运行：

```bash
python3 code/main.py
```

追踪涵盖：顺序团队通过 `context` 串联输出，分层团队管理者调度（研究员、写手、编辑，后“done”），Flow 按显式主题(`researched`、`drafted`、`edited`)运行三步，工具经 `@tool` 调用，长期内存两次启动间持续。

Crew trace 是流动式的；manager（管理者）原则上可以重新排序。Flow trace 是固定的。这个选择就是教训。

## 使用方法

- **CrewAI Flow** 用于生产环境。即使 Flow 只有一步调用 `Crew.kickoff()`。Flow 提供了审计边界。
- **CrewAI Crew（顺序式）** 用于明确定序的协作工作，尤其是初稿和审阅循环。
- **CrewAI Crew（层级式）** 当路由取决于输出且有四名或更多专家时使用。
- **LangGraph**（第13课）用于显式状态机、持久化恢复、严格排序。
- **AutoGen v0.4**（第14课）用于 actor-model（行为者模型）并发和故障隔离。
- **OpenAI Agents SDK**（第16课）用于以 OpenAI 为先的产品，支持任务交接和防护机制。
- **Claude Agent SDK**（第17课）用于以 Claude 为先的产品，支持子代理和会话存储。

## 部署

`outputs/skill-crew-or-flow.md` 根据任务选择 Crew 还是 Flow，并搭建最简实现。严格拒绝无背景故事的 Crew，无明确主题的 Flow，以及专家不足三人的层级式。

## 陷阱

- **背景故事作为风格。** 它塑造输出。每个代理测试三个变体；差异真实存在。选择一个，固定它。
- **跳过 `expected_output`。** 没有每个任务的契约，下游任务将获取任意 LLM（大语言模型）产生的内容。Crew 运行，审计失败。
- **内存始终开启。** 长期写入每次运行。向量数据库膨胀。检索变得嘈杂。限定写入范围于事实持久的任务。
- **管理者提示漂移。** 层级式的管理者提示是隐式的。路由异常时，开启详细模式打印并阅读提示内容。
- **工具副作用在 Crew 里。** Crew 可能调用工具超出预期次数。POST、DELETE、支付只能放在 Flow 步骤里，绝不放在 Crew 工具。

## 练习

1. 把顺序式 Crew 转换成 Flow。统计变异点数量。标记可读性下降的地方。
2. 给 Crew 增加实体内存：客户事实跨多次 kickoff 持续。验证检索能拉取正确实体。
3. 实现一个层级流程，其中管理者拒绝将任务路由给编辑，直到写作者输出不少于三段。跟踪重试过程。
4. 为 `BaseTool` 子类绑定（模拟的）网页搜索。比较 trace 形态与 `@tool` 装饰器版本的区别。
5. 给编辑任务添加 `output_pydantic=Brief`，其中 `Brief` 含有 `title`、`summary`、`sections`。让写作者任务输出一次格式错误的 JSON；验证 CrewAI 追踪中的重试行为。
6. 阅读 CrewAI 文档介绍。把玩具移植到真实的 `crewai` API。标记标准库版本缺失的保证。
7. 用 AgentOps 或 Langfuse（第24课）连接实际运行。标记标准库版本缺失的 trace。

## 关键词

| 术语 | 通常说法 | 实际含义 |
|------|----------|----------|
| Agent | “角色（persona）” | 角色 + 目标 + 背景故事 + 工具 |
| Task | “工作单元” | 任务描述 + 期望输出 + 受让人 + 可选结构化输出 |
| Crew | “代理团队” | Agent、Task、Process（流程）的容器 |
| Process | “执行策略” | 顺序式 / 层级式 / 一致式（计划中） |
| Flow | “确定性工作流” | 事件驱动、代码掌控、可测试 |
| Backstory | “角色提示” | 语气和判断塑造代理 |
| `@tool` | “函数工具” | 把函数装饰成代理可调用的工具 |
| `BaseTool` | “类工具” | 基于类的工具，带参数模式、重试、异步支持 |
| Entity memory | “按实体记忆” | 针对客户／账户／问题的内存 |
| Long-term memory | “跨次运行记忆” | 由向量支撑的持久内存 |
| Contextual memory | “即时检索记忆” | 代理需要时拉取的内存 |
| Manager LLM | “路由代理” | 层级流程中选择下一任务的额外 LLM |
| `expected_output` | “任务契约” | 告诉代理（和审计）返回数据形态的字符串 |

## 拓展阅读

- [CrewAI 文档简介](https://docs.crewai.com/en/introduction)：概念及推荐生产路径
- [CrewAI Flows 指南](https://docs.crewai.com/en/concepts/flows)：事件驱动形态，`@start`，`@listen`
- [CrewAI 工具参考](https://docs.crewai.com/en/concepts/tools)：`@tool`，`BaseTool`，内置工具包
- [CrewAI 内存](https://docs.crewai.com/en/concepts/memory)：短期、长期、实体、上下文
- [Anthropic，打造高效代理](https://www.anthropic.com/research/building-effective-agents)：多代理何时有助，何时无用
- [LangGraph 概览](https://docs.langchain.com/oss/python/langgraph/overview)：状态机的替代方案
