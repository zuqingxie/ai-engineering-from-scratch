# LangGraph — 代理的状态机

> 手写的 ReAct 循环是一个 `while True`。用 LangGraph 写的 ReAct 循环是一个你可以保存检查点、中断、分支和时空穿梭的图。代理没变，外围框架变了。

**类型：** 构建  
**语言：** Python  
**先决条件：** 阶段 11 · 09（函数调用）、阶段 11 · 14（模型上下文协议）  
**时间：** 约 75 分钟

## 问题

你发布了一个函数调用（function-calling）代理。它运行三轮正常，接着出现问题：模型尝试调用一个返回 500 的工具，用户在任务中途改变主意，或者代理在无人审核的情况下决定退款。`while True:` 循环没有钩子。你不能暂停它，不能倒带它，也不能分支成“如果模型选了另一个工具会怎样。” 一旦你把这个环节推过演示，代理就成了一个黑箱，要么成功，要么失败。

下一步显而易见，代理本质上已经是一个状态机——系统提示加消息历史，加上待处理的工具调用和下一步动作。把状态机显式化：用节点表示“模型思考”、“工具运行”、“人工审核”，以及它们之间的条件转移边。图一旦显式化，外围框架自动获得四大能力：检查点（在步骤间保存状态）、中断（暂停等待人工）、流式输出（推送 token 和中间事件）、时空穿梭（倒转到之前的状态尝试不同分支）。

LangGraph 是提供这种抽象的库。它不是 LangChain 意义上的 Agent 框架（“这里有个 AgentExecutor，祝你好运”），它是一个拥有一等状态、一等持久化和一等中断支持的图执行引擎。代理循环是你绘制的，而不是手写的。

## 概念

![LangGraph StateGraph: nodes, edges, and the checkpointer](../assets/langgraph-stategraph.svg)

`StateGraph` 有三样东西：

1. **状态（State）。** 一个类型化字典（TypedDict 或 Pydantic 模型），在图中流动。每个节点都接收完整状态，返回部分更新，LangGraph 用每字段定义的 *reducer* 对更新做合并——列表用 `operator.add` 做累加，否则默认覆盖。
2. **节点（Nodes）。** Python 函数，形式为 `state -> partial_state`。每个节点是一个离散步骤：“调用模型”、“运行工具”、“总结”。
3. **边（Edges）。** 节点间转移。静态边唯一去向。条件边使用路由器函数 `state -> next_node_name` 允许图根据模型输出分支。

你编译图。编译会绑定拓扑结构，关联可选且生产必需的检查点模块，返回一个可运行实体。调用时传入初始状态和 `thread_id`。执行的每一步都会持久化一个以 `(thread_id, checkpoint_id)` 为键的检查点。

### 四大全能

**检查点。** 每个节点转移都会把新状态写入存储（测试用内存存储，生产用 Postgres/Redis/SQLite）。用同样的 `thread_id` 再调用图即可从中断处恢复。

**中断。** 在节点上标记 `interrupt_before=["human_review"]`，执行将在该节点运行前暂停。状态被保存。API 向用户反馈“等待审核”。后续以相同 `thread_id` 并携带 `Command(resume=...)` 的请求会恢复执行。

**流式。** `graph.stream(state, mode="updates")` 实时产出状态差异。`mode="messages"` 流式传输模型节点内的 LLM token。`mode="values"` 输出完整状态快照。你选择在 UI 里呈现何种内容。

**时空穿梭。** `graph.get_state_history(thread_id)` 返回完整检查点日志。给 `graph.invoke` 传入任何历史的 `checkpoint_id`，即可从该点分叉。适合调试（“如果模型选了工具 B 会怎样？”）和播放生产轨迹做回归测试。

### Reducer 是关键

每个状态字段都有一个 reducer。大多数默认都适用——新值覆盖旧值。但消息列表用 `operator.add`，新消息追加而非替换。并行边通过 reducer 合并更新。如果两个节点都改 `messages`，但忘记用了 `Annotated[list, add_messages]`，第一个更新会被第二个默默覆盖，导致漏掉半轮对话。reducer 是库里唯一稍显微妙的设计；用对了后面能顺畅组合。

### 四节点 ReAct 图

典型的生产 ReAct 代理四个节点，两条边：

1. `agent` — 用当前消息历史调用 LLM，返回助理消息（可能带有 tool_calls）。
2. `tools` — 执行上一条助理消息里的所有 tool_calls，把工具结果追加为工具消息。
3. `agent` 节点条件边，根据最后消息是否带 tool_calls 路由到 `tools` 或 `END`。
4. `tools` 返回到 `agent` 的静态边。

这就是完整 ReAct 循环（思考→行动→观察→思考……）加上检查点、中断和流式输出，大约 40 行代码。

### StateGraph 与 Send（分发）

`Send(node_name, state)` 让节点派发并行子图。例：代理决定同时查询三个检索器。每个 `Send` 启动目标节点的并行执行；输出通过状态 reducer 合并。这是 LangGraph 实现 orchestrator-workers 模式无需线程原语的机制。

### 子图

编译后图可以作为另一图的节点。外层图视为单个节点，内部图有自己的状态和检查点。团队用它搭建监督者-工人代理：监督图路线用户意图到按领域划分的工人子图。

## 构建

### 步骤 1：状态和节点

```python
from typing import Annotated, TypedDict
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def agent_node(state: State) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def should_continue(state: State) -> str:
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END

tool_node = ToolNode(tools=[search_web, read_file])

graph = StateGraph(State)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")

app = graph.compile(checkpointer=MemorySaver())
```

`add_messages` 是使消息列表累加而非覆盖的 reducer。忘了它是 LangGraph 最常见的 bug。

### 步骤 2：用线程执行

```python
config = {"configurable": {"thread_id": "user-42"}}
for event in app.stream(
    {"messages": [HumanMessage("find the Anthropic headquarters address")]},
    config,
    stream_mode="updates",
):
    print(event)
```

每次更新都是 `{node_name: state_delta}` 的字典。前端可以实时展示“代理在思考…调用 search_web…得到了结果…回答中”。

### 步骤 3：加入人机交互中断

给节点标记，执行在运行该节点前暂停。

```python
app = graph.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["tools"],  # 在每次工具调用前暂停
)

state = app.invoke({"messages": [HumanMessage("delete the production database")]}, config)
# state["__interrupt__"] 设为 True。检查拟执行的工具调用。
# 如果批准：
from langgraph.types import Command
app.invoke(Command(resume=True), config)
# 如果拒绝：写入驳回消息再继续
app.update_state(config, {"messages": [AIMessage("Blocked by human reviewer.")]})
```

状态、检查点和线程在中断期间全部持久化。内存里只在运行时临时存在。

### 步骤 4：时空穿梭调试

```python
history = list(app.get_state_history(config))
for snapshot in history:
    print(snapshot.values["messages"][-1].content[:80], snapshot.config)

# 从历史检查点分叉
target = history[3].config  # 回退三步
for event in app.stream(None, target, stream_mode="values"):
    pass  # 从该点重新播放
```

传 `None` 作为输入是从指定检查点重播；传入状态则作为更新接入该检查点状态后恢复。这样即可重现某次失败的代理执行而不必重新运行整个对话。

### 步骤 5：使用生产级检查点存储

```python
from langgraph.checkpoint.postgres import PostgresSaver

with PostgresSaver.from_conn_string("postgresql://...") as checkpointer:
    checkpointer.setup()
    app = graph.compile(checkpointer=checkpointer)
```

集成了 SQLite、Redis、Postgres。`MemorySaver` 仅用于测试。需要跨重启持久化就必须用真正存储。

## 技能点

> 你构建的是图形化代理，而不是 `while True` 循环。

用 LangGraph 前，先做 60 秒设计：

1. **命名节点。** 每个离散决策或有副作用的动作都是节点：“代理思考”、“工具运行”、“审核通过”、“响应流式发送”。不能列出来说明任务还未变成代理形态。
2. **声明状态。** 最小 TypedDict，每个列表字段定义 reducer。不要把所有都塞 `messages`，把任务相关字段（工作用的 `plan`、`budget` 计数器、`retrieved_docs` 列表）提成顶层。
3. **画边。** 静态边除非下一步依赖模型输出。每条条件边配路由器函数指定命名分支。
4. **提前选检查点。** 测试用 `MemorySaver`，正式用 Postgres/Redis/SQLite。无检查点别发布——无检查点无恢复、无中断、无时间旅行。
5. **中断点选在工具运行前。** 审核放边上进入右侧副作用节点，避免副作用执行后才取消；校验放模型输出边上，可廉价拒绝坏调用。
6. **默认流式。** UI 用 `mode="updates"`，模型节点内 token 流用 `mode="messages"`，评估用快照流 `mode="values"`。

拒绝发布无检查点的 LangGraph 代理。拒绝发布副作用后才中断的代理。拒绝发布 reducer 不是 `add_messages` 的 `messages` 字段。

## 练习

1. **简单。** 用计算器工具和网络搜索工具实现上面四节点 ReAct 图。验证 `list(app.get_state_history(config))` 对两轮对话至少产生四个检查点。
2. **中等。** 加一个 `planner` 节点，先于 `agent` 运行，向状态写入结构化 `plan: list[str]`，让 `agent` 标记计划步骤完成。断言在检查点恢复时 `plan` 不丢失（避免 reducer 写错）。
3. **困难。** 用 `Send` 构建一个监督者图在三个子图（`researcher`、`writer`、`reviewer`）间路由。每个子图有自己的状态和检查点。在外层图上加 `interrupt_before=["writer"]`，让人可以审核研究摘要。确认用历史检查点时，时间旅行只重跑分叉的那条分支。

## 关键术语

| 术语 | 俗称 | 实际含义 |
|------|-----------------|-----------------------|
| StateGraph | “LangGraph 图” | 编译前你向其添加节点和边的构建对象。 |
| Reducer | “字段是如何合并的” | 一个函数 `(old, new) -> merged`，在节点返回该字段更新时应用；默认是覆盖，`add_messages` 是追加。 |
| Thread | “会话 ID” | 一个用于限定单个会话所有检查点的 `thread_id` 字符串。 |
| Checkpoint | “暂停状态” | 节点转换后持久化的完整图状态快照，键为 `(thread_id, checkpoint_id)`。 |
| Interrupt | “等待人工介入” | `interrupt_before` / `interrupt_after` 在节点边界暂停执行；重启时用 `Command(resume=...)`。 |
| Time-travel | “从先前步骤分叉” | `graph.invoke(None, config_with_old_checkpoint_id)` 从该检查点开始重新执行。 |
| Send | “并行子图分发” | 节点返回的构造器，用来生成目标节点的 N 个并行执行。 |
| Subgraph | “作为节点的已编译图” | 一个已编译的 StateGraph 用作另一个图中的节点；保留其独立状态域。 |

## 进一步阅读

- [LangGraph 文档](https://langchain-ai.github.io/langgraph/) — StateGraph、Reducer、检查点、Interrupt 的官方参考。
- [LangGraph 概念：状态、Reducer、检查点](https://langchain-ai.github.io/langgraph/concepts/low_level/) — 本课使用的思维模型，直接来源官方。
- [LangGraph 持久化与检查点](https://langchain-ai.github.io/langgraph/concepts/persistence/) — 关于 Postgres/SQLite/Redis 存储、检查点命名空间、线程 ID 的细节。
- [LangGraph 人机循环](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/) — 介绍 `interrupt_before`，`interrupt_after`，`Command(resume=...)` 及编辑状态模式。
- [Yao 等，“ReAct：语言模型中推理与行动的协同”（ICLR 2023）](https://arxiv.org/abs/2210.03629) — 每个 LangGraph agent 实现的模式；阅读以了解推理轨迹背后的原理。
- [Anthropic — 构建高效 Agent（2024 年 12 月）](https://www.anthropic.com/research/building-effective-agents) — 何时采用哪种图结构（链式、路由器、协调者-工作者、评估者-优化器）。
- Phase 11 · 09（函数调用）— 所有 LangGraph agent 节点重用的工具调用原语。
- Phase 11 · 14（模型上下文协议）— 可插入 LangGraph `ToolNode` 的外部工具发现机制 MCP 适配器。
- Phase 11 · 17（Agent 框架权衡）— 何时选择 LangGraph 而非 CrewAI、AutoGen 或 Agno。
