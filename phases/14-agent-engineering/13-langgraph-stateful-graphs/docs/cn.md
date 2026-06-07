# LangGraph：有状态图和持久执行

> LangGraph 是 2026 年面向低级有状态编排的参考标准。Agent 是状态机；节点是函数；边是转移；状态不可变且在每一步后进行检查点保存。可以准确从任意故障处恢复。

**类型：** 学习 + 构建  
**语言：** Python（标准库）  
**先决条件：** 第 14 阶段 · 01（Agent 循环），第 14 阶段 · 12（工作流模式）  
**时长：** 约 75 分钟

## 学习目标

- 描述 LangGraph 的核心模型：具有不可变状态、函数节点、有条件边和后步检查点的状态机。  
- 说明文档强调的四大能力：持久执行、流式处理、人与机器协作、全面内存。  
- 解释 LangGraph 支持的三种编排拓扑结构：监督者（supervisor）、对等（peer-to-peer，群集 swarm）、分层（hierarchical，嵌套子图）。  
- 实现一个标准库的有状态图，具备不可变状态、有条件边、检查点与恢复周期。

## 问题背景

Agents 和工作流都面临这样的问题：当 40 步的运行在第 38 步失败时，你希望从第 38 步继续执行，而不是重新开始。劣质的状态模型往往让操作者不得不在假设新运行的库外乱搞重试机制。

LangGraph 的设计解决方案是：状态是一级类型化对象，变更显式，且每个节点后都会持久化检查点。恢复调用即是 `load_state(session_id)`。

## 核心概念

### 图

一个图由以下定义：

- **状态类型。** 一个类型化的字典（或 Pydantic 模型），每个节点读取并修改该状态。  
- **节点。** 纯函数 `(state) -> state_update`，返回的更新合并到状态中。  
- **边。** 节点间的有条件或直接转移。  
- **入口和出口。** `START` 和 `END` 哨兵节点标记边界。

示例：包含 `classify`、`refund`、`bug`、`sales`、`done` 节点的代理——作为一个图形的路由工作流。

### 持久执行

每个节点返回后，运行时将序列化状态并写入检查点保存器（SQLite、Postgres、Redis、自定义存储等）。若第 N 步失败，运行时可调用 `resume(session_id)`，从第 N+1 步以及准确状态继续执行。

LangGraph 文档明确强调真正使用此特性的企业：Klarna、Uber、J.P. Morgan。重点不在图的形状，而是图结构加上检查点机制使得恢复非常经济高效。

### 流式处理

每个节点可以产生部分输出流。图会将每个节点的增量事件流式传递给调用方，使得 UI 随图执行即时更新。

### 人机协作（Human-in-the-loop）

可在节点之间检查和修改状态。实现方式：在关键节点前暂停，将状态呈现给人工，接受修改后恢复。检查点机制使这成为易事，因为状态已被序列化。

### 内存

短期内存（运行时内——如对话历史存储于状态中）和长期内存（跨运行——通过检查点保存器和单独的长期存储）。LangGraph 可通过工具集成外部内存系统（如 Mem0、自定义实现）。

### 三种拓扑结构

1. **监督者（Supervisor）。** 中央路由 LLM 负责分派给专门子代理。`langgraph-supervisor` 包里有 `create_supervisor()` 方法（但到 2026 年，LangChain 团队建议直接使用工具调用以获得更多上下文控制）。  
2. **群集/对等网络（Swarm / peer-to-peer）。** Agent 通过共享工具界面直接交接，没有中央路由。  
3. **分层（Hierarchical）。** 监督者管理次级监督者，通过嵌套子图实现。

### 此模式的缺陷

- **检查点太小。** 只检查点对话轮次，会导致工具状态和内存写入无法恢复，必须序列化完整状态。  
- **非确定性节点。** 恢复假设节点输入产生相同的状态更新。随机种子、时钟、外部 API 等必须捕获。  
- **过度使用条件边。** 每条边都加条件将形成难以推理的状态机，偏好多分支的线性链路。

## 实战构建

`code/main.py` 实现了标准库的有状态图：

- `State` — 一个类型化字典，包含 `messages`、`step`、`route`、`output`、`human_approval`。  
- `Node` — 可调用，接受状态并返回更新字典。  
- `StateGraph` — 节点 + 边 + 条件边 + 运行 + 恢复。  
- `SQLiteCheckpointer`（内存假存储）— 在每个节点后序列化状态；`load(session_id)` 恢复状态。  
- 一个演示图：classify -> 分支（refund / bug / sales）-> 人机网关 -> 发送。

运行：

```text
python3 code/main.py
```

跟踪显示第一次运行在人机网关处失败，持久化后恢复运行产出最终结果。

## 使用指南

- **LangGraph** — 参考实现、生产就绪。使用 `create_react_agent`、`create_supervisor` 或构建自定义图。  
- **AutoGen v0.4**（第 14 课）— 高并发场景的 actor 模型替代方案。  
- **Claude Agent SDK**（第 17 课）— 含内置会话存储的托管环境。  
- **自定义** — 需要精确控制状态形状或检查点后端时。

## 部署实现

`outputs/skill-state-graph.md` 生成符合 LangGraph 形状的有状态图，任意目标运行环境中具备检查点与恢复功能。

## 练习题

1. 添加条件边：当分类置信度低于阈值时，从 `classify` 直接跳转到 `end`。在人工手动设置 `route` 后恢复运行。  
2. 将 SQLite 风格的假存储替换为真实 SQLite 检查点保存器，测量每步序列化开销。  
3. 实现并行边：两个节点并发运行，合并时调用自定义合并函数。使用不可变状态有什么优势？  
4. 阅读 `langgraph-supervisor` 参考，实现玩具示例迁移到 `create_supervisor`，比较跟踪形态。  
5. 添加流式处理：每个节点在运行时产生部分状态，实时打印增量。

## 关键词

| 术语             | 常见说法               | 实际含义                        |
|-----------------|----------------------|-----------------------------|
| State graph（状态图）   | “Agent 作为状态机”         | 类型化状态 + 节点 + 边 + 合并函数       |
| Checkpointer（检查点器） | “持久化后端”              | 每节点后序列化状态，支持恢复                |
| Reducer（合并器）      | “状态合并器”              | 结合当前状态与节点更新的函数                |
| Conditional edge（有条件边） | “分支”                   | 由状态函数决定选取的边                   |
| Subgraph（子图）       | “嵌套图”                | 作为另一个图的节点使用的图                |
| Durable execution（持久执行） | “故障恢复”                | 从最后成功节点及准确状态重新启动           |
| Supervisor（监督者）    | “路由 LLM”               | 专家子代理的中央调度者                   |
| Swarm（群集）         | “点对点代理”              | 代理通过共享工具直接交接，无中央路由         |

## 延伸阅读

- [LangGraph 概览](https://docs.langchain.com/oss/python/langgraph/overview) — 参考文档  
- [langgraph-supervisor 参考](https://reference.langchain.com/python/langgraph/supervisor/) — 监督者模式 API  
- [AutoGen v0.4，微软研究院](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — actor 模型替代方案  
- [Claude Agent SDK 概览](https://platform.claude.com/docs/en/agent-sdk/overview) — 会话存储与子代理
