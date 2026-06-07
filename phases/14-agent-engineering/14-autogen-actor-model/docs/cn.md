# AutoGen v0.4: Actor Model（角色模型）和 Agent Framework（代理框架）

> AutoGen v0.4（微软研究院，2025年1月）围绕角色模型重新设计了代理编排。支持异步消息交换、事件驱动代理、故障隔离和自然并发。该框架目前处于维护模式，微软 Agent Framework（2025年10月公测）成为继任者。

**类型：** 学习 + 构建  
**语言：** Python（标准库）  
**先决条件：** 第14阶段 · 01（Agent Loop（代理循环）），第14阶段 · 12（Workflow Patterns（工作流模式））  
**时长：** ~75分钟

## 学习目标

- 描述角色模型：代理作为角色，消息是唯一的进程间通信（IPC），每个角色故障隔离。
- 说出 AutoGen v0.4 的三个 API 层——Core（核心）、AgentChat、Extensions（扩展）——及其用途。
- 解释为何将消息传递与消息处理解耦实现了故障隔离和自然并发。
- 用 Python 实现一个标准库角色运行时，并将一个两代理代码审查流程移植到该运行时。

## 问题

大多数代理框架是同步的：一个代理生产，一个代理消费，在调用栈中运行。失败会导致调用栈崩溃。并发是事后加上的。分布式需要重写。

AutoGen v0.4 的解决方案是角色模型。每个代理是一个有私有收件箱的角色。消息是唯一的交互方式。运行时将消息递送与消息处理解耦。失败只影响单个角色。并发是内建的。分布只是不同的传输方式。

## 概念

### 角色（Actors）

一个角色具有：

- 私有状态（外部永远不能直接访问）。
- 收件箱（消息队列）。
- 处理器：`receive(message) -> effects`，其中效果可以是“回复”、“发送给其他角色”、“产生新角色”、“更新状态”、“停止自身”。

两个角色不能共享内存，只能发送消息。

### AutoGen v0.4 中的三层 API

1. **Core（核心）**。低级角色框架。包含 `AgentRuntime`、`Agent`、`Message`、`Topic`。支持异步消息交换，事件驱动。
2. **AgentChat**。任务驱动的高级 API（替代 v0.2 的 ConversableAgent）。包含 `AssistantAgent`、`UserProxyAgent`、`RoundRobinGroupChat`、`SelectorGroupChat`。
3. **Extensions（扩展）**。集成——OpenAI、Anthropic、Azure、工具、记忆。

### 解耦的重要性

在 v0.2 模型中，调用 `agent_a.chat(agent_b)` 是同步阻塞的，`agent_a` 会等待 `agent_b` 返回。在 v0.4 中，调用 `send(agent_b, msg)` 会将消息放入 `agent_b` 的收件箱后立即返回。运行时随后负责递送。带来三大影响：

- **故障隔离。** 代理 B 崩溃不会导致代理 A 崩溃——运行时捕获 B 的处理器故障并决定如何处理（记录、重试、死信）。
- **自然并发。** 多个消息可以同时在途，角色可以并行处理收件箱。
- **分布就绪。** 收件箱加上传输构成相同抽象，无论角色是在进程内还是远程主机上。

### 拓扑结构

- **RoundRobinGroupChat（轮询群聊）**。代理按固定轮换顺序发言。
- **SelectorGroupChat（选择者群聊）**。选择者代理根据对话上下文选择谁下一步发言。
- **Magentic-One。** 用于网页浏览、代码执行、文件处理的参考多代理团队，基于 AgentChat 构建。

### 可观测性

内置 OpenTelemetry 支持。每条消息生成一个 span；工具调用携带 `gen_ai.*` 属性，遵循 2026 年的 OTel GenAI 语义规范（第23课）。

### 状态：维护模式

2026年初：AutoGen v0.7.x 稳定，适用于研究和原型开发。微软将主动开发转向 Microsoft Agent Framework（2025年10月1日公测，目标2026年第一季度末 GA）。AutoGen 的模式可以无缝移植——角色模型是持久的核心思想。

## 构建

`code/main.py` 实现了一个标准库角色运行时：

- `Message` —— 带有 `sender`、`recipient`、`topic`、`body` 的类型化载荷。
- `Actor` —— 抽象基类，包含 `receive(message, runtime)`。
- `Runtime` —— 基于共享队列的事件循环，实现消息递送和故障隔离。
- 一个两角色演示：`ReviewerAgent` 负责代码审查，`ChecklistAgent` 运行检查表；双方消息交换直至达成共识。

运行：

```text
python3 code/main.py
```

追踪显示消息递送过程，模拟一角色失败但不影响另一角色，以及达成一致判定。

## 使用建议

- **AutoGen v0.4/v0.7**（维护）——适合研究、原型开发、多代理模式。
- **Microsoft Agent Framework**（公测）——未来路径，采用同样的角色模型理念，但 API 更现代。
- **LangGraph swarm topology（第13课）**——通过共享工具交接实现类似模式。
- **自定义角色运行时**——需要特定传输协议（NATS、RabbitMQ、gRPC）时自建。

## 发布

`outputs/skill-actor-runtime.md` 生成一个最小的角色运行时加上一个团队模板（轮询或选择者），适用于指定多代理任务。

## 练习

1. 添加死信队列（dead-letter queue）：当处理器抛出异常时，将失败的消息存档以供人工检查。你的示例程序中死信队列有多频繁被触发？
2. 实现 `SelectorGroupChat`：一个选择者代理根据对话状态挑选下一条消息的处理者。
3. 添加分布式传输：将进程内队列替换为基于 JSON-over-HTTP 的服务器，让角色能在不同进程运行。
4. 为每条消息连接一个 OpenTelemetry span（或使用空操作代替），发出符合第23课要求的 `gen_ai.agent.name` 和 `gen_ai.operation.name`。
5. 阅读 AutoGen v0.4 的架构文章。将你的示例程序迁移到真正的 `autogen_core` API。思考你跳过了哪些生产环境必不可少的部分？

## 关键词

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| Actor（角色） | “Agent（代理）” | 私有状态 + 收件箱 + 处理器；没有共享内存 |
| Message（消息） | “Event（事件）” | 类型化载荷；角色唯一的交互方式 |
| Inbox（收件箱） | “Mailbox（邮箱）” | 每个角色待处理消息的队列 |
| Runtime（运行时） | “Agent host（代理主机）” | 路由消息并隔离故障的事件循环 |
| Topic（话题） | “Channel（频道）” | 角色间的命名发布-订阅路径 |
| Fault isolation（故障隔离） | “Let it crash（让它崩溃）” | 一个角色失败不会影响其他角色 |
| RoundRobinGroupChat（轮询群聊） | “Fixed-rotation team（固定轮换团队）” | 代理依次轮流 |
| SelectorGroupChat（选择者群聊） | “Context-routed team（基于上下文路由团队）” | 选择者决定下一位 |
| Magentic-One | “Reference team（参考团队）” | 用于网页、代码和文件的多代理团队 |

## 拓展阅读

- [AutoGen v0.4，微软研究院](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — 重设计文章  
- [LangGraph 概览](https://docs.langchain.com/oss/python/langgraph/overview) — 图结构替代方案  
- [OpenTelemetry GenAI 语义规范](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — AutoGen 默认发出的 spans
