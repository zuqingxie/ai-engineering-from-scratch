# 构建 MCP 客户端 — 发现、调用、会话管理

> 大多数 MCP 内容都提供了服务器教程，但对客户端只带过。客户端代码才是复杂的编排所在：进程生成、能力协商、跨多个服务器的工具列表合并、采样回调、重连以及命名空间冲突解决。本课构建一个多服务器客户端，将三个不同的 MCP 服务器提升为一个扁平的工具命名空间，供模型调用。

**类型：** 构建  
**语言：** Python（标准库，多服务器 MCP 客户端）  
**先决条件：** 阶段 13 · 07（构建 MCP 服务器）  
**时间：** ~75 分钟

## 学习目标

- 生成 MCP 服务器作为子进程，完成 `initialize` 并发送 `notifications/initialized`。
- 维护每个服务器的会话状态（capabilities（能力）、工具列表、最后一次看到的通知 ID）。
- 合并多个服务器的工具列表为一个命名空间，并处理冲突。
- 路由工具调用至所属服务器，并重新组装响应。

## 问题描述

一个真实的代理宿主（如 Claude Desktop、Cursor、Goose、Gemini CLI）会同时加载多个 MCP 服务器。用户可能同时运行一个文件系统服务器、一个 Postgres 服务器和一个 GitHub 服务器。客户端的工作是：

1. 启动每个服务器。
2. 独立完成握手。
3. 对每个服务器调用 `tools/list`，并将结果扁平化。
4. 当模型发出 `notes_search` 时，在合并后的命名空间中查找并路由到正确服务器。
5. 处理来自任何服务器的通知（如 `tools/list_changed`），且不阻塞。
6. 在传输故障时重新连接。

手工实现所有这些，是“玩具”与“可用”之间的分水岭。官方 SDK 把这些封装起来，但思维模型必须是你自己的。

## 概念讲解

### 子进程生成

使用 `subprocess.Popen`，设置 `stdin=PIPE, stdout=PIPE, stderr=PIPE`。设置 `bufsize=1`，开启文本模式（text mode）逐行读取。每个服务器对应一个进程；客户端持有每个服务器一个 `Popen` 实例。

### 每服务器会话状态

每个服务器拥有一个 `Session` 对象，包含：

- `process` — Popen 句柄。
- `capabilities` — 服务器在 `initialize` 阶段声明的能力。
- `tools` — 最近一次 `tools/list` 的结果。
- `pending` — 请求 ID 映射到一个等待响应的 promise/future。

请求本质上是异步的；如果向服务器 A 发送 `tools/call`，而服务器 B 正在处理中，调用不能阻塞。可以使用线程+队列或 asyncio。

### 合并命名空间

客户端将多个服务器的工具列表合并时，工具名称可能冲突。两个服务器都可能提供 `search` 工具。客户端有三种选择：

1. **加服务器名前缀。** 例如 `notes/search`、`files/search`。清晰但不美观。
2. **无声优先法。** 后加载的服务器的 `search` 覆盖之前的。风险较大，隐藏冲突。
3. **冲突拒绝。** 拒绝加载第二个服务器，并通知用户。对安全敏感的主机来说最安全。

Claude Desktop 采用前缀法。Cursor 采用冲突拒绝并给出清晰错误。VS Code MCP 也采用前缀法。

### 路由

合并后，客户端有一个调度表，映射 `tool_name -> session`。模型根据名称发起调用；客户端查找对应 session，向该服务器的 stdin 写入 `tools/call` 消息，然后等待响应。

### 采样回调

如果服务器在 `initialize` 阶段声明了 `sampling` 能力，服务器可能发送 `sampling/createMessage` 请求客户端运行其 LLM。客户端要：

1. 阻塞该服务器的后续请求直到采样完成，或者如果实现支持并发则排队执行。
2. 调用它的 LLM 提供者。
3. 将响应发送回服务器。

第 11 课覆盖采样端到端实现，本课仅提供存根以保持完整。

### 通知处理

`notifications/tools/list_changed` 需要重新调用 `tools/list`。`notifications/resources/updated` 表示如果资源在用中，需重新读取。通知不能产生回复——不要尝试确认它们。

常见客户端错误是：当通知消息在流中时，阻塞读取循环以等待 `tools/call` 响应。解决方案是使用后台读取线程，将所有消息推入队列；主线程从队列出队并分发。

### 重连

传输可能失败：服务器崩溃、OS 杀掉进程、stdio 管道断开。客户端检测 stdout EOF，认定该 session 已死亡。处理方式有：

- 静默重启服务器并重新握手。适合纯只读服务器。
- 向用户报告故障。适合有用户可见会话的有状态服务器。

阶段 13 · 09 讨论 Streamable HTTP 的重连语义；stdio 简单得多。

### 保活和会话 ID

Streamable HTTP 使用 `Mcp-Session-Id` 头。stdio 无会话 ID——进程身份即会话身份。保持活跃的 Ping 是可选的；stdio 管道不会因长时间无活动而断开。

## 使用示例

`code/main.py` 生成三个模拟 MCP 服务器作为子进程，握手每个服务器，合并它们的工具列表，并将工具调用路由至正确服务器。这些“服务器”实际上是执行简单响应器的其他 Python 进程（不是真正的 LLM）。运行它会看到：

- 三次初始化，每次声明不同的能力集。
- 三个 `tools/list` 结果被合并为一个拥有 7 个工具的命名空间。
- 基于工具名称的路由决策。
- 命名空间前缀式冲突防止。

可以重点关注：

- `Session` 数据类清晰管理每服务器状态。
- 后台读取线程不阻塞主线程，逐行出队 stdout。
- 调度表由简单的 `dict[str, Session]` 实现。
- 冲突处理明确：两服务器声明相同名字时，后者加前缀重命名。

## 交付成果

本课产出 `outputs/skill-mcp-client-harness.md`。基于声明式 MCP 服务器列表（名称、命令、参数），该技能生成一个挂载框架，启动它们、合并工具列表、生成带冲突解决的路由函数。

## 练习

1. 运行 `code/main.py`，观察服务器启动日志。用 SIGTERM 终止其中一个模拟服务器，观察客户端如何检测 EOF 并标记该会话为死亡。

2. 实现命名空间前缀。在两个服务器都暴露 `search` 时，将第二个重命名为 `<server>/search`。更新调度表并验证调用路由正确。

3. 为服务器重启添加连接池式退避：连续失败时指数退避，最大 30 秒，三次失败后通知用户。

4. 设计支持 100 个并发 MCP 服务器的客户端。什么数据结构替代简单调度字典？（提示：使用前缀树管理命名空间，再加一个每服务器工具计数的度量指标。）

5. 将客户端移植到官方 MCP Python SDK。SDK 封装了 `stdio_client` 和 `ClientSession`。代码行数应由约 200 行缩减到约 40 行，同时保留多服务器路由能力。

## 关键词汇

| 术语          | 常见说法         | 实际含义                        |
|---------------|------------------|--------------------------------|
| MCP client    | “代理宿主”       | 生成服务器并编排工具调用的进程 |
| Session      | “每服务器状态”    | 能力、工具列表和挂起请求管理    |
| Merged namespace | “统一工具列表” | 合并所有活跃服务器的扁平工具名集合 |
| Namespace collision | “两个服务器同工具” | 客户端必须加前缀、拒绝或先到先得处理 |
| Routing      | “谁处理这个调用？” | 从工具名调度到对应服务器         |
| Background reader | “非阻塞 stdout” | 后台线程或任务，读取服务器 stdout 丢入队列 |
| Sampling callback | “LLM 即服务”  | 客户端对服务器 `sampling/createMessage` 的处理 |
| `notifications/*_changed` | “数据被修改” | 通知客户端需重新发现或重读           |
| Reconnection policy | “服务器挂掉时” | 传输失败时重启的策略               |
| Stdio session | “进程即会话”    | 无会话 ID；子进程生命周期即会话     |

## 深入阅读

- [Model Context Protocol — Client 规范](https://modelcontextprotocol.io/specification/2025-11-25/client) — 标准客户端行为  
- [MCP — 快速启动客户端指南](https://modelcontextprotocol.io/quickstart/client) — 使用 Python SDK 的 Hello World 客户端教程  
- [MCP Python SDK — 客户端模块](https://github.com/modelcontextprotocol/python-sdk) — 参考 `ClientSession` 与 `stdio_client`  
- [MCP TypeScript SDK — 客户端](https://github.com/modelcontextprotocol/typescript-sdk) — TypeScript 版本  
- [VS Code — MCP 扩展指南](https://code.visualstudio.com/api/extension-guides/ai/mcp) — VS Code 如何在单编辑器宿主里多路复用多个 MCP 服务器
