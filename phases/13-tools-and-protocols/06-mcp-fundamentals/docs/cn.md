# MCP 基础 — 原语、生命周期、JSON-RPC 基础

> 在 MCP 之前，每个集成都都是一次性的。Model Context Protocol（模型上下文协议），由 Anthropic 于 2024 年 11 月首次发布，现在由 Linux 基金会的 Agentic AI Foundation 维护，标准化了发现和调用流程，使任何客户端都能与任何服务器通信。2025-11-25 版本规范定义了六个原语（三个服务器端，三个客户端）、三阶段生命周期，以及 JSON-RPC 2.0 的消息格式。掌握这些内容，本阶段 MCP 章节剩余部分就变成阅读理解。

**类型：** 学习  
**语言：** Python（stdlib，JSON-RPC 解析器）  
**先备知识：** 第 13 阶段 · 01 至 05（工具接口和函数调用）  
**时长：** 约 45 分钟

## 学习目标

- 命名 MCP 所有六个原语（服务器端的 tools（工具）、resources（资源）、prompts（提示）；客户端的 roots（根）、sampling（采样）、elicitation（引导））并为每个举一个用例。
- 理解三阶段生命周期（initialize（初始化）、operation（运行）、shutdown（关闭）），说明每个阶段谁发送哪些消息。
- 解析和生成 JSON-RPC 2.0 请求、响应和通知消息。
- 解释 `initialize` 阶段的能力协商（capability negotiation）是什么，以及没有它会导致什么问题。

## 问题背景

MCP 之前，每个使用工具的智能体都有自己的一套协议。Cursor 拥有一个类似 MCP 但不兼容的系统。Claude Desktop 配备了不同的解决方案。VS Code 的 Copilot 扩展又是第三种协议。一个团队开发的“Postgres 查询”工具在不同宿主 API 上写了三份代码。复用工具意味着代码复制。

结果是一次性集成爆炸式增长，生态系统发展受限。

MCP 通过统一消息格式解决了这个问题。单一 MCP 服务器可在所有 MCP 客户端中使用：Claude Desktop、ChatGPT、Cursor、VS Code、Gemini、Goose、Zed、Windsurf，以及 2026 年 4 月预计超过 300 个客户端。月度 SDK 下载量 1.1 亿次。公共服务器超过 10,000 台。Linux 基金会于 2025 年 12 月正式将其纳入 Agentic AI Foundation 旗下。

本阶段使用的规范版本是 **2025-11-25**。新增了异步任务（Async Tasks，SEP-1686）、URL 模式引导（SEP-1036）、带工具采样（SEP-1577）、增量范围授权（SEP-835）和 OAuth 2.1 资源指示标签语义。第 13 阶段 · 09 至 16 涵盖这些扩展。本课只讲基础。

## 核心概念

### 三个服务器端原语

1. **Tools（工具）。** 可调用的动作。与第 13 阶段 · 01 所述的四步循环相同。
2. **Resources（资源）。** 对外暴露的数据。只读，通过 URI 可寻址，如 `file:///path`、`db://query/...` 及自定义方案。
3. **Prompts（提示）。** 可复用模板。宿主 UI 中的斜线命令；服务器提供模板，客户端传入参数。

### 三个客户端原语

4. **Roots（根）。** 服务器允许访问的 URI 集合。客户端声明，服务器遵守。
5. **Sampling（采样）。** 服务器请求客户端的模型执行补全。使得服务器托管的智能体循环无需服务器端 API 密钥。
6. **Elicitation（引导）。** 服务器在运行时向客户端用户请求结构化输入。可以是表单或 URL（SEP-1036）。

MCP 中的每项能力正好属于其中一个原语。第 13 阶段 · 10 至 14 对它们进行了详细讲解。

### 消息格式：JSON-RPC 2.0

每条消息均为包含以下字段的 JSON 对象：

- 请求（Request）：`{jsonrpc: "2.0", id, method, params}`。
- 响应（Response）：`{jsonrpc: "2.0", id, result | error}`。
- 通知（Notification）：`{jsonrpc: "2.0", method, params}` — 无 `id`，不期望响应。

基础规范定义了约 15 种方法，按原语分组。重要方法：

- `initialize` / `initialized`（握手）
- `tools/list`, `tools/call`
- `resources/list`, `resources/read`, `resources/subscribe`
- `prompts/list`, `prompts/get`
- `sampling/createMessage`（服务器到客户端）
- `notifications/tools/list_changed`, `notifications/resources/updated`, `notifications/progress`

### 三阶段生命周期

**第一阶段：initialize（初始化）**

客户端发送 `initialize`，携带其 `capabilities` 和 `clientInfo`。服务器响应其自身 `capabilities`、`serverInfo` 和使用的规范版本。客户端接受响应后，发送 `notifications/initialized`。此后双方可根据协商的能力发送请求。

**第二阶段：operation（运行）**

双向通信。客户端调用 `tools/list` 以发现工具，再调用 `tools/call` 以执行。服务器若声明支持采样能力，可发送 `sampling/createMessage`。服务器工具列表发生变更时，发送 `notifications/tools/list_changed`。用户更改根作用域时，客户端可发送 `notifications/roots/list_changed`。

**第三阶段：shutdown（关闭）**

任何一方关闭连接。MCP 无结构化关闭方法；关闭信号通过传输层（stdio 或 Streamable HTTP，第 13 阶段 · 09）传达。

### 能力协商

`initialize` 握手中的 `capabilities` 是双方的能力协议。服务器示例：

```json
{
  "tools": {"listChanged": true},
  "resources": {"subscribe": true, "listChanged": true},
  "prompts": {"listChanged": true}
}
```

服务器声明它能发送 `tools/list_changed` 通知并支持 `resources/subscribe`。客户端按如下回应：

```json
{
  "roots": {"listChanged": true},
  "sampling": {},
  "elicitation": {}
}
```

若客户端不声明 `sampling`，服务器不能调用 `sampling/createMessage`。反之，若服务器不声明 `resources.subscribe`，客户端不得尝试订阅。

这防止了生态系统的偏离。客户端不支持采样依然是有效 MCP 客户端；服务器不调用采样依然是有效 MCP 服务器。只是不使用该功能而已。

### 结构化内容及错误格式

`tools/call` 返回 `content` 数组，包含类型化块：`text`、`image`、`resource`。第 13 阶段 · 14 加入了 MCP 应用（`ui://` 交互式 UI）作为类型。

错误使用 JSON-RPC 错误码。规范新增：`-32002` “资源未找到”，`-32603` “内部错误”，以及 MCP 特有的错误信息放在 `error.data`。

### 客户端能力与工具调用详情的区别

常见误区：`capabilities.tools` 是客户端是否支持工具列表变更通知的能力标志；客户端是否会调用特定工具是运行时由模型驱动的选择，不是能力标志。能力标志代表规范级别的协议，模型的选择是独立的。

### 为什么选 JSON-RPC 而非 REST？

JSON-RPC 2.0（2010）是轻量且支持双向通信的协议。REST 是客户端发起的。MCP 需要服务器发起消息（采样，通知），因此 JSON-RPC 的对称请求/响应形式更合适。且 JSON-RPC 易于基于 stdio、WebSocket 或 Streamable HTTP 组合，避免重新设计 HTTP 请求结构。

## 实践操作

`code/main.py` 提供了一个最简的 JSON-RPC 2.0 解析器和生成器，手动演示 `initialize` → `tools/list` → `tools/call` → `shutdown` 序列，打印所有消息。无真实传输，仅消息形态。可对照后续阅读中提供的规范验证每个消息。

关注点：

- `initialize` 双方声明能力；响应消息含 `serverInfo` 和 `protocolVersion: "2025-11-25"`。
- `tools/list` 返回 `tools` 数组；每项包含 `name`、`description`、`inputSchema`。
- `tools/call` 使用 `params.name` 和 `params.arguments`。
- 响应中的 `content` 是 `{type, text}` 类型的数组。

## 上线部署

本课产生文件 `outputs/skill-mcp-handshake-tracer.md`。根据 MCP 客户端-服务器交互的 pcap 风格日志，技能会注释每条消息对应的原语、生命周期阶段及相关能力。

## 练习题

1. 运行 `code/main.py`。找出能力协商发生的代码行，说明若服务器不声明 `tools.listChanged` 会发生什么改变。

2. 扩展解析器支持 `notifications/progress`。消息格式：`{method: "notifications/progress", params: {progressToken, progress, total}}`。在长时间执行的 `tools/call` 中发出此通知，确认客户端处理程序能显示进度条。

3. 通读 MCP 2025-11-25 规范（约 80 页）。找出多数服务器不需要声明的能力标志。提示：与资源订阅相关。

4. 纸上绘制一个假设的“定时任务”功能属于哪个原语。（提示：服务器想让客户端定时调用它。目前六个原语中没有合适的。）MCP 的 2026 路线图中已有该功能的草案 SEP。

5. 解析 GitHub 上一个开源 MCP 服务器的某次会话日志。统计请求、响应、通知消息数。计算生命周期消息占总流量的比例。

## 关键词汇

| 术语 | 通俗说法 | 实际含义 |
|------|----------|----------|
| MCP | “模型上下文协议” | 用于模型对工具发现与调用的开放协议 |
| Server primitive（服务器原语） | “服务器暴露的是什么” | tools（动作）、resources（数据）、prompts（模板） |
| Client primitive（客户端原语） | “客户端允许服务器使用的” | roots（作用域）、sampling（LLM 回调）、elicitation（用户输入） |
| JSON-RPC 2.0 | “底层消息格式” | 对称的请求/响应/通知消息封装 |
| `initialize` handshake | “能力协商” | 首次消息对；服务器与客户端声明支持的功能 |
| `tools/list` | “发现” | 客户端请求服务器当前工具集合 |
| `tools/call` | “调用” | 客户端请求服务器运行某工具并传参数 |
| `notifications/*_changed` | “变更事件” | 服务器告诉客户端某原语列表已变更 |
| Content block（内容块） | “类型化结果” | 工具结果中的 `{type: "text" | "image" | "resource" | "ui_resource"}` |
| SEP | “规范演进提案” | 命名的草案提案（如异步任务 SEP-1686） |

## 拓展阅读

- [Model Context Protocol — Specification 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25) — 官方规范文档
- [Model Context Protocol — Architecture concepts](https://modelcontextprotocol.io/docs/concepts/architecture) — 六原语模型概念
- [Anthropic — Introducing the Model Context Protocol](https://www.anthropic.com/news/model-context-protocol) — 2024 年 11 月发布文章
- [MCP blog — First MCP anniversary](https://blog.modelcontextprotocol.io/posts/2025-11-25-first-mcp-anniversary/) — 一周年回顾及 2025-11-25 版本更新介绍
- [WorkOS — MCP 2025-11-25 spec update](https://workos.com/blog/mcp-2025-11-25-spec-update) — SEP-1686、1036、1577、835 和 1724 总结
