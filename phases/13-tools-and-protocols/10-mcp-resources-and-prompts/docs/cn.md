# MCP 资源和提示 — 超越工具的上下文暴露

> 工具获得了 90% 的 MCP 关注度。其余两个服务器原语解决不同的问题。资源用于暴露可读取的数据；提示（prompt）作为斜杠命令暴露可复用的模板。许多服务器应使用资源替代将读取封装为工具，使用提示替代在客户端提示中硬编码工作流。本课介绍决策规则，并介绍 `resources/*` 和 `prompts/*` 消息。

**类型:** 构建  
**语言:** Python（stdlib，资源和提示处理器）  
**先决条件:** 13 阶段 · 07（MCP 服务器）  
**时间:** ~45 分钟

## 学习目标

- 在给定领域决定将功能暴露为工具、资源还是提示。
- 实现 `resources/list`、`resources/read`、`resources/subscribe` 并处理 `notifications/resources/updated`。
- 实现带参数模板的 `prompts/list` 和 `prompts/get`。
- 识别主机何时将提示作为斜杠命令展示与自动注入上下文。

## 问题

一个简单的笔记应用 MCP 服务器把所有功能都暴露为工具：`notes_read`，`notes_list`，`notes_search`。这将每次数据访问都包装为模型驱动的工具调用。结果是：

- 模型不得不决定是否对每个可能受益于上下文的查询调用 `notes_read`。
- 只读内容无法订阅或流式传输到主机侧边栏。
- 客户端 UI（Claude Desktop 的资源附件面板，Cursor 的“包含文件”选择器）无法显示数据。

正确的划分：将数据暴露为资源，将变更或计算操作暴露为工具，将可复用的多步骤工作流暴露为提示。每种原语都有其 UX 便利和访问模式。

## 概念

### 工具 vs 资源 vs 提示 — 决策规则

| 功能 | 原语 |
|------------|-----------|
| 用户想搜索、过滤或转换数据 | 工具 |
| 用户希望主机包含此数据作为上下文 | 资源 |
| 用户想要可以重复运行的模板化工作流 | 提示 |

指导原则：如果模型会重用调用以获取每个相关查询的益处，它是工具；如果用户会将其附加到对话中受益，是资源；如果用户希望重用整个多步骤工作流单元，是提示。

### 资源

`resources/list` 返回 `{resources: [{uri, name, mimeType, description?}]}`。`resources/read` 接受 `{uri}` 并返回 `{contents: [{uri, mimeType, text | blob}]}`。

URI 可以是任何可寻址的：

- `file:///Users/alice/notes/mcp.md`
- `postgres://my-db/query/SELECT ...`
- `notes://note-14`（自定义方案）
- `memory://session-2026-04-22/recent`（服务器特定）

`contents[]` 支持文本和二进制。二进制使用 base64 编码的字符串 `blob` 加上 `mimeType`。

### 资源订阅

在能力中声明 `{resources: {subscribe: true}}`。客户端调用 `resources/subscribe {uri}`。当资源变更时，服务器发送 `notifications/resources/updated {uri}`。客户端重新读取。

用例：笔记服务器的资源是磁盘文件；文件监视器触发更新通知；Claude Desktop 在主机外编辑时重新拉取文件上下文。

### 资源模板（2025-11-25 新增）

`resourceTemplates` 允许暴露参数化的 URI 模式：如 `notes://{id}`，其中 `id` 是补全目标。客户端可以在资源选择器中自动补全 id。

### 提示（Prompts）

`prompts/list` 返回 `{prompts: [{name, description, arguments?}]}`。`prompts/get` 接受 `{name, arguments}` 并返回 `{description, messages: [{role, content}]}`。

提示是生成消息列表的模板，主机将其传给模型。例如 `code_review` 提示接受 `file_path` 参数，返回一个包含三条消息的序列：系统消息、包含文件内容的用户消息，以及带有推理模板的助理启动消息。

### 主机和提示

Claude Desktop、VS Code 和 Cursor 在聊天 UI 中将提示作为斜杠命令暴露。用户输入 `/code_review` 并从表单中选择参数。服务器的提示是“用户快捷操作”与“发送至模型的完整提示”之间的契约。

并非所有客户端都支持提示 — 请检查能力协商。声明提示能力但客户端不支持提示的服务器，用户将看不到斜杠命令。

### “列表变更”通知

资源和提示均在集合变动时发出 `notifications/list_changed`。一个刚导入 20 条新笔记的笔记服务器发出 `notifications/resources/list_changed`；客户端重新调用 `resources/list` 获取新增项。

### 内容类型约定

文本：`mimeType: "text/plain"`，`text/markdown`，`application/json`。  
二进制：`image/png`，`application/pdf`，附加 `blob` 字段。  
MCP 应用（第 14 课）：`text/html;profile=mcp-app`，使用 `ui://` URI。

### 动态资源

资源 URI 不必对应静态文件。`notes://recent` 每次读取返回最新五条笔记。`db://query/users/active` 可以执行参数化查询。服务器可动态计算内容。

规则：若客户端能按 URI 缓存，URI 必须稳定。若仅一次性计算，URI 应包含时间戳或随机数，避免缓存过时。

### 订阅与轮询

支持订阅的客户端通过 `notifications/resources/updated` 接收服务器推送。未订阅或不支持订阅的客户端通过重复读取轮询。两种方式都符合规范。服务器声明的能力通知客户端支持哪种。

订阅的成本：服务器上的每会话状态（谁订阅了什么）。应限制订阅集合大小；断开客户端应超时清理。

### 提示 vs 系统提示

MCP 中提示不是系统提示。主机的系统提示（其自身操作指令）与 MCP 的服务器提示模板并存。行为良好的客户端不会让服务器提示覆盖自身系统提示，而是叠加使用。

## 练习使用

`code/main.py` 在第 07 课笔记服务器基础上扩展：

- 支持按笔记资源（`notes://note-1` 等）及 `resources/subscribe`。
- 一个 `review_note` 提示，渲染成三条消息模板。
- 模拟文件监视器，笔记修改时发出 `notifications/resources/updated`。
- 一个动态资源 `notes://recent`，始终返回最新五条笔记。

运行演示查看完整流程。

## 交付物

本课输出 `outputs/skill-primitive-splitter.md`。给定一个拟议 MCP 服务器，该技能将每个能力分类为工具 / 资源 / 提示并提供理由。

## 练习

1. 运行 `code/main.py`。观察初始资源列表，编辑笔记触发，确认 `notifications/resources/updated` 事件触发。

2. 添加 `resources/list_changed` 通知：创建新笔记时发送该通知，使客户端重新发现资源。

3. 为 GitHub MCP 服务器设计三个提示：`summarize_pr`，`triage_issue`，`release_notes`。每个带参数模式。提示体应能直接运行，无需改动。

4. 选一个第 07 课服务器中的工具，判断是否应保留为工具或拆分为资源+工具组合，并用一句话说明理由。

5. 阅读规范中 `server/resources` 和 `server/prompts` 部分。找出 `resources/read` 中很少填充但规范支持的字段。提示：查看资源内容的 `_meta`。

## 关键术语

| 术语 | 常用说法 | 实际含义 |
|------|-----------|----------|
| 资源 | “暴露数据” | 主机可读取的 URI 可寻址内容 |
| 资源 URI | “数据指针” | 带方案前缀的标识符（`file://`，`notes://` 等） |
| `resources/subscribe` | “监听变更” | 客户端主动订阅特定 URI 的服务器推送更新 |
| `notifications/resources/updated` | “资源变化” | 通知客户端所订阅资源内容已更新 |
| 资源模板 | “参数化 URI” | 带补全提示的 URI 模式供主机选取 |
| 提示 | “斜杠命令模板” | 命名的多消息模板，带参数槽位 |
| 提示参数 | “模板输入” | 主机在渲染前收集的类型化参数 |
| `prompts/get` | “渲染模板” | 服务器返回填充好的消息列表 |
| 内容块 | “类型区块” | `{type: text \| image \| resource \| ui_resource}` |
| 斜杠命令 UX | “用户快捷方式” | 主机将提示作为以 `/` 开头的命令展示 |

## 进一步阅读

- [MCP — 概念：资源](https://modelcontextprotocol.io/docs/concepts/resources) — 资源 URI，订阅和模板  
- [MCP — 概念：提示](https://modelcontextprotocol.io/docs/concepts/prompts) — 提示模板和斜杠命令集成  
- [MCP — 服务器资源规范 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/server/resources) — 完整的 `resources/*` 消息参考  
- [MCP — 服务器提示规范 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/server/prompts) — 完整的 `prompts/*` 消息参考  
- [MCP — 协议信息站：资源](https://modelcontextprotocol.info/docs/concepts/resources/) — 社区指南，扩展官方文档
