# Roots 和 Elicitation（范围限定与中途用户输入）

> 硬编码路径在用户打开不同项目时会失效。预填充工具参数在用户参数不足时会失败。Roots（根路径）将服务器的作用范围限定在用户控制的 URI 集合内；elicitation（引导）在工具调用中途暂停，通过表单或 URL 让用户提供结构化输入。两个客户端原语，两个常见 MCP 失败模式的解决方案。SEP-1036（URL 模式引导，2025-11-25）在 2026 年上半年仍为实验性——依赖前请检查 SDK 版本。

**类型：** 构建  
**语言：** Python（标准库，roots + elicitation 演示）  
**前置条件：** 第 13 阶段 · 07（MCP 服务器）  
**时间：** 约 45 分钟

## 学习目标

- 声明 `roots` 并响应 `notifications/roots/list_changed` 通知。
- 限制服务器对文件的操作仅作用于声明的 root 集内的 URI。
- 使用 `elicitation/create` 在工具调用中途询问用户确认或结构化输入。
- 在表单模式和 URL 模式引导中选择（后者为实验性；有漂移风险提醒）。

## 问题

notes MCP 服务器在生产环境中遇到的两个具体失败案例。

**路径假设失效。** 服务器是针对 `~/notes` 编写的。用户在另一台机器上，其 notes 位于 `~/Documents/Notes`，工具调用失败且无声（找不到文件），更糟的是写入了错误的位置。

**用户知道但缺失的参数。** 用户请求“删除旧的 TPS 报告笔记”。模型调用了 `notes_delete(title: "TPS report")`，但有三个匹配的笔记，分别来自 2023、2024 和 2025。工具无法猜测。以“模糊不清”失败令人恼火；对三个都执行则造成严重错误。

Roots 解决了第一个问题：客户端在 `initialize` 阶段声明服务器可触及的 URI 集。Elicitation 解决了第二个问题：服务器暂停工具调用，发送 `elicitation/create` 让用户选择。

## 概念

### Roots

客户端在 `initialize` 阶段声明 root 列表：

```json
{
  "capabilities": {"roots": {"listChanged": true}}
}
```

服务器可以调用 `roots/list`：

```json
{"roots": [{"uri": "file:///Users/alice/Documents/Notes", "name": "Notes"}]}
```

服务器必须将 roots 视为边界：任何 root 集外的文件读写请求都将被拒绝。客户端不会强制执行（服务器依然是用户信任的代码），但符合规范的服务器会遵循此规则。

当用户添加或移除 root，客户端发送 `notifications/roots/list_changed`。服务器重新调用 `roots/list` 并更新边界。

### 为什么 roots 是客户端原语

Roots 由客户端声明，因为它代表用户的授权模型。用户告诉 Claude Desktop：“允许此 notes 服务器访问这两个目录”。服务器不能扩大该范围。

### Elicitation：表单模式默认值

`elicitation/create` 需要一个表单 schema 以及一个自然语言提示：

```json
{
  "method": "elicitation/create",
  "params": {
    "message": "Delete 'TPS report'? Multiple notes match; pick one.",
    "requestedSchema": {
      "type": "object",
      "properties": {
        "note_id": {
          "type": "string",
          "enum": ["note-3", "note-7", "note-14"]
        },
        "confirm": {"type": "boolean"}
      },
      "required": ["note_id", "confirm"]
    }
  }
}
```

客户端渲染表单，收集用户回答，返回：

```json
{
  "action": "accept",
  "content": {"note_id": "note-14", "confirm": true}
}
```

有三种可能动作：`accept`（用户填写完成），`decline`（用户关闭表单），`cancel`（用户取消整个工具调用）。

表单 schema 必须是扁平的——v1 不支持嵌套对象。SDK 通常拒绝比单层结构更复杂的 schema。

### Elicitation：URL 模式（SEP-1036，实验性）

2025-11-25 新增。服务器发送 URL 而非 schema：

```json
{
  "method": "elicitation/create",
  "params": {
    "message": "Sign in to GitHub",
    "url": "https://github.com/login/oauth/authorize?client_id=..."
  }
}
```

客户端在浏览器打开该 URL，等待完成，用户返回后结束。适合 OAuth 流程、支付授权及需要文档签名的场景，表单不足以满足时使用。

漂移风险提示：SEP-1036 的响应格式尚未稳定；部分 SDK 返回回调 URL，部分返回完成令牌。生产使用前请务必查看 SDK 发布说明。

### 何时使用 elicitation

- 破坏性操作前的用户确认（破坏性提示 + 引导）。
- 消歧义（从多个候选中选一个）。
- 首次运行设置（API 密钥、目录、偏好）。
- 类 OAuth 流程（URL 模式）。

### 何时不使用 elicitation

- 工具必须参数，模型本可用自然语言询问的。此时用普通重提问，而不是引导对话框。
- 高频调用。引导中断对话，不应在循环内触发。
- 任何服务器可事后验证的操作。验证失败返回错误，让模型用文本向用户询问。

### 人机协作桥梁

Elicitation 和采样（sampling）共同支持 MCP 的“人机协作”模型。服务器代理循环可在等待用户输入（elicitation）或模型推理（sampling）时暂停。第 13 阶段 · 11 涵盖采样；本课涵盖引导。合用实现循环中完全控制。

## 练习范例

`code/main.py` 擴展了 notes 服务器功能：

- `roots/list` 响应，服务器会在收到根列表变更通知后重新查询。
- `notes_delete` 工具，当匹配多个笔记时使用 `elicitation/create` 进行歧义消除。
- `notes_setup` 工具，使用 URL 模式的引导打开首次配置页面（模拟）。
- 边界检查，拒绝操作根目录外的 URI。

演示运行三个场景：正常路径（匹配一个）、歧义消除（三个匹配，触发引导）、根外写入被拒绝。

## 发布内容

本课输出 `outputs/skill-elicitation-form-designer.md`。给定需要用户确认或消歧义的工具，技能设计引导表单 schema 和消息模板。

## 练习

1. 运行 `code/main.py`。触发歧义路径；确认模拟用户回答反馈到工具。

2. 新增工具 `notes_archive`，每次执行需要引导确认（破坏性提示）。体验用户体验：与模型以文本重询相比如何？

3. 为首次 OAuth 流程实现 URL 模式引导。留意漂移风险，增加 SDK 版本检测。

4. 扩展 `roots/list` 处理：通知到达时，服务器应原子性重新读取并重新扫描可能超出作用域的已打开文件句柄。

5. 阅读 GitHub 上 SEP-1036 讨论。找出影响服务器如何处理 URL 模式回调的一个未决问题。

## 关键词汇

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| Root | “授权边界” | 客户端允许服务器访问的 URI |
| `roots/list` | “服务器请求作用域” | 客户端返回当前的根路径集合 |
| `notifications/roots/list_changed` | “用户变更作用域” | 客户端通知根路径集合已改变 |
| Elicitation | “中途询问用户” | 服务器发起的结构化用户输入请求 |
| `elicitation/create` | “该方法” | 针对引导请求的 JSON-RPC 方法 |
| Form mode | “基于 Schema 的表单” | 在客户端 UI 中渲染的扁平 JSON Schema 表单 |
| URL mode | “浏览器重定向” | SEP-1036 实验性；打开 URL 并等待结果 |
| `accept` / `decline` / `cancel` | “用户响应结果” | 服务器处理的三种分支 |
| Disambiguation | “选一个” | 工具具有多个候选时的常用引导场景 |
| Flat form | “仅顶层属性” | 引导 schema 不支持嵌套 |

## 深入阅读

- [MCP — 客户端 Roots 规范](https://modelcontextprotocol.io/specification/draft/client/roots) — 根路径权威参考  
- [MCP — 客户端 Elicitation 规范](https://modelcontextprotocol.io/specification/draft/client/elicitation) — 引导权威参考  
- [Cisco — MCP 引导、结构化内容及 OAuth 增强新特性](https://blogs.cisco.com/developer/whats-new-in-mcp-elicitation-structured-content-and-oauth-enhancements) — 2025-11-25 新增特性讲解  
- [MCP — GitHub SEP-1036](https://github.com/modelcontextprotocol/modelcontextprotocol) — URL 模式引导提案（实验，存在漂移风险）  
- [The New Stack — 引导如何为 AI 工具带来人机协作](https://thenewstack.io/how-elicitation-in-mcp-brings-human-in-the-loop-to-ai-tools/) — 用户体验解析
