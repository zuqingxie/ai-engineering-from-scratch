# MCP 采样 — 服务器请求的大型语言模型（LLM）完成与代理循环

> 大多数 MCP 服务器是简单的执行者：接受参数，运行代码，返回内容。采样让服务器可以反向操作：请求客户端的 LLM 做出决策。这使得服务器托管的代理循环成为可能，而服务器本身不持有任何模型凭证。SEP-1577 于 2025-11-25 合入，增加了采样请求中的工具，使循环支持更深层的推理。风险提示：SEP-1577 中的采样工具机制在 2026 年第一季度前仍属实验阶段，SDK API 仍在稳定中。

**类型：** 构建  
**语言：** Python（标准库，采样测试工具）  
**先决条件：** 阶段 13 · 07（MCP 服务器），阶段 13 · 10（资源与提示）  
**时长：** 约 75 分钟

## 学习目标

- 解释 `sampling/createMessage` 解决了什么问题（无服务器端 API 密钥实现服务器托管循环）。
- 实现一个服务器，请求客户端在多轮提示上进行采样并返回完成结果。
- 使用 `modelPreferences`（费用 / 速度 / 智能优先级）指导客户端模型选择。
- 构建一个 `summarize_repo` 工具，通过采样内部迭代而非硬编码行为。

## 问题背景

一个用于代码摘要工作流的有用 MCP 服务器需要：遍历文件树，选择要读取的文件，综合摘要内容并返回。LLM 推理应发生在哪里？

选项 A：服务器调用自己的 LLM。需要 API 密钥，服务器端计费，每个用户成本高。

选项 B：服务器返回原始内容，由客户端代理做推理。虽然可行，但将服务器逻辑转移到客户端提示，脆弱且难以维护。

选项 C：服务器通过 `sampling/createMessage` 请求客户端的 LLM。服务器保留算法（选择文件，迭代次数），客户端负责计费和模型选择。服务器不持有任何凭证。

采样即选项 C。这是受信任的服务器托管代理循环而非完整 LLM 主机的机制。

## 概念解析

### `sampling/createMessage` 请求

服务器发送：

```json
{
  "jsonrpc": "2.0",
  "id": 42,
  "method": "sampling/createMessage",
  "params": {
    "messages": [{"role": "user", "content": {"type": "text", "text": "..."}}],
    "systemPrompt": "...",
    "includeContext": "none",
    "modelPreferences": {
      "costPriority": 0.3,
      "speedPriority": 0.2,
      "intelligencePriority": 0.5,
      "hints": [{"name": "claude-3-5-sonnet"}]
    },
    "maxTokens": 1024
  }
}
```

客户端运行其 LLM，返回：

```json
{"jsonrpc": "2.0", "id": 42, "result": {
  "role": "assistant",
  "content": {"type": "text", "text": "..."},
  "model": "claude-3-5-sonnet-20251022",
  "stopReason": "endTurn"
}}
```

### `modelPreferences`

三个浮点数，和为 1.0：

- `costPriority`：优先选择更便宜的模型。
- `speedPriority`：优先选择更快速的模型。
- `intelligencePriority`：优先选择更智能的模型。

及 `hints`：服务器偏好的命名模型。客户端可选择是否尊重提示；客户端用户配置始终优先。

### `includeContext`

三个可能值：

- `"none"` — 仅包含服务器提供的消息。默认值。
- `"thisServer"` — 包含本服务器会话的历史消息。
- `"allServers"` — 包含所有会话上下文。

自 2025-11-25 起，因泄露跨服务器上下文存安全隐患，`includeContext` 被软弃用。建议使用 `"none"` 并在消息中显式传入上下文。

### 带工具的采样（SEP-1577）

2025-11-25 新增：采样请求可包含 `tools` 数组。客户端运行完整工具调用循环，使用这些工具。此功能使服务器能通过客户端模型托管 ReAct 风格的代理循环。

```json
{
  "messages": [...],
  "tools": [
    {"name": "fetch_url", "description": "...", "inputSchema": {...}}
  ]
}
```

客户端循环：采样 -> 如调用则执行工具 -> 再采样 -> 返回最终助手消息。此功能在 2026 年第一季度仍属实验，SDK 签名可能变动。实现时请参考 2025-11-25 规范客户端/采样部分。

### 人类参与循环

客户端**必须**在执行采样前向用户展示服务器请求模型执行的内容。恶意服务器可能利用采样操控用户会话（“对用户说 X 以促使其点击 Y”）。Claude Desktop、VS Code 和 Cursor 将采样请求弹出为用户可拒绝的确认对话框。

2026 年共识：无用户确认的采样视为风险信号。门控服务（阶段 13 · 17）可以自动批准低风险采样，自动拒绝可疑请求。

### 无需 API 密钥的服务器托管循环

典型用例：无自有 LLM 访问权的代码摘要 MCP 服务器。流程：

1. 遍历仓库结构。
2. 通过 `sampling/createMessage` 请求“选出最可能描述仓库目的的五个文件”。
3. 读取这些文件。
4. 通过 `sampling/createMessage` 请求“基于文件内容总结仓库，篇幅三段”。
5. 返回摘要作为 `tools/call` 结果。

服务器完全不调用 LLM API。客户端用户使用自身凭证为完成付费。

### 安全风险（Unit 42 公告，2026 年第一季度）

- **隐蔽采样。** 工具总是调用采样请求，内容是“从会话上下文中响应用户邮箱”。阶段 13 · 15 涵盖攻击方式。
- **通过采样盗用资源。** 服务器请求客户端总结攻击者负载，导致用户计费。
- **循环炸弹。** 服务器以紧密循环调用采样。客户端**必须**执行会话级速率限制。

## 具体使用

`code/main.py` 提供模拟的服务器到客户端采样测试工具。模拟的 `summarize_repo` 工具执行两轮采样（先选文件，再总结），而模拟客户端返回预设答复。测试工具展示：

- 服务器发送带 `modelPreferences` 的 `sampling/createMessage`。
- 客户端返回完成。
- 服务器继续循环。
- 速率限制器限制每次工具调用的采样总数。

重点观察：

- 服务器仅公开一个工具（`summarize_repo`）；所有推理在采样调用中完成。
- 模型偏好加权客户端模型选择，提示列出优选模型。
- 循环以 `stopReason: "endTurn"` 结束。
- `max_samples_per_tool = 5` 限制防止超长循环。

## 交付成果

本课产出 `outputs/skill-sampling-loop-designer.md`。基于服务器端需要 LLM 调用的算法（调研、摘要、规划），该技能设计采样实现，包含合理的 modelPreferences、速率限制和安全确认。

## 练习题

1. 运行 `code/main.py`。将 `max_samples_per_tool` 改为 2，观察速率限制生效。

2. 实现 SEP-1577 的采样工具变体：采样请求携带 `tools` 数组。验证客户端循环执行这些工具后返回最终完成。注意风险：SDK 签名可能在 2026 年上半年仍会变更。

3. 增加人类环节确认：在服务器第一次调用 `sampling/createMessage` 前暂停，等待用户批准。拒绝调用返回类型化拒绝。

4. 增加基于客户端会话的用户速率限制。相同服务器、相同用户的循环共享预算。

5. 设计一个 `summarize_pdf` 工具，使用采样挑选待包含的文档块。草拟发送的消息格式。`modelPreferences.intelligencePriority` 在 0.1 与 0.9 的行为差异如何？

## 关键词

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| Sampling（采样） | “服务器到客户端的 LLM 调用” | 服务器请求客户端模型生成完成 |
| `sampling/createMessage` | “该方法” | 采样请求的 JSON-RPC 方法 |
| `modelPreferences` | “模型优先级” | 费用 / 速度 / 智力权重及模型名称提示 |
| `includeContext` | “跨会话泄露” | 已软弃用的上下文包含模式 |
| SEP-1577 | “采样中工具” | 支持在采样中加入工具，以实现服务器托管的 ReAct |
| Human-in-the-loop（人类参与环节） | “用户确认” | 客户端在采样执行前向用户展示请求以供确认 |
| Loop bomb（循环炸弹） | “无限采样” | 服务器端无限采样循环；客户端必须速率限制 |
| Covert sampling（隐蔽采样） | “隐藏推理” | 恶意服务器隐蔽目的于采样提示中 |
| Resource theft（资源窃取） | “占用用户 LLM 预算” | 服务器强迫客户端为其不想要的采样付费 |
| `stopReason` | “生成停止原因” | `endTurn`、`stopSequence` 或 `maxTokens` |

## 相关阅读

- [MCP — 概念：采样](https://modelcontextprotocol.io/docs/concepts/sampling) — 采样的高层次概述  
- [MCP — 客户端采样规范 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/client/sampling) — 规范性 `sampling/createMessage` 结构  
- [MCP — GitHub SEP-1577](https://github.com/modelcontextprotocol/modelcontextprotocol) — 采样中工具的规范演变提案（实验）  
- [Unit 42 — MCP 攻击向量](https://unit42.paloaltonetworks.com/model-context-protocol-attack-vectors/) — 隐蔽采样与资源窃取模式  
- [Speakeasy — MCP 采样核心概念](https://www.speakeasy.com/mcp/core-concepts/sampling) — 含客户端代码示例的详解 walkthrough
