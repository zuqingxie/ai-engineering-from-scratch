# A2A — 代理到代理协议

> MCP 是代理到工具。A2A（Agent2Agent）是代理到代理——一个开放协议，允许基于不同框架构建的不透明代理协作。由 Google 于 2025 年 4 月发布，2025 年 6 月捐赠给 Linux 基金会，2026 年 4 月达到 v1.0，拥有包含 AWS、Cisco、Microsoft、Salesforce、SAP 和 ServiceNow 在内的 150+ 支持者。它吸收了 IBM 的 ACP 并添加了 AP2 支付扩展。本课覆盖代理卡（Agent Card）、任务生命周期和两种传输绑定。

**类型：** 构建  
**语言：** Python（标准库，代理卡 + 任务测试框架）  
**先决条件：** 第 13 阶段 · 06（MCP 基础），第 13 阶段 · 08（MCP 客户端）  
**时间：** ~75 分钟  

## 学习目标

- 区分代理到工具（MCP）与代理到代理（A2A）的使用场景。
- 在 `/.well-known/agent.json` 发布包含技能和端点元数据的代理卡。
- 演示任务生命周期（submitted → working → input-required → completed / failed / canceled / rejected）。
- 使用带有 Parts（文本、文件、数据）的消息和工件作为输出。

## 问题描述

客户服务代理需要将报告撰写任务委派给专业的写作代理。A2A 出现前的选项有：

- 自定义 REST API。可行但每对配对都是独立开发。
- 共享代码库。要求两个代理运行相同框架。
- MCP。不适用：MCP 用于调用工具，不适合两个代理协作且保持各自内部推理不透明。

A2A 填补了这一空白。它将交互建模为一个代理向另一个代理发送任务，包含生命周期、消息和工件。被调用代理的内部状态保持不透明——调用方只看到任务状态转换和最终输出。

A2A 是“让跨框架代理相互通信”的协议，不替代 MCP；两者互补。

## 概念

### 代理卡（Agent Card）

每个符合 A2A 规范的代理在 `/.well-known/agent.json` 发布代理卡：

```json
{
  "schemaVersion": "1.0",
  "name": "research-agent",
  "description": "Summarizes academic papers and drafts citations.",
  "url": "https://research.example.com/a2a",
  "version": "1.2.0",
  "skills": [
    {
      "id": "summarize_paper",
      "name": "Summarize a paper",
      "description": "Read a paper PDF and produce a 3-paragraph summary.",
      "inputModes": ["text", "file"],
      "outputModes": ["text", "artifact"]
    }
  ],
  "capabilities": {"streaming": true, "pushNotifications": true}
}
```

发现基于 URL：抓取该卡，获取 A2A 端点 URL，列举技能。

### 签名代理卡（AP2）

AP2 扩展（2025 年 9 月）为代理卡添加加密签名。发布者用 JWT 签名自己的卡，消费者验证。防止冒充。

### 任务生命周期

```text
submitted -> working -> completed | failed | canceled | rejected
             -> input_required -> working（通过消息循环）
```

客户端通过 `tasks/send` 发起。被调用代理状态变化；客户端通过 SSE 订阅状态更新或轮询。

### 消息与 Parts

消息携带一个或多个 Parts：

- `text` — 纯文本内容。
- `file` — 带 mimeType 的 base64 二进制块。
- `data` — 类型化 JSON 负载（结构化输入供被调用代理）。

示例：

```json
{
  "role": "user",
  "parts": [
    {"type": "text", "text": "Summarize this paper."},
    {"type": "file", "file": {"name": "paper.pdf", "mimeType": "application/pdf", "bytes": "..."}},
    {"type": "data", "data": {"targetLength": "3 paragraphs"}}
  ]
}
```

### 工件（Artifacts）

输出是工件，不是原始字符串。工件是有名称、有类型的输出：

```json
{
  "name": "summary",
  "parts": [{"type": "text", "text": "..."}],
  "mimeType": "text/markdown"
}
```

工件可分块流式传输。调用方聚合块。

### 两种传输绑定

1. **基于 HTTP 的 JSON-RPC。** `/a2a` 端点，POST 请求，支持可选 SSE 流。默认绑定。  
2. **gRPC。** 适合企业环境的原生 gRPC。

两种绑定携带相同逻辑消息格式。

### 不透明性保护

关键设计原则：被调用代理的内部状态不透明。调用方仅见任务状态和工件。被调用代理的思路链、工具调用、子代理委派等全不可见。这区别于 MCP，后者工具调用透明。

原因：A2A 允许竞争对手协作而不泄露内部机制。A2A 可“调用某客户服务代理”而无需调用者知道该代理的具体实现。

### 时间线

- **2025-04-09。** Google 发布 A2A。  
- **2025-06-23。** 捐赠至 Linux 基金会。  
- **2025-08。** 吸收 IBM 的 ACP。  
- **2025-09。** 发布 AP2 扩展（代理支付）。  
- **2026-04。** v1.0 发布，获得 150+ 组织支持。

### 与 MCP 的关系

| 维度        | MCP                  | A2A                      |
|-------------|----------------------|--------------------------|
| 使用场景    | 代理到工具           | 代理到代理               |
| 不透明性    | 工具调用透明         | 内部推理不透明           |
| 典型调用方  | 代理运行时           | 另一个代理               |
| 状态        | 工具调用结果         | 带生命周期的任务         |
| 授权        | OAuth 2.1（第 13 阶段 · 16）| JWT 签名代理卡（AP2）     |
| 传输        | 标准输入输出 / 可流 HTTP | 基于 HTTP 的 JSON-RPC / gRPC |

调用特定工具时用 MCP。委派整个任务时用 A2A。许多生产系统两者并用：代理用 MCP 作为工具层，用 A2A 作为协作层。

## 使用示例

`code/main.py` 实现了一个最小 A2A 测试框架：研究代理发布代理卡，写作代理接收带 PDF 和文本指令的 `tasks/send`，状态转变工作中 → 需要输入 → 工作中 → 完成，返回文本工件。全部标准库实现；用内存传输聚焦消息格式。

重点观察：

- 代理卡 JSON 格式。  
- 任务 ID 分配和状态转换。  
- 混合类型的消息 Parts。  
- 任务中途的 input-required 分支。  
- 完成时返回工件。  

## 交付成果

本课生成 `outputs/skill-a2a-agent-spec.md`。针对应能被其他代理调用的新代理，该技能输出代理卡 JSON、技能模式和端点蓝图。

## 练习

1. 运行 `code/main.py`。跟踪完整任务生命周期，包括被调用代理请求澄清时的 input-required 暂停。

2. 添加签名代理卡。使用 HMAC 对代理卡规范 JSON 签名。编写验证器，并确认篡改后验证失败。

3. 实现任务流：写作代理通过 SSE 发送三个增量工件块，调用方累积它们。

4. 设计一个封装 MCP 服务器的 A2A 代理。将每个 MCP 工具映射为一个 A2A 技能。注意权衡 —— 会丧失什么不透明性？

5. 阅读 A2A v1.0 公告，识别截至 2026 年 4 月尚未有任何框架实现的功能。（提示：与多跳任务委派相关。）

## 关键词

| 术语       | 大众称呼                  | 实际含义                          |
|------------|---------------------------|----------------------------------|
| A2A        | “代理到代理协议”          | 用于不透明代理协作的开放协议      |
| 代理卡（Agent Card） | “`.well-known/agent.json`” | 描述代理技能和端点的发布元数据      |
| 技能（Skill） | “可调用单元”              | 代理支持的命名操作（类似 MCP 工具） |
| 任务（Task） | “委派单元”                | 带生命周期和最终工件的工作项        |
| 消息（Message） | “任务输入”                | 携带 Parts（文本、文件、数据）    |
| 部分（Part） | “类型化块”                | 消息里的 `text` / `file` / `data` 元素 |
| 工件（Artifact） | “任务输出”                | 任务完成时返回的命名且类型化输出     |
| AP2        | “代理支付协议”             | 代理卡签名扩展，用于信任与支付       |
| 不透明性（Opacity） | “黑盒协作”                | 被调用代理的内部对调用方隐藏         |
| 需要输入（Input-required） | “任务暂停”                | 任务生命周期状态，代理需更多信息时   |

## 延伸阅读

- [a2a-protocol.org](https://a2a-protocol.org/latest/) — A2A 正式规范  
- [a2aproject/A2A — GitHub](https://github.com/a2aproject/A2A) — 参考实现和 SDK  
- [Linux Foundation — A2A 发布新闻稿](https://www.linuxfoundation.org/press/linux-foundation-launches-the-agent2agent-protocol-project-to-enable-secure-intelligent-communication-between-ai-agents) — 2025 年 6 月治理转移  
- [Google Cloud — A2A 协议升级](https://cloud.google.com/blog/products/ai-machine-learning/agent2agent-protocol-is-getting-an-upgrade) — 路线图及合作伙伴动态  
- [Google Dev — A2A 1.0 里程碑](https://discuss.google.dev/t/the-a2a-1-0-milestone-ensuring-and-testing-backward-compatibility/352258) — v1.0 发布及向后兼容指南
