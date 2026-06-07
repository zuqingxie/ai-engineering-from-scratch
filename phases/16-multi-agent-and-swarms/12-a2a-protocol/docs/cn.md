# A2A — Agent-to-Agent 协议（Agent-to-Agent Protocol）

> Google 于 2025 年 4 月发布了 A2A；到 2026 年 4 月，规范已发布在 https://a2a-protocol.org/latest/specification/ ，并且获得了 150 多家机构的支持。A2A 是 MCP（Lesson 13）的横向补充：MCP 是垂直的（agent ↔ 工具），而 A2A 是点对点的（agent ↔ agent）。它定义了 Agent Card（发现机制）、带有工件（文本、结构化数据、视频）的任务、不透明的任务生命周期和身份认证。生产系统越来越多地将 MCP 与 A2A 配合使用。Google Cloud 在 2025-2026 年间将 A2A 支持集成到了 Vertex AI Agent Builder 中。

**类型:** 学习 + 构建  
**语言:** Python（stdlib，`http.server`，`json`）  
**前提:** 阶段16 · 04（原始模型）  
**时长:** 约75分钟

## 问题

你的 agent 需要调用另一台系统上的 agent。怎么做？你可以开启一个 HTTP 端点，定义一个定制的 JSON 结构，然后希望另一端能理解它。每对 agent 之间变成了一个定制的集成。

A2A 是该调用的通用传输协议。标准化的发现、标准化的任务模型、标准化的传输、标准化的工件。就像 HTTP+REST，但代理是一级公民。

## 概念

### 四个组成部分

**Agent Card。** 在 `/.well-known/agent.json` 的 JSON 文档，描述 agent：名称、技能、端点、支持的模态（modalities）、身份认证要求。通过读取 Agent Card 实现发现。

```text
GET https://agent.example.com/.well-known/agent.json
→ {
    "name": "code-review-agent",
    "skills": ["review-python", "review-typescript"],
    "endpoints": {
      "tasks": "https://agent.example.com/tasks"
    },
    "auth": {"type": "bearer"},
    "modalities": ["text", "structured"]
  }
```

**Task（任务）。** 工作单元。一个带生命周期的异步、有状态对象：`submitted → working → completed / failed / canceled`。客户端发送任务，并轮询或订阅更新。

**Artifact（工件）。** 任务生成的结果类型。文本、结构化 JSON、图片、视频、音频。工件有类型区分，使不同模态成为一级公民。

**Opaque lifecycle（不透明生命周期）。** A2A 不规定远程 agent 如何解决任务。客户端看到状态变迁和工件；实现层可以自由选择任何框架。

### MCP/A2A 的分工

- **MCP** （Lesson 13）：agent ↔ 工具。agent 通过 JSON-RPC 读写工具服务器。默认无状态。  
- **A2A**：agent ↔ agent。点对点协议；双方都是拥有自我推理能力的 agent。

生产多代理系统通常两者兼用。A2A 对等方调用本地的 MCP 工具。此分工保持职责清晰。

### 发现流程

```text
Client                     Agent server
  ├──GET /.well-known/agent.json──>
  <──Agent Card JSON─────────────
  ├──POST /tasks {skill, input}──>
  <──201 task_id, state=submitted
  ├──GET /tasks/{id}──────────────>
  <──state=working, 42% done──────
  ├──GET /tasks/{id}──────────────>
  <──state=completed, artifacts──
```

或者使用流式：通过 SSE 订阅 `/tasks/{id}/events` 以推送更新。

### 身份认证

A2A 支持三种常见模式：

- **Bearer token** — OAuth2 或不透明令牌。  
- **mTLS** — 双向 TLS，组织相互证明身份。  
- **Signed requests** — 对负载做 HMAC 签名。  

认证信息在 Agent Card 中声明；客户端可发现并遵守。

### 到 2026 年 4 月，150+ 家机构支持

企业的广泛采纳推动了 A2A 的规模。核心情况是：A2A 成为企业代理系统跨信任边界的通用方式。Google Cloud 推出了 Vertex AI Agent Builder 的 A2A 支持；微软的 Agent Framework 支持它；多数主流框架（LangGraph，CrewAI，AutoGen）都发布了 A2A 适配器。

### A2A 的优势所在

- **跨组织调用。** 公司 A 的 agent 调用公司 B 的 agent；没有 A2A，每对都需要定制协议。  
- **异构框架兼容。** LangGraph agent 调用 CrewAI agent，再调用自研 Python agent。A2A 实现了一致接口。  
- **类型化工件。** 视频结果、结构化 JSON、音频——都是一级公民。  
- **长时任务。** 不透明生命周期 + 轮询让数小时任务实现变得直接。

### A2A 的限制

- **对延迟敏感的微调用。** A2A 生命周期是异步的。子毫秒的 agent 间调用不适合，用直接 RPC。  
- **紧耦合同进程 agent。** 若两个 agent 在同一 Python 进程内运行，A2A 的 HTTP 往返过于繁重。  
- **小团队。** 规范开销真实存在；内部专用 agent 可能不需如此正规。

### A2A vs ACP, ANP, NLIP

2024-2026 年间出现数个相关规范：

- **ACP**（IBM/Linux 基金会）—— A2A 的前身，适用范围较窄。  
- **ANP**（Agent Network Protocol）—— 偏重对等发现，去中心化优先。  
- **NLIP**（Ecma 自然语言交互协议，2025 年 12 月标准化）—— 自然语言内容类型。

截至 2026 年 4 月，A2A 是采用最多的对等协议。详见 arXiv:2505.02279（Liu 等，《Agent Interoperability Protocols 调研》）的对比。

## 实践构建

`code/main.py` 实现了一个简化的 A2A 服务器和客户端，使用 `http.server` 和 JSON。服务器：

- 提供 `/.well-known/agent.json`，  
- 支持 `POST /tasks`，  
- 管理任务状态，  
- 在 `GET /tasks/{id}` 返回工件。

客户端：

- 获取 Agent Card，  
- 提交任务，  
- 轮询直到完成，  
- 读取工件。

运行：

```text
python3 code/main.py
```

脚本启动服务器后台线程，再对其运行客户端，展现完整流程：发现、提交、轮询、获取工件。

## 使用方法

`outputs/skill-a2a-integrator.md` 设计了一个 A2A 集成：Agent Card 内容、任务模式、认证选择、流式 vs 轮询。

## 部署指南

检查表：

- **锁定规范版本。** A2A 仍在演进，Agent Card 应声明协议版本。  
- **任务创建幂等。** 重复提交（网络重试）不应产生多个任务。  
- **工件模式。** 声明 agent 返回的形态；消费者应做校验。  
- **速率限制 + 认证。** A2A 面向公共；应用标准 Web 安全。  
- **失败任务死信队列。** 长期观测模式，识别重复故障类。

## 练习

1. 运行 `code/main.py`。确认客户端发现服务器并获得正确工件。  
2. 给服务器添加第二个技能（例如 “summarize”）。更新 Agent Card。编写客户端根据任务类型选择技能。  
3. 实现 SSE 流式端点：`/tasks/{id}/events` 输出状态变更。客户端需要做哪些调整？  
4. 阅读 A2A 规范（https://a2a-protocol.org/latest/specification/）。指出此示范未实现但规范要求的三项内容。  
5. 比较 A2A（Agent Card 发现）与 MCP（服务器端能力列表通过 `listTools`）。自描述 agent 与能力探测，有什么权衡？

## 关键术语

| 术语 | 人们的说法 | 真实含义 |
|------|----------------|------------------------|
| A2A | “Agent-to-agent” | 跨系统 agent 调用 agent 的对等协议。Google 2025 年发布。 |
| Agent Card | “Agent 的名片” | 存在 `/.well-known/agent.json`，描述技能、端点、认证。 |
| Task | “工作单元” | 异步有状态对象，带生命周期；完成后生成工件。 |
| Artifact | “结果” | 带类型输出：文本、结构化 JSON、图片、视频、音频。一流水平媒体。 |
| Opaque lifecycle | “怎么解决是 agent 自己的事” | 客户端看到状态变迁；服务器端可自由选框架与工具。 |
| Discovery | “发现 agent” | 通过 `GET /.well-known/agent.json` 返回 Agent Card。 |
| MCP vs A2A | “工具 vs 对等” | MCP: 垂直层 agent ↔ 工具。A2A: 水平层 agent ↔ agent。 |
| ACP / ANP / NLIP | “兄弟协议” | 相关规范；2026 年 A2A 为采用最多。 |

## 拓展阅读

- [A2A 规范](https://a2a-protocol.org/latest/specification/) — 官方规范  
- [Google Developers 博客 — A2A 发布](https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/) — 2025 年 4 月启动文章  
- [A2A GitHub 仓库](https://github.com/a2aproject/A2A) — 参考实现与 SDK  
- [Liu 等 — Agent Interoperability Protocols 调研](https://arxiv.org/html/2505.02279v1) — MCP、ACP、A2A、ANP 比较
