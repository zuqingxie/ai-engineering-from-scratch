# 毕业设计 — 构建完整的工具生态系统

> 第13阶段涵盖了所有部分。本毕业设计将这些部分整合成一个生产形态的系统：一个包含工具+资源+提示+任务+UI 的 MCP 服务器，边缘部署 OAuth 2.1，RBAC 网关，多服务器客户端，A2A 子代理调用，OTel 跟踪集成到收集器，CI 中的工具中毒检测，以及 AGENTS.md + SKILL.md 打包。完成后，你可以为每一个架构选择进行辩护。

**类型：** 构建  
**语言：** Python（stdlib，端到端生态系统驱动）  
**前置条件：** 第13阶段·01至21  
**时间：** ~120分钟

## 学习目标

- 组合一个 MCP 服务器，暴露工具、资源、提示和一个带 `ui://` 应用的任务。
- 使用 OAuth 2.1 网关作为服务器前端，实施 RBAC 和哈希固定。
- 编写一个多服务器客户端，实现带 OTel GenAI 属性的端到端跟踪。
- 将部分工作委派给 A2A 子代理；验证不透明性得以保留。
- 使用 AGENTS.md + SKILL.md 打包整个堆栈，使其他代理可以驱动它。

## 问题描述

交付“研究与报告”系统：

- 用户询问：“总结2026年三篇引用最多的arXiv关于代理协议的论文。”
- 系统：通过 MCP 搜索 arXiv；通过 A2A 委派论文摘要给专门的写作代理；聚合结果；以 MCP Apps `ui://` 资源渲染交互报告；将每一步记录到 OTel。

所有第13阶段的原语都会出现。这不是一个玩具——2026年Anthropic（Claude Research产品）、OpenAI（带 Apps SDK 的 GPTs）和第三方发布的生产研究助手系统正是这个形态。

## 概念

### 架构

```text
[user] -> [client] -> [gateway (OAuth 2.1 + RBAC)] -> [research MCP server]
                                                      |
                                                      +- MCP 工具：arxiv_search（纯功能）
                                                      +- MCP 资源：notes://recent
                                                      +- MCP 提示：/research_topic
                                                      +- MCP 任务：generate_report（长任务）
                                                      +- MCP Apps UI：ui://report/current
                                                      +- A2A 调用：writer-agent（tasks/send）
                                                      |
                                                      +- OTel GenAI 跟踪片段
```

### 跟踪层级

```text
agent.invoke_agent
 ├── llm.chat（启动）
 ├── mcp.call -> tools/call arxiv_search
 ├── mcp.call -> resources/read notes://recent
 ├── mcp.call -> prompts/get research_topic
 ├── a2a.tasks/send -> writer-agent
 │    └── 任务状态变迁（内部不透明）
 ├── mcp.call -> tools/call generate_report（任务增强）
 │    └── tasks/status 轮询
 │    └── tasks/result（完成，返回 ui:// 资源）
 └── llm.chat（最终合成）
```

一个追踪ID。每个片段都拥有正确的 `gen_ai.*` 属性。

### 安全态势

- OAuth 2.1 + PKCE，使用资源指标固定受众为网关。
- 网关持有上游凭证，用户永远看不到它们。
- RBAC：`alice` 拥有 `research:read`、`research:write` 权限，可调用所有工具。`bob` 只有 `research:read`，不可调用 `generate_report`。
- 固定描述清单：任何工具哈希变化的服务器都会被剔除。
- 双重规则审计：不允许工具同时结合不可信输入、敏感数据和关键操作。

### 渲染

最终的 `generate_report` 任务返回内容块和 `ui://report/current` 资源。客户端宿主（如 Claude Desktop 等）在沙箱 iframe 中渲染交互式仪表板。仪表板包含排序的论文列表、引用数以及用户点击任何论文时调用 `host.callTool('summarize_paper', {arxiv_id})` 的按钮。

### 打包

整个项目的目录结构：

```text
research-system/
  AGENTS.md                     # 项目约定
  skills/
    run-research/
      SKILL.md                  # 顶层工作流
  servers/
    research-mcp/               # MCP 服务器
      pyproject.toml
      src/
  agents/
    writer/                     # A2A 代理
  gateway/
    config.yaml                 # RBAC + 固定清单
```

用户通过 `docker compose up` 部署。Claude Code、Cursor、Codex 和 opencode 用户可通过调用 `run-research` skill 驱动系统。

### 第13阶段各课贡献

| 课程   | 毕业设计使用内容                               |
|--------|---------------------------------------------|
| 01-05  | 工具接口，提供者可移植性， 并行调用，模式，静态分析        |
| 06-10  | MCP 原语，服务器，客户端，传输协议，资源 + 提示               |
| 11-14  | 采样，根节点 + 引导，异步任务，`ui://` 应用             |
| 15-17  | 工具中毒，OAuth 2.1，网关 + 注册表                     |
| 18     | A2A 子代理委派                                  |
| 19     | OTel GenAI 跟踪                                |
| 20     | LLM 层路由网关                                  |
| 21     | SKILL.md + AGENTS.md 打包                        |

## 使用方法

`code/main.py` 将之前课程的模式拼接成完整的可运行演示。全部使用 stdlib，内存内执行，便于端到端阅读。运行了完整的研究与报告流程：与网关握手，模拟 OAuth 2.1，合并工具列表，将 generate_report 作为任务执行，A2A 调用写作代理，返回 ui:// 资源，发出 OTel 跟踪片段。

重点观察：

- 所有调用环节共享同一 trace id。
- 网关策略阻止第二位用户写操作。
- 任务生命周期为进行中 → 完成，返回文本和 ui:// 内容。
- A2A 调用的内部状态对协调器是不透明的。
- AGENTS.md 和 SKILL.md 是其他代理重现工作流所需的唯二文件。

## 交付成果

本课生成 `outputs/skill-ecosystem-blueprint.md`。基于产品需求（研究、总结、自动化），该技能描述完整架构：涉及哪些 MCP 原语、哪些网关控制、哪些 A2A 调用、哪些遥测以及如何打包。

## 练习

1. 运行 `code/main.py`。观察单一 trace id 和 span 的嵌套情况。统计演示涉及了多少第13阶段原语。

2. 扩展演示：添加第二个后端 MCP 服务器（如 `bibliography`），确认网关将其工具合并到同一命名空间。

3. 将伪造的 A2A 写作代理替换成在子进程中运行的真实代理。使用第19课的测试框架。

4. 在协调器与 LLM 之间的路由网关中添加 PII（个人身份信息）脱敏步骤。确认用户查询中的邮箱被擦除。

5. 为维护该系统的团队成员编写 AGENTS.md。它应在五分钟内阅读完毕，提供驱动毕业设计所需的所有信息，支持 Cursor 或 Codex 使用。

## 关键词

| 术语             | 通俗说法                        | 实际含义                                     |
|------------------|-------------------------------|----------------------------------------------|
| 毕业设计         | “第13阶段集成演示”             | 使用所有原语的端到端系统                     |
| 研究与报告       | “演示场景”                     | 搜索、总结、渲染模式                         |
| 生态系统         | “所有部分集合”                 | 服务器 + 客户端 + 网关 + 子代理 + 遥测 + 打包 |
| 跟踪层级         | “单一 trace id”                | 每个调用环节 span 共享同一 trace，父子关系通过 span id|
| 网关签发令牌     | “传递认证”                    | 客户端只见网关 token，网关持有上游凭证        |
| 合并命名空间     | “所有工具合为一组”             | 多服务器在网关处合并，哈希冲突时添加前缀      |
| 不透明边界       | “A2A 调用隐藏内部”             | 子代理的推理过程对协调器不可见                 |
| 三层堆栈         | “AGENTS.md + SKILL.md + MCP”  | 项目上下文 + 工作流 + 工具                    |
| 深度防御         | “多重安全层”                  | 固定哈希，OAuth，RBAC，双重规则，审计日志     |
| 规范符合矩阵     | “交付内容与规范对比表”         | 将交付项目映射到 2025-11-25 规范要求          |

## 延伸阅读

- [MCP — 规范 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25) — 综合参考  
- [MCP 博客 — 2026 路线图](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) — 协议发展方向  
- [a2a-protocol.org](https://a2a-protocol.org/latest/) — A2A v1.0 参考  
- [OpenTelemetry — GenAI 语义约定](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — 权威跟踪约定  
- [Anthropic — Claude Agent SDK 概览](https://code.claude.com/docs/en/agent-sdk/overview) — 生产代理运行时模式
