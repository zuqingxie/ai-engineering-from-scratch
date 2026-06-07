# 结业项目 13 — 带注册和治理的 MCP 服务器

> Model Context Protocol（模型上下文协议）在 2026 年不再是未来，而成为默认的工具使用规范。Anthropic、OpenAI、Google 及所有主流 IDE 都内置了 MCP 客户端。Pinterest 发布了内部的 MCP 服务器生态系统。AAIF Registry 在 `.well-known` 定义了能力元数据格式。AWS ECS 发布了参考的无状态部署方案。Block 的 goose-agent 将相同协议嵌入托管助手中。2026 年的生产形态是：StreamableHTTP 传输、OAuth 2.1 范围（scopes）、OPA 策略门控，以及一个让平台团队能发现、验证和启用服务器的注册中心。打造这个端到端系统。

**类型：** 结业项目  
**语言：** Python（服务器，通过 FastMCP）或 TypeScript（`@modelcontextprotocol/sdk`），Go（注册中心服务）  
**先决条件：** 第 11 阶段（LLM 工程）、第 13 阶段（工具和 MCP）、第 14 阶段（代理）、第 17 阶段（基础设施）、第 18 阶段（安全）  
**涉及阶段：** P11 · P13 · P14 · P17 · P18  
**时长：** 25 小时

## 问题

MCP 已成为工具使用的通用语言。Claude Code、Cursor 3、Amp、OpenCode、Gemini CLI 及所有托管代理现在都消费 MCP 服务器。生产中的挑战不在于编写服务器（FastMCP 让这变得简单），而是满足企业要求的规模化部署：每租户的 OAuth 范围、对破坏性工具的 OPA 策略、StreamableHTTP 无状态水平扩展、用于发现的注册中心、针对每次工具调用的审计日志。Pinterest 的内部 MCP 生态和 AAIF 注册中心规范定义了 2026 年的标杆。

你将构建一个 MCP 服务器，暴露 10 个内部工具（Postgres 只读、S3 列表、Jira、Linear、Datadog 等）、一个用于平台发现的注册中心 UI 以及对破坏性工具的人类审批关卡。负载测试展示 StreamableHTTP 的水平扩展能力。审计轨迹满足企业安全审查。

## 概念

MCP 2026 版规定 StreamableHTTP 为默认传输协议。与早期基于 stdio 和 SSE 的设计不同，StreamableHTTP 默认无状态：单一 HTTP 端点接受 JSON-RPC 请求，流式返回响应，支持长连接以接收通知。无状态意味着可在负载均衡器后面水平扩展。

授权采用 OAuth 2.1，每个工具使用独立的 scope。令牌携带诸如 `jira:read`、`s3:list`、`postgres:query:readonly` 等权限。MCP 服务器在工具调用时检查权限，而不仅是会话启动时。对于高风险工具，服务器会拒绝未在最近 N 分钟内通过 Slack 审核卡提升到 `approved:by:human` 范围的调用。

注册中心是独立服务。每个 MCP 服务器都会暴露 `.well-known/mcp-capabilities` 文档，包括工具清单、传输 URL、认证需求。注册中心会轮询、验证并索引这些信息。平台团队通过注册中心 UI 看到可用工具、所需权限以及拥有团队。

## 架构

```text
MCP 客户端（Claude Code、Cursor 3 等）
          |
          v
基于 HTTPS 的 StreamableHTTP（JSON-RPC + 流式）
          |
          v
负载均衡器后的 MCP 服务器（FastMCP）
          |
   +------+------+---------+----------+------------+
   v             v         v          v            v
Postgres    S3 列表    Jira       Linear     Datadog
（只读）    （分页）    （读取）   （读取）   （查询）
          |
   +------+-------------+
   v                    v
 OPA 策略门控     破坏性工具 MCP（独立服务器）
                        |
                        v
                   通过 Slack 的人工审批
                        |
                        v
                   审计日志（追加且每租户隔离）

  注册中心服务
     |
     v  从各服务器获取 GET /.well-known/mcp-capabilities
     v
     UI：搜索 / 验证 / 启用-禁用 / 所有权管理
```

## 技术栈

- 服务器框架：FastMCP（Python）或 `@modelcontextprotocol/sdk`（TypeScript）  
- 传输：基于 HTTPS 的 StreamableHTTP（无状态）  
- 认证：OAuth 2.1，采用 SPIFFE/SPIRE 实现工作负载身份  
- 策略：每工具使用 OPA / Rego 规则；每次请求调用策略决策服务  
- 注册中心：自托管，消费各服务器的 `.well-known/mcp-capabilities` 清单  
- 人工审批：针对破坏性工具，使用 Slack 交互式消息实现  
- 部署：AWS ECS Fargate 或 Fly.io，每租户单服务器或共享服务器但租户隔离  
- 审计：每租户存储结构化 JSONL，记录每次调用的全流程  

## 实现步骤

1. **工具接口。** 暴露 10 个内部工具：Postgres 只读查询、S3 列出对象、Jira 搜索与获取、Linear 搜索与获取、Datadog 指标查询、PagerDuty 值班查询、GitHub 只读、Notion 搜索、Slack 搜索、Salesforce 读取。每个工具均定义类型化 schema 和对应权限标签。

2. **FastMCP 服务器。** 挂载工具，配置 StreamableHTTP 传输，增加 OAuth 令牌检测和权限校验中间件。

3. **OPA 策略。** 为每个工具编写 Rego 策略：定义调用权限范围，个人敏感信息（PII）脱敏规则，载荷大小限制。每次调用时决策服务执行策略。

4. **注册中心服务。** 使用 Go 或 TypeScript 实现，周期轮询注册服务器的 `.well-known/mcp-capabilities`，用 JSON Schema 验证，并提供列表、搜索、验证、启用/禁用及所有权管理 UI。

5. **能力清单。** 每个服务器暴露 `.well-known/mcp-capabilities`，包括工具列表、认证需求、传输 URL、拥有团队、SLO（服务水平目标）。

6. **破坏性工具隔离。** 会修改状态的工具（如 Jira 创建，Linear 创建，Postgres 写入）运行在第二个 MCP 服务器上，采用更严格的认证流程：令牌必须含有由 Slack 卡片审批后 15 分钟内有效的 `approved:by:human` 权限。

7. **审计日志。** 追加式 JSONL 按租户存储：`{timestamp, user, tool, args_redacted, response_redacted, outcome}`。写入前通过 Presidio 进行 PII 脱敏。

8. **负载测试。** 100 个客户端并发访问 StreamableHTTP。通过增加第二个副本演示水平扩展，展示负载均衡器无会话粘性地分发请求。

9. **合规测试。** 使用官方 MCP 合规测试套件对两个服务器进行测试，全部强制性部分通过。

## 使用示例

```bash
$ curl -H "Authorization: Bearer eyJhbGc..." \
       -X POST https://mcp.internal.example.com/ \
       -d '{"jsonrpc":"2.0","method":"tools/call",
            "params":{"name":"postgres.readonly","arguments":{"sql":"SELECT 1"}}}'
[registry]   capability validated: postgres.readonly v1.2
[policy]    scope postgres:query:readonly present; allowed
[audit]     logged: user=u42 tool=postgres.readonly outcome=ok
response:    { "result": { "rows": [[1]] } }
```

## 交付物

`outputs/skill-mcp-server.md` 描述了交付物。一个生产级别的 MCP 服务器 + 注册中心 + 审计层，支持内部工具的 OAuth 2.1 权限及 OPA 策略门控。

| 权重 | 标准 | 测量方式 |
|:-:|---|---|
| 25 | 规范符合 | StreamableHTTP + 能力清单通过 MCP 合规测试 |
| 20 | 安全 | 全工具权限执行，OPA 策略覆盖，密钥管理规范 |
| 20 | 可观察性 | 每次调⽤的审计日志，含 PII 脱敏 |
| 20 | 可扩展性 | 100 客户端负载测试证明水平扩展 |
| 15 | 注册中心用户体验 | 支持发现 / 验证 / 启用-禁用工作流 |
| **100** | | |

## 练习

1. 新增工具（Confluence 搜索）。通过注册中心验证流程交付，无需改动核心服务器。

2. 编写 OPA 策略，对包含 `email`、`ssn`、`phone` 字段的 Postgres 查询结果进行脱敏。用探测查询测试。

3. 本地对比 StreamableHTTP 与 stdio 的延迟性能，汇报每次调用的 p50/p95 延时。

4. 实现每租户限额：每租户每工具每分钟最大调用次数。通过第二条 OPA 规则执行强制。

5. 使用 [mcp-conformance-tests](https://github.com/modelcontextprotocol/conformance) 运行 MCP 合规套件，修复所有失败。

## 关键词解释

| 术语 | 俗称 | 实际含义 |
|------|--------|------------|
| StreamableHTTP | “2026 MCP 传输” | 无状态 HTTP + 流式；取代 SSE + stdio 用于网络服务器 |
| 能力清单（Capability manifest） | “Well-known 文档” | `.well-known/mcp-capabilities`，包含工具列表、认证、传输 URL |
| OPA / Rego | “策略引擎” | Open Policy Agent，用于基于外部规则授权工具调用 |
| 权限提升（Scope elevation） | “人工审批” | 通过 Slack 审批临时授予的短效权限，破坏性工具必需 |
| 注册中心（Registry） | “工具发现” | 通过能力清单索引 MCP 服务器的服务 |
| 工作负载身份（Workload identity） | “SPIFFE / SPIRE” | 用于 OAuth 令牌签发的密码学服务身份 |
| 合规套件（Conformance suite） | “规范测试” | 官方 MCP 测试套件，验证 StreamableHTTP 与工具清单准确性 |

## 延伸阅读

- [Model Context Protocol 2026 路线图](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) — StreamableHTTP、能力元数据、注册中心  
- [AAIF MCP 注册中心规范](https://github.com/modelcontextprotocol/registry) — 2026 年注册中心规范  
- [AWS ECS 参考部署](https://aws.amazon.com/blogs/containers/deploying-model-context-protocol-mcp-servers-on-amazon-ecs/) — 参考生产部署  
- [Pinterest 内部 MCP 生态系统](https://www.infoq.com/news/2026/04/pinterest-mcp-ecosystem/) — 参考内部部署  
- [Block `goose` MCP 使用](https://block.github.io/goose/) — 参考代理消费模式  
- [FastMCP](https://github.com/jlowin/fastmcp) — Python 服务器框架  
- [Open Policy Agent](https://www.openpolicyagent.org/) — 策略引擎参考  
- [SPIFFE / SPIRE](https://spiffe.io) — 工作负载身份参考
