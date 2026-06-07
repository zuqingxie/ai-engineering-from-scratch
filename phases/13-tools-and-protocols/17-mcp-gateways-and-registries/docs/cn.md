# MCP 网关和注册表 — 企业控制平面

> 企业不能让每个开发者随意安装 MCP 服务器。网关负责集中认证、RBAC（基于角色的访问控制）、审计、限流、缓存和工具投毒检测，然后将合并后的工具表面作为单一 MCP 端点暴露。官方 MCP 注册表（Anthropic + GitHub + PulseMCP + Microsoft，命名空间验证）是权威上游。本课介绍网关的位置，演示最小实现，并调研 2026 年供应商生态。

**类型：** 学习  
**语言：** Python（标准库，最小网关）  
**先决条件：** 第 13 阶段 · 15 节（工具投毒），第 13 阶段 · 16 节（OAuth 2.1）  
**时长：** ~45 分钟

## 学习目标

- 解释 MCP 网关的位置（介于 MCP 客户端和多个后端 MCP 服务器之间）。
- 实现网关的五大职责：认证（auth），RBAC，审计，限流，策略。
- 在网关层强制执行固定工具哈希清单（manifest）。
- 区分官方 MCP 注册表与元注册表（Glama、MCPMarket、MCP.so、Smithery、LobeHub）。

## 问题描述

一家财富 500 强公司拥有 30 个批准的 MCP 服务器，5000 名开发者，合规和审计需求，以及希望集中策略的安全团队。允许所有开发者在 IDE 内随意安装服务器是行不通的。

网关模式：

1. 网关作为一个可流式 HTTP 端点运行，供开发者连接。
2. 网关持有每个后端 MCP 服务器的凭证。
3. 开发者的每个请求通过网关自身的 OAuth 进行认证和限权。
4. 网关将调用路由到后端服务器，并应用策略。
5. 所有调用均被记录用于审计。

Cloudflare MCP Portals、Kong AI Gateway、IBM ContextForge、MintMCP、TrueFoundry、Envoy AI Gateway —— 都在 2025-2026 年发布了网关或网关功能。

与此同时，官方 MCP 注册表作为权威上游上线： curated（策划），经过命名空间验证、反向 DNS 命名的服务器，网关可以拉取。元注册表（Glama、MCPMarket、MCP.so、Smithery、LobeHub）聚合多个来源的服务器。

## 概念

### 五大网关职责

1. **认证（Auth）。** 使用 OAuth 2.1 识别开发者；映射到用户角色。
2. **基于角色的访问控制（RBAC）。** 按用户策略：允许访问的服务器、工具和权限范围。
3. **审计（Audit）。** 每次调用都记录调用者、内容、时间、结果。
4. **限流（Rate limit）。** 针对用户/工具/服务器的调用上限，防止滥用。
5. **策略（Policy）。** 拒绝被投毒的描述，执行“二人规则”（Rule of Two），脱敏个人身份信息（PII）。

### 网关作为单一端点

对开发者而言，网关看起来像一个 MCP 服务器。实际上内部会路由到多个后端。会话 ID（第 13 阶段 · 09 节）在边界处重写。

### 凭证保管（Credential vaulting）

开发者永远看不到后端的令牌。网关保管（或代理给身份提供商）。一个在网关拥有 `notes:read` 权限的开发者可以通过网关的后端凭证间接访问 notes MCP 服务器 —— 但必须遵守绑定了转发访问的策略。

### 网关层的工具哈希固定（Tool-hash pinning）

网关保存批准的工具描述的清单（SHA256 哈希）。在发现时，拉取每个后端的 `tools/list`，比对哈希，移除描述有变的工具。这是来自第 13 阶段 · 15 节的防拉地毯（rug-pull）攻击的中心化防护。

### 策略即代码（Policy-as-code）

高级网关使用 OPA/Rego、Kyverno 或 Styra 来表达策略。比如 “用户 `alice` 只允许在 org `acme` 的仓库调用 `github.open_pr`” 这样的规则以声明式编码。简单网关用手写 Python 实现，两者都有效。

### 会话感知路由（Session-aware routing）

当用户会话包含多个服务器时，网关进行多路复用：开发者端一个 MCP 会话包含 N 个后端会话，每个服务器一个。任何后端的通知都通过网关传递至开发者会话。

### 命名空间合并

网关合并所有后端的工具命名空间，通常使用冲突前缀。例：`github.open_pr`、`notes.search`。保证路由无歧义。

### 注册表

- **官方 MCP 注册表 (`registry.modelcontextprotocol.io`)。** 由 Anthropic、GitHub、PulseMCP、Microsoft 管理。命名空间验证（反向 DNS：`io.github.user/server`）。预先过滤基础质量。
- **Glama。** 聚焦搜索的元注册表，聚合众多来源。
- **MCPMarket。** 偏商业的目录，含供应商列表。
- **MCP.so。** 社区目录，开放提交。
- **Smithery。** 类包管理器的安装流程。
- **LobeHub。** 集成于其 LobeChat 应用的 UI 注册表。

企业网关默认从官方注册表拉取，允许管理员从元注册表添加，拒绝未固定哈希的内容。

### 反向 DNS 命名

官方注册表要求公有服务器以反向 DNS 命名：`io.github.alice/notes`。命名空间防止占位，明确信任委托。

### 供应商调研（2026 年 4 月）

| 供应商                | 优势                             |
|-------------------|--------------------------------|
| Cloudflare MCP Portals | 边缘托管；集成 OAuth；免费层          |
| Kong AI Gateway       | Kubernetes 原生；细粒度策略；日志支持 OpenTelemetry |
| IBM ContextForge      | 企业身份访问管理（IAM）；合规；审计导出       |
| TrueFoundry           | 偏向 DevOps；指标优先                   |
| MintMCP               | 面向开发者平台                      |
| Envoy AI Gateway      | 开源；可定制过滤器                    |

第 17 阶段（生产基础设施）将更深入探讨网关操作。

## 使用它

`code/main.py` 提供 ~150 行的最小网关实现：用伪造 Bearer 令牌认证用户，持有每用户 RBAC 策略，将请求路由至两个后端 MCP 服务器，写入审计日志，执行限流，拒绝描述哈希与固定清单不符的后端工具。

重点查看：

- 以 `user_id` 为键，授权 `server_tool` 的 `RBAC` 字典。
- `AUDIT_LOG` 是只能追加的事件列表。
- 限流对每个用户使用令牌桶算法。
- 固定清单是 `server::tool -> hash` 字典。

## 部署它

本课产生 `outputs/skill-gateway-bootstrap.md`。基于企业 MCP 规划（用户、后端、合规），该技能生成网关配置规范。

## 练习

1. 运行 `code/main.py`。分别以允许用户、禁止用户、超出限流的爆发情形发起调用，验证三种流程。

2. 添加一条策略，返回给客户端结果前脱敏 PII。用简单正则匹配类似社保号（SSN）的字符串；注意遗漏（邮箱、电话）。

3. 扩展审计日志以输出 OpenTelemetry GenAI 跟踪。第 13 阶段 · 20 节有对应属性。

4. 设计一个 50 人团队的 RBAC 策略，管理五个后端（notes、github、postgres、jira、slack）。谁拥有读权限？谁拥有写权限？

5. 通读 Cloudflare 企业 MCP 文章。指出官方 stdlib 网关未实现的 Cloudflare 功能。

## 关键词

| 词汇               | 大众描述           | 实际含义                        |
|------------------|------------------|-----------------------------|
| 网关（Gateway）       | “MCP 代理”         | 客户端与后端间的集中服务器            |
| 凭证保管（Credential vaulting） | “后端令牌仅保存在服务器端” | 开发者永不触及上游令牌               |
| 会话感知路由（Session-aware routing） | “多后端会话”        | 网关为每个开发者会话复用多个后端会话        |
| 工具哈希固定（Tool-hash pinning）   | “批准清单”          | 每个批准工具描述的 SHA256，中心管控防止拉地毯攻击 |
| RBAC               | “基于用户策略”        | 对工具和服务器的基于角色访问控制           |
| 策略即代码（Policy-as-code）      | “声明式规则”         | 网关层执行 OPA/Rego、Kyverno、Styra 策略   |
| 审计日志（Audit log）          | “谁、什么、何时”       | 追加式事件日志，用于合规                   |
| 限流（Rate limit）           | “每用户令牌桶”        | 每分钟调用上限，防止滥用                   |
| 官方 MCP 注册表（Official MCP Registry） | “权威上游”          | `registry.modelcontextprotocol.io`，命名空间验证 |
| 反向 DNS 命名（Reverse-DNS naming） | “注册表命名空间”       | `io.github.user/server` 规范               |

## 延伸阅读

- [官方 MCP 注册表](https://registry.modelcontextprotocol.io/) — 权威上游，命名空间验证  
- [Cloudflare — 企业 MCP](https://blog.cloudflare.com/enterprise-mcp/) — 含 OAuth 与策略的网关模式  
- [agentic-community — MCP 网关注册表](https://github.com/agentic-community/mcp-gateway-registry) — 开源参考网关  
- [TrueFoundry — 什么是 MCP 网关？](https://www.truefoundry.com/blog/what-is-mcp-gateway) — 功能比较文章  
- [IBM — MCP Context Forge](https://github.com/IBM/mcp-context-forge) — IBM 的企业网关
