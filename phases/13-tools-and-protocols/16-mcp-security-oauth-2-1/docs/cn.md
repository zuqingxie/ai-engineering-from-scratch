# MCP 安全 II — OAuth 2.1、资源指示器、增量权限范围

> 远程 MCP 服务器需要授权，而不仅仅是身份验证。2025-11-25 规范对齐 OAuth 2.1 + PKCE + 资源指示器（RFC 8707）+ 受保护资源元数据（RFC 9728）。SEP-835 增加了带有 403 WWW-Authenticate 的增量权限范围同意的提升授权。本课实现了一个作为状态机的提升流程，让你能看到每个跳转。

**类型：** 构建  
**语言：** Python（标准库，OAuth 状态机模拟器）  
**先决条件：** 第 13 阶段 · 09（传输），第 13 阶段 · 15（安全 I）  
**时间：** ~75 分钟

## 学习目标

- 区分资源服务器与授权服务器的职责。
- 演练受 PKCE 保护的 OAuth 2.1 授权码流程。
- 使用 `resource`（RFC 8707）和受保护资源元数据（RFC 9728）防止混淆代理（confused deputy）攻击。
- 实现提升授权：服务器响应 403 并携带 WWW-Authenticate，要求更高权限范围；客户端重新提示用户确认并重试。

## 问题描述

早期 MCP（2025年前）向远程服务器发布的是临时的 API 密钥，甚至无认证。2025-11-25 规范通过完整的 OAuth 2.1 配置弥补了这一差距。

三个实际需求：

- **普通远程服务器。** 用户安装可访问其 Notion / GitHub / Gmail 的远程 MCP 服务器。OAuth 2.1 结合 PKCE 是合适的方案。
- **权限提升。** 笔记服务器授予 `notes:read` 权限后，针对某些操作可能需要 `notes:write`。无需重做整个流程，提升（SEP-835）请求附加权限。
- **混淆代理防御。** 客户端持有限定给服务器 A 的令牌。服务器 A 恶意尝试将该令牌用于服务器 B。资源指示器（RFC 8707）将令牌绑定到其目标受众。

OAuth 2.1 本身不新颖，MCP 的新亮点是其配置：强制使用授权码 + PKCE，不支持隐式与默认客户端凭证；每个令牌请求必须带资源指示器；发布受保护资源元数据使客户端知道目标。

## 概念解析

### 角色

- **客户端（Client）。** MCP 客户端（Claude Desktop，Cursor 等）。
- **资源服务器（Resource server）。** MCP 服务器（笔记、GitHub、Postgres 等）。
- **授权服务器（Authorization server）。** 签发令牌，可能与资源服务器同属一体，也可能是独立身份提供者（Auth0、Keycloak、Cognito）。

MCP 配置中，资源服务器和授权服务器可以是同一主机，但应通过 URL 区分。

### 授权码 + PKCE

流程：

1. 客户端生成 `code_verifier`（随机）和 `code_challenge`（SHA256）。
2. 客户端重定向用户至 `/authorize?response_type=code&client_id=...&redirect_uri=...&scope=notes:read&code_challenge=...&resource=https://notes.example.com`。
3. 用户同意，授权服务器重定向回 `redirect_uri?code=...`。
4. 客户端通过 POST 到 `/token?grant_type=authorization_code&code=...&code_verifier=...&resource=...`。
5. 授权服务器验证 `code_verifier` 与保存的 `code_challenge`，签发访问令牌。
6. 客户端使用令牌：每次请求资源服务器均携带 `Authorization: Bearer ...`。

PKCE 可防止授权码被截取攻击。资源指示器保证令牌只对预期服务器有效。

### 受保护资源元数据（RFC 9728）

资源服务器发布 `.well-known/oauth-protected-resource` 文档：

```json
{
  "resource": "https://notes.example.com",
  "authorization_servers": ["https://auth.example.com"],
  "scopes_supported": ["notes:read", "notes:write", "notes:delete"]
}
```

客户端由资源服务器发现授权服务器，减少配置需求——客户端只需知道资源 URL。

### 资源指示器（RFC 8707）

令牌请求中的 `resource` 参数固定令牌目标受众。签发的令牌中含有 `aud: "https://notes.example.com"`。其他 MCP 服务器收到该令牌时，会校验 `aud` 并拒绝无效令牌。

### 权限范围模型

权限范围以空格分隔字符串表示，MCP 常用约定：

- `notes:read`，`notes:write`，`notes:delete`
- `admin:*` 表示管理员权限（慎用）
- `profile:read` 表示身份信息读取权限

权限选择应遵循最小权限原则：需求什么权限即申请什么，之后需要再提升。

### 提升授权（SEP-835）

用户授予 `notes:read` 后，请求代理删除笔记。服务器响应：

```text
HTTP/1.1 403 Forbidden
WWW-Authenticate: Bearer error="insufficient_scope",
    scope="notes:delete", resource="https://notes.example.com"
```

客户端检测到 `insufficient_scope` 错误，弹出额外权限确认对话框，完成小型 OAuth 流程获取新令牌，重试请求。

### 令牌受众验证

每个请求，服务器检查 `token.aud == self.resource_url`。不匹配即返回 401。防止跨服务器令牌复用。

### 短期令牌与轮换

访问令牌应当为短期令牌（默认 1 小时）。刷新令牌在每次刷新时应轮换，客户端后台无感刷新处理。

### 不允许令牌透传

采样服务器（第 13 阶段 · 11）不得将客户端令牌透传给其他服务，采样请求即边界。

### 混淆代理防御

令牌绑定到 `aud`，客户端绑定到 `client_id`。每个请求都要校验两者。规范明确禁止了预 MCP 远程工具生态中典型的“传令牌”模式。

### 客户端 ID 发现

每个 MCP 客户端在固定 URL 发布元数据。授权服务器可抓取客户端元数据文档发现重定向 URI 和联系信息，消除手动注册。

### 网关与 OAuth

第 13 阶段 · 17 展示了企业网关如何处理 OAuth：网关持有上游服务器的凭证，对客户端颁发令牌，上游令牌不离开网关。信任模式反转——用户只与网关认证一次，网关处理 N 个服务器的授权。

## 实践应用

`code/main.py` 模拟了完整 OAuth 2.1 提升流程状态机。实现：

- PKCE code-verifier / challenge 生成。
- 授权码流带资源指示器。
- 受保护资源元数据端点。
- 含受众校验令牌验证。
- 针对 `insufficient_scope` 的提升授权。

本课无 HTTP 服务器，状态机内存运行，方便追踪每步。第 13 阶段 · 17 的网关课将其接入实际传输。

## 输出内容

本课生成 `outputs/skill-oauth-scope-planner.md`。给定远程 MCP 服务器及其工具，skill 设计权限范围集合、绑定规则和提升策略。

## 练习

1. 运行 `code/main.py`。跟踪两权限提升流程，注意哪些跳转在提升时重复。

2. 增加刷新令牌轮换：每次刷新签发新刷新令牌，失效旧令牌。模拟被盗刷新令牌在轮换后被使用，确认失败。

3. 使用标准库 http.server 实现受保护资源元数据端点为真实 HTTP 响应，镜像第 09 课的 /mcp 端点。

4. 为 GitHub MCP 服务器设计权限层级：读仓库，写 PR，批准 PR，合并 PR，管理员。各级间使用提升授权。

5. 阅读 RFC 8707 和 RFC 9728。找出 9728 中 MCP 使用方式与 RFC 示例不同的字段。（提示：涉及 `scopes_supported`。）

## 关键词汇

| 术语 | 常见说法 | 实际含义 |
|------|-----------|-----------|
| OAuth 2.1 | “现代 OAuth” | 合并的 RFC，强制 PKCE，禁止隐式流程 |
| PKCE | “拥有证明” | 代码验证器 + 挑战，防止授权码截取 |
| 资源指示器 | “令牌受众” | RFC 8707 中 `resource` 参数，绑定令牌至单一服务器 |
| 受保护资源元数据 | “发现文档” | RFC 9728 `.well-known/oauth-protected-resource` |
| 提升授权 | “增量同意” | SEP-835 按需追加权限流程 |
| `insufficient_scope` | “403 带 WWW-Authenticate” | 服务器要求重新同意更大权限 |
| 混淆代理 | “跨服务令牌复用” | 攻击，受信令牌持有者错误转发令牌 |
| 短期令牌 | “访问令牌 TTL” | 快速过期的 Bearer 令牌，刷新令牌续签 |
| 权限层级 | “最小权限堆栈” | 按级递增的权限集及权限提升 |
| 客户端 ID 元数据 | “客户端发现文档” | 客户端发布自身 OAuth 元数据的 URL |

## 相关阅读

- [MCP — 授权规范](https://modelcontextprotocol.io/specification/draft/basic/authorization) — MCP OAuth 标准规范
- [den.dev — MCP 11 月授权规范](https://den.dev/blog/mcp-november-authorization-spec/) — 2025-11-25 变更讲解
- [RFC 8707 — OAuth 2.0 资源指示器](https://datatracker.ietf.org/doc/html/rfc8707) — 受众绑定 RFC
- [RFC 9728 — OAuth 2.0 受保护资源元数据](https://datatracker.ietf.org/doc/html/rfc9728) — 发现文档 RFC
- [Aembit — MCP OAuth 2.1、PKCE 与 AI 授权未来](https://aembit.io/blog/mcp-oauth-2-1-pkce-and-the-future-of-ai-authorization/) — 实用提升流程演练
