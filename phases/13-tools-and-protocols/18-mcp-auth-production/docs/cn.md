# 生产环境中的 MCP 认证 — 注册、JWKS 刷新、受众绑定令牌

> 第16课在内存中搭建了 OAuth 2.1 状态机。到2026年，你交付给真实组织的每个 MCP 服务器都部署在生产认证后面：客户端注册要能扩展到无限量客户端（优先使用 Client ID Metadata Documents（客户端 ID 元数据文档），作为向后兼容的备用方案使用动态客户端注册（Dynamic Client Registration，DCR）），授权服务器元数据发现（RFC 8414 或 OpenID Connect 发现），JWKS 缓存刷新必须不会中断凌晨3点的令牌验证，使用受众绑定（audience-pinned）令牌以拒绝跨资源重放。本课用三个角色模拟完整表面：授权服务器、资源服务器（MCP 服务器）和客户端，帮助你追踪从发现到验证的每个跳点。
>
> **规范说明（2025-11-25）：** 2025年11月的 MCP 授权规范将动态客户端注册从 `SHOULD` 降级为 `MAY`，并将 **Client ID Metadata Documents（CIMD，客户端 ID 元数据文档）** 作为推荐的默认注册机制。本课按规范优先级讲授两者，且代码保持 DCR 以便演示，因为它完全自包含于一个进程中。

**类型：** 构建  
**语言：** Python（标准库）  
**先决条件：** 第13阶段 · 第16课（OAuth 2.1 状态机），第13阶段 · 第17课（网关）  
**时间：** 约90分钟

## 学习目标

- 通过 RFC 8414 元数据发现授权服务器并验证协议。
- 实现 RFC 7591 动态客户端注册，使 MCP 客户端无管理员干预即可注册。
- 定时缓存并刷新 JWKS 密钥，确保签名验证在密钥轮换时不中断。
- 使用 RFC 8707 资源指示器将令牌绑定到单个 MCP 资源，拒绝混淆代理（confused-deputy）重用。
- 清晰划分三个角色——授权服务器、资源服务器、客户端，确保各自只执行属于自身的检查。
- 读取身份提供者（IdP）能力矩阵，若 IdP 无法满足 MCP 的认证配置则拒绝部署。

## 问题描述

第16课的模拟器在内存中运行 OAuth 2.1。生产环境存在三个运营上的漏洞，内存模拟器看不到它们。

第一个漏洞是注册。真实组织运行数百个 MCP 服务器和数千个 MCP 客户端，运维不会逐个为每个 Cursor 用户手工注册 OAuth 客户端。2025-11-25 规范给客户端提供了解决方案的优先顺序：如果已有预注册的 `client_id`，则使用它；否则使用 **客户端 ID 元数据文档（CIMD）**（客户端用它控制的 HTTPS URL 作为身份标识，授权服务器主动拉取元数据）；否则回退到 **RFC 7591 动态客户端注册（DCR）**（客户端发起 `POST /register` 并即时获取 `client_id`）；否则提示用户。CIMD 被推荐为默认方案，因为它彻底消除了每台服务器的注册需求，且保持了基于 DNS 的信任模型；DCR 保留以保证向后兼容。两者的入口点均从授权服务器的元数据中发现：`client_id_metadata_document_supported` 表明 CIMD 支持，`registration_endpoint` 表明支持 DCR。

第二个漏洞是密钥轮换。JWT 验证依赖授权服务器的签名密钥，这些密钥作为 JSON Web Key Set（JWKS）发布。授权服务器定期轮换密钥（通常是每小时，有时在应急响应中更快）。只在启动时获取一次 JWKS 的 MCP 服务器能正常验证直至密钥轮换窗口，之后所有请求都会失败直至重启。生产环境将 JWKS 作为缓存值，并设定刷新任务，在旧密钥过期前覆盖缓存，同时在缓存未命中的情况下尽快获取更新，以应对令牌签名密钥高于缓存的情况。

第三个漏洞是受众绑定。第16课引入了 RFC 8707 资源指示器。在生产环境中，这个指示器变成每个请求必须做的硬性校验。MCP 服务器将 `token.aud` 与自身的标准资源 URL 比较，不匹配时返回 HTTP 401。此为唯一防御措施，用以阻止上游 MCP 服务器（或持有本服务器令牌的恶意客户端）将令牌重放到另一个同信任网格中的服务器。

本课将每个漏洞映射到具体实现上。元数据文档是一个 HTTP 端点。JWKS 缓存刷新是一个定时任务加键值缓存。JWT 验证是资源服务器派发任何工具前执行的例程。保持三个角色分离，每个角色只执行属于自己的检查：授权服务器负责签发和轮换密钥；资源服务器负责缓存和验证；客户端负责发现和注册。

## 概念介绍

### RFC 8414 — OAuth 授权服务器元数据

`/.well-known/oauth-authorization-server` 处的文档描述客户端所需的所有信息：

```json
{
  "issuer": "https://auth.example.com",
  "authorization_endpoint": "https://auth.example.com/authorize",
  "token_endpoint": "https://auth.example.com/token",
  "jwks_uri": "https://auth.example.com/.well-known/jwks.json",
  "registration_endpoint": "https://auth.example.com/register",
  "response_types_supported": ["code"],
  "grant_types_supported": ["authorization_code", "refresh_token"],
  "code_challenge_methods_supported": ["S256"],
  "scopes_supported": ["mcp:tools.read", "mcp:tools.invoke"],
  "token_endpoint_auth_methods_supported": ["none", "private_key_jwt"]
}
```

客户端拿到 MCP 资源 URL 后，按链式发现：从 RFC 9728 的 `oauth-protected-resource`（资源服务器文档）找到发行者（issuer），再从本 RFC 的 `oauth-authorization-server` 发现每个端点。客户端不再硬编码授权 URL。

信任 IdP 之前需验证协议：

- `code_challenge_methods_supported` 包含 `S256`（依据 RFC 7636 的 PKCE）。规范明确：如果该字段**缺失**，说明授权服务器不支持 PKCE，客户端**必须**拒绝继续。
- `grant_types_supported` 包含 `authorization_code`，且必须拒绝 `password` 和 `implicit`。
- 至少有一条注册路径：`client_id_metadata_document_supported: true`（CIMD，推荐）**或** `registration_endpoint`（RFC 7591 DCR，备用）。满足任一即可，无需强制要求 DCR。
- `response_types_supported` 精确为 `["code"]`，符合 OAuth 2.1。

如果缺少 `S256`，MCP 服务器拒绝对该 IdP 部署——PKCE 无降级模式。如果两个注册路径都未公布且没有预注册 `client_id`，则无法注册，说明部署清单有误，而非代码问题。

### RFC 9728（回顾） — 受保护资源元数据

第16课涵盖了 RFC 9728。生产环境区别在于：此文档是客户端寻找所谓 *本 MCP 服务器* 信任的授权服务器的唯一位置。单个 MCP 服务器可能接受多个 IdP 的令牌（例如一个服务员工，一个服务合作伙伴）。RFC 9728 声明该集合；RFC 8414 说明各自功能。

```json
{
  "resource": "https://notes.example.com",
  "authorization_servers": ["https://auth.example.com", "https://partners.example.com"],
  "scopes_supported": ["mcp:tools.invoke"],
  "bearer_methods_supported": ["header"],
  "resource_documentation": "https://notes.example.com/docs"
}
```

### Client ID Metadata Documents（推荐默认方案）

CIMD 将注册过程从*推送*变为*拉取*。客户端不由授权服务器生成 `client_id`，而是使用自己控制的 HTTPS URL **作为** `client_id`。该 URL 指向 JSON 元数据文档，授权服务器在 OAuth 流程中按需拉取。信任根植于 DNS：如果服务器运营商信任 `app.example.com`，那么也就信任 `https://app.example.com/client.json` 的客户端。无需注册交互，无 `client_id` 命名空间耗尽，也无需单服务器状态同步。

客户端托管的元数据文档：

```json
{
  "client_id": "https://app.example.com/oauth/client.json",
  "client_name": "Example MCP Client",
  "client_uri": "https://app.example.com",
  "redirect_uris": ["http://127.0.0.1:7333/callback", "http://localhost:7333/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "response_types": ["code"],
  "token_endpoint_auth_method": "none"
}
```

文档中的 `client_id` **必须**等于其服务的 URL（授权服务器对此验证，若不匹配会被拒绝）。授权服务器在其 RFC 8414 元数据中通过 `client_id_metadata_document_supported: true` 宣告支持。

规范毫不含糊地提出两个安全事实：

- **SSRF（服务器端请求伪造）。** 授权服务器需要拉取攻击者提供的 URL，必须防备服务器端请求伪造攻击（禁止访问内部或管理员端点）。
- **localhost 冒充。** CIMD 无法阻止本地攻击者声称合法客户端的元数据 URL 并绑定任意 `localhost` 回调 URI。授权服务器**必须**在授权界面清晰显示回调 URI 主机名，**应当**对纯 `localhost` 回调做出警告。

因为 CIMD 不需要服务器状态，故不像 DCR 那样必须设置登记端点。客户端是只读：从静态 HTTPS 端点提供元数据文档，让授权服务器拉取即可。

### RFC 7591 — 动态客户端注册（后备 / 向后兼容）

DCR 现为 `MAY`，保留用于兼容2025-11-25 之前的部署和不支持 CIMD 的 IdP。如果都没有（无 DCR、无 CIMD、无预注册），则每个 MCP 客户端（Cursor、Claude Desktop、自定义代理）都必须线下与 IdP 管理员交换数据。使用 DCR，客户端提交：

```json
POST /register
Content-Type: application/json

{
  "redirect_uris": ["http://127.0.0.1:7333/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "response_types": ["code"],
  "token_endpoint_auth_method": "none",
  "scope": "mcp:tools.invoke",
  "client_name": "Cursor",
  "software_id": "com.cursor.cursor",
  "software_version": "0.42.0"
}
```

服务器返回 `client_id` 和供后续更新用的 `registration_access_token`：

```json
{
  "client_id": "c_3e7f1a",
  "client_id_issued_at": 1769472000,
  "redirect_uris": ["http://127.0.0.1:7333/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "registration_access_token": "regt_b2...",
  "registration_client_uri": "https://auth.example.com/register/c_3e7f1a"
}
```

对于运行在用户设备上的 MCP 客户端，`token_endpoint_auth_method: none` 是合适默认。它们只拥有 `client_id`，没有可被泄露的 `client_secret`。PKCE 提供公众客户端所需的持有证明（proof-of-possession）。

三个生产风险点：

- 注册端点必须按源 IP 限速。否则恶意用户可脚本化数百万伪造注册，耗尽 `client_id` 名字空间。应在登记器处理请求前执行限速检查。
- `software_statement`（一个为客户端背书的签名 JWT）是部分企业 IdP 的要求。本课模拟跳过该部分；生产环境连接的验证步骤会拒绝非 `localhost` 重定向 URI 的未签名注册。
- `registration_access_token` 必须以哈希形式存储而非明文。该令牌一旦被盗，攻击方即可重写客户端的重定向 URI。

### RFC 8707（回顾）— 资源指示器

第16课确定了形式。生产规则：每个令牌请求都必须包含 `resource=<canonical-mcp-url>`，并且 MCP 服务器在每次调用时验证 `token.aud` 是否与自己的资源 URL 匹配。规范 URI 是服务器的*最具体*标识符：它使用小写方案和主机名，不带片段，通常也不带尾部斜杠。路径组件**不会**被规则剥离 —— 规范在需要标识单个 MCP 服务器时保留它。`https://mcp.example.com`、`https://mcp.example.com/mcp`、`https://mcp.example.com:8443` 和 `https://mcp.example.com/server/mcp` 都是有效的规范 URI。每个服务器选择一个，并将 `aud` 精确固定到该 URI。（本课的模拟使用裸主机作为 audience，如 `https://notes.example.com` 以简洁起见；部署中若多个 MCP 服务器共存于一个源，会通过路径区分它们。）

### RFC 7636（回顾）— PKCE

PKCE 是 OAuth 2.1 中的必需项。本课中的授权码流程始终携带 `code_challenge` 和 `code_verifier`。服务器会拒绝没有 verifier 或 verifier 计算后哈希与存储的挑战不匹配的任何令牌请求。

### MCP 规范 2025-11-25 身份验证配置文件

MCP 规范（2025-11-25）明确规定 MCP 服务器的授权层必须做的事：

- 实现 RFC 9728 保护资源元数据，并通过 401 响应中的 `WWW-Authenticate: Bearer resource_metadata="..."` 头或者 `.well-known` URI `/.well-known/oauth-protected-resource` 提供其位置（SEP-985 使该响应头可选，使用了 well-known fallback）。元数据中的 `authorization_servers` 字段**必须**至少指定一个服务器。
- 只能通过请求头 `Authorization: Bearer ...` 接收令牌 —— **每次请求**都必须验证，绝不可通过查询字符串，且不可只在会话开始时验证。
- 每次请求都必须验证 `aud`（受众）、`iss`（发行者）、`exp`（过期时间）和所需作用域。服务器**必须**验证令牌是专门为它签发的（受众绑定）；若缺失或者受众不匹配，必须拒绝，绝不可将缺失的 `aud` 视为通配符。
- 对于 401/403 响应，要返回 `WWW-Authenticate: Bearer`，携带 `error=...`、`resource_metadata="<PRM-URL>"` 参数（元数据文档的 URL，*不是*裸资源 URL），以及在 `insufficient_scope`（403）时返回 `scope="..."`。注意：参数名是 `resource_metadata`，这是一个发现指针 —— 质询中没有 `resource` 参数。
- 授权服务器发现支持**任意** RFC 8414 OAuth 元数据**或** OpenID Connect Discovery 1.0；客户端必须按优先顺序尝试两种 well-known 后缀。
- 客户端（而非服务器）负责防御**混淆攻击（mix-up attacks）**：它在重定向前记录预期的 `issuer`，并在兑换授权码前验证授权响应中的 `iss` 参数（RFC 9207）。仅用 PKCE 无法防止混淆攻击，因为客户端都会把它的 `code_verifier` 给任何它被引导去的令牌端点。

OAuth 2.1 草案是底层基础；RFC 8414/7591/8707/9728/9207 + RFC 7636 + CIMD 组成表层；MCP 规范是整个配置文件。

### IdP 功能矩阵

并非所有 IdP 都支持完整的 MCP 配置文件。下表记录了截至 2025-11-25 规范的实际能力声明。这是*部署门槛*，不是推荐。

CIMD 在 2025-11-25 规范中发布，而底层 OAuth 草案仅于 2025 年 10 月被采纳，厂商支持仍在推进 —— 下面的 “CIMD” 视作“目前的状态，需在你的租户环境中验证”，而非永久声明。

| IdP 类别 | AS 元数据 (8414/OIDC) | CIMD | RFC 7591 DCR | RFC 8707 资源 | RFC 7636 S256 PKCE | 备注 |
|---|---|---|---|---|---|---|
| 自托管（Keycloak） | 有 | 初现 | 有 | 有（自 24.x 起） | 有 | 本课 MCP 配置文件的参考 IdP；完整 DCR 流程，CIMD 跟踪最新规范。 |
| 企业 SSO（Microsoft Entra ID） | 有 | 初现 | 有（高级版） | 有 | 有 | DCR 功能依租户级别不同；部署前请在目标租户确认。 |
| 企业 SSO（Okta） | 有 | 初现 | 有（Okta CIC / Auth0） | 有 | 有 | DCR 在 Auth0（现为 Okta CIC）支持；传统 Okta 机构需管理员预注册。 |
| 社交登录 IdP（通用） | 不同 | 无 | 很少 | 很少 | 有 | 多数社交 IdP 将客户端视作静态合作伙伴，无自主注册服务。仅用作身份来源，授权层需自行实现 MCP 授权服务器。 |
| 自研 / 定制 | 视情况而定 | 视情况而定 | 视情况而定 | 视情况而定 | 视情况而定 | 若自研，务必实现完整配置文件并优先使用 CIMD。跳过 PKCE 或受众绑定即破坏 MCP 授权契约。 |

部署清单拒绝规则：若选定的 IdP 在 `code_challenge_methods_supported` 中未列出 `S256`，MCP 服务器拒绝启动 —— PKCE 无降级模式。注册为软门槛：只需*一个*有效路径（预注册的 `client_id`、`client_id_metadata_document_supported: true` 或 `registration_endpoint`）。单缺少 DCR 不再拒绝，因为 CIMD 或预注册可补充。

### JWKS 刷新模式（在 AS 轮换，在资源服务器刷新）

必须明确区分两个动作，否则在生产中是真实的漏洞：

- **轮换（Rotate）** 是*授权服务器*执行：生成新的签名密钥，发布在 JWKS 中，稍后废弃旧密钥。资源服务器不参与此环节，且无法执行 —— 它不持有 IdP 的私钥。
- **刷新（Refresh）** 是*资源服务器*执行：重新 `GET` 发布的 JWKS 到其缓存。这是资源服务器执行的唯一 JWKS 行为。

生产失败模式是缓存过期。解决方案是定期刷新任务加键值缓存。资源服务器运行一个定时任务（cron、计时器或运行时提供的类似功能），在固定间隔内获取 `<issuer>/.well-known/jwks.json` 并覆盖 `cache[issuer] = {keys, fetched_at}`。校验器从该缓存读取。若令牌中的 `kid` 在缓存中缺失，则触发**一次**同步刷新作为备选，再次检查。此方案同时处理定期刷新和密钥覆盖窗口，当新密钥签发的令牌到达，但尚未到下次定期刷新时。

备选方案**必须是重新获取，而不是旋转**。若缓存未命中时强制轮换生成新密钥，会导致两种故障：(1) 新密钥 `kid` 仍与令牌不匹配，查找失败；(2) 攻击者大量随机 `kid` 噪声令牌，导致无限创建密钥的 DoS。重新获取是幂等的，假 `kid` 最多只浪费一次拉取请求。

缓存结构示例：

```json
{
  "https://auth.example.com": {
    "keys": [
      {"kid": "k_2026_03", "kty": "RSA", "n": "...", "e": "AQAB", "alg": "RS256", "use": "sig"},
      {"kid": "k_2026_04", "kty": "RSA", "n": "...", "e": "AQAB", "alg": "RS256", "use": "sig"}
    ],
    "fetched_at": 1772668800
  }
}
```

同时有两个密钥是常态。授权服务器通过先引入新密钥（`k_2026_04`）再废弃旧密钥（`k_2026_03`）来轮换，旧密钥签发的令牌在过期前依然有效。缓存保持两者的并集；校验器根据 `kid` 选择。

### 校验流程

MCP 服务器在分发任何工具前先运行校验。`code/main.py` 使用的示例：

```python
result = server.validate(bearer_token, required_scope="mcp:tools.invoke")
if not result["valid"]:
    return {"status": result["status"], "WWW-Authenticate": result["www_authenticate"]}
```

`validate` 解码 JWT，从 JWKS 缓存加载签名密钥（缓存缺失时刷新一次），验证签名，然后检查 `iss` 是否在允许列表中，`aud` 是否匹配本服务器规范资源，`exp`（过期时间）和所需作用域 —— 任何一次失败都返回 `WWW-Authenticate` 质询。将所有检查统一为单个例程保证每个入口点（每次工具调用、每次传输）都经过相同校验；不存在未校验就访问工具的路径。

### 受众重放攻击演练（访问令牌权限限制）

服务器 A（`notes.example.com`）和服务器 B（`tasks.example.com`）均注册到同一授权服务器。服务器 A 被攻破。攻击者拿了用户的 notes 令牌，尝试重放给服务器 B。

服务器 B 的校验器流程：

1. 解码 JWT，通过 `kid` 获取 JWKS，验证签名。
2. 检查 `iss` 是否属于其保护资源元数据中的 `authorization_servers` 列表。（通过 —— 同一个 IdP。）
3. 检查 `aud == "https://tasks.example.com"`。（失败 —— 令牌 `aud` 是 `https://notes.example.com`。）
4. 返回 401，附带 `WWW-Authenticate: Bearer error="invalid_token", error_description="audience mismatch", resource_metadata="https://tasks.example.com/.well-known/oauth-protected-resource"`。

在协议层面，受众声明是此攻击的唯一防御。跳过它以提升性能是最常见的生产错误；校验器必须对每个请求执行，不仅限于会话开始时。规范称之为**访问令牌权限限制**：MCP 服务器`必须`拒绝任何未将自身列为受众的令牌。

> **命名说明。** 规范将*confused deputy*（混淆副手）用于相关但不同的问题：MCP 服务器作为 OAuth **代理**调用第三方 API，使用静态客户端 ID，转发令牌且未取得每客户端的用户同意。受众绑定防止上述重放；混淆副手修复需要每客户端同意**加上**绝不将入站令牌传给上游 API（MCP 服务器`必须`获取自己单独的上游令牌）。

### 混淆攻击（客户端防御，服务器无法提供）

客户端一生中会与多个授权服务器交互。恶意授权服务器可以诱导客户端在攻击者的令牌端点兑换来自正当授权服务器的授权码。受众绑定无效 —— 攻击发生在令牌生成之前。防御应在客户端（RFC 9207）：

1. 重定向前，客户端记录验证过的授权服务器元数据中的预期 `issuer`。
2. 授权响应时，客户端对比返回的 `iss` 参数和之前记录的发行者（简单字符串比较，无需归一化），若不符或当 AS 宣布支持 `authorization_response_iss_parameter_supported` 却无 `iss` 时 —— 拒绝请求，不显示 `error` 字段。
3. 不匹配即拒绝。

单独用 PKCE 无法阻止混淆攻击，因为客户端会把自己的 `code_verifier` 直接交给任何引导去的令牌端点。这就是为什么规范要求为每次请求记录 `issuer`，与 PKCE 验证器和 `state` 一起。

### 故障模式

- **JWKS 过期。** 当授权服务器轮换密钥后，校验器拒绝有效令牌。解决方案是上述定时刷新 + 缓存未命中时重新获取模式。绝不可缓存 JWKS 且不作刷新。
- **以轮换代替备选。** 缓存未命中路径若是轮换和重新生成密钥，会导致找不到缺失的 `kid`，且允许攻击者用任意 `kid` 触发无限密钥创建的 DoS。备选方案必须是幂等的 `refresh-jwks`。
- **缺失 `aud` 声明。** 部分 IdP 默认省略 `aud`，除非请求的令牌里有 `resource`。校验器必须拒绝缺少 `aud` 的令牌，绝不可将缺失当做通配符。
- **缺少 `iss` 校验导致混淆攻击。** 客户端不验证 RFC 9207 的授权响应 `iss` 参数，未与预记录的发行者比对，可能被引导用正主授权码到攻击者令牌端点。此为客户端缺陷，资源服务器无补救措施。
- **Scope 升级竞争。** 同一用户的两个并发提权流程都能成功，产生两个不同作用域的令牌。校验器必须使用当前请求携带的令牌，不得依赖“用户当前作用域”查询，避免 TOCTOU（时间点到竞争）漏洞。
- **注册令牌泄露。** 泄露 `registration_access_token` 允许攻击者修改重定向 URI。应加密存储，客户端每次更新均需明文认证，怀疑泄漏时立即轮换。
- **未固定 `iss`。** 允许任意 `iss` 的校验器，使攻击者搭建自己的授权服务器，注册面向目标受众的客户端并发行令牌。保护资源元数据中的 `authorization_servers` 列表即允许列表，必须强制执行。

## 使用它

`code/main.py` 通过标准库 Python 和三个角色——`AuthorizationServer`（授权服务器），`ResourceServer`（资源服务器）和 `Client`（客户端）演示完整的生产流程。流程如下：

1. 授权服务器在 `/.well-known/oauth-authorization-server` 发布 RFC 8414 元数据。
2. MCP 客户端调用元数据端点并检查其注册选项（CIMD 的 `client_id_metadata_document_supported`，DCR 的 `registration_endpoint`）和 `S256` PKCE 支持。
3. 演示采用 DCR 备用路径：客户端向 `/register`（RFC 7591）发送请求，获得 `client_id`。（CIMD 客户端则呈现自己的 HTTPS `client_id` URL 并跳过此步骤。）
4. MCP 客户端运行带有 `resource` 指示器（RFC 8707）的 PKCE 保护授权码流程（RFC 7636）。
5. MCP 客户端使用 `Authorization: Bearer ...` 调用 MCP 服务器上的工具。
6. MCP 服务器运行 `validate`，通过 JWKS 缓存解析签名密钥。
7. IdP 轮换密钥；计划刷新任务重新拉取 JWKS 到缓存。
8. 下一次调用使用刷新后的密钥进行验证，无需重启，并且在重叠窗口内旧令牌依然有效。
9. 针对不同 MCP 资源的 audience 重放尝试返回 401，带有 `audience mismatch` 和指向 `resource_metadata` 的指针。

这里的 JWT 使用 HS256 和共享密钥（因此本课仅使用标准库）。生产环境使用 RS256 或 EdDSA，配合上述 JWKS 模式；验证逻辑其它相同。因 IdP 和资源服务器在同一进程，`refresh_jwks` 直接读取授权服务器的密钥列表；网络层面是 HTTP 对 `jwks_uri` 的 `GET` 调用。

## 交付它

本课生成 `outputs/skill-mcp-auth.md`。给定 MCP 服务器配置及 IdP 能力集，技能输出认证接口信息以启用服务——包括受保护资源元数据、使用的注册路径（CIMD、预注册或 DCR 备用）、JWKS 刷新周期、作用域映射，以及 IdP 不支持完整 RFC 配置时的拒绝规则。

## 练习

1. 运行 `code/main.py`。跟踪流程。注意步骤 6 中 IdP 轮换密钥，计划的 `refresh_jwks` 重新拉取已发布密钥集合，以及重叠窗口内旧令牌和新令牌均能验证且无需重启。
2. 在 protected-resource 元数据的 `authorization_servers` 列表中添加新的 IdP。发行该 IdP 签发的令牌，确认验证器接受。发行未列出的 IdP 签发的令牌，确认验证器拒绝并带 `WWW-Authenticate: Bearer error="invalid_token", error_description="iss not allowed"`。
3. 在 `register_client` 中添加速率限制检查，运行于注册方接受请求前。对每个源 IP 使用一个令牌桶，将其保存在以 IP 为键的字典中。
4. 阅读 RFC 7591，找出本课的 `/register` 处理函数未验证的两个字段。添加相应验证。（提示：`software_statement` 和 `redirect_uris` 的 URI 方案。）
5. 添加 Client ID Metadata Document 路径。提供 `client.json`，其 `client_id` 等于其自身 URL，授权服务器拉取并验证（`client_id` ≠ URL 则拒绝）。确认 CIMD 客户端注册时无需调用 `register_client`。
6. 验证 DoS 修复。向验证器发送带随机 `kid` 的令牌，确认 `refresh_jwks` 最多运行一次，授权服务器的密钥数量不增长。然后故意改成 rotate-and-mint 的回退逻辑，观察假令牌导致密钥数量上升——之后恢复重新拉取。
7. 实现客户端侧 RFC 9207 的 `iss` 检查（混淆攻击章节）：在授权请求前记录期望的发行者，拒绝 `iss` 不匹配的授权响应。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| ASM | "OAuth 元数据文档" | RFC 8414 `/.well-known/oauth-authorization-server` JSON |
| CIMD | "客户端元数据 URL" | Client ID Metadata Document（客户端 ID 元数据文档）——用作 `client_id` 的 HTTPS URL；授权服务器拉取 JSON。2025-11-25 起推荐默认使用 |
| DCR | "自助客户端注册" | RFC 7591 的 `POST /register` 流程；在 2025-11-25 降级为 `MAY` 备用方案 |
| JWKS | "JWT 验证公钥集合" | JSON Web Key Set，从 `jwks_uri` 拉取，按 `kid` 索引 |
| 轮换 vs 刷新 | "密钥更新" | *轮换* 指授权服务器生成/废弃签名密钥；*刷新* 指资源服务器重新拉取已发布密钥集合。资源服务器仅执行刷新 |
| 资源指示器 | "受众参数" | RFC 8707 的 `resource` 参数，将令牌限定给单个服务器 |
| `aud` 声明 | "受众" | JWT 中验证者对照的 canonical 资源 URL |
| 受众重放 | "令牌重放" | 用于服务器 A 的令牌被用于服务器 B；通过受众验证防护（规范：访问令牌权限限制） |
| 代理困境 | "代理令牌误用" | MCP 代理带固定客户端 ID 转发令牌且无客户端同意；区别于受众重放 |
| 混淆攻击 | "错误令牌端点" | 客户端被引导向攻击者端点兑换真授权服务器的代码；通过客户端 RFC 9207 的 `iss` 校验防护 |
| `iss` 允许列表 | "信任的授权服务器" | protected-resource 元数据中命名的 `authorization_servers` 集合 |
| `resource_metadata` | "PRM 文档位置" | 401/403 响应中 `WWW-Authenticate` 参数指向的 RFC 9728 元数据 URL |
| 公开客户端 | "本地或浏览器客户端" | 无 `client_secret` 的 OAuth 客户端；由 PKCE 规避风险 |
| `WWW-Authenticate` | "401/403 响应头" | 携带 `Bearer error=...` 指令，引导客户端恢复操作 |

## 延伸阅读

- [MCP — 授权规范（2025-11-25）](https://modelcontextprotocol.io/specification/2025-11-25/basic/authorization) — 本课实现的 MCP 认证配置文件
- [MCP 博客 — MCP 一周年：2025-11-25 版本发布](https://blog.modelcontextprotocol.io/posts/2025-11-25-first-mcp-anniversary/) — 2025-11-25 的变更（CIMD、XAA、DCR 降级）
- [Aaron Parecki — 2025 年 11 月 MCP 授权规范中的客户端注册](https://aaronparecki.com/2025/11/25/1/mcp-authorization-spec-update) — CIMD 替代 DCR 的理由
- [OAuth Client ID Metadata Document (draft-ietf-oauth-client-id-metadata-document-00)](https://datatracker.ietf.org/doc/html/draft-ietf-oauth-client-id-metadata-document-00) — CIMD
- [RFC 8414 — OAuth 2.0 授权服务器元数据](https://datatracker.ietf.org/doc/html/rfc8414) — 发现协定
- [RFC 7591 — OAuth 2.0 动态客户端注册协议](https://datatracker.ietf.org/doc/html/rfc7591) — DCR（备用路径）
- [RFC 7636 — Proof Key for Code Exchange (PKCE)](https://datatracker.ietf.org/doc/html/rfc7636) — 公开客户端持有证明
- [RFC 8707 — OAuth 2.0 资源指示器](https://datatracker.ietf.org/doc/html/rfc8707) — 受众绑定
- [RFC 9728 — OAuth 2.0 受保护资源元数据](https://datatracker.ietf.org/doc/html/rfc9728) — 资源服务器发现
- [RFC 9207 — OAuth 2.0 授权服务器发行者识别](https://datatracker.ietf.org/doc/html/rfc9207) — 防混淆攻击的 `iss` 参数
- [OAuth 2.1 草案](https://datatracker.ietf.org/doc/html/draft-ietf-oauth-v2-1) — OAuth 统一标准
