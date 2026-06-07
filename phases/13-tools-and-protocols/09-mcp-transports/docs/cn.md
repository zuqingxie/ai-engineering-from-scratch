# MCP 传输 — stdio 与 Streamable HTTP 及 SSE 迁移

> stdio 仅限本地使用。Streamable HTTP（2025-03-26）是远程标准。旧的 HTTP+SSE 传输已被弃用，预计在 2026 年中期移除。选择错误的传输方式将导致迁移成本；选择正确的方式则获得一个支持会话连续性且具备 DNS 重绑定防护的远程可托管 MCP 服务器。

**类型：** 学习  
**语言：** Python（标准库，Streamable HTTP 端点骨架）  
**先决条件：** 13 阶段 · 07, 08（MCP 服务器和客户端）  
**时间：** 约 45 分钟

## 学习目标

- 根据部署形态（本地与远程，单进程与集群）选择 stdio 或 Streamable HTTP。  
- 实现 Streamable HTTP 单端点模式：POST 用于请求，GET 用于会话流。  
- 强制执行 `Origin` 验证与会话 ID 语义，防止 DNS 重绑定攻击。  
- 迁移旧版 HTTP+SSE 服务器到 Streamable HTTP，赶在 2026 年中期停止支持前完成。

## 问题

首次 MCP 远程传输（2024-11）采用的是 HTTP+SSE：两个端点，一个用于客户端 POST，另一个为服务端推送(Server-Sent-Events)频道实现服务器到客户端的数据流。这种模式能工作，但也笨拙：每个会话两个端点，一些 CDN 前的缓存失效，且强依赖长连接 SSE，部分 WAF 会对其强制断开。

2025-03-26 规范将其替换为 Streamable HTTP：单端点，POST 用于客户端请求，GET 用以建立会话流，两者共享 `Mcp-Session-Id` 头。从那时起，所有新建或迁移的服务器均使用 Streamable HTTP。旧的 SSE 模式被弃用——Atlassian Rovo 已于 2026 年 6 月 30 日移除；Keboola 于 2026 年 4 月 1 日移除；大多数其它企业服务器将于 2026 年底前移除。

本地服务器仍需 stdio 支持。Claude Desktop、VS Code 及所有 IDE 客户端都通过 stdio 启动服务器。正确的思路是：stdio 用于“本机”，Streamable HTTP 用于“网络”。两者不交叉。

## 概念

### stdio

- 子进程通信。客户端启动服务器，通过 stdin/stdout 通信。  
- 每行一个 JSON 对象，换行分隔。  
- 无会话 ID，进程身份即为会话。  
- 无需身份验证（子进程继承父进程信任边界）。  
- 不用于远程服务器——远程需要 SSH 或 socat 隧道，届时应使用 Streamable HTTP。

### Streamable HTTP

单端点 `/mcp`（或任何路径），支持三种 HTTP 方法：

- **POST /mcp。** 客户端发送 JSON-RPC 消息。服务器回复单个 JSON 响应，或一个或多个响应构成的 SSE 流（适用于批量响应及该请求相关通知）。  
- **GET /mcp。** 客户端打开长期 SSE 频道。服务器用于服务器到客户端请求（采样、通知、提取）。  
- **DELETE /mcp。** 客户端显式终止会话。

会话通过服务器在首响应中设置，客户端后续请求中回显的 `Mcp-Session-Id` 头标识。会话 ID 必须是密码学随机值（128 位以上）；客户端自选 ID 出于安全原因被拒绝。

### 单端点 vs 双端点

旧规范的双端点模式在 2026 年仍可调用——规范声明其“兼容旧版”。但所有新服务器应采用单端点模式。官方 SDK 默认发出单端点请求；仅与未迁移远程服务器通信时使用旧模式。

### `Origin` 验证与 DNS 重绑定

浏览器目前不是 MCP 客户端，但攻击者可构造网页让浏览器向 `localhost:1234/mcp`（用户本地 MCP 服务器监听端口）发送 POST 请求。如果服务器不检查 `Origin`，浏览器的同源策略不会生效，因为 `Origin: http://evil.com` 是合法的跨源。

2025-11-25 规范要求服务器拒绝不在允许列表中的 `Origin` 请求。允许列表通常包含 MCP 客户端主机（`https://claude.ai`，`vscode-webview://*`）及本地界面相关的 localhost 变体。

### 会话 ID 生命周期

1. 客户端首次请求不带 `Mcp-Session-Id`。  
2. 服务器分配随机 ID，设置 `Mcp-Session-Id` 响应头。  
3. 客户端在后续所有请求及 GET /mcp 流请求中回显该头。  
4. 会话可被服务器撤销；客户看到 404 并须重新初始化。  
5. 客户端可以显式 DELETE 会话，实现干净关闭。

### 保持连接与重连

SSE 连接会断开。客户端通过再次使用相同 `Mcp-Session-Id` 的 GET 请求重建连接。服务器必须在合理时间窗内缓存掉线期间的事件，并通过客户端回显的 `last-event-id` 头重放。

第 13 阶段 · 第 13 课涵盖 Task，支持长时间任务即使全会话重连后依然存活。

### 向后兼容探测

支持新旧服务器的客户端使用：

1. POST 到 `/mcp`。  
2. 如果响应为 200 OK 且为 JSON 或 SSE 流，则为 Streamable HTTP。  
3. 如果响应为 200 OK，`Content-Type: text/event-stream` 且含有指向次级端点的 `Location` 头，则为旧的 HTTP+SSE；按 `Location` 访问。

### Cloudflare、ngrok 与托管

2026 年生产环境远程 MCP 服务器运行于 Cloudflare Workers（配合 MCP Agents SDK）、Vercel Functions 或容器化 Node/Python。关键：托管需支持 SSE GET 长连接。Vercel 免费层限制 10 秒，不适用。Cloudflare Workers 支持无限制流。

### 网关组合

当前置多个 MCP 服务器用网关（13 阶段 · 17 课），网关为单一 Streamable HTTP 端点，负责改写会话 ID 并进行上游复用。工具在网关层合并，客户端看见的是单一逻辑服务器。

### 传输故障模式

- **stdio SIGPIPE。** 子进程写入中途死掉导致 SIGPIPE，服务器应干净退出。客户端检测 EOF 并标记会话失效。  
- **HTTP 502 / 504。** Cloudflare、nginx 等代理上游失败返回。Streamable HTTP 客户端应短暂重试一次。  
- **SSE 连接断开。** TCP RST、代理超时或客户端网络变更关闭流。客户端带同一 `Mcp-Session-Id` 和可选 `last-event-id` 重连。  
- **会话撤销。** 服务器作废会话 ID，客户端下次请求见 404，须重新握手。  
- **时钟偏移。** 客户端资源 TTL 计算与服务器偏离，客户端应信服务器时间戳。

### 何时跳过 Streamable HTTP

部分企业在内网用 gRPC 或消息队列作为 MCP 传输。这非标准——MCP 规范未正式定义。网关可对外暴露合规的 Streamable HTTP 表面，内部用 gRPC 转换。外部接口保持规范，翻译由网关负责。

## 使用它

`code/main.py` 实现了基于 `http.server`（标准库）的极简 Streamable HTTP 端点。支持 `/mcp` 的 POST、GET、DELETE，设置首响应的 `Mcp-Session-Id`，校验 `Origin`，拒绝非允许名单的请求。处理器重用第 07 课笔记服务器的分发逻辑。

重点查看：

- POST 处理器读取 JSON-RPC，调用分发，写单响应 JSON（单响应模式，SSE 模式结构相似）。  
- `Origin` 校验拒绝默认的 `http://evil.example` 探测，接受 `http://localhost`。  
- 会话 ID 是随机 128 位十六进制字符串，服务器内存保存每会话状态。

## 发布它

本课生成 `outputs/skill-mcp-transport-migrator.md`。给定一个 HTTP+SSE（旧式）MCP 服务器，生成带会话 ID 连续性、Origin 校验和向后兼容探测支持的 Streamable HTTP 迁移方案。

## 练习

1. 运行 `code/main.py`。用 `curl` POST 一个 `initialize` 请求，观察响应头 `Mcp-Session-Id`。再 POST 第二个请求携带同一 header，验证会话连续性。

2. 添加入 GET 处理器，打开 SSE 流。每 5 秒发送一个 `notifications/progress` 事件。再次用相同会话 ID 重新 GET 以重连，确认服务器接受。

3. 实现 `last-event-id` 重放机制。重连时，重放自该 ID 以来生成的事件。

4. 扩展 `Origin` 验证，支持通配符模式（`https://*.example.com`），验证允许 `https://app.example.com`，拒绝 `https://evil.example.com.attacker.net`。

5. 从官方注册库挑一个旧版 HTTP+SSE 服务器，设计迁移方案：端点处理、会话 ID 生成、头语义的变化。

## 关键词条

| 术语                  | 大众描述           | 实际含义                                    |
|-----------------------|--------------------|---------------------------------------------|
| stdio transport       | “本地子进程”       | 通过 stdin/stdout 的 JSON-RPC，换行分隔      |
| Streamable HTTP       | “远程传输”         | 单端点 POST + GET + 可选 SSE，2025-03-26 规范 |
| HTTP+SSE              | “旧版”             | 两端点模型，将于 2026 年中期移除             |
| `Mcp-Session-Id`      | “会话头”           | 服务器分配随机 ID，后续请求回显               |
| `Origin` allowlist    | “DNS 重绑定防护”   | 拒绝 Origin 不在允许名单的请求                |
| 单端点                | “一个 URL”          | `/mcp` 处理 POST / GET / DELETE 等所有会话操作 |
| `last-event-id`       | “SSE 重放”         | 用于重连时恢复断开流中的事件                   |
| 向后兼容探测          | “旧新检测”         | 通过响应形态自动选择传输方式                   |
| 长连接 HTTP           | “SSE 流”           | 服务器在 TCP 连接上推送数分钟或数小时的事件     |
| 会话撤销              | “强制重新初始化”   | 服务器作废会话 ID，客户端须重新握手             |

## 延伸阅读

- [MCP — 基础传输规范 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/basic/transports) — stdio 与 Streamable HTTP 的权威参考  
- [MCP — 基础传输规范 2025-03-26](https://modelcontextprotocol.io/specification/2025-03-26/basic/transports) — 引入 Streamable HTTP 的修订版  
- [Cloudflare — MCP 传输](https://developers.cloudflare.com/agents/model-context-protocol/transport/) — Workers 托管的 Streamable HTTP 模式  
- [AWS — MCP 传输机制](https://builder.aws.com/content/35A0IphCeLvYzly9Sw40G1dVNzc/mcp-transport-mechanisms-stdio-vs-streamable-http) — 各种部署形态对比  
- [Atlassian — HTTP+SSE 弃用公告](https://community.atlassian.com/forums/Atlassian-Remote-MCP-Server/HTTP-SSE-Deprecation-Notice/ba-p/3205484) — 具体迁移截止实例
