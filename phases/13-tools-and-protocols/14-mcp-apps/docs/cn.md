# MCP 应用 — 通过 `ui://` 交互式 UI 资源

> 仅文本工具输出限制了 agent 能展示的内容。MCP 应用（SEP-1724，官方发布时间 2026 年 1 月 26 日）允许工具返回沙箱内联渲染的交互式 HTML，在 Claude Desktop、ChatGPT、Cursor、Goose 和 VS Code 中显示。仪表板、表单、地图、3D 场景，全都通过一个扩展实现。本课介绍 `ui://` 资源方案、`text/html;profile=mcp-app` MIME 类型、iframe 沙箱 postMessage 协议，以及服务器渲染 HTML 所带来的安全面。

**类型：** 构建  
**语言：** Python（stdlib，UI 资源发射器）、HTML（示例应用）  
**先决条件：** 第 13 阶段 · 07（MCP 服务器）、第 13 阶段 · 10（资源）  
**时间：** 约 75 分钟

## 学习目标

- 从工具调用返回 `ui://` 资源，并设置正确的 MIME 和元数据。
- 使用 `_meta.ui.resourceUri`、`_meta.ui.csp` 和 `_meta.ui.permissions` 声明工具关联的 UI。
- 实现 iframe 沙箱 postMessage JSON-RPC，用于 UI 与宿主的通信。
- 应用 CSP 和 permissions-policy 默认策略，防御来自 UI 的攻击。

## 问题背景

2025 年代的 `visualize_timeline` 工具可能返回“这里是按时间顺序排列的 14 条笔记：...”。这只是一个段落。用户实际上想要的是交互式时间线。在 MCP 应用出现前，方案是：特定客户端的 widget API（Claude 材料、OpenAI 自定义 GPT HTML），或者根本没有 UI。

MCP 应用（SEP-1724，2026 年 1 月 26 日发布）标准化了这一契约。工具结果包含一个 URI 为 `ui://...` 的 `resource`，MIME 是 `text/html;profile=mcp-app`。宿主在受限 CSP 和无网络访问（除非显式授权）的沙箱 iframe 中渲染它。iframe 内的 UI 通过极小的 postMessage JSON-RPC 方言向宿主发送消息。

所有兼容客户端（Claude Desktop、ChatGPT、Goose、VS Code）都以一致方式渲染相同的 `ui://` 资源。一个服务器，一个 HTML 包，通用 UI。

## 概念解析

### `ui://` 资源方案

工具返回：

```json
{
  "content": [
    {"type": "text", "text": "Here is your notes timeline:"},
    {"type": "ui_resource", "uri": "ui://notes/timeline"}
  ],
  "_meta": {
    "ui": {
      "resourceUri": "ui://notes/timeline",
      "csp": {
        "defaultSrc": "'self'",
        "scriptSrc": "'self' 'unsafe-inline'",
        "connectSrc": "'self'"
      },
      "permissions": []
    }
  }
}
```

宿主随后调用 `resources/read`，请求 `ui://notes/timeline` URI，返回：

```json
{
  "contents": [{
    "uri": "ui://notes/timeline",
    "mimeType": "text/html;profile=mcp-app",
    "text": "<!doctype html>..."
  }]
}
```

### Iframe 沙箱

宿主将 HTML 渲染到带沙箱属性的 `<iframe>` 中，属性包括：

- `sandbox="allow-scripts allow-same-origin"`（或依服务器声明更严格）
- 服务器通过响应头施加的 CSP。
- 不带宿主源的 cookie 和 localStorage。
- 网络访问限制在 CSP 的 `connectSrc`。

### postMessage 协议

iframe 使用 `window.postMessage` 与宿主通信。采用简化的 JSON-RPC 2.0 方言：

务必将 `targetOrigin` 锁定为通信对端的精确 origin，接收方验证 `event.origin` 是否在白名单内后才处理载荷。绝不可使用 `"*"` 作为通道任一端的参数 —— 消息体承载工具调用和资源读取。

```js
// iframe 向宿主发送（锁定到宿主 origin）
window.parent.postMessage({
  jsonrpc: "2.0",
  id: 1,
  method: "host.callTool",
  params: { name: "notes_update", arguments: { id: "note-14", title: "..." } }
}, "https://host.example.com");

// 宿主向 iframe 发送（锁定到 iframe origin）
iframe.contentWindow.postMessage({
  jsonrpc: "2.0",
  id: 1,
  result: { content: [...] }
}, "https://iframe.example.com");

// 双方监听接收
window.addEventListener("message", (event) => {
  if (event.origin !== "https://expected-peer.example.com") return;
  // 安全地处理 event.data
});
```

UI 可调用的宿主方法：

- `host.callTool(name, arguments)` — 调用服务器工具。
- `host.readResource(uri)` — 读取 MCP 资源。
- `host.getPrompt(name, arguments)` — 获取提示模板。
- `host.close()` — 关闭 UI。

所有调用仍通过 MCP 协议，继承服务器权限。

### 权限请求

`_meta.ui.permissions` 列表请求额外能力：

- `camera` — 访问用户摄像头（用于扫码文档 UI）。
- `microphone` — 语音输入。
- `geolocation` — 定位权限。
- `network:*` — 比单单 `connectSrc` 允许更广泛的网络访问。

每项权限都会在 UI 渲染前向用户弹出提示。

### 安全风险

iframe 中的 HTML 依然是 HTML，新增攻击面包括：

- **UI 诱导的提示注入。** 恶意服务器 UI 可展示看似系统消息的文本，欺骗用户。宿主渲染应明显区分服务器 UI 与宿主 UI。
- **通过 `connectSrc` 数据泄露。** 若 CSP 放宽为 `connect-src: *`，UI 可能向任意外部发送数据。默认应设为严格策略。
- **点击劫持。** UI 可能遮盖宿主界面。宿主需阻止 z-index 操作和强制不透明度规则。
- **抢占焦点。** UI 可能抢占键盘焦点并截获后续输入。宿主必须拦截。

第 13 阶段 · 15 详细覆盖 MCP 安全性；本课为入门介绍。

### `ui/initialize` 握手

iframe 加载完成后，通过 postMessage 发送 `ui/initialize`：

```json
{"jsonrpc": "2.0", "id": 0, "method": "ui/initialize",
 "params": {"theme": "dark", "locale": "en-US", "sessionId": "..."}}
```

宿主返回功能能力和会话令牌。UI 后续调用均携带此令牌。

### AppRenderer / AppFrame SDK 原语

ext-apps SDK 提供两个便捷原语：

- `AppRenderer`（服务器端）— 包装 React / Vue / Solid 组件，发射带正确 MIME 和元数据的 `ui://` 资源。
- `AppFrame`（客户端）— 接收资源，挂载 iframe，并中介 postMessage 通信。

也可手写 HTML 和 JSON-RPC。

### 生态状态

MCP 应用于 2026 年 1 月 26 日发布。2026 年 4 月客户端支持情况：

- **Claude Desktop。** 2026 年 1 月起全面支持。
- **ChatGPT。** 通过 Apps SDK 全面支持（底层 MCP Apps 协议）。
- **Cursor。** 测试版，需设置启用。
- **VS Code。** 仅 Insider 版本支持。
- **Goose。** 全面支持。
- **Zed, Windsurf。** 路线图规划中。

生产环境服务器示例：仪表板、地图可视化、数据表、图表构建器、沙箱 IDE 预览。

## 实践操作

`code/main.py` 扩展了笔记服务器，添加了返回 `ui://notes/timeline` 资源的 `visualize_timeline` 工具，以及对应 `resources/read` 处理器，返回一个带 SVG 时间线的小型完整 HTML 包。HTML 由 stdlib 模板生成，无需构建工具。postMessage 通过 JS 注释展示，因 stdlib 不能驱动浏览器。

观察重点：

- 工具响应中的 `_meta.ui` 包含 resourceUri、CSP、权限。
- HTML 渲染无网络访问，所有数据均内嵌。
- JS 通过 `window.parent.postMessage` 调用 `host.callTool`（演示文档中说明但未激活）。

## 发布

本课生成文件 `outputs/skill-mcp-apps-spec.md`。给定带交互 UI 需求的工具，技能产出完整 MCP 应用契约：`ui://` URI、CSP、权限、postMessage 入口点和安全清单。

## 练习

1. 运行 `code/main.py`，检查生成的 HTML。将 HTML 文件直接在浏览器打开，确认 SVG 正确渲染。然后设计 UI 如何使用 postMessage 调用 `host.callTool("notes_update", ...)`。

2. 收紧 CSP：去掉 `'unsafe-inline'`，改用基于 nonce 的脚本策略。HTML 生成代码需做哪些改动？

3. 新增第二个 UI 资源 `ui://notes/editor`，提供笔记就地编辑表单。用户提交时，iframe 调用 `host.callTool("notes_update", ...)`。

4. 审计 UI 攻击面。恶意服务器可能在哪些地方注入内容？iframe 沙箱可防御哪些攻击，哪些不能？

5. 阅读 SEP-1724 规格，找出该玩具实现未利用的 MCP 应用 SDK 能力之一。（提示：组件级状态同步）

## 关键词

| 术语 | 常用说法 | 实际含义 |
|------|----------|----------|
| MCP Apps | “交互式 UI 资源” | 2026-01-26 发布的 SEP-1724 扩展 |
| `ui://` | “应用 URI 方案” | UI 资源包的资源方案 |
| `text/html;profile=mcp-app` | “MIME 类型” | MCP 应用 HTML 内容类型 |
| Iframe 沙箱 | “渲染容器” | 通过 CSP 与权限对 UI 进行浏览器沙箱隔离 |
| postMessage JSON-RPC | “UI 到宿主通道” | 用于宿主调用的极小 postMessage JSON-RPC 方言 |
| `_meta.ui` | “工具与 UI 绑定” | 将工具结果与 UI 资源关联的元数据 |
| CSP | “内容安全策略” | 声明脚本、网络、样式的允许来源 |
| AppRenderer | “服务端 SDK 原语” | 将框架组件转换为 `ui://` 资源 |
| AppFrame | “客户端 SDK 原语” | 挂载 iframe 并中介 postMessage 通信 |
| `ui/initialize` | “握手协议” | UI 向宿主发送的首条 postMessage |

## 延伸阅读

- [MCP ext-apps — GitHub](https://github.com/modelcontextprotocol/ext-apps) — 参考实现与 SDK
- [MCP Apps 规范 2026-01-26](https://github.com/modelcontextprotocol/ext-apps/blob/main/specification/2026-01-26/apps.mdx) — 正式规范文档
- [MCP — Apps 扩展概述](https://modelcontextprotocol.io/extensions/apps/overview) — 高层文档
- [MCP 博客 — MCP Apps 启动](https://blog.modelcontextprotocol.io/posts/2026-01-26-mcp-apps/) — 2026 年 1 月启动文章
- [MCP Apps API 参考](https://apps.extensions.modelcontextprotocol.io/api/) — JSDoc 格式 SDK 参考
