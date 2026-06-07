# 构建 MCP 服务器 — Python + TypeScript SDK

> 大多数 MCP 教程只展示 stdio 的 hello-world。真正的服务器会暴露工具、资源和提示，处理能力协商，输出结构化错误，并且在各个 SDK 上表现一致。本课将从头到尾构建一个笔记服务器：stdlib stdio 传输，JSON-RPC 分发，三个服务器原语，以及可以无缝切换到 Python SDK 的 FastMCP 或 TypeScript SDK 的纯函数风格。

**类型：** 构建  
**语言：** Python（stdlib，stdio MCP 服务器）  
**前置知识：** 阶段 13 · 06（MCP 基础）  
**时间：** ~75 分钟  

## 学习目标

- 实现 `initialize`、`tools/list`、`tools/call`、`resources/list`、`resources/read`、`prompts/list` 和 `prompts/get` 方法。
- 编写一个分发循环，从 stdin 读取 JSON-RPC 消息，并向 stdout 写响应。
- 根据 JSON-RPC 2.0 规范和 MCP 的额外代码发出结构化错误响应。
- 将 stdlib 实现无缝毕业到 FastMCP（Python SDK）或 TypeScript SDK，无需重写工具逻辑。

## 问题背景

在使用远程传输（阶段 13 · 09）或认证层（阶段 13 · 16）前，你需要一个干净的本地服务器。本地意味着 stdio：服务器由客户端作为子进程生成，消息通过 stdin/stdout 按换行分隔流动。

2025-11-25 规范规定 stdio 消息以 JSON 对象编码，并使用显式的 `\n` 分隔。不使用 SSE；SSE 是旧远程模式，将于2026年中旬移除（Atlassian 的 Rovo MCP 服务器于 2026 年 6 月 30 日弃用，Keboola 于 2026 年 4 月 1 日弃用）。stdio 的线格式即为每行一个 JSON 对象。

笔记服务器形态良好，因为它涵盖所有三个服务器原语。工具进行变更（`notes_create`），资源暴露数据（`notes://{id}`），提示传递模板（`review_note`）。本课的形态可推广至任何领域。

## 概念

### 分发循环

```text
循环:
  line = stdin.readline()
  msg = json.loads(line)
  if 有 id:
    处理请求 -> 写响应
  else:
    处理通知 -> 无响应
```

三条规则：

- 不允许向 stdout 打印非 JSON-RPC 封包内容。调试日志输出到 stderr。
- 每个请求必须匹配带相同 `id` 的响应。
- 通知不能响应。

### 实现 `initialize`

```python
def initialize(params):
    return {
        "protocolVersion": "2025-11-25",
        "capabilities": {
            "tools": {"listChanged": True},
            "resources": {"listChanged": True, "subscribe": False},
            "prompts": {"listChanged": False},
        },
        "serverInfo": {"name": "notes", "version": "1.0.0"},
    }
```

只声明支持的功能。客户端依赖能力集合来控制功能开启。

### 实现 `tools/list` 和 `tools/call`

`tools/list` 返回 `{tools: [...]}`，每条包括 `name`、`description`、`inputSchema`。`tools/call` 接收 `{name, arguments}`，返回 `{content: [blocks], isError: bool}`。

内容块有类型。最常见的：

```json
{"type": "text", "text": "Found 2 notes"}
{"type": "resource", "resource": {"uri": "notes://14", "text": "..."}}
{"type": "image", "data": "<base64>", "mimeType": "image/png"}
```

工具错误分两类。协议级错误（未知方法，错误参数）为 JSON-RPC 错误。工具级错误（调用有效但工具失败）返回 `{content: [...], isError: true}`。模型可在上下文中看到失败。

### 实现资源

资源默认只读。`resources/list` 返回清单，`resources/read` 返回内容。URI 可为 `file://...`、`http://...` 或自定义方案如 `notes://`。

将数据作为资源暴露而非工具时：

- 模型不会“调用”它；客户端可按用户请求将其注入上下文。
- 订阅允许服务器在资源变更时推送更新（阶段 13 · 10）。
- 阶段 13 · 14 通过 `ui://` 扩展交互式资源。

### 实现提示

提示是带命名参数的模板。宿主将其作为斜杠命令展现。`review_note` 提示可能带 `note_id` 参数，生成可由客户端模型多消息输入的提示模板。

### stdio 传输细节

- 换行分隔的 JSON，无长度前缀帧。
- 不缓冲。每次写后调用 `sys.stdout.flush()`。
- 客户端控制生命周期。stdin 关闭（EOF）时正常退出。
- 不静默处理 SIGPIPE，需记录日志并退出。

### 注释信息

每个工具可携带描述安全属性的 `annotations`：

- `readOnlyHint: true` — 纯读取，安全重试。
- `destructiveHint: true` — 不可逆副作用，客户端应确认。
- `idempotentHint: true` — 相同输入产出相同结果。
- `openWorldHint: true` — 与外部系统交互。

客户端据此决定 UX（确认框、状态指示）和路由（阶段 13 · 17）。

### 毕业路径

`code/main.py` 中的 stdlib 服务器约 180 行。FastMCP（Python）将逻辑浓缩为装饰器风格：

```python
from fastmcp import FastMCP
app = FastMCP("notes")

@app.tool()
def notes_search(query: str, limit: int = 10) -> list[dict]:
    ...
```

TypeScript SDK 也有类似形态。毕业路径只需替换，概念（能力，分发，内容块）不变。

## 使用指南

`code/main.py` 是完整的 stdio 笔记 MCP 服务器，仅 stdlib 实现。支持 `initialize`、`tools/list`、`tools/call` 的三个工具（`notes_list`、`notes_search`、`notes_create`），支持每条笔记的 `resources/list` 和 `resources/read`，以及 `review_note` 提示。你可通过管道传递 JSON-RPC 消息驱动之：

```text
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' | python main.py
```

关注点：

- 分发器是以方法名为键的 `dict[str, Callable]`。
- 每个工具执行器返回内容块列表，而非裸字符串。
- 当执行器抛异常时设 `isError: true`。

## 发布指南

本课生成 `outputs/skill-mcp-server-scaffolder.md`。鉴于领域（笔记、工单、文件、数据库），该技能脚手架 MCP 服务器，含合理的工具/资源/提示拆分及 SDK 毕业路径。

## 练习

1. 运行 `code/main.py` 并通过手写 JSON-RPC 消息驱动。调用 `notes_create`，再用 `resources/read` 读取新笔记。

2. 增加带 `annotations: {destructiveHint: true}` 的 `notes_delete` 工具。验证客户端会弹出确认框（需真实宿主；Claude Desktop 支持）。

3. 实现 `resources/subscribe`，使服务器在笔记修改时推送 `notifications/resources/updated`。添加保持活动任务。

4. 将服务器迁移到 FastMCP。Python 文件缩减至 80 行以内。线行为须保持一致；用相同 JSON-RPC 测试套件验证。

5. 阅读规范的 `server/tools` 部分，识别本课服务器未实现的工具定义字段之一。（提示：有多个，任选一项添加。）

## 关键词汇

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| MCP server | “暴露工具的东西” | 通过 stdio 或 HTTP 使用 MCP JSON-RPC 通信的进程 |
| stdio transport | “子进程模型” | 服务器被客户端派生；通过 stdin/stdout 通信 |
| Dispatcher（分发器） | “方法路由器” | JSON-RPC 方法名到处理函数的映射 |
| Content block（内容块） | “工具结果块” | 工具响应 `content` 数组中的类型化元素 |
| `isError` | “工具级失败” | 表示工具调用失败；区别于 JSON-RPC 错误 |
| Annotations（注释信息） | “安全提示” | 只读 / 破坏性 / 幂等 / 开放世界 标识 |
| FastMCP | “Python SDK” | 基于装饰器的 MCP 协议高级框架 |
| Resource URI（资源 URI） | “可寻址数据” | `file://`、`db://` 或自定义方案标识资源 |
| Prompt template（提示模板） | “斜杠命令简述” | 服务器提供带参数槽的模板给宿主 UI |
| Capability declaration（能力声明） | “功能开关” | 在 `initialize` 中声明的每个原语标志 |

## 深入阅读

- [Model Context Protocol — Python SDK](https://github.com/modelcontextprotocol/python-sdk) — 参考 Python 实现  
- [Model Context Protocol — TypeScript SDK](https://github.com/modelcontextprotocol/typescript-sdk) — 并行 TS 实现  
- [FastMCP — 服务器框架](https://gofastmcp.com/) — 基于装饰器风格的 Python MCP API  
- [MCP — 快速入门服务器指南](https://modelcontextprotocol.io/quickstart/server) — 使用任一 SDK 的端到端教程  
- [MCP — 服务器工具规范](https://modelcontextprotocol.io/specification/2025-11-25/server/tools) — tools/* 消息完整参考文档
