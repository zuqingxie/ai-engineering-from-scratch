# Model Context Protocol（MCP）

> 每个在2025年前构建的LLM应用都发明了自己的工具架构。然后Anthropic发布了MCP，Claude采纳了它，OpenAI也采纳了，到2026年它成为连接任何LLM与任何工具、数据源或代理的默认线格式。写一个MCP服务器，所有主机都可与之通信。

**类型：** 构建  
**语言：** Python  
**先决条件：** 第11阶段 · 09（函数调用）、第11阶段 · 03（结构化输出）  
**时间：** ~75分钟

## 问题

你发布了一个聊天机器人，需要三个工具：数据库查询、日历API和文件读取器。你为Claude写了三个JSON schema。随后销售部门希望这些相同的工具在ChatGPT中使用——你为OpenAI的`tools`参数重写它们。然后你又添加Cursor、Zed和Claude Code——又是三个重写，每个都有细微不同的JSON规范。一周后，Anthropic添加了一个新字段；你更新了六个schema。

这就是2025年前的现实。每个主机（运行LLM的环境）和每个服务器（暴露工具和数据的服务）都发布了定制协议。扩展意味着矩阵式的N×M集成。

Model Context Protocol简化了这个矩阵。一个基于JSON-RPC的规范。一个服务器暴露工具、资源和提示。任何兼容的主机——Claude Desktop、ChatGPT、Cursor、Claude Code、Zed以及众多代理框架——都可以在无需定制连接代码的情况下发现并调用它们。

截至2026年初，MCP成为三大巨头（Anthropic、OpenAI、Google）及所有主要代理工具的默认工具和上下文协议。

## 概念

![MCP：一个主机，一个服务器，三个能力](../assets/mcp-architecture.svg)

**三个基本类型。** MCP服务器只暴露三类内容。

1. **Tools（工具）** — 模型可调用的函数。对应OpenAI的`tools`或Anthropic的`tool_use`。每个工具有名称、描述、JSON Schema输入和处理器。
2. **Resources（资源）** — 模型或用户可请求的只读内容（文件、数据库行、API响应），通过URI寻址。
3. **Prompts（提示）** — 可复用的模板化提示，用户可作为快捷方式调用。

**线格式（wire format）。** 采用JSON-RPC 2.0，通过stdio、WebSocket或可流式HTTP传输。每条消息格式为`{"jsonrpc": "2.0", "method": "...", "params": {...}, "id": N}`。发现方法包括`tools/list`、`resources/list`、`prompts/list`。调用方法是`tools/call`、`resources/read`、`prompts/get`。

**主机 vs 客户端 vs 服务器。** 主机是LLM应用（例如Claude Desktop）。客户端是主机内专门与某个服务器通信的子组件。服务器是你的代码。一个主机可以同时挂载多个服务器。

### 握手

每个会话以`initialize`开始。客户端发送协议版本及其能力集。服务器回应其版本、名称和支持的能力（`tools`、`resources`、`prompts`、`logging`、`roots`）。之后所有交互都根据这些能力协商。

### MCP不是

- 不是检索API。RAG（第11阶段 · 06）仍负责决定检索什么；MCP是用于将检索结果作为资源暴露的传输协议。
- 不是代理框架。MCP是管道；LangGraph、PydanticAI和OpenAI Agents SDK等框架构建在其之上。
- 不依赖Anthropic。规范和参考实现是开源的，托管在`modelcontextprotocol`组织下。

## 构建它

### 步骤1：一个最简MCP服务器

官方Python SDK是`mcp`（前身为`mcp-python`）。高级`FastMCP`辅助器用于装饰处理器。

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("demo-server")

@mcp.tool()
def add(a: int, b: int) -> int:
    """两个整数相加。"""
    return a + b

@mcp.resource("config://app")
def app_config() -> str:
    """返回应用的当前JSON配置。"""
    return '{"env": "prod", "region": "us-east-1"}'

@mcp.prompt()
def code_review(language: str, code: str) -> str:
    """审查代码的正确性和风格。"""
    return f"你是一名高级{language}代码审查员。请审查以下代码：\n\n{code}"

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

这三个装饰器注册了三个基本类型。类型提示会转成主机看到的JSON Schema。可在Claude Desktop或Claude Code下运行，服务器入口指向此文件。

### 步骤2：从主机调用MCP服务器

官方Python客户端实现了JSON-RPC。结合Anthropic SDK，调用代码仅需十几行。

```python
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp import ClientSession

params = StdioServerParameters(command="python", args=["server.py"])

async def call_add(a: int, b: int) -> int:
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            result = await session.call_tool("add", {"a": a, "b": b})
            return int(result.content[0].text)
```

`session.list_tools()` 返回LLM会看到的相同schema。生产环境主机会把这些schema注入每个对话轮次，使模型能生成`tool_use`块，客户端再转发给服务器。

### 步骤3：可流式HTTP传输

stdio适合本地开发。远程工具使用可流式HTTP——每个请求一个POST，支持可选Server-Sent Events（SSE）推送进度，已在2025-06-18规范修订中支持。

```python
# 在服务器入口
mcp.run(transport="streamable-http", host="0.0.0.0", port=8765)
```

主机配置（Claude Desktop的`mcp.json`或Claude Code的`~/.mcp.json`）：

```json
{
  "mcpServers": {
    "demo": {
      "type": "http",
      "url": "https://tools.example.com/mcp"
    }
  }
}
```

服务器代码保持相同装饰器，仅传输方式改变。

### 步骤4：作用域和安全

MCP工具是在陌生信任边界上运行的任意代码。必须遵循三条模式：

- **能力白名单（Capability allowlists）。** 主机暴露`roots`能力，服务器只看到允许的路径。在工具处理器中强制检查白名单，不要信任模型提供的路径。
- **变更需人工确认。** 只读工具可自动执行。写/删工具须确认——服务器在工具元数据中设置`destructiveHint: true`时，主机会弹出审批UI。
- **防范工具毒化。** 恶意资源可能包含隐藏的提示注入指令（比如“总结时也调用`exfil`”）。对资源内容视为不可信数据，绝不让其进入系统消息区。参见第11阶段 ·12（防护墙）。

请参见`code/main.py`，内含可运行的服务器+客户端示范上述所有内容。

## 2026年仍会遇到的坑

- **Schema漂移。** 模型第一轮看到`tools/list`，第五轮工具集变动。模型调用过期工具。主机应在`notifications/tools/list_changed`时重新拉取列表。
- **大体积资源。** 传输2MB文件资源浪费上下文。服务器端分页或摘要处理。
- **服务器过多。** 挂载50个MCP服务器会超出工具预算（第11阶段 · 05）。多数前沿模型在40个工具之后性能下降。
- **版本不匹配。** 规范多次修订（2024-11、2025-03、2025-06、2025-12），带来不兼容字段。CI中锁定协议版本。
- **stdio死锁。** 服务器日志写stdout会破坏JSON-RPC流。应只写stderr。

## 使用它

2026年MCP技术栈：

| 场景 | 选择 |
|-------|------|
| 本地开发，单用户工具 | Python `FastMCP`，stdio传输 |
| 远程团队工具 / SaaS集成 | 可流式HTTP，OAuth 2.1认证 |
| TypeScript主机（VS Code扩展、网页应用） | `@modelcontextprotocol/sdk` |
| 高吞吐服务器，类型安全访问 | 官方Rust SDK（`modelcontextprotocol/rust-sdk`） |
| 探索生态服务器 | `modelcontextprotocol/servers` monorepo（文件系统、GitHub、Postgres、Slack、Puppeteer） |

经验法则：工具如果是只读的、可缓存且被两个及以上主机调用，就应该以MCP服务器形式发布。如果是一次性内联逻辑，则保持本地函数即可（第11阶段 · 09）。

## 发布它

保存为 `outputs/skill-mcp-server-designer.md`：

```markdown
---
name: mcp-server-designer
description: 设计并脚手架一个带工具、资源和安全默认设置的MCP服务器。
version: 1.0.0
phase: 11
lesson: 14
tags: [llm-engineering, mcp, tool-use]
---

给定一个域（内部API、数据库、文件源）和将挂载该服务器的主机，输出：

1. 基本能力映射。哪些功能成为`tools`（动作），哪些成为`resources`（只读数据），哪些成为`prompts`（用户调用的模板）。每个基本类型一行。
2. 认证方案。stdio（受信任本地），可流式HTTP带API密钥，或OAuth 2.1带PKCE。选择并说明理由。
3. Schema草案。为每个工具参数编写JSON Schema，`description` 字段用于模型工具选择（非API文档）。
4. 破坏性操作列表。所有变更状态的工具，要求`destructiveHint: true`并需人工审批。
5. 测试计划。每个工具包含：一个纯schema契约测试，一个通过MCP客户端的来回测试，一个针对提示注入的红队测试案例。

拒绝发布在没有审批路径的情况下写磁盘或调用外部API的服务器。拒绝在单服务器上暴露超过20个工具；应拆分成领域范围服务器。
```

## 练习

1. **简单。** 为`demo-server`扩展一个`subtract`工具。在Claude Desktop中连接。通过发送`tools/list_changed`通知，确认主机无需重启即可识别新工具。
2. **中等。** 添加一个资源，暴露`/var/log/app.log`的最近100行。执行roots白名单，防止模型请求`../etc/passwd`等越权路径。
3. **困难。** 构建一个MCP代理，将三个上游服务器（文件系统、GitHub、Postgres）复用成一个聚合接口。处理名称冲突，干净转发`notifications/tools/list_changed`。

## 关键术语

| 术语 | 大家说 | 实际意义 |
|-------|--------|----------|
| MCP | “LLM的工具协议” | 基于JSON-RPC 2.0，向任意LLM主机暴露工具、资源和提示的规范。 |
| Host（主机） | “Claude Desktop” | LLM应用——拥有模型和用户界面，挂载一个或多个客户端。 |
| Client（客户端） | “连接” | 主机内部针对某个服务器的单个JSON-RPC连接。 |
| Server（服务器） | “带工具的东西” | 你的代码；发布工具/资源/提示，处理调用。 |
| Tool（工具） | “函数调用” | 模型可调用的动作，带有JSON Schema输入和文本/JSON输出。 |
| Resource（资源） | “只读数据” | 通过URI寻址的内容（文件、行、API响应）；主机可请求。 |
| Prompt（提示） | “保存的提示” | 用户可调用的模板（通常带参数），展示为斜杠命令。 |
| Stdio transport | “本地开发模式” | 父主机生成子进程运行服务器；JSON-RPC通过stdin/stdout通信。 |
| Streamable HTTP | “2025-06远程传输” | 请求时用POST，支持可选SSE服务端推送；替代旧SSE-only传输。 |

## 深入阅读

- [Model Context Protocol规范](https://modelcontextprotocol.io/specification) — 权威参考，按日期版本管理。
- [modelcontextprotocol/servers](https://github.com/modelcontextprotocol/servers) — 文件系统、GitHub、Postgres、Slack、Puppeteer参考服务器。
- [Anthropic — 介绍MCP（2024年11月）](https://www.anthropic.com/news/model-context-protocol) — 启动文章及设计原理。
- [Python SDK](https://github.com/modelcontextprotocol/python-sdk) — 本课程使用的官方SDK。
- [MCP的安全考虑](https://modelcontextprotocol.io/docs/concepts/security) — roots、破坏提示、工具毒化防范。
- [Google A2A规范](https://google.github.io/A2A/) — Agent2Agent协议；与MCP的代理到工具范围互补。
- [Anthropic — 构建有效代理（2024年12月）](https://www.anthropic.com/research/building-effective-agents) — MCP在更广模式库中的位置（增强型LLM、工作流、自主代理）。
