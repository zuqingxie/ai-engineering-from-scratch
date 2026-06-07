# Claude Agent SDK：子代理和会话存储

> Claude Agent SDK 是 Claude Code 框架的库形式。内置工具、用于上下文隔离的子代理、钩子、W3C 跟踪传播、会话存储并保持一致。Claude Managed Agents 是用于长时间异步工作的托管替代方案。

**类型：** 学习 + 构建  
**语言：** Python（标准库）  
**先决条件：** 第14阶段 · 01（代理循环）、第14阶段 · 10（技能库）  
**时间：** 约75分钟

## 学习目标

- 解释 Anthropic Client SDK（原始 API）与 Claude Agent SDK（框架形态）之间的区别。  
- 描述子代理——并行化与上下文隔离——以及何时使用它们。  
- 列出 Python SDK 的会话存储接口（`append`、`load`、`list_sessions`、`delete`、`list_subkeys`）及 `--session-mirror` 的作用。  
- 实现一个带内置工具、带隔离上下文的子代理生成、生命周期钩子和会话存储的标准库框架。  

## 问题背景

原始 LLM API 只提供一次往返调用。生产级代理需要工具执行、MCP 服务器、生命周期钩子、子代理生成、会话持久化、跟踪传播。Claude Agent SDK 以库形式提供此框架——也是 Claude Code 使用的相同框架，向自定义代理开放。

## 概念

### 客户端 SDK 与代理 SDK

- **客户端 SDK (`anthropic`)。** 原始消息 API。你自己管理循环、工具和状态。  
- **代理 SDK (`claude-agent-sdk`)。** 内置工具执行、MCP 连接、钩子、子代理生成、会话存储。Claude Code 循环的库实现。

### 内置工具

SDK 开箱即用包含 10 多种工具：文件读写、shell 命令、grep、glob、网页抓取等。自定义工具通过标准的工具模式接口注册。

### 子代理

Anthropic 文档列出了两个用途：

1. **并行化。** 并发运行独立任务。例：找到 20 个模块的测试文件，是 20 个并行子代理任务。  
2. **上下文隔离。** 子代理使用自己的上下文窗口；只有结果回传给主控。主控的上下文预算得以保留。

Python SDK 新增了：`list_subagents()`、`get_subagent_messages()` 用于读取子代理对话记录。

### 会话存储

与 TypeScript 保持协议一致：

- `append(session_id, message)` — 添加一个回合。  
- `load(session_id)` — 恢复对话。  
- `list_sessions()` — 枚举会话。  
- `delete(session_id)` — 级联删除子代理会话。  
- `list_subkeys(session_id)` — 列出子代理键。

`--session-mirror`（命令行标志）在对话流式传输时，将对话记录镜像到外部文件，用于调试。

### 钩子

可注册的生命周期钩子：

- `PreToolUse`、`PostToolUse` — 控制或审计工具调用。  
- `SessionStart`、`SessionEnd` — 初始化和清理。  
- `UserPromptSubmit` — 处理用户输入，在模型接收前。  
- `PreCompact` — 上下文压缩前运行。  
- `Stop` — 代理退出时清理。  
- `Notification` — 侧通道通知。

钩子是像专业工作流（第14阶段课程参考）等系统添加横切行为的方式。

### W3C 跟踪上下文

OTel 追踪跨度在调用者端激活，经过 W3C 跟踪上下文头传导进 CLI 子进程。整个多进程跟踪在后端展现为单个追踪。

### Claude Managed Agents

托管版本（Beta 版本标头为 `managed-agents-2026-04-01`）。支持长时间异步工作，内置提示缓存和压缩。以管理基础设施为代价换取控制权。

### 本模式的误区

- **子代理过度生成。** 为 100 个微小任务生成 100 个子代理，开销过大。应批量处理。  
- **钩子泛滥。** 各组添加钩子，启动时间膨胀。需季度审查钩子。  
- **会话膨胀。** 会话不断累积，大小增长。要配合 `list_sessions` 和过期策略使用。

## 构建它

`code/main.py` 实现了标准库中的 SDK 框架：

- `Tool`、`ToolRegistry` 带内置 `read_file`、`write_file`、`list_dir` 工具。  
- `Subagent` — 私有上下文、隔离运行、返回结果。  
- `SessionStore` — 支持 append、load、list、delete、list_subkeys。  
- `Hooks` — 包括 `pre_tool_use`、`post_tool_use`、`session_start`、`session_end`。  
- 一个演示：主代理并行生成 3 个子代理（各自隔离），汇总结果，持久化会话。

运行：

```text
python3 code/main.py
```

追踪显示子代理上下文隔离（主控上下文大小保持有限）、钩子执行和会话持久化。

## 使用它

- **Claude Agent SDK** 适合想要 Claude Code 框架形态的 Claude 优先产品。  
- **Claude Managed Agents** 适合托管的长时间异步工作。  
- **OpenAI Agents SDK**（第16课）是面向 OpenAI 的对应实现。  
- **LangGraph + 自定义工具** 若想要图形状态机则选此。

## 部署它

`outputs/skill-claude-agent-scaffold.md` 脚手架生成包含子代理、钩子、会话存储、MCP 服务器连接和 W3C 跟踪传播的 Claude Agent SDK 应用。

## 练习

1. 添加一个子代理生成器，把 20 个任务分批成 5 个并行子代理。测量主控上下文大小对比单任务生成。  
2. 实现一个 `PreToolUse` 钩子，限制每会话 `write_file` 调用速率（每分钟5次）。跟踪行为变化。  
3. 使用 `list_subkeys` 绘制子代理树。深层嵌套是什么样？  
4. 将示例移植到真实的 `claude-agent-sdk` Python 包，工具注册有什么变化？  
5. 阅读 Claude Managed Agents 文档，何时应从自托管切换到托管？

## 关键词

| 术语           | 大众说法           | 实际含义                                     |
|----------------|--------------------|----------------------------------------------|
| Agent SDK      | “Claude Code 库”   | 框架形态：工具、MCP、钩子、子代理、会话存储       |
| Subagent      | “子代理”           | 独立上下文，自有预算；结果向上冒泡                 |
| Session store | “对话数据库”       | 持久化、加载、列举、删除回合，支持子代理级联         |
| Hook          | “生命周期回调”     | 钩子接口：工具前后、会话开始结束、提交输入、压缩、停止 |
| W3C trace context | “跨进程跟踪”    | 父跨度传导进 CLI 子进程                           |
| Managed Agents | “托管框架”         | Anthropic 托管的长时间异步工作                      |
| `--session-mirror` | “对话镜像”      | 实时写会话回合到外部文件                            |
| MCP server    | “工具表面”          | 外部工具/资源源挂载代理                            |

## 扩展阅读

- [Claude Agent SDK 概述](https://platform.claude.com/docs/en/agent-sdk/overview) — Claude Code 的库形式  
- [Anthropic，使用 Claude Agent SDK 构建代理](https://www.anthropic.com/engineering/building-agents-with-the-claude-agent-sdk) — 生产模式  
- [Claude Managed Agents 概述](https://platform.claude.com/docs/en/managed-agents/overview) — 托管替代方案  
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) — 对应实现
