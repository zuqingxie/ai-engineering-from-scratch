# 异步任务（SEP-1686）—— 长时间运行工作的即时调用，稍后取回

> 真实的智能体工作需要几分钟到几小时：CI 运行、深度研究合成、批量导出。同步工具调用会导致连接中断、超时或阻塞UI。SEP-1686 于 2025-11-25 合并，添加了 Tasks 原语：任何请求都可以增强为任务，并且结果可以稍后获取或通过状态通知流式传输。漂移风险提示：任务在 2026 年上半年仍属实验性质；SDK 接口仍围绕规范设计。

**类型：** 构建  
**语言：** Python（标准库、异步任务状态机）  
**先决条件：** 第13阶段 · 07（MCP 服务器），第13阶段 · 09（传输层）  
**时间：** 约 75 分钟  

## 学习目标

- 识别何时将工具从同步转换为任务增强（服务器端工作超过30秒）。
- 理解任务生命周期：`working` → `input_required` → `completed` / `failed` / `cancelled`。
- 持久化任务状态，防止崩溃导致正在进行的工作丢失。
- 正确轮询 `tasks/status` 并获取 `tasks/result`。

## 问题描述

一个 `generate_report` 工具运行一个多分钟的数据提取管道。同步模型下的选项：

1. 维持连接三分钟。远程传输会断开；客户端超时；UI 冻结。
2. 立即返回占位符；要求客户端轮询自定义端点。破坏 MCP 的统一性。
3. 发送即忘；无结果。

都不好。SEP-1686 引入了第四种方案：任务增强。任何请求（通常是 `tools/call`）都可以标记为任务。服务器立即返回任务ID。客户端轮询 `tasks/status` 并在完成时获取 `tasks/result`。服务器端状态可持久化并抵抗重启。

## 概念

### 任务增强

通过设置 `params._meta.task.required: true`（或 `optional: true`，由服务器决定），请求变成任务。服务器立即响应：

```json
{
  "jsonrpc": "2.0", "id": 1,
  "result": {
    "_meta": {
      "task": {
        "id": "tsk_9f7b...",
        "state": "working",
        "ttl": 900000
      }
    }
  }
}
```

`ttl` 是服务器承诺保留状态的时间（毫秒）；过期后任务结果丢弃。

### 每工具选择加入

工具注释里可声明任务支持情况：

- `taskSupport: "forbidden"` — 此工具始终同步运行。适合快速工具。
- `taskSupport: "optional"` — 客户端可请求任务增强。
- `taskSupport: "required"` — 客户端必须使用任务增强。

`generate_report` 工具应设为 `required`。`notes_search` 工具设为 `forbidden`。

### 状态机

```text
working  -> input_required -> working  （通过询问循环）
working  -> completed
working  -> failed
working  -> cancelled
```

状态机为追加式：一旦为 `completed`、`failed` 或 `cancelled`，任务即终止。

### 方法接口

- `tasks/status {taskId}` — 返回当前状态和进度提示。
- `tasks/result {taskId}` — 阻塞等待或返回 404（未完成时）。
- `tasks/cancel {taskId}` — 幂等操作；终态忽略。
- `tasks/list` — 可选；列举活跃和最近完成的任务。

### 状态变更流式推送

服务器支持时，客户端可订阅状态通知：

```text
server -> notifications/tasks/updated {taskId, state, progress?}
```

流式比轮询更好，轮询始终是最低门槛支持。

### 持久化状态

规范要求声明支持任务的服务器必须持久化状态。崩溃时不丢失 ttl 内完成任务结果。存储方案有 SQLite、Redis、文件系统。第13课用的是文件系统。

### 取消语义

`tasks/cancel` 是幂等的。若任务执行中，服务器会尝试停止（检查执行器协作取消）。若已终止，则请求无操作。

### 崩溃恢复

服务器重启时：

1. 加载所有持久化任务状态。
2. 标记所有执行进程死掉的 `working` 任务为 `failed`，错误为 `CRASH_RECOVERY`。
3. 保持 `completed` / `failed` / `cancelled` 状态并存 ttl 时间。

### 异步任务加采样

任务内部可以调用 `sampling/createMessage`。这就是长时间运行研究任务的方式：服务器任务线程根据需求采样客户端模型，同时客户端UI显示任务为 `working` 并定期更新进度。

### 为什么还属实验性

SEP-1686 于2025-11-25发布，但整体路线图指出三个待解决问题：持久化订阅原语、子任务（父子任务关系）、结果 ttl 标准化。规范预计在2026年持续演进。生产代码应对任务只视为常见情况稳定，并预防未来 SDK 因子任务改动。

## 使用说明

`code/main.py` 实现了一个耐用的任务存储（基于文件系统）与一个在后台线程运行的 `generate_report` 工具。客户端调用工具，立即获得任务ID，轮询 `tasks/status` 期间工作线程更新进度，完成时获取 `tasks/result`。支持取消；崩溃恢复通过杀死工作线程并重载状态模拟。

重点关注：

- 任务状态 JSON 持久化到 `/tmp/lesson-13-tasks/<id>.json`。
- 工作线程更新 `progress` 字段；轮询显示进度递增。
- 客户端取消时设置事件；工作线程检测后提前退出。
- 崩溃重载时将进行中任务标记为带 `CRASH_RECOVERY` 的 `failed`。

## 发布说明

本课产出 `outputs/skill-task-store-designer.md`。针对长运行工具（研究、构建、导出），设计技能的任务存储（状态形态、ttl、持久性），选取合适的 taskSupport 标志，并草拟进度通知。

## 练习

1. 运行 `code/main.py`。启动一个 `generate_report` 任务，轮询状态，然后获取结果。

2. 在运行时添加一次 `tasks/cancel` 调用。验证工作线程响应并任务状态变为 `cancelled`。

3. 模拟崩溃恢复：杀死工作线程，重启加载程序，观察出现 `CRASH_RECOVERY` 失败模式。

4. 将存储扩展为 SQLite。持久化优势不变，提供查询选项（列出会话X的所有任务）。

5. 阅读 MCP 2026 路线图文章。找出最可能影响明年 SDK API 设计的任务相关开放问题。

## 关键词

| 术语 | 通俗说法 | 实际含义 |
|------|----------|----------|
| Task | “长时间运行的工具调用” | 通过 `_meta.task` 增强以异步执行的请求 |
| SEP-1686 | “任务规范” | 2025-11-25 添加任务的规范提案（Spec Evolution Proposal） |
| `_meta.task` | “任务封装” | 包含 id、状态、ttl 的每请求元数据 |
| taskSupport | “工具标志” | 每工具的 `forbidden` / `optional` / `required` |
| `tasks/status` | “轮询方法” | 获取当前状态和可选进度提示 |
| `tasks/result` | “获取结果” | 返回完成的负载，未完成时返回 404 |
| `tasks/cancel` | “取消它” | 幂等的取消请求 |
| ttl | “保留期” | 服务器承诺保存任务状态的毫秒数 |
| `notifications/tasks/updated` | “状态推送” | 服务器发起的状态变更事件 |
| Durable store | “防崩状态” | 文件系统 / SQLite / Redis 持久层 |

## 延伸阅读

- [MCP — GitHub SEP-1686 issue](https://github.com/modelcontextprotocol/modelcontextprotocol/issues/1686) — 原始提案与完整讨论  
- [WorkOS — MCP async tasks for AI agent workflows](https://workos.com/blog/mcp-async-tasks-ai-agent-workflows) — 设计流程及原理  
- [DeepWiki — MCP task system and async operations](https://deepwiki.com/modelcontextprotocol/modelcontextprotocol/2.7-task-system-and-async-operations) — 机制与状态机  
- [FastMCP — Tasks](https://gofastmcp.com/servers/tasks) — SDK层任务实现模式  
- [MCP blog — 2026 roadmap](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) — 待解决问题及2026重点，包含子任务
