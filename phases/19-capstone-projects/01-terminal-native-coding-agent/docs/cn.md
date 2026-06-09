# Capstone 01 — 终端原生编码代理

> 到2026年，编码代理的形态已定型。一个基于终端用户界面（TUI） 的框架（harness），一个有状态的计划，一个沙箱工具表面，一个循环包含计划、执行、观察和恢复。Claude Code、Cursor 3 和 OpenCode 从 50 英尺外看起来都一模一样。本毕业设计要求你从头到尾构建一个——命令行输入，拉取请求输出——并在 SWE-bench Pro 上与 mini-swe-agent 和 Live-SWE-agent 进行对比评测。你将学习为什么难点不在模型调用，而在工具循环、沙箱及50轮运行的成本上限。

**类型：** 毕业设计  
**语言：** TypeScript / Bun（框架），Python（评测脚本）  
**前置条件：** 阶段11（LLM 工程）、阶段13（工具与协议）、阶段14（代理）、阶段15（自治系统）、阶段17（基础设施）  
**所用阶段：** P0 · P5 · P7 · P10 · P11 · P13 · P14 · P15 · P17 · P18  
**时间：** 35 小时

## 问题

编码代理于2026年成为主流的 AI 应用类别。Claude Code（Anthropic）、带有 Composer 2 和 Agent Tabs（Cursor）的 Cursor 3、Amp（Sourcegraph）、OpenCode（11.2万星）、Factory Droids 和 Google Jules 都采用相似架构：终端框架、带权限的工具表面、沙箱及基于前沿模型的计划-执行-观察循环。前沿虽窄——Live-SWE-agent 使用 Opus 4.5 在 SWE-bench Verified 达到79.2%——但工程工艺广泛。大多数失败模式并非模型错误，而是工具循环不稳定、上下文污染、代币（token）成本失控、破坏性文件系统操作。

你无法从外部推理这些代理。必须亲自构建，观察循环在第47轮时因 ripgrep 返回8MB匹配结果而崩溃，然后重建截断层。这正是本课题的重点。

## 概念

框架有四个表面。**Plan（计划）**维护一个 TodoWrite 风格的状态对象，模型每轮重写。**Act（执行）**调度工具调用（读取、编辑、运行、搜索、Git）。**Observe（观察）**捕获 stdout / stderr / 退出码，截断并反馈摘要。**Recover（恢复）**处理工具错误，避免上下文窗口爆炸或无限循环。2026年的形态新增了一项：**钩子（hooks）**。`PreToolUse`、`PostToolUse`、`SessionStart`、`SessionEnd`、`UserPromptSubmit` 、`Notification`、`Stop` 和 `PreCompact` —— 这些可配置扩展点允许操作者注入策略、遥测和保护措施。

沙箱是 E2B 或 Daytona。每个任务在新的 devcontainer 中运行，并挂载带读写权限的 git 工作树。框架永远不触碰宿主文件系统。工作树任务成功或失败后被销毁。成本通过三层控制：每轮代币上限、每会话美元预算及硬性轮数限制（通常为50轮）。可观察性层基于 OpenTelemetry，带 GenAI 语义规范，发送至自托管 Langfuse。

## 架构

```text
  用户 CLI  ->  框架（Bun + Ink TUI）
                  |
                  v
           计划 / 执行 / 观察 循环  <--->  Claude Sonnet 4.7 / GPT-5.4-Codex / Gemini 3 Pro
                  |                          （通过 OpenRouter，模型无关）
                  v
           工具调度器（MCP StreamableHTTP 客户端）
                  |
     +------------+------------+----------+
     v            v            v          v
  读取/编辑   ripgrep      tree-sitter    git/运行
     |            |            |          |
     +------------+------------+----------+
                  |
                  v
           E2B / Daytona 沙箱 （工作树隔离）
                  |
                  v
           钩子：Pre/Post、会话、提示、压缩
                  |
                  v
           OpenTelemetry -> Langfuse（跨度、代币、花费）
                  |
                  v
           通过 GitHub 应用发起 PR
```

## 技术栈

- 框架运行时：Bun 1.2 + Ink 5（终端中的 React）
- 模型访问：OpenRouter 统一 API，支持 Claude Sonnet 4.7、GPT-5.4-Codex、Gemini 3 Pro、Opus 4.5（最难任务）
- 工具传输：模型上下文协议（MCP）StreamableHTTP（2026年修订版）
- 沙箱环境：E2B 沙箱（JS SDK）或 Daytona devcontainer
- 代码搜索：ripgrep 子进程，支持 17 种语言的 tree-sitter 解析器（预编译）
- 隔离：每任务使用 `git worktree add`，成功/失败后清理
- 评测框架：SWE-bench Pro（验证子集）+ Terminal-Bench 2.0 + 自定义30任务保留集
- 可观察性：OpenTelemetry SDK，带 `gen_ai.*` 语义规范，发送至自托管 Langfuse
- PR 提交：GitHub 应用，使用细粒度 token，限制访问目标仓库

## 构建步骤

1. **TUI 和命令循环。** 使用 Ink 搭建 Bun 项目。支持命令 `agent run <repo> "<task>"`。打印分割视图：计划窗格（顶部）、工具调用流（中部）、代币预算（底部）。支持 Ctrl-C 取消，退出前触发 `SessionEnd` 钩子。

2. **计划状态。** 定义带类型的 TodoWrite 结构（待办 / 进行中 / 已完成条目，附带备注）。模型每轮完整重写状态（作为工具调用），不允许增量修改。将计划持久化到 `.agent/state.json`，保证崩溃后可恢复。

3. **工具表面。** 定义六个工具：`read_file`、`edit_file`（带 diff 预览）、`ripgrep`、`tree_sitter_symbols`、`run_shell`（带超时）、`git`（状态 / diff / commit / push）。通过 MCP StreamableHTTP 暴露，确保框架与传输无关。所有工具输出截断，最多4千代币。

4. **沙箱封装。** 每个任务启动一个 E2B 沙箱。使用 `git worktree add -b agent/$TASK_ID` 新分支。所有工具调用均在沙箱执行，宿主文件系统无法访问。

5. **钩子。** 实现 2026 年的八种钩子。至少接入四个用户自定义钩子：（a）`PreToolUse` 破坏性命令保护，阻止工作树外的 `rm -rf`；（b）`PostToolUse` 代币计费；（c）`SessionStart` 预算初始化；（d）`Stop` 写入最终追踪包。

6. **评测循环。** 克隆 SWE-bench Pro 的 30 个 Python 任务子集。对每个运行你的框架。与 mini-swe-agent（极简基线）对比 pass@1、每任务轮数及每任务成本，结果写入 `eval/results.jsonl`。

7. **成本控制。** 硬性截断：50轮、20万个上下文代币、每任务5美元预算。`PreCompact` 钩子在到达15万个代币时，将旧轮次总结为先前状态块，释放空间供新观测但不丢失计划。

8. **PR 发布。** 成功时，最后一步执行 `git push` 并调用 GitHub API，创建包含计划和差异摘要的拉取请求。

## 使用示例

```text
$ agent run ./my-repo "Fix the race condition in worker.rs"
[plan]  1 确定 worker.rs 并列出互斥锁用法
        2 识别争用的共享状态
        3 提出修复方案，验证测试
[tool]  ripgrep mutex.*lock -t rust           （44个匹配，已截断）
[tool]  read_file src/worker.rs 120..180
[tool]  edit_file src/worker.rs (+8 -3)
[tool]  run_shell cargo test worker::          （通过）
[plan]  1 完成 · 2 完成 · 3 完成
[done]  PR 已开：#482   轮数=9   代币=38k   成本=$0.41
```

## 交付物

交付技能文档位于 `outputs/skill-terminal-coding-agent.md`。给定仓库路径和任务描述，它在沙箱中运行完整的计划-执行-观察循环，返回拉取请求 URL 与追踪包。本毕业设计评分标准：

| 权重 | 评判标准 | 测量方式 |
|:-:|---|---|
| 25 | SWE-bench Pro pass@1 相比基线 | Matched 30-task Python comparison against mini-swe-agent |
| 20 | 架构清晰度 | Plan/execute/observe separation, hook surface, and tool structure against Live-SWE-agent layout |
| 20 | 安全性 | Sandbox escape tests, permission prompts, and destructive-command red-team tests |
| 20 | 可观察性 | Trace completeness for all tool calls and per-turn token accounting accuracy |
| 15 | 开发者体验 | Cold start under 2 seconds, crash recovery plan, and clean Ctrl-C tool-call interruption |
| **100** |  |  |

## 练习

1. 将后端模型从 Claude Sonnet 4.7 替换为在 vLLM 上服务的 Qwen3-Coder-30B。比较 pass@1 和每任务花费。报告开放模型的性能劣势位置。

2. 新增一个 `reviewer` 子代理，在 PR 发布前阅读差异，可请求修订循环。测量是否假阳性评价导致 SWE-bench 通过率低于单代理基线（提示：通常是）。

3. 进行沙箱压力测试：编写一个尝试 `curl` 外部 URL 的任务和一个试图写出工作树的任务。确认二者都被 `PreToolUse` 钩子阻止。记录尝试。

4. 实现带小模型（Haiku 4.5）的 `PreCompact` 摘要。测量3倍压缩后计划保真度损失。

5. 将 MCP StreamableHTTP 传输替换为 stdio。基准测试冷启动和调用延迟，选出本地专用方案优胜者。

## 关键词

| 术语 | 通俗说法 | 实际含义 |
|------|---------|---------|
| 框架（Harness） | “代理循环” | 围绕模型调度工具、维护计划状态和执行预算的代码 |
| 钩子（Hook） | “代理事件监听器” | 用户自定义脚本，在八个生命周期事件之一由框架调用 |
| 工作树（Worktree） | “Git 沙箱” | 在独立路径上的链接 git 检出；可丢弃不影响主仓库 |
| TodoWrite | “计划状态” | 带类型的待办 / 进行中 / 完成条目列表，模型每轮重写 |
| StreamableHTTP | “MCP 传输” | 2026 MCP 版本：长连接双向流式 HTTP，替代 SSE |
| 代币上限（Token ceiling） | “上下文预算” | 每轮或每会话对输入+输出代币的限制，引发压缩或终止 |
| pass@1 | “单次尝试通过率” | SWE-bench 任务首次运行成功的比例，无重试或窥视测试集 |

## 拓展阅读

- [Claude Code 文档](https://docs.anthropic.com/en/docs/claude-code) — Anthropic 参考框架  
- [Cursor 3 更新日志](https://cursor.com/changelog) — Agent Tabs 和 Composer 2 产品笔记  
- [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) — 用于 SWE-bench 框架对比的极简基线  
- [Live-SWE-agent](https://github.com/OpenAutoCoder/live-swe-agent) — 使用 Opus 4.5 在 SWE-bench Verified 得分 79.2%  
- [OpenCode](https://opencode.ai) — 开源框架，11.2万星  
- [SWE-bench Pro 排行榜](https://www.swebench.com) — 本毕业设计的评测目标  
- [模型上下文协议 2026 路线图](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) — StreamableHTTP、能力元数据  
- [OpenTelemetry GenAI 语义规范](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — 工具调用和代币使用的跨度模式
