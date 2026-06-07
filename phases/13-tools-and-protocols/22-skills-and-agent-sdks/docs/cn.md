# Skills 和 Agent SDKs — Anthropic Skills、AGENTS.md、OpenAI Apps SDK

> MCP 表示“有哪些工具”。Skills 表示“如何完成一项任务”。2026 年栈将两者结合。Anthropic 的 Agent Skills（开放标准，2025 年 12 月）以 SKILL.md 形式发布，支持渐进式展示。OpenAI 的 Apps SDK 是 MCP 加上 widget 元数据。AGENTS.md（现有 60,000+ 仓库采用）位于仓库根目录，作为项目级别的 agent 上下文。本课介绍各自覆盖内容，并构建一个可在不同 agent 之间流动的最小 SKILL.md + AGENTS.md 打包包。

**类型：** 学习  
**语言：** Python（标准库、SKILL.md 解析器和加载器）  
**先决条件：** Phase 13 · 07（MCP 服务器）  
**时长：** 约 45 分钟

## 学习目标

- 区分三层结构：AGENTS.md（项目上下文）、SKILL.md（可复用知识）、MCP（工具）。
- 编写带 YAML frontmatter 和渐进式展示的 SKILL.md。
- 在 agent 运行时以文件系统方式加载 skills。
- 结合 MCP 服务器和 AGENTS.md 组成一个技能包，使其在 Claude Code、Cursor 和 Codex 中都能工作。

## 问题背景

一位工程师将发布说明编写工作流提炼成多步提示：“读取最新合并的 PR，按领域分组，分别总结，按照团队风格写 changelog 条目，发布 Slack 草稿”。他们把流程写进 Notion 文档供团队使用。

现在他们希望流程能在 Claude Code、Cursor 和 Codex CLI 中复用。每个 agent 加载指令方式不同：Claude Code 使用斜杠命令，Cursor 用规则，Codex 用 `.codex.md`。结果工程师复制了三份流程且维护三份内容。

AGENTS.md 和 SKILL.md 结合解决此问题：

- **AGENTS.md** 位于仓库根目录，每个兼容 agent 在会话开始时读取。回答“该项目如何运作？有哪些约定？哪些命令跑测试？”
- **SKILL.md** 是可携带包：YAML frontmatter（名称、描述）+ markdown 内容 + 可选资源。支持技能的 agent 按需根据名称加载。
- **MCP**（Phase 13 · 06-14）管理技能所需调用的工具。

三层，一份可携带的工件。

## 概念介绍

### AGENTS.md （agents.md）

2025 年底推出，截至 2026 年 4 月被 60,000+ 仓库采纳。一个文件放仓库根目录。格式示例：

```markdown
# Project: my-service

## Conventions
- TypeScript 严格模式。
- Python 端使用 Pydantic 定义模型。
- 使用 `pnpm test` 运行测试。

## Build and run
- 本地开发服务器用 `pnpm dev`。
- 生产环境构建用 `pnpm build`。
```

Agents 会在会话开始读取它，以调整该项目的行为配置。2026 年所有编码 agent 都支持 AGENTS.md：Claude Code、Cursor、Codex、Copilot Workspace、opencode、Windsurf、Zed。

### SKILL.md 格式

Anthropic 的 Agent Skills（2025 年 12 月作为开放标准发布）：

```markdown
---
name: release-notes-writer
description: 按本项目风格为最新合并的 PR 撰写发布说明条目。
---

# Release notes writer

调用时执行以下步骤：

1. 列出自上次标签以来合并的 PR，使用 `gh pr list --base main --state merged`。
2. 按标签分组：feature、fix、chore、docs。
3. 针对每组中的每个 PR，写一行：`- <title> (#<num>)`。
4. 草拟发布说明，暂存至 CHANGELOG.md。

若用户说 “ship”，执行 `git tag vX.Y.Z` 和 `gh release create`。

## Notes

- 不包含没有 PR 的提交。
- 公开发布说明中跳过 “chore” 条目。
```

Frontmatter 声明技能身份。正文是技能加载时展现给模型的提示。

### 渐进式展示

技能可以引用子资源，agent 仅在需要时拉取。示例：

```text
skills/
  release-notes-writer/
    SKILL.md
    style-guide.md
    template.md
    scripts/
      generate.sh
```

SKILL.md 里写“详见 style-guide.md 风格指南”。agent 仅在该技能运行时拉取 style-guide.md，避免提示装载太多模型可能不需要的细节。

### 文件系统发现

Agent 运行时扫描已知目录的 SKILL.md 文件：

- `~/.anthropic/skills/*/SKILL.md`
- 项目 `./skills/*/SKILL.md`
- `~/.claude/skills/*/SKILL.md`

加载以文件夹名和 frontmatter 中的 `name` 为键。Claude Code、Anthropic Claude Agent SDK 和跨 agent 的 SkillKit 都遵循此规范。

### Anthropic Claude Agent SDK

`@anthropic-ai/claude-agent-sdk`（TypeScript）和 `claude-agent-sdk`（Python）会话开始时加载技能，提供可调用的“agent”接口。agent 循环在用户调用时转派到对应技能。

### OpenAI Apps SDK

2025 年 10 月推出，基于 MCP 构建。统一了 OpenAI 之前的 Connectors 和 Custom GPT Actions。Apps SDK 应用是：

- MCP 服务器（工具、资源、提示）。
- 额外提供 ChatGPT UI 的 widget 元数据。
- 可选 MCP Apps 的 `ui://` 资源用于交互界面。

协议统一，体验升级。

### 跨 Agent 可移植性通过 SkillKit

SkillKit 等工具及类似跨 agent 分发层将单一 SKILL.md 转换至 32+ 种 AI agent 的原生格式（Claude Code、Cursor、Codex、Gemini CLI、OpenCode 等）。单一真源，多客户端共用。

### 三层栈结构

| 层级      | 文件          | 载入时机    | 目的               |
|-----------|---------------|-------------|--------------------|
| AGENTS.md | 仓库根目录    | 会话开始    | 项目级约定         |
| SKILL.md  | skills 目录   | 技能调用    | 可复用工作流       |
| MCP 服务器| 外部进程      | 需要工具时  | 可调用动作工具     |

三者协作：agent 会在会话开始读取 AGENTS.md，用户调用技能，技能指令包含 MCP 工具调用，agent 通过 MCP 客户端转派执行。

## 使用方法

`code/main.py` 提供了一个标准库级 SKILL.md 解析器和加载器。它会在 `./skills/` 目录下发现技能，解析 YAML frontmatter 和 markdown 正文，生成按技能名索引的 dict。随后模拟 agent 循环，按名称调用 `release-notes-writer`。

重点关注：

- 使用最简标准库解析 YAML frontmatter（无 pyyaml 依赖）。
- 技能正文原文保存，调用时 prepended 到系统提示中。
- 通过 `read_subresource` 函数演示渐进式展示，仅需时拉取引用的文件。

## 发布

本课输出 `outputs/skill-agent-bundle.md`。给定工作流，生成合并的 SKILL.md + AGENTS.md + MCP-服务器蓝图打包，可跨 agent 便携使用。

## 练习

1. 运行 `code/main.py`。在 `skills/` 下新增第二个技能，确认加载器能识别。

2. 为本课程仓库编写一个 AGENTS.md，包含测试命令、风格约定以及 Phase 13 思维模型。

3. 将团队内部文档中的多步骤工作流迁移为 SKILL.md，验证能在 Claude Code 中加载。

4. 手工将该技能翻译成 Cursor 和 Codex 的原生规则格式，比较两种格式差异——这正是 SkillKit 自动化处理的转换层。

5. 阅读 Anthropic Agent Skills 博文，指出 Claude Agent SDK 中本课加载器未覆盖的一个功能。（提示：agent 子调用）

## 关键术语

| 术语            | 常用表述               | 实际含义                                    |
|-----------------|------------------------|---------------------------------------------|
| SKILL.md        | “技能文件”             | 包含 YAML frontmatter 和 markdown 正文，由 agent 运行时加载         |
| AGENTS.md       | “仓库根 agent 上下文”  | 项目级约定文件，会话开始被读取                             |
| 渐进式展示      | “懒加载子资源”         | 技能正文引用的文件仅在需要时拉取                            |
| Frontmatter     | “顶部 YAML 块”         | 元数据，格式为 `---` 包裹的 name、description 等信息        |
| Claude Agent SDK| “Anthropic 的技能运行时”| `@anthropic-ai/claude-agent-sdk`，加载技能并路由调用           |
| OpenAI Apps SDK | “MCP + widget 元”      | 基于 MCP、带 ChatGPT UI 钩子的 OpenAI 开发平台                |
| 技能发现        | “文件系统扫描”         | 扫描已知目录寻找 SKILL.md，按名称索引                         |
| 跨 agent 可移植 | “一技能多 agent”       | 通过 SkillKit 等工具将单一 SKILL.md 转换给 32+ 种 agent          |
| Agent Skill     | “可携带知识”           | MCP 工具定义以外的、可复用任务模板                              |
| Apps SDK        | “MCP 加 ChatGPT UI”    | 连接器和定制 GPT 统一在 MCP 上                                   |

## 延伸阅读

- [Anthropic — Agent Skills 发布公告](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills) — 2025 年 12 月发布  
- [Anthropic — Agent Skills 文档](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview) — SKILL.md 格式参考  
- [OpenAI — Apps SDK](https://developers.openai.com/apps-sdk) — 基于 MCP 的 ChatGPT 开发平台  
- [agents.md](https://agents.md/) — AGENTS.md 格式与采纳列表  
- [Anthropic — anthropics/skills GitHub](https://github.com/anthropics/skills) — 官方技能示例
