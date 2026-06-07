# 交接与例程 — 无状态编排

> OpenAI 的 Swarm（2024 年 10 月）将多代理编排提炼为两个原语：**例程**（指令 + 作为系统提示的工具）和**交接**（返回另一个 Agent 的工具）。没有状态机，也没有分支 DSL —— 大型语言模型（LLM）通过调用正确的交接工具进行路由。OpenAI Agents SDK（2025 年 3 月）是其生产级继任者。Swarm 本身仍是最简洁的概念参考 —— 整个源码仅几百行。该模式具有传播性，因为 API 表面大致是“agent = prompt + tools；handoff = 返回 agent 的函数”。限制：无状态，因此记忆由调用方负责。

**类型：** 学习 + 构建  
**语言：** Python（标准库）  
**先决条件：** 阶段 16 · 04（原始模型）  
**时间：** ~60 分钟

## 问题

每个多代理框架都希望你学习它的 DSL：LangGraph 节点和边，CrewAI 的 crews 和任务，AutoGen GroupChat 和管理者。这些 DSL 是实际的抽象，但会让整体看起来比实际需要的更复杂。

Swarm 则往相反方向发展：利用模型已有的调用工具能力。交接变成工具调用。编排者是当前持有对话的 agent。状态机隐含于 agents 的系统提示中。

## 概念

### 两个原语

**例程。** 定义 agent 角色和可用工具的系统提示。把它当作一组作用域内的指令：“你是一个分诊 agent；如果用户询问退款，交接给退款 agent。”

**交接。** agent 可以调用的一个工具，返回新的 Agent 对象。Swarm 运行时检测到 Agent 返回值后，切换下一轮的活跃 agent。

这就是全部抽象。

```python
def transfer_to_refunds():
    return refund_agent  # Swarm 看到 Agent 返回 → 切换活跃 agent

triage_agent = Agent(
    name="triage",
    instructions="Route the user to the right specialist.",
    functions=[transfer_to_refunds, transfer_to_sales, transfer_to_support],
)
```

分诊 agent 的系统提示使其根据用户消息选择正确的交接。LLM 的调用工具功能完成路由。

### 为什么该模式病毒式传播

- **API 简洁。** 只需学习两个概念。
- **利用模型已具备能力。** 工具调用已是跨供应商的生产级功能。
- **无状态机负担。** 你不必描述图结构；agents 的提示描述交接对象。

### 无状态的折衷

Swarm 明确在运行间无状态。框架在一次运行中保留消息历史，但不持久化任何内容。记忆、连续性、长时任务 —— 都由调用方负责。

在生产环境中（OpenAI Agents SDK，2025 年 3 月），这是主要改变之一：SDK 增加了内置会话管理、保护机制和追踪功能，同时保留了交接原语。

### 适用场景

- **分诊模式。** 前线代理将用户引导给专家。
- **基于技能的交接。** “若任务需编程，调用程序员；需调研，调用调研员。”
- **短期有界对话。** 客服、FAQ 转工单、简单工作流。

### 不适用场景

- **长期会话与共享记忆。** 交接时会将对话状态重置为新 agent 的提示加历史。无调用方管理的持久状态无法跨 agent 保持。
- **并行执行。** 交接一次只切换一个活跃 agent。并行需要调用方协调多个 Swarm 运行。
- **审计和重放。** 无状态运行难以精确重复；LLM 的交接选择非确定性。

### OpenAI Agents SDK（2025 年 3 月）

生产级继任者添加：

- **会话状态。** 跨运行的持久线程。
- **保护机制。** 输入/输出校验钩子。
- **追踪。** 记录每次工具调用和交接。
- **交接过滤。** 控制交接时上下文的转移内容。

交接原语保留；生产环境改进围绕其构建。

### Swarm 与 GroupChat 对比

两者都用 LLM 驱动路由，但其“谁选下一位”不同：

- GroupChat：选择器（函数或 LLM）从外部挑选下一发言者。
- Swarm：当前 agent 通过调用交接工具选择下一位。

Swarm 是“agent 决定下一步”；GroupChat 是“管理者决定下一步”。Swarm 的决策在活跃 agent 的工具调用里；GroupChat 在 `GroupChatManager` 项目中。

## 构建它

`code/main.py` 从头实现了 Swarm：一个 Agent 数据类、交接机制（工具返回 Agent），以及检测代理切换的运行循环。

演示：分诊 agent 路由至退款、销售或支持专家。每个专家有自己的工具。运行循环打印每次交接。

运行：

```text
python3 code/main.py
```

## 使用它

`outputs/skill-handoff-designer.md` 设计给定任务的交接拓扑结构：有哪些 agent，能调用哪些交接，转移哪些上下文。

## 发布它

检查清单：

- **交接日志。** 记录每次交接事件，包括来源 agent、目标 agent、上下文快照。
- **上下文转移规则。** 决定交接时携带的内容：完整历史（代价高）、最近 N 条消息，或者摘要。
- **交接保护。** 交接到具有不同工具权限的专家必须经过认证 —— 否则提示注入可能造成恶意交接。
- **环路检测。** 两个 agent 来回互传是常见失败；用简单的最后 K 次循环检测避免。
- **备用代理。** 交接目标不存在时，退回安全默认。

## 练习

1. 运行 `code/main.py`，实现分诊到退款 agent。确认第二轮活跃 agent 是退款 agent。
2. 增加环路检测规则：如果同两代理连续交接 3 次，强制退出。设计备用方案。
3. 阅读 OpenAI Agents SDK 关于交接过滤的文档。实现一种“交接时摘要”版本：离开 agent 在接手 agent 接管前将上下文压缩为项目符号摘要。
4. 比较 Swarm 交接与 GroupChatManager 选择器。哪种模式更容易发生提示注入，原因何在？
5. 阅读 Swarm 示例手册（https://developers.openai.com/cookbook/examples/orchestrating_agents）。找出 Swarm 做出的一个明确设计决策，且 OpenAI Agents SDK 是如何变更或保留的。

## 关键词

| 术语 | 常说的含义 | 实际含义 |
|------|------------|----------|
| Routine（例程） | “agent 的提示” | 系统提示 + 工具列表。定义角色和可用交接。 |
| Handoff（交接） | “传给另一个 agent” | 活跃 agent 调用的工具，返回新 Agent。运行时切换活跃 agent。 |
| Stateless（无状态） | “运行间无记忆” | Swarm 不持久化内容；记忆由调用方负责。 |
| Active agent（活跃代理） | “当前发言者” | 当前持有对话的 agent。交接会变更它。 |
| Context transfer（上下文转移） | “交接时传递什么” | 新 agent 可见的历史策略：完整、最近 N 条或摘要。 |
| Handoff loop（交接环路） | “代理乒乓互传” | 两个 agent 持续交接导致失败模式。 |
| OpenAI Agents SDK | “生产级 Swarm” | 2025 年 3 月继任者，基于交接原语增加会话、保护、追踪。 |
| Handoff filter（交接过滤） | “转接门控” | SDK 功能，检查并修改交接边界上下文。 |

## 深入阅读

- [OpenAI cookbook — Orchestrating Agents: Routines and Handoffs](https://developers.openai.com/cookbook/examples/orchestrating_agents) — 权威说明  
- [OpenAI Swarm 代码库](https://github.com/openai/swarm) — 原始实现，保留为概念参考  
- [OpenAI Agents SDK 文档](https://openai.github.io/openai-agents-python/) — 带会话和追踪的生产级继任者  
- [Anthropic 在 Claude 中的交接笔记](https://docs.anthropic.com/en/docs/claude-code) — Claude Code 子代理如何通过 `Task` 用类似交接的模式
