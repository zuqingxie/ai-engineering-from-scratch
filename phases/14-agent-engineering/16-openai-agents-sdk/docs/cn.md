# OpenAI Agents SDK：切换、护栏和追踪

> OpenAI Agents SDK 是建立在 Responses API 之上的轻量级多代理框架。包含五个基本原语：Agent（代理）、Handoff（切换）、Guardrail（护栏）、Session（会话）、Tracing（追踪）。切换表现为名为 `transfer_to_<agent>` 的工具。护栏会在输入或输出时触发。追踪默认开启。

**类型：** 学习 + 构建  
**语言：** Python（标准库）  
**先修知识：** 第14阶段 · 01 （Agent 循环），第14阶段 · 06 （工具使用）  
**时间：** 约75分钟

## 学习目标

- 说出 OpenAI Agents SDK 的五个基本原语名称。
- 解释切换：为什么将其建模为工具，模型看到的名称格式，以及上下文是如何传递的。
- 区分输入护栏、输出护栏和工具护栏；解释 `run_in_parallel` 与阻塞模式的区别。
- 实现一个带有切换 + 护栏 + 类span风格追踪的标准库运行时。

## 问题背景

不能干净地进行委派的代理最终会将所有内容塞进一个提示中。没有护栏的代理会泄露个人身份信息（PII）、产生违反政策的输出或无限循环。OpenAI 的 SDK 将使多代理工作变得可管理的三大原语具体化。

## 概念介绍

### 五个基本原语

1. **Agent（代理）。** 大语言模型（LLM）+ 指令 + 工具 + 切换。  
2. **Handoff（切换）。** 委派给另一个代理。表现为模型视图中的一个名为 `transfer_to_<agent_name>` 的工具。  
3. **Guardrail（护栏）。** 针对输入（仅首个代理）、输出（仅最后代理）或工具调用（每个函数工具）的验证。  
4. **Session（会话）。** 跨轮次的自动对话历史存储。  
5. **Tracing（追踪）。** 对 LLM 生成、工具调用、切换、护栏的内置 span。

### 切换作为工具

模型在其工具列表中看到 `transfer_to_billing_agent`。调用它会向运行时发出信号：

1. 复制对话上下文（或通过 `nest_handoff_history` 测试版方法折叠上下文）。  
2. 初始化目标代理及其指令。  
3. 继续由目标代理执行运行。  

这就是第13课／第28课中的负责人模式的产品化实现。

### 护栏

三种类型：

- **输入护栏。** 在首个代理的输入上执行。拒绝不安全或超出范围的请求，防止进行任何 LLM 调用。  
- **输出护栏。** 在最后一个代理的输出上执行。捕获个人身份信息泄露、政策违规、格式错误的响应。  
- **工具护栏。** 针对每个函数工具执行。验证参数、检查权限、审核执行。  

模式：

- **并行**（默认）。护栏 LLM 与主 LLM 并行运行。降低尾延迟。如果触发，主 LLM 的工作被丢弃（浪费 token）。  
- **阻塞**（`run_in_parallel=False`）。护栏 LLM 先运行。如果触发，主调用无 token 浪费。  

触发护栏时会抛出 `InputGuardrailTripwireTriggered` / `OutputGuardrailTripwireTriggered` 异常。

### 追踪

默认开启。每次 LLM 生成、工具调用、切换和护栏都会发出一个 span。设置环境变量 `OPENAI_AGENTS_DISABLE_TRACING=1` 可关闭追踪。`add_trace_processor(processor)` 可将 span 同步到自有后端，除 OpenAI 外并行接收。

### 会话

`Session` 将对话历史存储在后端（SQLite、Redis、自定义）。通过调用 `Runner.run(agent, input, session=session)` 自动加载并追加历史。

### 此模式的陷阱

- **切换漂移。** 代理 A 切换给代理 B，但 B 又切换回 A。需加跳数计数器。  
- **护栏绕过。** 工具护栏仅对函数工具生效；内置工具（文件读取、网页抓取）需要单独策略。  
- **过度追踪。** span 中有敏感内容。可配合 OpenTelemetry GenAI 内容捕捉规则（第23课）—外部存储并通过 ID 引用。

## 构建它

`code/main.py` 实现了标准库中的 SDK 框架：

- `Agent`，`FunctionTool`，`Handoff`（作为带转移语义的函数工具）。  
- `Runner`，支持输入/输出/工具护栏、切换分派和跳数计数。  
- 简单的 span 发射器，展示追踪结构。  
- 一个根据用户查询切换至计费或客服的分流代理；示例中有一个输入触发护栏。  

运行：

```text
python3 code/main.py
```

追踪展示了两个成功切换，一个输入护栏触发和一个与真实 SDK 一致的 span 树。

## 使用它

- **OpenAI Agents SDK** 适合 OpenAI 优先的产品。  
- **Claude Agent SDK**（第17课）适合 Claude 优先的产品。  
- **LangGraph**（第13课）适合需要显式状态和持久恢复的场景。  
- **自定义** 适合需要精确控制（语音、多供应商、联合部署）的场景。

## 交付它

`outputs/skill-agents-sdk-scaffold.md` 搭建一个包含分流代理、切换、输入/输出/工具护栏、会话存储和追踪处理器的 Agents SDK 应用脚手架。

## 练习

1. 添加切换跳数计数器：超过 N 次转接时拒绝。追踪该行为。  
2. 实现 `nest_handoff_history` 选项 — 转接前将之前消息折叠为一条摘要。  
3. 编写阻塞式输出护栏。对触发与未触发的提示比较延迟。  
4. 连接 `add_trace_processor` 到 JSON 日志记录器。每个 span 发出什么样的结构？  
5. 阅读 SDK 文档。将你的标准库玩具版本移植到 `openai-agents-python`。有哪些建模错误？

## 关键词

| 术语              | 常见说法          | 实际含义                        |
|-------------------|-------------------|--------------------------------|
| Agent（代理）      | “大语言模型 + 指令” | SDK 中代理类型，拥有工具和切换   |
| Handoff（切换）    | “转移”            | 模型调用以委派给另一个代理的工具 |
| Guardrail（护栏）  | “政策校验”        | 对输入 / 输出 / 工具调用的验证    |
| Tripwire（触发线） | “护栏触发”        | 护栏拒绝时抛出的异常             |
| Session（会话）    | “历史存储”        | 在运行间持久化的对话记忆         |
| Tracing（追踪）    | “span”            | LLM + 工具 + 切换 + 护栏的内建可观测 |
| Blocking guardrail（阻塞护栏） | “顺序校验”  | 护栏先运行；触发时无 token 浪费  |
| Parallel guardrail（并行护栏） | “并发校验”  | 护栏与主 LLM 并行；触发浪费 token |

## 延伸阅读

- [OpenAI Agents SDK 文档](https://openai.github.io/openai-agents-python/) — 基本原语、切换、护栏、追踪  
- [Claude Agent SDK 概述](https://platform.claude.com/docs/en/agent-sdk/overview) — Claude 风格对应实现  
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — 何时使用切换  
- [OpenTelemetry GenAI 语义规范](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — 标准 Agents SDK 的 span 映射
