# OpenTelemetry GenAI — 端到端追踪工具调用

> 一个 agent 调用五个工具、三个 MCP 服务器和两个子 agent。你需要一个跨越所有调用的追踪（trace）。OpenTelemetry GenAI 语义约定（在 v1.37 及以上版本中为稳定属性）是 2026 年的标准，得到 Datadog、Langfuse、Arize Phoenix、OpenLLMetry 和 AgentOps 的原生支持。本课介绍必需的属性，演示跨度（span）层次结构（agent → LLM → tool），并提供一个标准库跨度发送器（stdlib span emitter），可以插入任何 OTel 导出器。

**类型:** 构建  
**语言:** Python（标准库，OTel 跨度发送器）  
**先决条件:** 第 13 阶段·07（MCP 服务器），第 13 阶段·08（MCP 客户端）  
**时间:** 约 75 分钟

## 学习目标

- 说明 LLM 跨度和工具执行跨度所需的 OTel GenAI 属性。
- 构建覆盖 agent 循环、LLM 调用、工具调用和 MCP 客户端调度的追踪层次结构。
- 判断捕获（选择加入）与脱敏（默认）内容的范围。
- 在不重写工具代码的情况下，将跨度发送到本地收集器（Jaeger、Langfuse）。

## 问题描述

2026 年 2 月的一次调试：用户报告“我的 agent 有时响应需要 30 秒，有时只要 3 秒。”但没有追踪数据。日志显示 LLM 调用，但没有工具调度、MCP 服务器往返，也没有子 agent。你只能猜测。最终发现：一个 MCP 服务器在冷启动时偶尔会卡住。

没有端到端追踪，你无法发现这个问题。OTel GenAI 解决了此难题。

2025-2026 年，OpenTelemetry 语义约定小组确立了这些规范。它们定义了稳定的属性名，使 Datadog、Langfuse、Phoenix、OpenLLMetry 和 AgentOps 都能解析相同的跨度。只需一次检测；即可发送到任何后台系统。

## 概念

### 跨度层次结构

```text
agent.invoke_agent  （顶层，INTERNAL 跨度）
 ├── llm.chat       （CLIENT 跨度）
 ├── tool.execute   （INTERNAL）
 │    └── mcp.call  （CLIENT 跨度）
 ├── llm.chat       （CLIENT 跨度）
 └── subagent.invoke （INTERNAL）
```

整个调用链嵌套在一个追踪 ID 下。跨度 ID 连接父子关系。

### 必需属性

依据 2025-2026 年 semconv：

- `gen_ai.operation.name` — `"chat"`，`"text_completion"`，`"embeddings"`，`"execute_tool"`，`"invoke_agent"`。
- `gen_ai.provider.name` — `"openai"`，`"anthropic"`，`"google"`，`"azure_openai"`。
- `gen_ai.request.model` — 请求的模型字符串（例如 `"gpt-4o-2024-08-06"`）。
- `gen_ai.response.model` — 实际提供的模型。
- `gen_ai.usage.input_tokens` / `gen_ai.usage.output_tokens`。
- `gen_ai.response.id` — 供应商的响应 ID，用于关联。

工具跨度：

- `gen_ai.tool.name` — 工具标识符。
- `gen_ai.tool.call.id` — 特定调用的 ID。
- `gen_ai.tool.description` — 工具描述（可选）。

Agent 跨度：

- `gen_ai.agent.name` / `gen_ai.agent.id` / `gen_ai.agent.description`。

### 跨度类型

- `SpanKind.CLIENT` 用于跨进程边界的调用（LLM 供应商，MCP 服务器）。
- `SpanKind.INTERNAL` 用于 agent 自身循环步骤和工具执行。

### 选择加入的内容捕获

默认情况下，跨度携带指标和时间信息——不包括提示（prompt）或完成结果（completion）。大型负载和个人身份信息（PII）默认关闭。设置 `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental` 并配置特定内容捕获环境变量以包含内容。在生产环境启用前请仔细审查。

### 跨度事件

令牌级事件可以作为跨度事件添加：

- `gen_ai.content.prompt` — 输入消息。
- `gen_ai.content.completion` — 输出消息。
- `gen_ai.content.tool_call` — 记录的工具调用。

事件在跨度内按时间顺序排列，用于详细回放。

### 导出器（Exporters）

OTel 跨度可以导出到：

- **Jaeger / Tempo.** 开源、自托管。
- **Langfuse.** 专注于 LLM 可观测性；可视化令牌使用。
- **Arize Phoenix.** 结合评估与追踪。
- **Datadog.** 商业产品；原生解析 `gen_ai.*` 属性。
- **Honeycomb.** 面向列存，便于查询。

所有导出器使用 OTLP（OpenTelemetry Line Protocol）通信，你的代码无需关心底层细节。

### MCP 之间的传播

当 MCP 客户端调用服务器时，将 W3C 规范的 `traceparent` 头注入请求。流式 HTTP 支持标准头。标准 I/O（stdio）本身不携带 HTTP 头；2026 年的规格路线图讨论为 JSON-RPC 调用添加 `_meta.traceparent` 字段。

在正式发布前：手动在每个请求的 `_meta` 字段中包含 `traceparent`。服务器记录追踪 ID。

### 指标（Metrics）

除了跨度，GenAI semconv 还定义了指标：

- `gen_ai.client.token.usage` — 直方图。
- `gen_ai.client.operation.duration` — 直方图。
- `gen_ai.tool.execution.duration` — 直方图。

这些用于不需要每次调用详细数据的仪表盘。

### AgentOps 层

AgentOps（2024 年成立）专注于 GenAI 可观测性。它封装了流行框架（LangGraph、Pydantic AI、CrewAI），自动发出 OTel 跨度。如果你的技术栈使用支持的框架，非常有用；否则进行手动检测。

## 使用示例

`code/main.py` 向标准输出发出符合 OTel 格式的跨度（OTLP-JSON 样式），模拟一个 agent 调用 LLM、调度两个工具并进行一次 MCP 往返。没有真实导出器；课程重点是跨度结构和属性集。将输出粘贴到兼容 OTLP 的查看器或直接阅读。

注意点：

- 所有跨度共享同一个追踪 ID。
- 父子关系通过 `parentSpanId` 编码。
- 填充了必须的 `gen_ai.*` 属性。
- 内容捕获默认关闭；一个场景通过环境变量开启。

## 部署指南

本课生成 `outputs/skill-otel-genai-instrumentation.md`。给定一个 agent 代码库，技能会生成检测计划：在哪里添加跨度，填写哪些属性，针对哪些导出器。

## 练习

1. 运行 `code/main.py`。统计跨度数量，区分哪个是 CLIENT，哪个是 INTERNAL。

2. 开启内容捕获（环境变量），确认出现 `gen_ai.content.prompt` 和 `gen_ai.content.completion` 事件。注意 PII 的影响。

3. 添加工具执行指标 `gen_ai.tool.execution.duration`，并以直方图样本形式发出。

4. 将父 agent 跨度中的 `traceparent` 传播到 MCP 请求的 `_meta.traceparent` 字段。验证 MCP 服务器能够看到相同的追踪 ID。

5. 阅读 OTel GenAI semconv 规范。找出本课代码未发出的属性，补充添加。

## 关键词汇

| 术语 | 俗称 | 实际含义 |
|------|------|----------|
| OTel | “OpenTelemetry” | 追踪、指标、日志的开放标准 |
| GenAI semconv | “GenAI 语义约定” | LLM、工具、agent 跨度的稳定属性名 |
| `gen_ai.*` | “属性命名空间” | 所有 GenAI 属性共享的前缀 |
| Span | “有时间的操作” | 带起止时间和属性的工作单元 |
| Trace | “跨跨度的祖先关系” | 共享追踪 ID 的跨度树 |
| SpanKind | “CLIENT / SERVER / INTERNAL” | 表示跨度方向的提示 |
| OTLP | “OpenTelemetry Line Protocol” | 导出器的传输格式 |
| Opt-in content | “提示/完成内容捕获” | 默认关闭；通过环境变量开启 |
| traceparent | “W3C 头” | 跨服务传播追踪上下文 |
| Exporter | “后端发送器” | 负责向 Jaeger / Datadog 等发送跨度的组件 |

## 参考资料

- [OpenTelemetry — GenAI semconv](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — GenAI 跨度、指标和事件的权威规范
- [OpenTelemetry — GenAI spans](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/) — LLM 和工具执行跨度属性列表
- [OpenTelemetry — GenAI agent spans](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-agent-spans/) — agent 级别的 `invoke_agent` 跨度
- [open-telemetry/semantic-conventions — GenAI spans](https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/gen-ai-spans.md) — GitHub 上的官方规范源码
- [Datadog — LLM OTel semantic convention](https://www.datadoghq.com/blog/llm-otel-semantic-convention/) — 生产环境集成流程
