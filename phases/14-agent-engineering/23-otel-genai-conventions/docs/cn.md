# OpenTelemetry GenAI 语义规范（Semantic Conventions）

> OpenTelemetry 的 GenAI 专项兴趣小组（SIG，2024年4月启动）定义了代理遥测的标准架构。跨供应商统一的 span 名称、属性和内容捕获规则，使得 Datadog、Grafana、Jaeger 和 Honeycomb 中的代理追踪语义一致。

**类型（Type）：** 学习 + 构建  
**语言（Languages）：** Python（stdlib 标准库）  
**先决条件（Prerequisites）：** 第14阶段·13（LangGraph），第14阶段·24（可观测性平台）  
**时长（Time）：** 约60分钟

## 学习目标

- 命名 GenAI 的 span 类别：model/client（模型/客户端）、agent（代理）、tool（工具）。
- 区分 `invoke_agent` 的 CLIENT 与 INTERNAL spans 及其适用场景。
- 列出顶层 GenAI 属性：provider name（提供者名称）、request model（请求模型）、data-source ID（数据源 ID）。
- 解释内容捕获约定：选择加入（opt-in）、`OTEL_SEMCONV_STABILITY_OPT_IN`、外部引用推荐。

## 问题描述

每个供应商自主命名其 span，导致运维团队不得不为每个框架单独构建仪表盘。OpenTelemetry 的 GenAI SIG 通过定义一个整个生态系统共同遵循的标准来解决此问题。

## 概念

### Span 类别

1. **Model / client spans（模型/客户端 span）。** 涉及原始大语言模型（LLM）调用。由提供者 SDK（如 Anthropic、OpenAI、Bedrock）和框架模型适配器产生。
2. **Agent spans（代理 span）。** 包括 `create_agent`（代理创建时）和 `invoke_agent`（代理运行时）。
3. **Tool spans（工具 span）。** 每次工具调用一个 span，通过父子关系与代理 span 关联。

### 代理 span 命名

- Span 名称：若有命名则为 `invoke_agent {gen_ai.agent.name}`；否则使用 `invoke_agent`。
- Span 类型（span kind）：
  - **CLIENT** — 用于远程代理服务（OpenAI 助手 API、Bedrock Agents）。
  - **INTERNAL** — 用于进程内代理框架（LangChain、CrewAI、本地 ReAct）。

### 关键属性

- `gen_ai.provider.name` — 如 `anthropic`、`openai`、`aws.bedrock`、`google.vertex`。
- `gen_ai.request.model` — 请求的模型 ID。
- `gen_ai.response.model` — 解析后使用的模型（可能因路由而异）。
- `gen_ai.agent.name` — 代理标识符。
- `gen_ai.operation.name` — `chat`、`completion`、`invoke_agent`、`tool_call`。
- `gen_ai.data_source.id` — 用于 RAG（检索增强生成）：查询的语料库或存储标识。

针对 Anthropic、Azure AI Inference、AWS Bedrock、OpenAI 有专门的技术约定。

### 内容捕获

默认规则：instrumentation（插装）*不*应默认捕获输入/输出。捕获须选择加入（opt-in），通过以下属性：

- `gen_ai.system_instructions`
- `gen_ai.input.messages`
- `gen_ai.output.messages`

推荐的生产模式：内容存储在外部（如 S3、日志存储），span 上只记录引用（指针 ID，而非文本内容）。这是课时27防止内容污染的可观测性措施。

### 稳定性

截至2026年3月，大多数规范仍为实验性。使用以下环境变量选择加入稳定预览：

```text
OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental
```

Datadog v1.37+ 原生映射 GenAI 属性到其 LLM 可观测性架构。其他后端（Grafana、Honeycomb、Jaeger）支持原始属性。

### 此模式的常见错误

- **直接在 span 中捕获完整提示。** 导致 PII（个人身份信息）、机密、客户数据暴露给运维。应存外部。
- **缺少 `gen_ai.provider.name`。** 多提供者仪表盘失去归因信息。
- **缺少父链 span。** 工具 span 成为孤儿。务必传递上下文。
- **不设置稳定性 opt-in。** 后端升级时属性可能被重命名。

## 构建示例

`code/main.py` 实现了符合 GenAI 规范的 stdlib span 发送器：

- `Span` 包含 GenAI 属性架构。
- `Tracer` 支持 `start_span`，嵌套上下文。
- 脚本化代理运行，发出：`create_agent`、`invoke_agent`（INTERNAL）、每个工具的 span、用于 LLM 调用的 `chat` span。
- 内容捕获模式，将提示存外部，并在 span 中记录 ID。

运行：

```text
python3 code/main.py
```

输出：带有所有必需 GenAI 属性的 span 树，以及展示选择加入内容引用的“外部存储”。

## 使用方法

- **Datadog LLM 可观测性**（v1.37+）原生映射属性。
- **Langfuse / Phoenix / Opik**（第24课）— 自动插装生态系统。
- **Jaeger / Honeycomb / Grafana Tempo** — 原生 OTel 追踪；基于 GenAI 属性构建仪表盘。
- **自托管** — 运行 OTel Collector 集成 GenAI 处理器。

## 部署

`outputs/skill-otel-genai.md` 将 OTel GenAI span 集成到现有代理，带内容捕获默认和外部引用存储。

## 练习

1. 使用 `invoke_agent`（INTERNAL）和每个工具 span，给第01课的 ReAct 循环插装。发送到 Jaeger 实例。
2. 加入“仅引用”模式的内容捕获：提示存 SQLite，span 属性仅带行 ID。
3. 阅读 `gen_ai.data_source.id` 规格，整合到第09课 Mem0 检索中。
4. 设置 `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental`，验证属性不会被 Collector 重命名。
5. 构建仪表盘：“哪个工具错误与哪些模型相关”，只用 GenAI 属性。

## 关键词

| 术语（Term） | 常说法（What people say） | 实际含义（What it actually means） |
|--------------|---------------------------|----------------------------------|
| GenAI SIG | “OpenTelemetry GenAI 组” | 定义架构的 OTel 工作组 |
| invoke_agent | “代理 span” | 代表代理运行的 span 名称 |
| CLIENT span | “远程调用” | 调用远程代理服务的 span |
| INTERNAL span | “进程内” | 进程内代理运行的 span |
| gen_ai.provider.name | “提供者” | anthropic / openai / aws.bedrock / google.vertex |
| gen_ai.data_source.id | “RAG 源” | 检索命中的语料库/存储 |
| Content capture（内容捕获） | “提示日志” | 选择加入消息捕获；生产存外部 |
| Stability opt-in（稳定性选择加入） | “预览模式” | 环境变量控制实验性规范 |

## 延伸阅读

- [OpenTelemetry GenAI 语义规范](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — 规范文档  
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) — 默认生成 GenAI spans  
- [AutoGen v0.4（微软研究院）](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — 内置 OTel spans  
- [Claude Agent SDK](https://platform.claude.com/docs/en/agent-sdk/overview) — W3C 追踪上下文传播
