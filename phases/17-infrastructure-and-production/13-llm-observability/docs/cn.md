# LLM Observability Stack 选择

> 2026 年的可观测性市场分为两类。开发平台（LangSmith、Langfuse、Comet Opik）将监控与评估、提示管理、会话回放捆绑在一起。网关/仪器工具（Helicone、SigNoz、OpenLLMetry、Phoenix）专注于遥测。Langfuse 是 MIT 许可证的核心，拥有强大的开源平衡（免费云服务支持 5 万事件/月）。Phoenix 是基于 Elastic 许可证 2.0 的 OpenTelemetry 原生工具，非常适合漂移和 RAG 可视化，但不是持久的生产后端。Arize AX 采用零拷贝 Iceberg/Parquet 集成，声称比单一的监控架构便宜 100 倍。LangSmith 领先 LangChain/LangGraph，价格为每用户每月 39 美元，自托管仅限企业版。Helicone 基于代理，设置时间为 15-30 分钟，免费 10 万请求/月，但代理跟踪深度较浅。常见生产模式是：网关（Helicone/Portkey）+ 评估平台（Phoenix/TruLens），通过 OpenTelemetry 粘合。

**类型：** 学习  
**语言：** Python（标准库，玩具追踪采样模拟器）  
**先决条件：** 阶段 17 · 08（推理指标）、阶段 14（Agent 工程）  
**时间：** 约 60 分钟  

## 学习目标

- 区分开发平台（捆绑：评估 + 提示 + 会话）和网关/遥测工具（仅跟踪 + 指标）。
- 将六个主要工具（Langfuse、LangSmith、Phoenix、Arize AX、Helicone、Opik）映射到它们的许可、定价和适用场景。
- 解释允许将网关工具与单独评估平台结合的 OpenTelemetry 粘合模式。
- 说出 2026 年成本区分点（Arize AX 的零拷贝方法与单一摄取）并指出约 100 倍的乘数。

## 问题描述

你交付了一个 LLM 功能。功能正常，但你无法查看提示失败、工具循环、延迟回归、成本暴涨或提示缓存命中率。你搜索“LLM 可观测性”，得到八个工具，声称它们解决同一个问题，但价格差异巨大。

它们解决的问题其实不一样。LangSmith 回答“为什么这个 LangGraph 执行失败？”Phoenix 回答“我的 RAG 流水线是否发生了漂移？”Helicone 回答“哪个应用在浪费令牌？”Langfuse 回答“我可以自托管整个系统吗？”不同工具，不同受众。

选择涉及四个维度：技术栈（LangChain？原始 SDK？多供应商？）、许可证容忍度（仅 MIT？Elastic 能行？商业许可？）、预算（免费层级？每月 100 美元？每月 1000 美元？）、和自托管需求（必须？可选？不要？）。

## 概念

### 两类工具

**开发平台** 将可观测性与评估、提示管理、数据集版本控制、会话回放捆绑。你进行实验，查看哪个提示有效，将新提示与旧版本做数据集回归测试。代表有 LangSmith、Langfuse、Comet Opik。

**网关/遥测工具** 对推理调用进行仪器化 — 包括提示、响应、令牌、延迟、模型、成本。代表有 Helicone、SigNoz、OpenLLMetry、Phoenix。简约型。可以通过 OpenTelemetry 与单独的评估工具组合。

### Langfuse — 开源许可平衡

- 核心 Apache / MIT 许可；可通过 Docker 自托管。
- 云免费层级：5 万事件/月。付费：29 美元/月/团队。
- 包括评估、提示管理、跟踪、数据集。对四个开发平台功能覆盖均衡。
- 适用场景：你想要 LangSmith 级别的功能，但必须自托管或坚持 OSS 许可。

### Phoenix（Arize）— 遥测优先，OpenTelemetry 原生

- Elastic 许可证 2.0；自托管非常容易。
- 在 RAG 和漂移可视化方面表现出色。嵌入空间散点图作为一等公民。
- 设计时不作为持久生产后端，主要用于开发时的可观测性。
- 适用场景：RAG 流水线开发、漂移调试，生产环境配合独立网关使用。

### Arize AX — 规模化方案

- 商业产品。通过 Iceberg/Parquet 实现零拷贝数据湖集成。
- 声称比单一监控架构（Datadog 级别）便宜约 100 倍，数学原理是：跟踪数据存储在你自己的 S3 上的 Parquet 文件，Arize 直接读取。
- 适用场景：每天追踪 >1000 万条，已有数据湖，想要不依赖 Datadog 的 LLM 专用仪表盘。

### LangSmith — 首选 LangChain/LangGraph

- 商业产品，39 美元/用户/月。自托管仅限企业版本。
- LangChain 和 LangGraph 技术栈的行业领先产品。如果你没有使用这两者，吸引力有限。
- 适用场景：团队使用 LangChain，且愿意付费。

### Helicone — 基于代理的最小可行方案

- 通过将 `OPENAI_API_BASE` 切换为 Helicone 代理，15-30 分钟即可完成设置。
- MIT 许可；免费 10 万请求/月，付费 20 美元/月起。
- 提供容错、缓存、限速，兼作网关。
- Agent / 多步骤跟踪深度有限。
- 适用场景：快速启动，单一技术栈应用，需要网关和可观测性二合一。

### Opik（Comet）— 开源开发平台

- Apache 2.0，完全开源。
- 功能集类似 Langfuse，继承自 Comet。
- 适用场景：已有 Comet 使用经验的 ML 团队，想要在同一视图中获得 LLM 可观测性。

### SigNoz — OpenTelemetry 优先的全功能 APM

- Apache 2.0。支持一般 APM 并通过 OpenTelemetry 实现 LLM 可观测。
- 适用场景：跨服务和 LLM 调用的统一可观测性。

### 粘合技术：OpenTelemetry + GenAI 语义规范

OpenTelemetry 于 2025 年底发布了 GenAI 语义规范（如 `gen_ai.system`、`gen_ai.request.model`、`gen_ai.usage.input_tokens`）。消费 OpenTelemetry 的工具可以互操作。生产模式如下：

1. 从每个 LLM 调用发出带有 GenAI 规范的 OpenTelemetry 数据。
2. 路由到日常使用的网关（Helicone/Portkey）。
3. 双向发送到评估平台（Phoenix/Langfuse），用于回归。
4. 存档进数据湖（Iceberg），供 Arize AX 或 DuckDB 进行长期分析。

### 陷阱：在错误层级进行仪器化

在你的 agent 框架内部做仪器化（例如添加 LangSmith 跟踪）会绑定到该框架。HTTP/OpenAI-SDK 层的仪器化（通过 OpenLLMetry 或你的网关）更具移植性。

### 采样 — 无法保留所有数据

在请求量超过每日 100 万时，完全保留所有跟踪费用高于 LLM 调用本身。按规则采样：错误 100%，高成本 100%，成功 5%。始终保留聚合数据，保留长尾的原始数据。

### 你需要记住的数字

- Langfuse 免费云服务：每月 5 万事件。
- LangSmith：39 美元/用户/月。
- Helicone 免费：每月 10 万请求。
- Arize AX 声称：大规模时比单一监控架构便宜约 100 倍。
- OpenTelemetry GenAI 规范：2025 年发布，2026 年广泛采用。

## 使用方法

`code/main.py` 模拟了 1 百万跟踪/天的保留策略（100% 摄取、采样、采样 + 错误）。报告存储成本和丢失的数据。

## 交付成果

本课件生成 `outputs/skill-observability-stack.md`，根据技术栈、规模、预算、许可证策略选择工具。

## 练习

1. 你的团队使用 LangChain，想要开源自托管的可观测性。选择 Langfuse 或 Opik 并说明理由。
2. 在每天 500 万条跟踪量且 Datadog 报价 15 万美元/月的情况下，计算 Arize AX 的盈亏平衡点。
3. 设计一个 OpenTelemetry GenAI 属性集，作为你们组织每次 LLM 调用的指南要求。
4. 论证单独使用 Phoenix 是否足够用于生产。什么时候不够？
5. Helicone 有 20 毫秒的代理开销，在 P99 TTFT 为 300 毫秒时是否可以接受？如果 SLA 是 100 毫秒又如何？

## 关键术语

| 术语             | 常见说法                           | 实际含义                                   |
|------------------|---------------------------------|--------------------------------------------|
| OpenLLMetry      | “LLM 的 OTel”                    | 用于 LLM 的开放源代码 OpenTelemetry 仪器  |
| GenAI conventions| “OTel 属性”                     | 面向 LLM 调用的标准 OpenTelemetry 属性名称  |
| LangSmith        | “LangChain 可观测性”               | 与 LangChain 生态捆绑的商业平台               |
| Langfuse         | “开源版 LangSmith”                | 类似功能集的 MIT 许可开源版本                 |
| Phoenix          | “Arize 开发工具”                  | OpenTelemetry 原生的开发/评估平台             |
| Arize AX         | “规模化可观测性”                   | 商业零拷贝 Iceberg/Parquet 可观测性工具       |
| Helicone         | “代理可观测性”                    | 收集 LLM 遥测数据的 HTTP 代理 + 网关功能       |
| Opik             | “Comet LLM”                      | 来自 Comet 的 Apache 2.0 开源开发平台          |
| Session replay   | “跟踪重放”                       | 重放带工具调用的完整 agent 会话                |
| Eval             | “离线测试”                       | 在标注数据集上运行候选模型/提示                 |

## 深入阅读

- [SigNoz — 2026 年主流 LLM 可观测性工具](https://signoz.io/comparisons/llm-observability-tools/)
- [Langfuse — Arize AX 替代方案分析](https://langfuse.com/faq/all/best-phoenix-arize-alternatives)
- [PremAI — 搭建 Langfuse、LangSmith、Helicone、Phoenix](https://blog.premai.io/llm-observability-setting-up-langfuse-langsmith-helicone-phoenix/)
- [OpenTelemetry GenAI 语义规范](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
- [Arize Phoenix 文档](https://docs.arize.com/phoenix)
- [Helicone 文档](https://docs.helicone.ai/)
