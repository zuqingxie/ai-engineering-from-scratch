# Agent Observability：Langfuse、Phoenix、Opik

> 三大开源 agent observability（代理可观测性）平台统治 2026 年市场。Langfuse（MIT 许可）——月安装量超过 600 万，支持 tracing（追踪）+ prompt management（提示管理）+ evals（评估）+ session replay（会话重播）。Arize Phoenix（Elastic 2.0 许可）——深度 agent 专属评估，RAG 相关性检索，OpenInference 自动检测仪。Comet Opik（Apache 2.0 许可）——自动化提示优化，guardrails（护栏），LLM-judge（大模型评判）幻觉检测。

**类型：** 学习  
**语言：** Python（标准库）  
**前置条件：** 第 14 阶段 · 第 23 课（OTel GenAI）  
**时间：** 约 45 分钟

## 学习目标

- 说出三大顶级开源 agent observability 平台及其许可证。
- 区分每个平台的优势：Langfuse（提示管理 + 会话），Phoenix（RAG + 自动检测仪），Opik（优化 + 护栏）。
- 解释为何 89% 的组织预计到 2026 年将具备 agent 可观测性。
- 实现一个基于标准库的 trace-to-dashboard（追踪到仪表盘）管道，含 LLM-judge 评估。

## 问题背景

OTel GenAI（第 23 课）提供了 schema（架构规范）。你仍需一个平台来摄取 span（跨度）、执行评估、存储提示版本并展示回归。三大竞品各自侧重生命周期的不同部分。

## 概念介绍

### Langfuse（MIT 许可）

- SDK 月安装量超 600 万，GitHub 星标 19k+。
- 功能：追踪、带版本控制的提示管理 + playground（实验场）、评估（LLM 作为评判者、用户反馈、自定义）、会话重播。
- 2025 年 6 月：原商业模块（LLM-judge、注释队列、提示实验、Playground）已开源，采用 MIT 许可。
- 最强项：从端到端的可观测性到紧密的提示管理闭环。

### Arize Phoenix（Elastic 2.0 许可）

- 更深层 agent 专属评估：trace clustering（追踪聚类）、异常检测、RAG 检索相关性。
- 原生 OpenInference 自动检测器。
- 搭配付费的 Arize AX 管理产品使用。
- 不支持提示版本控制——定位为配合更大平台进行漂移/行为回归检测的工具。
- 最强项：RAG 相关性、行为漂移、异常检测。

### Comet Opik（Apache 2.0 许可）

- 通过 A/B 测试实现自动提示优化。
- 护栏（PII（个人身份信息）掩码、主题约束）。
- LLM-judge 幻觉检测。
- Comet 自身测评对比：Opik 记录 + 评估用时 23.44 秒 vs Langfuse 327.15 秒（约 14 倍差距）——供应商基准仅供参考。
- 最强项：优化闭环、自动实验、护栏执行。

### 行业数据

据 Maxim（2026 年行业分析）：89% 组织已部署 agent observability；质量问题是最主要的生产障碍（32% 受访者指出）。

### 如何选择

| 需求 | 推荐平台 |
|------|----------|
| 全功能提示管理 | Langfuse |
| 深度 RAG 评估 + 漂移 | Phoenix |
| 自动优化 + 护栏 | Opik |
| 开源许可证，无 ELv2 限制 | Langfuse（MIT）或 Opik（Apache 2.0） |
| Datadog / New Relic 集成 | 任意——三者均支持导出 OTel |

### 该模式的局限

- **无评估策略。** 仅有追踪而无评估只是昂贵的日志记录。
- **自主构建的 LLM-judge 缺乏验证依据。** 应用 CRITIC 模式（第 05 课）——评判需要外部工具进行事实验证。
- **提示版本未与追踪绑定。** 生产回归时无法精确定位哪个提示导致问题。

## 实现示例

`code/main.py` 实现了一个基于标准库的追踪收集器 + LLM-judge 评估器：

- 摄取 GenAI 格式的 span。
- 按会话分组，标记失败运行（触发护栏、低置信度评估）。
- 一个基于脚本的 LLM-judge，根据评分标准打分代理响应。
- 类仪表盘汇总：失败率、主要失败原因、评估分布。

运行：

```text
python3 code/main.py
```

输出：每会话的评估分数和失败分类，模拟 Langfuse/Phoenix/Opik 展示内容。

## 使用方法

- **Langfuse**：自托管或云服务；通过 OTel 或其 SDK 进行接入。
- **Arize Phoenix**：自托管；自动检测 OpenInference。
- **Comet Opik**：自托管或云服务；自动化优化闭环。
- **Datadog LLM Observability**：适合已运行 Datadog 的运维与 ML 混合团队。

## 部署指引

`outputs/skill-obs-platform-wiring.md` 选择平台并将追踪、评估、提示版本接入现有 agent。

## 练习任务

1. 导出一周的 OTel 追踪到 Langfuse 云端（免费层）。哪些会话失败？原因是什么？
2. 为你的领域编写 LLM-judge 评分标准（事实准确性、语气、范围遵守）。在 50 条追踪中测试。
3. 对比 Langfuse 提示版本管理与 Phoenix 追踪聚类。哪个能更快告诉你故障原因？
4. 阅读 Opik 护栏文档。为一个 agent 运行接入 PII 掩码护栏。
5. 在你的语料上对三者进行基准测试。忽略供应商公布数据，自行测量。

## 关键词定义

| 术语 | 通常说法 | 实际含义 |
|------|----------|----------|
| Tracing（追踪） | “Spans collector（跨度收集器）” | 摄取 OTel / SDK 的跨度；按会话索引 |
| Prompt management（提示管理） | “Prompt CMS（提示内容管理系统）” | 版本化提示，与追踪绑定 |
| LLM-as-judge（大模型评判） | “Automated eval（自动评估）” | 独立大模型根据评分标准评估输出 |
| Session replay（会话重播） | “Trace playback（追踪回放）” | 逐步回放历史运行进行调试 |
| RAG relevancy（RAG 相关性） | “Retrieval quality（检索质量）” | 检索到的上下文是否匹配查询 |
| Trace clustering（追踪聚类） | “Behavioral grouping（行为分组）” | 聚类相似运行，用于漂移检测 |
| Guardrail enforcement（护栏执行） | “Policy at log time（日记时策略）” | 对日志内容做 PII/毒性/范围检查 |

## 延伸阅读

- [Langfuse 文档](https://langfuse.com/) — 追踪、评估、提示管理  
- [Arize Phoenix 文档](https://docs.arize.com/phoenix) — 自动检测、漂移分析  
- [Comet Opik](https://www.comet.com/site/products/opik/) — 优化 + 护栏  
- [OpenTelemetry GenAI 语义规范](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — 三者共用的架构规范
