# Capstone 11 — 大型语言模型（LLM）可观测性与评估仪表盘

> Langfuse 采用开源核心模式。Arize Phoenix 发布了 2026 年生成式 AI 语义约定映射。Helicone 和 Braintrust 双双加大了按用户成本归因投入。Traceloop 的 OpenLLMetry 成为事实标准的 SDK 自动埋点工具。生产形态是 ClickHouse 用于 traces，Postgres 用于元数据，Next.js 用于 UI，和一大批基于采样 traces 运行的评估作业（DeepEval，RAGAS，LLM-judge）。构建一个自托管系统，至少接入四个 SDK 家族的数据，并展示在五分钟内捕获注入的回归问题。

**类型：** Capstone  
**语言：** TypeScript（UI），Python / TypeScript（采集 + 评估），SQL（ClickHouse）  
**先决条件：** 阶段 11（LLM 工程），阶段 13（工具），阶段 17（基础设施），阶段 18（安全）  
**涉及阶段：** P11 · P13 · P17 · P18  
**时间：** 25 小时

## 问题

每个在 2026 年运行生产流量的 AI 团队都维持一个与模型并行的可观测性平面。成本归因、幻觉检测、数据漂移监控、越狱信号、服务级别目标（SLO）仪表盘、个人识别信息（PII）泄露警报。开源参考项目 —— Langfuse、Phoenix、OpenLLMetry —— 均采用 OpenTelemetry 生成式AI语义约定（GenAI semconv）作为采集模板。你现在可以用同一个 SDK 为 OpenAI、Anthropic、Google、LangChain、LlamaIndex 和 vLLM 生成兼容的 spans 并传输。

你将构建一个自托管的仪表盘，从至少四个 SDK 家族采集数据，针对采样的 traces 运行一小批评估作业，检测漂移并发出警报。考核标准是：给定一个故意注入的回归（例如一个开始生产 PII 的 prompt），仪表盘能在五分钟内捕获并发送告警。

## 概念

采集采用 OTLP HTTP。SDK 生成 GenAI-semconv spans：`gen_ai.system`，`gen_ai.request.model`，`gen_ai.usage.input_tokens`，`gen_ai.response.id`，`llm.prompts`，`llm.completions`。Spans 存入 ClickHouse 以便列式分析；元数据（用户、会话、应用）存储到 Postgres。

评估作为批处理作业运行在采样的 traces 上。DeepEval 评判答案的准确性（faithfulness）、有害性（toxicity）和相关性。RAGAS 在带有检索上下文的 traces 上评判检索指标。自定义 LLM-judge 运行领域特定检查（PII 泄露，违规响应）。评估结果写回同一个 ClickHouse，作为与父 trace 关联的评估 span。

漂移检测监控嵌入空间分布随时间的变化（利用 PSI 或 KL 散度对 prompt 嵌入进行比较）以及评估分数趋势。警报通过 Prometheus Alertmanager 发送到 Slack/ PagerDuty。UI 使用 Next.js 15 搭配 Recharts。

## 架构

```text
生产应用：
  OpenAI SDK  +  Anthropic SDK  +  Google GenAI SDK
  LangChain + LlamaIndex + vLLM
       |
       v
  带 GenAI 语义约定的 OpenTelemetry SDK
       |
       v  OTLP HTTP
  收集器（采集，采样，分发）
       |
       +-------------+-----------+
       v             v           v
  ClickHouse    Postgres    S3 归档
  （spans）      （元数据）  （原始事件）
       |
       +---> 评估作业（DeepEval，RAGAS，LLM-judge）
       |     采样或全量 traces
       |     评估 span 写回
       |
       +---> 漂移检测器（对 prompt 嵌入计算 PSI / KL）
       |
       +---> Prometheus 指标 -> Alertmanager -> Slack / PagerDuty
       |
       v
  Next.js 15 仪表盘（Recharts）
```

## 技术栈

- 采集：OpenTelemetry SDK + GenAI 语义约定；OTLP HTTP 传输  
- 收集器：包含 tail-sampling（尾部采样）处理器的 OpenTelemetry Collector （控制成本）  
- 存储：ClickHouse 存 spans，Postgres 存元数据，S3 存原始事件归档  
- 评估：DeepEval，RAGAS 0.2，Arize Phoenix 评估包，自定义 LLM-judge  
- 漂移：基于 sentence-transformers 池化 prompt 嵌入的 PSI/KL 指标，按周计算  
- 告警：Prometheus Alertmanager -> Slack / PagerDuty  
- UI：Next.js 15 应用路由 + Recharts + server actions  
- 支持 SDK（开箱即用）：OpenAI，Anthropic，Google GenAI，LangChain，LlamaIndex，vLLM

## 构建步骤

1. **收集器配置。** 配置 OpenTelemetry Collector，开启 OTLP HTTP 接收器，tail-sampler 保留 100% 的错误 traces 和 10% 的成功 traces，导出到 ClickHouse 和 S3。

2. **ClickHouse 表结构。** 创建 `spans` 表，列名对应 GenAI semconv：`gen_ai_system`、`gen_ai_request_model`、`input_tokens`、`output_tokens`、`latency_ms`、`prompt_hash`、`trace_id`、`parent_span_id`，及用于存储超长载荷的 JSON 字段。建立按 `user_id` 和 `app_id` 的辅助索引。

3. **SDK 覆盖测试。** 编写小型客户端程序，分别使用 OpenAI、Anthropic、Google、LangChain、LlamaIndex、vLLM 六个 SDK，通过 OpenLLMetry 自动埋点。验证每个 SDK 生成的 GenAI span 能成功保存至 ClickHouse。

4. **评估作业。** 定时作业读取最近 15 分钟采样的 traces，运行 DeepEval 对准确性、毒性和答案相关性评分。评估结果作为子 span 写回到对应的父 trace。

5. **自定义 LLM-judge。** 实现一个 PII 泄露判定器：对回答调用守卫 LLM 给出 PII 泄露的概率评分。高分响应进入复核队列。

6. **漂移检测。** 每周计算本周与前4周基线的 pooled prompt 嵌入间的 PSI，超过阈值触发告警。

7. **仪表盘。** 使用 Next.js 15 构建，页面包括：总体视图（spans/秒、用户成本、95分位延迟）、traces（搜索 + 瀑布视图）、评估（准确性走向、有害性）、漂移（PSI 时间趋势）、告警。

8. **告警链路。** Prometheus 导出和读取评估分数汇总与延迟百分位数据，Alertmanager 分别将告警路由到 Slack（警告）和 PagerDuty（严重告警）。

9. **回归探针。** 注入缺陷：评估的聊天机器人开始以 1% 频率泄露假社保号。测量 MTTR（从 bug 部署到 Slack 警报触发时间）。

## 使用示例

```text
$ curl -X POST https://my-otel-collector/v1/traces -d @trace.json
[collector]  接收 1 个 trace，3 个 span
[clickhouse] 插入 3 个 spans（app=chat, user=u_42）
[eval]       DeepEval 准确率 0.82，有害性 0.03
[drift]      周度 PSI 0.08（阈值 0.2 以下）
[ui]         仪表盘上线 https://obs.example.com
```

## 交付物

`outputs/skill-llm-observability.md` 是交付文件。给定一个 LLM 应用，仪表盘能采集其 traces，运行评估，告警漂移，并在 Next.js 中展示用户成本细分。

| 权重 | 评判标准 | 评估方式 |
|:-:|---|---|
| 25 | Trace schema 覆盖度 | 生产标准 GenAI span 的 SDK 家族数量（目标：6 个及以上） |
| 20 | 评估准确度 | DeepEval / RAGAS 评分与人工标注的对比 |
| 20 | 仪表盘用户体验 | 注入回归的平均修复时间（MTTR，目标 5 分钟内） |
| 20 | 成本与规模 | 持续支持 1000 spans/秒采集且无积压 |
| 15 | 告警与漂移检测 | Prometheus/Alertmanager 全链路演练 |
| **100** | | |

## 练习

1. 为 Haystack 框架添加自定义埋点。验证生成的标准 span 含 `gen_ai.*` 属性且成功写入 ClickHouse。

2. 用 Phoenix 评估器替换 DeepEval，比较两者在相同 trace 上得分的漂移情况。

3. 优化漂移检测：按 app-id 单独计算 PSI，展示各应用的漂移轨迹。

4. 增加“用户影响”页面：展示用户成本和失败率的火花图（sparklines）。

5. 实现一个尾采样策略，保留所有毒性 > 0.5 的 traces 以及其余 traces 的 10% 分层样本。测量样本偏差。

## 关键术语

| 术语 | 俗称 | 实际含义 |
|------|-------|--------------|
| GenAI semconv | “OTel LLM 属性” | 2025 年 OpenTelemetry 对 LLM span 属性（系统、模型、令牌）的规范 |
| 尾采样（Tail sampling） | “后期采样” | 收集器在 trace 完成后决定保留或丢弃，可以查看错误信息 |
| PSI | “人口稳定指数” | 衡量两个分布差异的漂移指标，超过 0.2 通常意味着显著漂移 |
| LLM-judge | “模型评估模型” | 用一个 LLM 对另一 LLM 的输出根据评判标准（准确性、有害性、PII）打分 |
| 尾采样策略 | “保留规则” | 定义哪些 traces 被持久化，哪些被丢弃的规则（包括错误和采样率） |
| 评估 span | “关联评估 trace” | 作为子 span 传递评估分数，链接到原始 LLM 调用的 span |
| 用户成本 | “单元经济” | 在一定时间窗口中按 user_id 计的花费，关键产品指标 |

## 延伸阅读

- [Langfuse](https://github.com/langfuse/langfuse) — 参考开源核心可观测平台  
- [Arize Phoenix](https://github.com/Arize-ai/phoenix) — 具有强漂移支持替代参考  
- [OpenLLMetry (Traceloop)](https://github.com/traceloop/openllmetry) — 自动埋点 SDK 家族  
- [OpenTelemetry GenAI 语义约定](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — 采集数据模式  
- [Helicone](https://www.helicone.ai) — 另一种托管可观测解决方案  
- [Braintrust](https://www.braintrust.dev) — 另一种以评估为核心的平台  
- [ClickHouse 文档](https://clickhouse.com/docs) — 列式 span 存储  
- [DeepEval](https://github.com/confident-ai/deepeval) — 评估库
