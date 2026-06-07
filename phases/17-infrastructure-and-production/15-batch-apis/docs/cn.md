# Batch APIs — 作为行业标准的五折优惠

> 每个主要提供商都发布了异步 batch API，享受 50% 折扣和约 24 小时的处理时间。OpenAI、Anthropic、Google 以及大多数推理平台（Fireworks batch 级别，Together batch）都采用了相同模式。将 batch 与 prompt caching（提示缓存）和夜间流水线结合，成本降至同步非缓存成本的约 10%。原则极其简单：如果不是交互式的，就属于 batch。内容生成流水线、文档分类、数据提取、报告生成、大规模标注、目录标签——任何能容忍 24 小时延迟的任务没有搬到 batch 都是在浪费资金。2026 年的生产模式是将每个新的 LLM 工作负载分类为三类：交互式（同步带缓存）、半交互式（异步队列带降级）、批处理（夜间、缓存输入叠加）。那些假装交互式但能容忍几分钟延迟的工作负载浪费了大部分成本。

**类型：** 学习  
**语言：** Python（标准库，演示 batch 和 sync 成本对比的模拟器）  
**先决条件：** 阶段 17·14（Prompt & Semantic Caching）  
**时间：** 约 45 分钟

## 学习目标

- 了解三大提供商的 batch API（OpenAI、Anthropic、Google），以及通用的 50% 折扣 + 24 小时处理承诺。  
- 计算叠加批处理和缓存输入后夜间分类工作负载的成本，并与同步非缓存基线进行比较。  
- 将工作负载分类为交互式 / 半交互式 / 批处理，并对此分类进行合理化说明。  
- 了解两大陷阱：部分交互性（用户期望快于 24 小时）和输出 Schema 漂移（批处理文件格式因提供商而异）。

## 问题介绍

你的团队交付了一个每晚运行的报告生成流水线。需要处理 50,000 份文档，先摘要每份文档，再对摘要进行聚类，最后起草执行简报。同步运行时耗时 4 小时，成本为 2,000 美元/晚。你听说了 batch API。

batch 让你享受了 50% 折扣。你还启用了系统提示的 prompt caching（所有 50,000 次调用共享）。叠加使用后，账单降至 180 美元/晚——约为基线的 9%。同一条流水线，改动三处配置。

Batch 是 LLM 成本工具中最便宜的杠杆，却鲜有人利用。主要原因是组织上的误解：团队想当然地认为 SLA 是“实时”，而实际 SLA 是“明早之前”。本课的重点是不让账单上剩下 90% 的钱。

## 概念讲解

### 三个 batch API

**OpenAI Batch API**：上传 JSONL 文件，内含请求列表。承诺 24 小时内处理（实际常见约 2-8 小时）。输入输出 token 均享 50% 折扣。访问 `/v1/batches` 端点。缓存 eligible 输入还额外享受缓存输入价格。

**Anthropic Message Batches**：JSONL 上传。24 小时内处理。50% 折扣。支持 `cache_control`，缓存写操作是显式的，读操作在批处理中自动发生。

**Google Vertex AI Batch Prediction**：BigQuery 或 GCS 作为输入。Gemini 同样享受 50% 折扣。整合 Vertex 流水线服务。

### 语义：异步不等于慢

Batch 是“我保证 24 小时内返回”，而不是“这得 24 小时”。典型 P50 是 2-6 小时。提供商会在离峰时段调度批处理作业，充分利用空闲 GPU 资源。

### 与缓存叠加使用

示例：50,000 文档摘要，使用同一 4K token 的系统提示：

- 同步非缓存：50000 ×（$input × 4000 + $output × 200），按全价计费。  
- 同步缓存：系统提示缓存写入后，剩余 49999 次调用输入成本降低 10 倍。  
- 批处理缓存：上述两项外加输入输出均享 50% 折扣。

叠加结果：batch + cache 成本约为同步非缓存的 10%。任何夜间运行且共享系统提示的工作负载都应使用此方案。

### 工作负载分流

**交互式** — 用户等待响应，重视 TTFT（首响应时间）。同步调用，带提示缓存。不可批处理。

**半交互式** — 用户提交任务，若干分钟后回来查看。异步队列，若 batch 不可用可降级为同步。适合中等量级的 RAG 索引。

**批处理** — 用户期望“明早”或“下小时”拿到结果。适用内容流水线、大规模分类、离线分析。必定 batch，必定叠加缓存。

常见错误：鉴于流水线属于生产环境，就将所有任务归为交互式。生产环境不是延迟标准，SLA 才是。

### 部分交互陷阱

有些功能看似交互式，但容忍 5-10 分钟延迟。例：每晚的客户健康报告有刷新按钮。用户点击刷新，等待 10 分钟可以接受。团队却将其做成同步调用。50 个并发刷新时，其成本是批处理+邮件投递的 10 倍。

问题是问：“24 小时对这个用户意味着什么？”如果答案是“用户不会察觉”，那就用 batch。

### 输出 Schema 陷阱

不同提供商的批处理文件格式不同：

- OpenAI：JSONL，每行一个请求。  
- Anthropic：JSONL，每行一个消息，响应格式包含其中。  
- Vertex：BigQuery 表或 GCS 前缀，使用 TFRecord 格式。

要写“一个批处理客户端”覆盖多家提供商，意味着每家写适配器。多提供商批处理网关（Portkey、LiteLLM 部分层级）仍只是在原始格式上做了薄包装。

### 你应该记住的数据

- Batch 折扣：提供商间统一输入输出均 50% 折扣。  
- 处理时间 SLA：保证 24 小时内完成，典型 P50 是 2-6 小时。  
- 叠加批处理+缓存输入：约为同步非缓存成本的 10%。  
- 工作负载分流规则：若能接受 24 小时延迟，一律 batch。

## 使用方法

`code/main.py` 脚本计算了针对 5 万文件工作负载的同步、同步+缓存、批处理及批处理+缓存成本，报告节省金额和百分比。

## 交付成果

本课生成 `outputs/skill-batch-triager.md`，根据工作负载特征进行交互/半交互/批处理分类，并预测节省。

## 练习

1. 运行 `code/main.py`。针对 10 万文档流水线，使用 3K token 系统提示和 500 token 输出，计算全叠加（batch + cache）相较同步基线的节省。  
2. 选择你熟悉的真实产品中的三个功能，分类为交互式/半交互式/批处理。  
3. 用户抱怨报告用了 3 小时。判断这是批处理错误分流还是合理交互？写下决策准则。  
4. 你的 batch API 承诺 24 小时返回，但 P99 是 20 小时。如何向用户传达？针对边缘情况下游系统如何表现？  
5. 计算盈亏平衡点：共享前缀长度达多少时，batch + cache 会比你自有预留 GPU 的夜间运行更便宜？

## 关键术语

| 术语 | 大众说法 | 实际含义 |
|------|----------|----------|
| Batch API | “异步折扣” | 输入输出均 50% 折扣，24 小时返还保证 |
| JSONL | “批处理格式” | 每行一个 JSON 请求；OpenAI/Anthropic 标准 |
| Message Batches | “Anthropic 批处理” | Anthropic 的批处理 API 产品名 |
| Batch prediction | “Vertex 批处理” | Vertex AI 的批处理产品 |
| Turnaround SLA | “24小时承诺” | 保证完成时间，不是典型时长；典型是 2-6 小时 |
| Workload triage | “交互决策” | 交互 / 半交互 / 批处理 路由选择 |
| Output schema | “响应格式” | 各提供商 JSONL 布局；不可移植 |
| Stacked discount | “批处理 + 缓存” | 两者叠加时约为非缓存同步成本的 10% |

## 深入阅读

- [OpenAI Batch API](https://platform.openai.com/docs/guides/batch) — JSONL 格式及 `/v1/batches` 语义。  
- [Anthropic Message Batches](https://docs.anthropic.com/en/docs/build-with-claude/batch-processing) — 批处理格式及 `cache_control` 交互。  
- [Vertex AI Batch Prediction](https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/batch-prediction) — Gemini 批处理语义。  
- [Finout — OpenAI vs Anthropic API Pricing 2026](https://www.finout.io/blog/openai-vs-anthropic-api-pricing-comparison)  
- [Zen Van Riel — LLM API Cost Comparison 2026](https://zenvanriel.com/ai-engineer-blog/llm-api-cost-comparison-2026/)
