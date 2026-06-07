# 托管大语言模型平台 — Bedrock、Vertex AI、Azure OpenAI

> 三大超级云服务商，三种截然不同的策略。AWS Bedrock 是一个模型市场 — Claude、Llama、Titan、Stability、Cohere 通过一个 API 接入。Azure OpenAI 是 OpenAI 独家合作加上 Provisioned Throughput Units（PTU，预配置吞吐单元）提供专用容量。Vertex AI 以 Gemini 优先，拥有最佳的长上下文和多模态能力。根据 2026 年 Artificial Analysis 的测评，Azure OpenAI 在 Llama 3.1 405B 等效模型上的中位延迟约为 50 ms，Bedrock 约为 75 ms — PTU 解释了差距，因为专用容量优于共享按需容量。选择规则不是“哪个最快”，而是“哪个模型目录和 FinOps 管理界面与我的产品匹配”。本课帮助你记录权衡点，做出明智选择，而非凭直觉。

**类型：** 学习  
**语言：** Python（标准库，简单成本与延迟比较器）  
**先修：** 第 11 阶段（大语言模型工程）、第 13 阶段（工具与协议）  
**时间：** 约 60 分钟

## 学习目标

- 说出三种平台策略（市场型 vs 独家合作 vs Gemini 优先），并为每种策略匹配相应的产品用例。
- 解释 Azure OpenAI 的 Provisioned Throughput Units（PTU）提供了什么，以及为啥在 405B 规模下 Bedrock 按需通常慢约 25 ms。
- 绘制每个平台的 FinOps 归属面：Bedrock 的应用推理配置文件 vs Vertex 的项目团队划分 vs Azure 的订阅范围加 PTU 预留。
- 写出“至少两个供应商”的政策，并解释为什么单一供应商锁定在 2026 年是昂贵的错误。

## 问题描述

你为产品选了 Claude 3.7 Sonnet，现在需要部署服务。你可以直接调用 Anthropic API，也可以通过 AWS Bedrock，或通过一个统一网关。直接使用 API 最简单；Bedrock 增加了业务协议（BAA）、VPC 终端节点、IAM 和 CloudWatch 归属。网关增加了故障切换、统一计费和跨供应商速率限制。

更深的问题是目录。如果你需要 Claude、Llama 和 Gemini 同时进产品，除非同时使用 Bedrock、Vertex 和 Azure OpenAI，否则无法在一个地方买全。超级云厂商不可互换 — 各自下注了谁主导模型层。

本课描绘三种下注、延迟差距、FinOps 差距和锁定风险。

## 概念介绍

### 三种策略

**AWS Bedrock** — 市场型。涵盖 Claude（Anthropic）、Llama（Meta）、Titan（AWS 自家）、Stability（图像）、Cohere（嵌入向量）、Mistral，另有图像和嵌入子目录。统一 API、统一 IAM 界面、统一 CloudWatch 导出。Bedrock 下注客户重视多样化选择多于单模型。

**Azure OpenAI** — 独家合作。提供 GPT-4 / 4o / 5 / o-series，DALL·E，Whisper 及在 Azure 数据中心内对 OpenAI 模型的微调。Azure OpenAI 服务目录不含非 OpenAI 模型 — 它们归属 Azure AI Foundry（独立产品）。Azure 下注 OpenAI 依然是前沿，客户看重该特定合作的企业级控制。

**Vertex AI** — Gemini 优先，其它其次。Gemini 1.5 / 2.0 / 2.5 Flash 和 Pro，外加 Model Garden（第三方模型）。Vertex 下注多模态长上下文 — 100 万令牌的 Gemini 上下文是差异点。

### 大规模下的延迟差距

Artificial Analysis 持续跑基准测试。在等效 Llama 3.1 405B 部署（共享按需）条件下，Azure OpenAI 的中位首令牌时间（TTFT）约 50 ms；Bedrock 约 75 ms。差距不是 AWS 失败，而是容量模型不同。Azure 提供 PTU（预配置吞吐单元），为租户保留 GPU 容量。Bedrock 也有类似预配置吞吐，但价格约 $21/小时起，大部分客户仍用共享按需。

共享按需容量与所有其他客户流量竞争，专用容量没有。如果产品 SLA 是 P99 首令牌时间 < 100ms，你要么买 Azure PTU，要么买 Bedrock 预配置吞吐，或者接受默认的波动。

### 预配置吞吐经济学

Azure PTU：推理计算的预留块。预测稳定负载时可节约约 70%。成本按小时固定，不论流量多少 — 你为预留付费即使闲置。盈亏平衡点通常在 40-60% 持续利用率。

Bedrock 预配置吞吐：$21-$50/小时，依模型和区域不同。类似计算，盈亏平衡点约为峰值利用率一半。需月度承诺。

Vertex 预配置容量按 Gemini SKU 出售，价格依模型和地域不同，不公开透明。

### FinOps 归属面 — 真实差异

**Bedrock 应用推理配置文件** 是市场上最干净的归属机制。通过给配置文件打标签（`team`、`product`、`feature`），所有模型调用均经过此配置文件，CloudWatch 可按配置文件统计成本，无需后期处理。2025 年新增，至今最细粒度的云厂商原生功能。

**Vertex** 归属为项目-团队方式，且资源处处打标签。每团队建一个 GCP 项目，所有资源标标签，用 BigQuery 计费导出+DataStudio 做汇总。工作量大，但能用任意 SQL 查询成本数据。

**Azure** 依赖订阅/资源组范围和标签，PTU 预留被作为一类成本对象。标签继承自资源组，不是请求本身，所以每请求细粒度归属需用 Application Insights 自定义指标或在网关插入请求头。

模式：Bedrock 原生最干净，Vertex 通过 BigQuery 最灵活，Azure 需额外监控才透明。

### 锁定风险是2026年的隐患

过去单云厂商锁定可行，因为一款模型占主导。2026 年前沿模型每季度变化 — 某季度是 Claude 3.7，下季度是 Gemini 2.5，再下一季度是 GPT-5。绑定一个平台意味着失去三分之二的前沿。

有效团队的做法是：任何重要 LLM 调用都至少用两个供应商。常见组合是 Bedrock 与 Azure OpenAI — Claude 在一边，GPT 在另一边，网关做故障切换。成本增加很小，因网关路由最优；且在故障（如 2025 年 1 月 Azure OpenAI 事件、AWS us-east-1 停机）时可显著提高可用性。

### 数据驻留、BAA 和监管行业

Bedrock：大部分区域有业务协议（BAA）；支持 VPC 端点；有使用护栏。金融科技默认首选。

Azure OpenAI：符合 HIPAA、SOC 2、ISO 27001；欧盟数据驻留；企业监管默认选。

Vertex：符合 HIPAA、GDPR，数据驻留按区域；Google Cloud 合规栈。

三者均满足基础检查项。差异在于数据保留政策、日志处理，以及滥用监控是否读取流量（大多数默认开启，企业可选择关闭）。

### 关键数据记忆点

- Azure OpenAI 在 Llama 3.1 405B 等效的中位 TTFT：约 50 ms（含 PTU）。
- Bedrock 按需中位 TTFT：约 75 ms。
- Bedrock 预配置吞吐：$21-$50/小时每单位。
- Azure PTU 盈亏平衡点：约 40-60% 持续利用率。
- 高利用率时 PTU 节省：最高可达 70%。

## 使用

`code/main.py` 在合成负载下比较三平台 — 模拟按需 vs PTU 经济学、TTFT 波动、成本归属准确度。运行试试看 PTU 何时划算，以及什么时候市场模型丰富性胜过 TTFT 差距。

## 实操

本课制作 `outputs/skill-managed-platform-picker.md`。根据负载特征（所需模型、TTFT SLA、日调用量、合规要求），推荐首选平台、备用平台及 FinOps 监测方案。

## 练习

1. 运行 `code/main.py`。对于 70B 级模型，Azure PTU 在持续利用率多少时优于按需？计算盈亏点，并与官方 40-60% 进行对比。
2. 产品需要 Claude 3.7 Sonnet 和 GPT-4o。设计双供应商部署方案 — 哪个模型对应哪个云厂商，前置什么网关，故障切换策略如何？
3. 一个受监管的医疗客户需 BAAs、美国东部数据驻留和 sub-100ms P99 TTFT。选一个平台并用三个具体特性说明理由。
4. 发现本月 Bedrock 账单上涨了 4 倍但流量没变。没有应用推理配置文件，如何排查原因？有了它，多长时间能定位？
5. 查看 Azure OpenAI 和 Bedrock 定价页。对于 1 亿令牌/月的 Claude 负载，哪种方案更便宜 — 直接用 Anthropic API，Bedrock 按需，还是 Bedrock 预配置吞吐？

## 关键术语

| 术语 | 俗称 | 实际含义 |
|------|---------------|---------------------------|
| Bedrock | “AWS 大语言模型服务” | Claude、Llama、Titan、Mistral、Cohere 的模型市场 |
| Azure OpenAI | “Azure 的 ChatGPT” | Azure 数据中心的独家 OpenAI 模型及企业级控制 |
| Vertex AI | “Google 的大语言模型” | Gemini 优先平台，含第三方模型的 Model Garden |
| PTU | “专用容量” | Provisioned Throughput Unit — 预留推理 GPU，按小时计费 |
| 应用推理配置文件（Application Inference Profile） | “Bedrock 标签” | 基于标签的单产品成本与使用归属，CloudWatch 原生 |
| Model Garden | “Vertex 目录” | Vertex AI 的第三方模型区，独立于 Gemini |
| 双供应商最低要求（Two-provider minimum） | “LLM 冗余” | 关键推理路径至少跨 ≥2 个超级云供应商的策略 |
| BAA | “HIPAA 业务协议” | 业务合作协议，PHI 需签署，各平台均提供 |
| 滥用监控 | “日志审查员” | 提供方对提示/输出的安全扫描，大多数默认开启，企业可关闭 |

## 进一步阅读

- [AWS Bedrock 定价](https://aws.amazon.com/bedrock/pricing/) — 权威价格表与预配置吞吐定价。  
- [Azure OpenAI 服务定价](https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/) — PTU 经济学与价格表。  
- [Vertex AI 生成式 AI 定价](https://cloud.google.com/vertex-ai/generative-ai/pricing) — Gemini 层级及 Model Garden 附加费。  
- [Artificial Analysis 大语言模型排行榜](https://artificialanalysis.ai/) — 各供应商持续延迟与吞吐基准测试。  
- [The AI Journal — 2026 年 AWS Bedrock vs Azure OpenAI CTO 指南](https://theaijournal.co/2026/03/aws-bedrock-vs-azure-openai/) — 企业决策框架。  
- [Finout — Bedrock vs Vertex vs Azure FinOps 对比](https://www.finout.io/blog/bedrock-vs.-vertex-vs.-azure-cognitive-a-finops-comparison-for-ai-spend) — 成本归属细节解析。
