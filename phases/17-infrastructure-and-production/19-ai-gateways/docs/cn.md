# AI 网关 — LiteLLM、Portkey、Kong AI Gateway、Bifrost

> 网关位于您的应用程序和模型提供商之间。核心功能包括提供商路由、回退、重试、速率限制、密钥引用、可观察性、保护措施。2026年市场划分：**LiteLLM** 是 MIT 开源软件，支持100+提供商，兼容 OpenAI，但在约2000 RPS 时性能下降（8 GB 内存，公开基准测试中出现级联失败）；适合 Python、<500 RPS、开发/原型设计。**Portkey** 定位为控制平面（保护措施、PII脱敏、越狱检测、审计轨迹），2026年3月开源 Apache 2.0，延迟开销20-40 ms，生产级别49美元/月。**Kong AI Gateway** 基于 Kong Gateway —— Kong 自有基准测试，12核 CPU 同等条件下，速度是 Portkey 的228%，LiteLLM 的859%；定价100美元/模型/月（Plus 级别最多支持5个模型）；若已使用 Kong，适合企业。**Bifrost**（Maxim AI）— 支持自动重试和可配置的指数退避，在 OpenAI 429 错误时回退至 Anthropic。**Cloudflare / Vercel AI Gateways** — 托管，零运维，基础重试。数据驻留需求是自托管决策的驱动因素；Portkey 和 Kong 兼具开源和可选托管的中间方案。

**类型：** 学习  
**语言：** Python（标准库，简单的网关路由模拟器）  
**先决条件：** 第17阶段 · 01（托管 LLM 平台）、第17阶段 · 16（模型路由）  
**时间：** ~60分钟

## 学习目标

- 枚举六个核心网关功能（路由、回退、重试、速率限制、密钥、可观察性、保护措施）。
- 将四个2026年主流网关（LiteLLM、Portkey、Kong AI、Bifrost）映射到规模上限和使用场景。
- 引用 Kong 基准测试数据（比 Portkey 快228%，比 LiteLLM 快859%）并解释为何对超过500 RPS至关重要。
- 根据数据驻留和运维预算，选择自托管或托管方案。

## 问题描述

您的产品调用 OpenAI、Anthropic 以及自托管的 Llama。每个提供商都有不同的 SDK、错误模型、速率限制和认证方案。您希望实现故障转移（如 OpenAI 返回 429 则尝试 Anthropic）、统一的凭证存储、统一的可观察性和租户级速率限制。

在应用层重新实现此逻辑会导致每个服务都与每个提供商耦合。网关层将其合并为一个进程，提供统一的 API（通常兼容 OpenAI），再分发给各提供商。

## 概念

### 六个核心功能

1. **提供商路由** — 通过一个 API 支持 OpenAI、Anthropic、Gemini、自托管等。
2. **回退** — 在429、5xx或质量故障时，尝试其他提供商。
3. **重试** — 指数回退，有限次数。
4. **速率限制** — 按租户、密钥、模型划分。
5. **密钥引用** — 运行时从凭证库中拉取密钥（决不放入应用中）。
6. **可观察性** — OpenTelemetry + 生成式 AI 属性（第17阶段 · 13）+ 成本归因。
7. **保护措施** — PII 脱敏、越狱检测、允许主题过滤。

### LiteLLM — MIT 开源，Python

- 支持100+提供商，兼容 OpenAI，路由配置、回退、基础可观察性。
- Kong基准测试中约2000 RPS 时性能下降；8 GB 内存占用，持续负载下出现级联故障。
- 最佳适用场景：Python 应用，<500 RPS，开发/预发布环境，实验性路由。
- 成本：零成本开源；云端有免费套餐。

### Portkey — 控制平面定位

- 2026年3月采 Apache 2.0 开源。保护措施、PII 脱敏、越狱检测、审计轨迹。
- 请求延迟开销20-40毫秒。
- 生产级别49美元/月，包含数据保留和 SLA。
- 适合受监管行业，需保护措施+可观察性集成场景。

### Kong AI Gateway — 针对规模应用

- 基于 Kong Gateway（成熟 API 网关产品，lua+OpenResty）。
- Kong 自身基准测试（12核 CPU 等效）：比 Portkey 快228%，比 LiteLLM 快859%。
- 定价：100美元/模型/月，Plus 级别最多支持5个模型。
- 适合已使用 Kong，需 >1000 RPS，愿意付费授权。

### Bifrost（Maxim AI）

- 支持自动重试和可配置退避。
- 在 OpenAI 429 时回退 Anthropic 是典型用法。
- 新兴商业产品。

### Cloudflare AI Gateway / Vercel AI Gateway

- 托管，零运维。基础重试和可观察性。
- 适合 Cloudflare/Vercel 上边缘部署的 JavaScript 应用。
- 保护措施和速率限制相较 Kong/Portkey 有所欠缺。

### 自托管还是托管

数据驻留需求是关键驱动力。医疗和金融默认自托管（LiteLLM、Portkey OSS 或 Kong）。消费类产品默认托管（Cloudflare AI Gateway）或中间层（Portkey托管）。混合用例：对受监管租户自托管，其他使用托管。

### 延迟预算

- LiteLLM：典型开销5-15毫秒。
- Portkey：延迟开销20-40毫秒。
- Kong：延迟开销3-8毫秒。
- Cloudflare/Vercel：延迟开销1-3毫秒（边缘优势）。

网关延迟直接叠加到 TTFT（首次标记时间）。若 TTFT P99 需小于100毫秒（SLA），推荐 Kong 或 Cloudflare；若 P99 < 500 毫秒，则均可。

### 速率限制语义的重要性

简单的令牌桶适用于中小规模。多租户需要滑动窗口 + 突发允许 + 租户分级。LiteLLM 使用令牌桶；Kong 使用滑动窗口；Portkey 提供分级策略。

### 网关 + 可观察性 + 路由的组合

第17阶段 · 13（可观察性）+ 16（模型路由）+ 19（网关）是生产环境中同一层。建议选择覆盖全部功能的工具，或精心整合同步：2026年多数部署结合 Helicone（可观察性）或 Portkey（保护措施）与 Kong（规模）实现角色分离。

### 需要记住的数据

- LiteLLM：约2000 RPS 性能瓶颈，8 GB 内存。
- Portkey：延迟开销20-40 ms，2026年3月起 Apache 2.0 开源。
- Kong：比 Portkey 快 228%，比 LiteLLM 快 859%。
- Kong 定价：100美元/模型/月，Plus 级别最多5个模型。
- Cloudflare/Vercel：边缘延迟开销1-3 ms。

## 使用示例

`code/main.py` 模拟网关路由及三提供商间429/5xx错误注入下的回退逻辑。报告延迟、重试率和回退命中率。

## 交付成果

本课输出 `outputs/skill-gateway-picker.md`。根据规模、运维姿态、合规要求、延迟预算，选择适合的网关。

## 练习

1. 运行 `code/main.py`。配置回退路径 OpenAI→Anthropic→自托管。假设提供商错误率为5%，预计回退命中率是多少？
2. 您的 SLA 要求 TTFT P99 < 200 毫秒，基线延迟300毫秒，哪些网关能满足预算？
3. 医疗客户需要自托管 + PII 脱敏 + 审计，选择 Portkey OSS 还是 Kong？
4. 比较 LiteLLM 与 Kong：团队应在什么 RPS 上限进行迁移？
5. 设计多租户 SaaS 的速率限制策略：免费层、试用层、付费层。采用令牌桶还是滑动窗口？

## 关键术语

| 术语              | 通俗说法       | 实际含义                               |
|-------------------|----------------|---------------------------------------|
| Gateway           | “API 经纪人”  | 位于应用和提供商之间的处理进程       |
| LiteLLM           | “MIT 版”       | Python 开源，支持100+提供商，2K RPS瓶颈 |
| Portkey           | “保护措施网关” | 控制平面 + 可观察性，Apache 2.0       |
| Kong AI Gateway   | “大规模解决方案”| 基于 Kong Gateway，基准性能领先       |
| Bifrost           | “Maxim的网关”  | 重试 + Anthropic 回退方案             |
| Cloudflare AI Gateway | “边缘托管”   | 边缘部署的托管网关，零运维            |
| PII redaction     | “数据清洗”      | 模型请求前使用正则表达式+命名实体识别掩码 |
| Jailbreak detection | “提示注入防护” | 对用户输入分类检测越狱行为            |
| Audit trail       | “合规日志”      | 不可变的每次 LLM 调用记录             |
| Token-bucket      | “简单速率限制”  | 基于令牌补充的速率限制器               |
| Sliding-window    | “精确速率限制”  | 基于时间窗口的速率限制，更公平         |

## 相关阅读

- [Kong AI Gateway 基准测试](https://konghq.com/blog/engineering/ai-gateway-benchmark-kong-ai-gateway-portkey-litellm)
- [TrueFoundry — 2026 年 AI 网关对比指南](https://www.truefoundry.com/blog/a-definitive-guide-to-ai-gateways-in-2026-competitive-landscape-comparison)
- [Techsy — 2026 年顶级 LLM 网关工具](https://techsy.io/en/blog/best-llm-gateway-tools)
- [LiteLLM GitHub](https://github.com/BerriAI/litellm)
- [Portkey GitHub](https://github.com/Portkey-AI/gateway)
- [Kong AI Gateway 文档](https://docs.konghq.com/gateway/latest/ai-gateway/)
