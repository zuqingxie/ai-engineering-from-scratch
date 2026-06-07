# LLM Routing Layer — LiteLLM, OpenRouter, Portkey

> 服务商锁定成本高昂。不同的工具调用工作负载适合不同的模型。路由网关提供统一的 API 界面、重试、故障切换、成本追踪和护栏机制。2026 年有三种主流原型：LiteLLM（开源自托管）、OpenRouter（托管 SaaS）、Portkey（生产级，2026 年 3 月开源）。本课介绍决策标准并演示一个标准库路由网关。

**类型：** 学习  
**语言：** Python（标准库，路由 + 故障切换 + 成本追踪）  
**先决条件：** 第 13 阶段 · 02（函数调用），第 13 阶段 · 17（网关）  
**时间：** 约 45 分钟

## 学习目标

- 区分自托管、托管和生产级路由选项。  
- 实现一个在提供商故障时按定义优先级顺序重试的故障切换链。  
- 跟踪跨提供商的每请求成本和令牌使用量。  
- 针对具体生产约束，选择 LiteLLM、OpenRouter 或 Portkey。

## 问题场景

需要提供商路由的场景：

1. **成本。** Claude Sonnet 的费用是 Haiku 的 3 倍。对初筛任务，Haiku 就够用；对综合任务，Sonnet 更有价值。按请求路由。

2. **故障切换。** OpenAI 出现故障，所有请求失败。你希望自动回退到 Anthropic，无需重新部署。

3. **延迟。** 实时聊天 UI 需要快速首令牌响应，批量摘要不需要。按延迟 SLA 路由。

4. **合规。** 欧盟用户必须留在欧盟区域。按区域路由。

5. **实验。** 同一工作负载对比两个模型。按测试组路由。

针对每个集成手写这套逻辑重复繁琐。路由网关提供统一的 OpenAI 兼容 API 并处理其余细节。

## 概念

### OpenAI 兼容代理形式

大家都使用 OpenAI 格式。路由网关暴露 `/v1/chat/completions`，接受 OpenAI 架构请求，内部代理到 Anthropic / Gemini / Cohere / Ollama / 任何后端。客户端无感知。

### 模型别名

不使用 `claude-3-5-sonnet-20251022`，代码中写 `our_smart_model`。网关将别名映射到真实模型。Anthropic 发布 Claude 4 时，在服务器端更新别名，代码无需变动。

### 故障切换链

```text
primary: openai/gpt-4o
on 5xx: anthropic/claude-3-5-sonnet
on 5xx: google/gemini-1.5-pro
on 5xx: refuse
```

网关在配置中定义，重试计入预算，防止故障切换链导致成本暴涨。

### 语义缓存

相同或近似相同提示命中缓存，而非调用提供商。对重复的智能代理循环节省 30% 至 60% 成本。缓存键基于嵌入，近似提示共享缓存槽。

### 护栏机制

网关层级：

- **PII（个人身份信息）脱敏。** 发送提示前通过正则或机器学习进行脱敏。  
- **策略违规。** 拒绝包含禁止内容的提示。  
- **输出过滤。** 清理输出防止泄漏。

Portkey 和 Kong 都内置意见性护栏。LiteLLM 选择性支持。

### 按密钥限流

一个 API key 对应一个团队。按密钥预算防止团队过度消耗共享配额。大多数网关支持此功能。

### 自托管与托管的权衡

| 因素       | LiteLLM（自托管）      | OpenRouter（托管）     | Portkey（生产级）       |
|------------|----------------------|-----------------------|-----------------------|
| 代码       | 开源，Python          | 托管 SaaS             | 开源（2026 年 3 月）+托管   |
| 部署       | 部署代理              | 注册账号               | 二者均可               |
| 提供商数量 | 100+                  | 300+                  | 100+                  |
| 计费       | 用户自有密钥          | OpenRouter 点数       | 用户自有密钥           |
| 可观测性   | OpenTelemetry         | 控制面板               | 完整 OpenTelemetry + PII 脱敏 |
| 适用场景   | 需要完全控制的团队    | 快速原型开发           | 需要合规和护栏的生产环境 |

当你拥有 SRE 团队并希望数据主权时，LiteLLM 是最佳选择。想要单一订阅无基础设施时选 OpenRouter。需要开箱即用的合规与护栏时选 Portkey。

### 成本追踪

每个请求携带 `provider`、`model`、`input_tokens`、`output_tokens`，乘以网关维护的模型每令牌价格表，汇总到用户 / 团队 / 项目。

### MCP 加路由

路由网关可同时路由 LLM 调用和 MCP 采样请求。当采样请求的 modelPreferences 指定特定模型时，网关映射到正确后端。此处第 13 阶段 · 17（MCP 网关）与本课路由网关有时合并成一个服务。

### 路由策略

- **静态优先。** 按列表顺序，错误时回退。  
- **负载均衡。** 轮询或加权。  
- **成本感知。** 选满足延迟/质量要求的最便宜模型。  
- **延迟感知。** 选最近 N 分钟内最快模型。  
- **任务感知。** 提示分类器将编码任务路由一个模型，摘要任务路由另一个。

## 实践示例

`code/main.py` 约 150 行实现了一个路由网关：接受 OpenAI 格式请求，转换成各提供商存根，运行优先级故障切换链，追踪单请求成本，输入前执行 PII 脱敏。用三种场景测试：正常请求、主提供商故障触发回退、PII 泄漏被脱敏拦截。

重点观察：

- `ROUTES` 字典：别名 → 按优先级排列的具体提供商列表。  
- 故障切换循环针对 5xx 错误重试。  
- 成本追踪器通过模型费率计算令牌使用费用。  
- PII 脱敏器转发前清理类似 SSN 的模式。

## 部署指南

本课产生 `outputs/skill-routing-config-designer.md`，根据工作负载配置（延迟、成本、合规），技能自动选择 LiteLLM / OpenRouter / Portkey 并生成路由配置。

## 练习

1. 运行 `code/main.py`。触发故障场景，确认回退到第二提供商且成本正确归因。

2. 添加语义缓存：使用提示的 SHA256 作为查找键，命中缓存即时返回。测量重复调用的成本节省。

3. 添加提示分类器，将以 "code ..." 开头的提示路由到偏重智能的别名，以 "summarize ..." 开头的提示路由到偏重速度的别名。

4. 设计团队预算：每团队设月度消费上限，达到上限网关拒绝请求。选择执行粒度（每请求或窗口计）。

5. 对比阅读 LiteLLM、OpenRouter 和 Portkey 文档，列出每个提供的独有功能。

## 关键词

| 术语           | 常说法            | 实际含义                                        |
|----------------|-------------------|------------------------------------------------|
| Routing gateway | “LLM 代理”        | 多提供商前的统一 API 层                          |
| OpenAI-compatible | “支持 OpenAI 架构” | 接受 `/v1/chat/completions` 格式，转换到任意后端 |
| Model alias    | “our_smart_model” | 代码中名称，网关映射至具体模型                   |
| Fallback chain | “重试列表”        | 失败时尝试的有序提供商列表                        |
| Semantic caching | “提示嵌入缓存”     | 缓存键为提示嵌入，近似重复共享缓存命中           |
| Guardrails    | “输入/输出过滤”    | 脱敏 PII、拒绝违规内容                            |
| Per-key rate limit | “团队预算”        | 针对 API key 的配额                              |
| Cost tracking | “按请求花费”       | 汇总令牌使用 x 模型价格                           |
| LiteLLM       | “开源代理”         | 自托管 OSS 路由网关                              |
| OpenRouter    | “托管 SaaS”       | 托管网关，采用点数计费                            |
| Portkey       | “生产级方案”       | 开源+托管，内置护栏                              |

## 延伸阅读

- [LiteLLM — 文档](https://docs.litellm.ai/) — 自托管路由网关  
- [OpenRouter — 快速开始](https://openrouter.ai/docs/quickstart) — 托管路由 SaaS  
- [Portkey — 文档](https://portkey.ai/docs) — 带护栏的生产路由  
- [TrueFoundry — LiteLLM vs OpenRouter](https://www.truefoundry.com/blog/litellm-vs-openrouter) — 决策指南  
- [Relayplane — 2026 年 LLM 网关对比](https://relayplane.com/blog/llm-gateway-comparison-2026) — 厂商调研
