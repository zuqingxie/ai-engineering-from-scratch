# 推理平台经济学 — Fireworks, Together, Baseten, Modal, Replicate, Anyscale

> 2026 年的推理市场不再是 GPU 时间租赁。它分化为定制硅（Groq, Cerebras, SambaNova）、GPU 平台（Baseten, Together, Fireworks, Modal）和 API 优先的市场（Replicate, DeepInfra）。Fireworks 于 2026 年 5 月 1 日将 GPU 价格上涨 1 美元/小时，10 万亿以上令牌/天的 40 亿美元估值告诉你，基于流量的模式有效。Baseten 于 2026 年 1 月完成 3 亿美元的 E 轮融资，估值 50 亿美元。竞争定位规则很简单：Fireworks 优化延迟，Together 优化目录广度，Baseten 优化企业打磨，Modal 优化 Python 原生开发体验（DX），Replicate 优化多模态覆盖，Anyscale 优化分布式 Python。本课给你一个可供创始人参考的矩阵。

**类型：** 学习  
**语言：** Python（标准库，简单调用经济对比器）  
**先决条件：** 第17阶段 · 01（托管 LLM 平台），第17阶段 · 04（vLLM 服务内部结构）  
**时长：** ~60 分钟

## 学习目标

- 说出三个市场细分（定制硅，GPU 平台，API 优先）并将每个供应商映射到对应细分。
- 解释为何“按令牌计费”的 API 定价模型趋近于服务引擎的成本曲线，而非硬件成本曲线。
- 计算至少三个供应商的有效请求成本，并解释何时按分钟计费（Baseten、Modal）优于按令牌计费。
- 判断哪个平台适合特定工作负载（无服务器突发、高稳定吞吐量、微调变体、多模态）。

## 问题描述

你评估了托管的超大规模云平台之后，决定需要更专注、更快速的供应商——Fireworks 优先考虑延迟，Together 优先考虑广度，Baseten 优先考虑微调定制模型。现在有六个真实选择，定价页面却完全不匹配。Fireworks 显示按百万令牌计费；Baseten 按分钟计费；Modal 按秒计费；Replicate 按预测计费。若不做工作负载建模，无法直接横向比较。

更糟的是，每个定价页面背后的商业模型不同。Fireworks 运行自有定制引擎（FireAttention）于共享 GPU 上；按令牌费率反映其利用率曲线。Baseten 提供 Truss + 专用 GPU；按分钟计费反映独享性质。Modal 真正实现 Python 级无服务器架构——按秒计费，冷启动时间不到秒。相同的输出（LLM 响应），三种不同的成本函数。

本课将建模这六个平台，并告诉你各自胜出的场景。

## 核心概念

### 三大细分

**定制硅（custom silicon）** — Groq（LPU）、Cerebras（WSE）、SambaNova（RDU）。通常在相同模型上解码速度比基于 GPU 的集群快 5-10 倍。按令牌价格较高（Groq 在 2025 年末 Llama-70B 模型大约 $0.99/百万令牌），但对延迟敏感场景无可匹敌。Groq 是语音代理和实时翻译的生产首选。

**GPU 平台** — Baseten、Together、Fireworks、Modal、Anyscale。运行于 NVIDIA（2026 年的 H100、H200、B200）偶尔 AMD。处于“裸 GPU 租赁”（RunPod、Lambda）和“超大规模云托管服务”（Bedrock）之间的经济层。

**API 优先市场** — Replicate、DeepInfra、OpenRouter、Fal。目录丰富，按预测或按秒计费，强调首次调用时间。

### Fireworks — 延迟优化的 GPU 平台

- 自研 FireAttention 引擎；宣传比 vLLM 在相同配置上延迟低 4 倍。
- 非交互式工作负载采用批量层，费用约为无服务器费率的 50%。
- 微调模型按基础模型费率计费——对其他对 LoRA 额外收费的供应商是明显差异化。
- 2026 年中期：2026 年 5 月 1 日起按需 GPU 租赁提价 $1/小时，量大可议价。
- 财务信号：40 亿美元估值，单日令牌处理超过 10 万亿。

### Together — 广度优化

- 超过 200 款模型，包括开源模型在上线几天内同步发布。
- 相较于 Replicate，在等效 LLM 模型上便宜 50%-70%——“AI 原生云”定位在于流量和目录。
- 在同一个 API 中集成推理、微调和训练。

### Baseten — 企业打磨优化

- Truss 框架：模型打包，包含依赖、密钥和服务配置的清单文件。
- GPU 覆盖从 T4 到 B200，按分钟计费，兼顾合理的冷启动缓解策略。
- SOC 2 类型 II，具备 HIPAA 准备。常见金融科技和医疗健康选型。
- 2026 年 1 月完成 5 亿美元估值的 E 轮融资（3 亿美元来自 CapitalG、IVP、NVIDIA）。

### Modal — Python 原生优化

- 纯 Python 的基础设施即代码。用 `@modal.function(gpu="A100")` 装饰函数，一条命令即可部署。
- 按秒计费，预热后冷启动 2-4 秒，小模型冷启动 <1 秒。
- 2025 年完成 8,700 万美元 B 轮融资，估值 11 亿美元。独立调查中开发者体验最高。

### Replicate — 多模态广度

- 按预测计费。图像、视频和音频模型的默认平台。
- 集成生态（Zapier、Vercel、CMS 插件）。
- 在 LLM 按令牌费率上竞争力稍弱，但多模态多样性获胜。

### Anyscale — Ray 原生

- 基于 Ray；RayTurbo 是 Anyscale 自研推理引擎（与 vLLM 竞争）。
- 最适合推理步骤只是更大图中一个节点的分布式 Python 工作负载。
- 托管 Ray 集群；与 Ray AIR 和 Ray Serve 紧耦合。

### 按令牌计费 VS 按分钟计费 — 何时胜出

当工作负载对延迟不敏感且突发时，按令牌计费更合理——你只为实际使用付费。当利用率高且可预测时，按分钟计费更划算——一旦饱和 GPU，按分钟计费反超按令牌计费。

粗略经验：当专用 GPU 的持续利用率超过约 30% 时，按分钟计费（Baseten、Modal）开始优于按令牌计费（Fireworks、Together）。低于此利用率时，按令牌计费优，因为可以避免闲置付费。

### 定制引擎是真正护城河

以上所有平台在 vLLM 与 SGLang 之上都宣称有自研引擎。FireAttention，RayTurbo，Baseten 的推理栈。定制引擎声称在营销中过于强调——诚实的说法是 vLLM + SGLang 覆盖了约 80% 的生产开源推理，而平台层的差异化在于开发体验（DX）、归因和服务等级协议（SLA）。

### 你应记住的数字

- Fireworks GPU 租赁：2026 年 5 月 1 日起每小时涨价 1 美元。
- Fireworks 声称：延迟比相同配置下的 vLLM 低 4 倍。
- Together：在 LLM 上比 Replicate 便宜 50%-70%。
- Baseten 估值：50 亿美元（2026 年 1 月，E 轮融资 3 亿美元）。
- Modal 估值：11 亿美元（2025 年，B 轮融资）。
- 持续利用率超过 ~30% 时，按分钟计费胜过按令牌计费。

## 使用方法

`code/main.py` 对六个供应商在综合定价模型上的合成工作负载进行对比。输出每日费用和有效每百万令牌费用。运行它，找出按令牌计费和按分钟计费的临界点。

## 交付成果

本课生成 `outputs/skill-inference-platform-picker.md`。根据工作负载特征、SLA 和预算，推荐首选推理平台并命名候补方案。

## 练习

1. 运行 `code/main.py`。在一个 H100 及 70B 模型上，Baseten（按分钟）在多大持续利用率下会超越 Fireworks（按令牌）？自行推导临界点并与经验法则比较。
2. 你的产品提供图像生成、聊天和语音转文本。为每种模态选择平台，并命名统一的网关模式。
3. Fireworks 对你主模型提高了 1 美元/小时价格。假设 40% 流量转入批量层（五折），模拟混合成本影响。
4. 一位受监管客户要求 SOC 2 类型 II + HIPAA + 专用 GPU。哪三个平台可行？哪一个在财务运营（FinOps）上最优？
5. 对 Llama 3.1 70B，在 Fireworks 无服务器、Together 按需、Baseten 专用和 Replicate API 平台上对比每 1,000 次预测的成本？10 次预测/天和 10,000 次预测/天时哪个最便宜？

## 关键术语

| 术语 | 常见表述 | 实际含义 |
|------|----------|----------|
| Custom silicon（定制硅） | “非 GPU 芯片” | Groq LPU、Cerebras WSE、SambaNova RDU——专门优化解码性能 |
| FireAttention | “Fireworks 引擎” | 自研注意力核；宣传比 vLLM 延迟低 4 倍 |
| Truss | “Baseten 格式” | 模型打包清单；依赖+密钥+服务配置 |
| Per-token（按令牌计费） | “API 计费” | 按消耗令牌数收费；避免闲置付费 |
| Per-minute（按分钟计费） | “专用计费” | 按 GPU 时钟时间计费；在高利用率时占优 |
| Per-prediction（按预测计费） | “Replicate 计费” | 按模型调用次数计费；图像/视频常见 |
| RayTurbo | “Anyscale 引擎” | 基于 Ray 的专有推理引擎；与 vLLM 在 Ray 集群竞争 |
| Batch tier（批量层） | “五折优惠” | 非交互式队列，费率降低；Fireworks、OpenAI 常见 |
| Fine-tuned at base rate（微调同基价） | “Fireworks LoRA” | 对 LoRA 服务请求按基础模型费率收费（差异化） |

## 拓展阅读

- [Fireworks 定价](https://fireworks.ai/pricing) — 按令牌费率，批量层，GPU 租赁。
- [Baseten 定价](https://www.baseten.co/pricing/) — 按分钟费率，承诺容量，企业档位。
- [Modal 定价](https://modal.com/pricing) — 按秒计费 GPU 费率及免费档。
- [Together AI 定价](https://www.together.ai/pricing) — 模型目录和按令牌计费。
- [Anyscale 定价](https://www.anyscale.com/pricing) — RayTurbo 与托管 Ray 价格。
- [Northflank — Fireworks AI 替代方案](https://northflank.com/blog/7-best-fireworks-ai-alternatives-for-inference) — 比较评估。
- [Infrabase — 2026 年 AI 推理 API 供应商](https://infrabase.ai/blog/ai-inference-api-providers-compared) — 供应商全景分析。
