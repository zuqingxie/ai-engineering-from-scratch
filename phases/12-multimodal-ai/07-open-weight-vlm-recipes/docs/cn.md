# 开源权重视觉语言模型（VLM）方案：真正重要的因素

> 2024-2026 年开源权重 VLM 文献是一片消融表的森林。Apple 的 MM1 测试了图像编码器、连接器和数据混合的 13 种组合。Allen AI 的 Molmo 证明详细的人类字幕优于 GPT-4V 蒸馏。Cambrian-1 进行过 20 多种编码器比较。Idefics2 形式化了五轴设计空间。Prismatic VLMs 在受控基准测试中比较了 27 种训练方案。在这些杂音中，一小部分结果在多篇论文中反复验证：图像编码器比连接器架构更重要，数据混合比两者都重要，详细的人类字幕优于蒸馏合成数据。本课解读这些表格，免你亲自翻阅。

**类型：** 学习 + 实验  
**语言：** Python（标准库、消融表解析器 + 方案选择器）  
**前置知识：** 第 12 阶段 · 05（LLaVA 基线）  
**时长：** 约 180 分钟  

## 学习目标

- 了解五轴 VLM 设计空间：图像编码器、连接器、LLM、数据混合、分辨率调度。
- 阅读 MM1 / Idefics2 / Cambrian-1 的消融表并预测调节哪个参数影响基准测试结果。
- 在给定计算预算和任务组合的情况下，为新的 VLM 选择一套方案（编码器、连接器、数据、分辨率）。
- 解释为什么详细的人类字幕在相同令牌数下优于 GPT-4V 蒸馏。

## 问题背景

目前存在数百种开源权重 VLM。大部分“好”与“先进”之间的差距并非架构设计，而是数据、分辨率调度和编码器的选择。知道当模型表现不佳时该首先调节哪一项，可以避免浪费 500 万 GPU 小时的错误。

2023 年浪潮（LLaVA-1.5、InstructBLIP、MiniGPT-4）基于字幕对预训练 + LLaVA-Instruct-150k。是个不错的基线，性能最高在 MMMU 达到约 35%。

2024 年浪潮（MM1、Idefics2、Molmo、Cambrian-1、Prismatic VLMs）则进行了详尽消融，结果既令人惊讶又很实用。

## 核心概念

### 五轴设计空间

Idefics2（Laurençon 等，2024）定义的五个轴：

1. 图像编码器。CLIP ViT-L/14、SigLIP SO400m/14、DINOv2 ViT-g/14、InternViT-6B。编码器差异体现在补丁大小、分辨率和预训练目标。
2. 连接器。MLP（2-4 层）、Q-Former（32 个查询 + 跨注意力）、Perceiver Resampler（64 个查询）、C-Abstractor（卷积 + 双线性池化）。
3. 语言模型。Llama-3 8B / 70B、Mistral 7B、Phi-3、Gemma-2、Qwen2.5。LLM 大小为主要参数成本。
4. 训练数据。字幕对（CC3M、LAION）、交织数据（OBELICS、MMC4）、指令数据（LLaVA-Instruct、ShareGPT4V、PixMo、Cauldron）。
5. 分辨率调度。固定 224/336/448，AnyRes，自然动态。训练中逐步升高或保持不变。

每一个生产级 VLM 都会在每个轴上做出选择。大部分 MMMU 分数的方差可由轴 1、4 和 5 解释——而非所选连接器。

### 轴 1：编码器 > 连接器

MM1 第 3.2 节显示：从 CLIP ViT-L/14 换成 SigLIP SO400m/14，MMMU 增加 3+ 分。连接器从 MLP 换成 Perceiver Resampler 增长不到 1 分。Idefics2 复现了该结论：SigLIP 优于 CLIP，Q-Former、MLP 和 Perceiver 在相同令牌数下表现相近。

Cambrian-1 的《Cambrian Vision Encoders Match-Up》（Tong 等，2024）测试了 20 多种编码器在视觉中心基准上的表现（CV-Bench）。榜单前列多为 DINOv2 和 SigLIP，CLIP 处于中游，ImageBind 和 ViT-MAE 较低。CLIP ViT-L 到 DINOv2 ViT-g/14 的差距约为 5-7 分。

2026 年开源 VLM 默认编码器为 SigLIP 2 SO400m/14，兼顾语义和密集特征，有时会与 DINOv2 ViT-g/14 特征拼接（Cambrian 的“Spatial Vision Aggregator”即如此）。

### 轴 2：连接器设计无显著差异

MM1、Idefics2、Prismatic 和 MM-Interleaved 均得出同一结论：在固定视觉令牌数的条件下，连接器架构差异不大。对补丁进行均值池化后使用 2 层 MLP，其表现与 32 查询的 Q-Former 在令牌预算相同情况下相差不到 1 分。

真正重要的是令牌数。更多视觉令牌 = 更多 LLM 计算 = 性能更好，直到出现收益递减。64 个令牌对于 OCR 较少；576-1024 个令牌是大多数开源 VLM 的最佳范围。2048+ 仅对文档和图表有帮助。

Q-Former 和 MLP 的选择是成本问题而非质量问题：Q-Former 无论图像分辨率如何，令牌数上限在 32-64；MLP 输出所有补丁令牌。高分辨率时 Q-Former 节省 LLM 上下文，对低分辨率影响微乎其微。

### 轴 3：LLM 大小决定上限

将 LLM 从 7B 翻倍到 13B，MMMU 一致增加 2-4 分，出现在每篇 VLM 论文中。达到 70B 后大部分基准趋于饱和。VLM 的多模态推理上限即是 LLM 纯文本推理上限——视觉编码器仅负责输入，无法推理。

这就是为什么 Qwen2.5-VL-72B 和 Claude Opus 4.7 横扫 MMMU-Pro 和 ScreenSpot-Pro：语言模型容量巨大。7B VLM 无法通过巧妙连接器设计替代 70B VLM。

### 轴 4：数据——详细人类字幕胜蒸馏

Molmo + PixMo（Deitke 等，2024）是 2024 年不可不读的结果。Allen AI 让人类标注者用 1-3 分钟密集语音转文字方式描述图像，得到 712K 条密集字幕。训练数据中没有任何 GPT-4V 蒸馏。

Molmo-72B 在 11 个基准中均击败 Llama-3.2-90B-Vision。差距非架构，而是字幕质量。详细人类字幕每张图含信息量是简短网页字幕的 5-10 倍，且能保持事实准确，GPT-4V 蒸馏经常出现幻觉。

ShareGPT4V（Chen 等，2023）和 Cauldron（Idefics2）采用人类 + GPT-4V 混合字幕，趋势明显：2026 年前沿关注字幕内容密度 > 字幕数量 > 蒸馏便利性。

### 轴 5：分辨率及调度

Idefics2 消融显示：384 -> 448 分辨率提升 1-2 分，448 -> 980（采用图像拆分 AnyRes）提升 3-5 分，特别是 OCR 基准。平分辨率训练在中等准确率处趋于饱和；分辨率逐步提升（起点 224，终点 448 或原生）可加速对齐训练并获得更好结果。

Cambrian-1 做过分辨率与令牌数的折中：固定 compute 预算下，低分辨率可使用更多令牌数，高分辨率则令牌数较少。高分辨率更有利 OCR，低分辨率多令牌更适合通用场景理解。

2026 年生产方案：Stage 1 固定 384 分辨率训练，Stage 2 动态分辨率最高至 1280，适合 OCR 任务。

### Prismatic 受控比较

Prismatic VLMs（Karamcheti 等，2024）论文控制所有轴参数，仅变化其中一个轴。相同 13B LLM，相同指令数据和评测。结果：

- 每图视觉令牌数解释约 60% 方差。
- 编码器选择解释约 20%。
- 连接器架构解释约 5%。
- 其余（数据混合、调度器、学习率）约 15%。

该拆分虽粗略，但为“我应先消融哪个参数？”提供了最清晰答案。

### 2026 方案选择器

证据显示，2026 年新项目默认开源 VLM 配置：

- 编码器：SigLIP 2 SO400m/14 自然分辨率，使用 NaFlex；如需分割/定位，拼接 DINOv2 ViT-g/14 密集特征。
- 连接器：补丁令牌上的 2 层 MLP。不限制令牌数则跳过 Q-Former。
- LLM：Qwen2.5 / Llama-3.1 / Gemma 2，根据目标延迟选 7B（成本优）或 70B（质量优）。
- 数据：PixMo + ShareGPT4V + Cauldron，补充任务特定指令数据。
- 分辨率：动态（长边最小 256，最大 1280 像素）。
- 调度：Stage 1 对齐（仅投影层），Stage 2 全参数微调，Stage 3 任务特定微调。

上述每项默认均可追溯至课尾引用论文中的消融实验。

## 使用方法

`code/main.py` 是个消融表解析器和方案选择器，编码 MM1 和 Idefics2 消融表（精简版），支持查询：

- “给定预算 X 和任务 Y，哪个方案胜出？”
- “我把 7B Llama 的 SigLIP 换为 CLIP，预期 MMMU 差异多少？”
- “我应优先消融哪个轴以达到 80% 置信度？”

输出为带有预期基准增减的方案排名及“优先消融”建议。

## 交付物

本课将产出 `outputs/skill-vlm-recipe-picker.md`。给定目标任务组合、计算预算和延迟目标，输出完整方案（编码器、连接器、LLM、数据混合、分辨率调度），并引用支持每个选择的消融研究。避免工程师每次启动新项目时重新造 Idefics2 消融表。

## 练习

1. 阅读 MM1 第 3.2 节。对于预算 5,000 万图像，固定 2B LLM，哪个编码器胜出？13B LLM 情况下结论是否逆转？为什么？

2. Cambrian-1 发现 DINOv2 + SigLIP 拼接在视觉中心基准上优于单独使用，但对 MMMU 无显著提升。预测哪类基准受益，哪类保持平稳。

3. 目标是基于 2B LLM 的移动 UI 代理。选择编码器、连接器、分辨率和数据混合。基于具体消融表论证选择理由。

4. Molmo 推出 4B 与 72B 模型。4B 可竞争闭源 7B VLM；72B 11/11 基准击败 Llama-3.2-90B-Vision。此现象对 LLM 大小平台期假说有何启示？

5. 设计消融表以分离数据质量与编码器质量对 7B VLM 的影响。最少训练轮数？提出四轴设置方案。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|---------|----------|
| 消融（Ablation） | “转动一个旋钮” | 多次训练实验中仅改变设计空间中某一轴，其他轴保持不变 |
| 连接器（Connector） | “桥梁” / “投影器” | 训练模块，将视觉编码器输出映射到 LLM 令牌空间（MLP、Q-Former、Perceiver） |
| 详细人类字幕（Detailed human caption） | “密集字幕” | 多句人工撰写描述（通常 80-300 令牌），信息量丰富，超过网页替代文本 |
| 蒸馏（Distillation） | “GPT-4V 字幕” | 由更强大闭源 VLM 生成的训练数据，易用但可能继承幻觉 |
| AnyRes / 动态分辨率 | “高分辨率路径” | 通过切片或 M-RoPE 支持超出编码器原生分辨率的图像输入策略 |
| 分辨率提升（Resolution ramp） | “课程方案” | 训练过程从低分辨率逐步提升，加速对齐学习 |
| 视觉中心基准（Vision-centric bench） | “CV-Bench / BLINK” | 侧重细粒度视觉感知的评测，而非偏重语言推理 |
| PixMo | “Molmo 的数据” | Allen AI 712K 张密集字幕图像数据集；人类语音转录为密集字幕 |

## 延伸阅读

- [McKinzie 等 — MM1 (arXiv:2403.09611)](https://arxiv.org/abs/2403.09611)
- [Laurençon 等 — Idefics2 / VLM 构建的关键因素 (arXiv:2405.02246)](https://arxiv.org/abs/2405.02246)
- [Deitke 等 — Molmo 和 PixMo (arXiv:2409.17146)](https://arxiv.org/abs/2409.17146)
- [Tong 等 — Cambrian-1 (arXiv:2406.16860)](https://arxiv.org/abs/2406.16860)
- [Karamcheti 等 — Prismatic VLMs (arXiv:2402.07865)](https://arxiv.org/abs/2402.07865)
