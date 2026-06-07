# Emu3：图像与视频生成的下一token预测（Next-Token Prediction）

> BAAI 的 Emu3（Wang 等，2024年9月）是2024年的成果，本应终结扩散（diffusion）与自回归（autoregressive）之争。仅用一个 Llama 风格的 Decoder-only Transformer，训练目标仅为下一token预测，统一文本 + VQ 图像 token + 3D VQ 视频 token 的词汇表，就在图像生成上超过了 SDXL，在感知（perception）上击败了 LLaVA-1.6。无 CLIP 损失，无扩散调度。推理时使用无分类器引导（classifier-free guidance）以提升质量，但核心训练目标依然是带教师强制（teacher forcing）的下一token预测。论文发表于 Nature。本课学习 Emu3 论文 —— 为什么更好的分词器（tokenizer）加上规模就是全部 —— 并对比扩散法。

**类型:** 学习  
**语言:** Python（标准库，3D 视频分词器数学 + 自回归采样器骨架）  
**先决条件:** 第12.11节（Chameleon）  
**时间:** 约120分钟

## 学习目标

- 解释为何 Emu3 的单一损失下一token目标，能突破长期认为扩散是图像质量必需的假设。
- 描述3D视频分词器：时空 VQ 码本（codebook）长什么样，为什么patch跨越时间维度。
- 对比 Emu3 与 Stable Diffusion XL 在训练计算量、推理成本、质量上线。
- 说明同一 Emu3 模型承担的三种角色：Emu3-Gen（图像生成）、Emu3-Chat（感知）、Emu3-Stage2（视频生成）。

## 问题背景

2024年前的常识：图像生成需要扩散。论点是：离散图像token丢失太多细节信息，自回归采样在数千token级别累计误差。Stable Diffusion、DALL-E 3、Imagen、Midjourney 都采用某种形式的扩散。Chameleon（第12.11课）在小规模下部分推翻这一点，但未能匹配 SDXL 的质量。

Emu3 正面挑战这一论点。主张是：更好的视觉分词器 + 足够规模 + 下一token损失 = 在同一个模型内超越扩散的图像生成和感知能力。

这一押注在发表时颇具争议。两年后，开源的统一生成系列（Emu3、Show-o、Janus-Pro、Transfusion）成为研究的默认路径；生产端的前沿模型也似乎采用某种变体。

## 核心概念

### Emu3 分词器

关键是视觉分词器。Emu3 训练了自定义的 IBQ 类分词器（Inverse Bottleneck Quantizer，SBER-MoVQGAN 系列），每个token实现8x8分辨率缩减。512x512图像转为64x64=4096个token，码本大小32768。

这比 Chameleon 在512x512时的1024 token（码本K=8192）多，但每个token成本更低（码本查找更小，编解码器更简单）。关键指标是重构峰值信噪比（PSNR）达到30.5 dB，与 Stable Diffusion 连续潜空间的32 dB 竞争力相当。

视频方面：3D VQ 分词器将时空patch（4x4x4像素）编码为一个整数。以4秒8FPS视频片段为例，有32帧；256x256分辨率下，空间缩减4倍、时间缩减4倍，token数为 (256/4) * (256/4) * (32/4) = 64 * 64 * 8 = 32768个token。

分词器质量是上限。Emu3的贡献部分在于“我们训练了一个非常好的分词器”。

### 单一损失训练

Emu3 只使用一个目标：跨文本token、2D图像token和3D视频token的共享词汇表的下一token预测。训练时，权重乘以各模态特定的因子以平衡贡献，但损失函数完全相同。

训练数据混合如下：
- 图像生成：`<text caption> <image> image_tokens </image>`
- 图像感知：`<image> image_tokens </image> <question> text_tokens`
- 视频生成：`<text caption> <video> video_tokens </video>`
- 视频感知：类似
- 纯文本：标准下一token预测

模型学习何时输出图像token或文本token，基于数据分布。生成过程由模型在 `<image>` 标签后预测图像token触发。

### 无分类器引导和温度调节

自回归图像生成在推理时用无分类器引导（CFG）显著提升质量。Emu3 采用该策略：生成两次，一次用完整标题，一次用空标题，利用引导权重（通常3.0-7.0）混合两者logits。这是扩散中同样的CFG技巧，被借用到自回归场景。

温度设置也关键：过高易出现伪影，过低导致模态崩溃。Emu3 推荐感知用温度1.0，图像生成用0.8。

### 三种角色，一模一样权重

Emu3 以三个功能不同的API发布，但基于同一个模型权重集：

- Emu3-Gen：图像生成，输入文本，输出图像token。
- Emu3-Chat：视觉问答（VQA）和描述，输入图像token，输出文本。
- Emu3-Stage2：视频生成和视频VQA，输入文本或视频，输出文本或视频。

无任务专属头，仅不同的提示模板，相同检查点。

### 基准测试

依据 Emu3 论文（2024年9月）：

- 图像生成：在 MJHQ-30K FID 指标上胜过 SDXL（5.4 vs 5.6），GenEval 总分接近（0.54 vs 0.55——统计学持平），Deep-Eval 综合指标持平。
- 图像感知：优于 LLaVA-1.6 在 VQAv2（75.1 vs 72.4），在 MMMU 上大致相当。
- 视频生成：4秒片段质量与 Sora时代公开基准模型 FVD 竞争。

数据并非全赢，Emu3 在部分指标上得一分、舍一分，但“仅靠下一token预测即可”的观点跨模态是有理据的。

### 计算成本

Emu3 使用7B参数模型，训练大约3000亿多模态token。GPU小时时间与 Llama-2-7B 预训练相当（2000-4000 GPU年A100级别）。扩散模型如 Stable Diffusion 3 预算类似，但需要额外文本编码器和更复杂流水线。

推理时，Emu3比SDXL每图像慢：4096个token以30 token/s速率约需2分钟生成512x512图像，SDXL只需2-5秒。推测解码和KV缓存优化缩小差距但未完全消除。自回归图像生成计算开销大，这是固有权衡。

### 重要意义

Emu3 的深远贡献在于理念。如果下一token预测扩展能匹配扩散的图像生成，统一模型路径（单一损失，单一骨干，任意模态）是可行的。未来模型不必单独文本编码器、扩散调度器和VAE。只需一个Transformer，每模态一个分词器，再加规模。

Show-o、Janus-Pro、InternVL-U等都基于或挑战该结论。中国研究机构（BAAI、DeepSeek）在2025年前在该方向发表更积极，较美国机构领先。

## 使用方法

`code/main.py` 实现两个小工具：

- 2D vs 3D VQ 分词器计数器：给定（分辨率，patch大小，剪辑时长，FPS），计算图像与视频token数。
- 自回归图像token采样器，带无分类器引导和温度调节。

CFG实现符合Emu3配方 —— 按引导权重混合有条件和无条件logits。

## 部署

本课生成 `outputs/skill-token-gen-cost-analyzer.md` 。给定生成产品规格（图像或视频，目标分辨率，质量等级，延迟预算），计算token数、推理成本，并选择Emu3系列还是扩散。

## 练习

1. Emu3于512x512图像产生4096个token，8x8缩减。计算1024x1024和2048x2048对应的token数，推理延迟会发生什么？

2. 阅读Emu3第3.3节视频分词器。描述3D VQ patch形状及为何是4x4x4而非8x8x1。

3. 无分类器引导权重5.0与3.0：视觉效果如何？追踪`code/main.py`中的计算过程。

4. 计算Emu3-7B在3000亿token训练的FLOPs，并与Stable Diffusion 3比较，哪一个更昂贵？

5. Emu3在FID上胜过SDXL，但在VQAv2上不敌专门化视觉语言模型（VLMs）。解释统一损失方法为何在不同基准上表现出与专家模型不同的优势。

## 关键术语

| 术语              | 常见说法             | 实际含义                                      |
|-------------------|----------------------|-----------------------------------------------|
| Next-token prediction | “NTP”               | 标准自回归损失：预测token[i+1]，给定token[0..i]；适用于所有分词模态。 |
| IBQ tokenizer      | “Inverse bottleneck quantizer” | 一类VQ-VAE，码本较大（32768+），重构优于Chameleon分词器。    |
| 3D VQ             | “Spatiotemporal quantizer” | 码本按（时间，行，列）索引；一个token覆盖4x4x4像素立方。       |
| Classifier-free guidance | “CFG”             | 用权重γ混合有条件与无条件logits，提升推理图像质量。             |
| Unified vocabulary | “Shared tokens”          | 文本+图像+视频共享同一整数词汇；模型预测任一模态的下一个token。    |
| MJHQ-30K           | “Image gen benchmark”     | Midjourney质量的30k提示基准测试；Emu3 在此报告FID。              |

## 相关阅读

- [Wang 等 — Emu3：下一token预测即是你所需 (arXiv:2409.18869)](https://arxiv.org/abs/2409.18869)
- [Sun 等 — Emu：多模态生成预训练 (arXiv:2307.05222)](https://arxiv.org/abs/2307.05222)
- [Liu 等 — LWM (arXiv:2402.08268)](https://arxiv.org/abs/2402.08268)
- [Yu 等 — MAGVIT-v2 (arXiv:2310.05737)](https://arxiv.org/abs/2310.05737)
- [Tian 等 — VAR (arXiv:2404.02905)](https://arxiv.org/abs/2404.02905)
