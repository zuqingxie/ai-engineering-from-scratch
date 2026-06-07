# Vision Transformers 和 Patch-Token 原语（Patch-Token 原语）

> 在多模态（multimodal）之前，图像必须先转换成 transformer 可以处理的令牌序列。2020 年的 ViT 论文给出的答案是用 16x16 像素的图块（patch），线性投影，加上位置嵌入。五年后，2026 年的每一个前沿模型（Claude Opus 4.7 原生 2576px，Gemini 3.1 Pro，Qwen3.5-Omni）依然从这一步开始——编码器从 ViT 变成了 DINOv2，再到 SigLIP 2，增加了寄存令牌（register tokens），位置编码变成了 2D-RoPE，但原语保持不变。本课将从头到尾阅读 patch-token 管线，并用标准库 Python 构建它，以便 Phase 12 的其余部分有一个具体的「视觉令牌（visual tokens）」的心理模型。

**类型（Type）：** 学习  
**语言（Languages）：** Python（标准库、patch tokenizer + 几何计算器）  
**先决条件（Prerequisites）：** Phase 7（Transformers），Phase 4（计算机视觉）  
**时间（Time）：** ~120 分钟

## 学习目标（Learning Objectives）

- 将一个形状为 HxWx3 的图像转换为带有正确位置编码的 patch 令牌序列。
- 计算给定（patch 大小，分辨率，隐藏维度，深度）的 ViT 的序列长度、参数数量和 FLOPs。
- 说出让 ViT 从 2020 年研究阶段走向 2026 年生产环境的三个升级：自监督预训练（DINO / MAE）、寄存令牌（register tokens）和原生分辨率的打包（native-resolution packing）。
- 对下游任务选择 CLS pooling、均值池化（mean pooling）或寄存令牌。

## 问题（The Problem）

Transformer 处理的是向量序列。文本已经是序列（字节或词元）。图像是一个二维像素网格，有三个颜色通道——不是序列。如果你把每个像素展平，一个 224x224 的 RGB 图像会变成 150,528 个令牌，这样长度的自注意力计算根本不可行（序列长度的二次方复杂度）。

2020 年之前的方法是在前端加上 CNN 特征提取器：ResNet 生成 7x7 的 2048 维向量特征图，把这 49 个令牌送入 transformer。这能用，但继承了 CNN 的偏见（平移等变性，局部感受野）且失去了 transformer 对尺度的渴望。

Dosovitskiy 等人（2020）大胆提出：如果跳过 CNN 呢？将图像分割为固定大小的 patch（比如 16x16 像素），线性投影每个 patch 到向量，加入位置嵌入，然后送入 vanilla transformer。当时这几乎是异端——视觉中不使用卷积。用足够数据（JFT-300M，然后是 LAION），其性能超过了 ImageNet 上的 ResNet，且持续提升。

到了 2026 年，ViT 原语已无可争议地成为基础。每个开源权重的视觉语言模型（VLM）的视觉塔都是此派系的后代（DINOv2、SigLIP 2、CLIP、EVA、InternViT）。问题不再是「是否用 patch？」而是「用什么 patch 大小，什么分辨率调度，什么预训练目标，什么位置编码？」

## 概念（The Concept）

### Patch 作为令牌

给定一个形状为 `(H, W, 3)` 的图像 `x` 和 patch 大小 `P`，你将图像切割为 `(H/P) x (W/P)` 个不重叠的补丁。每个 patch 是一个 `P x P x 3` 的像素块。将每个块展平成一个 `3 P^2` 的向量。对每个 patch 应用共享的线性投影矩阵 `W_E`，形状为 `(3 P^2, D)`，将其映射到模型的隐藏维度 `D`。

对 ViT-B/16 的典型配置：  
- 分辨率 224，patch 大小 16 → 网格为 14x14 → 196 个 patch 令牌。  
- 每个 patch 是 `16 x 16 x 3 = 768` 个像素值，投影到 `D = 768`。  
- 加上一个可学习的 `[CLS]` 令牌 → 序列长度为 197。

Patch 投影在数学上等价于一个卷积核大小为 `P`，步幅为 `P`，输出通道数为 `D` 的二维卷积。生产代码就是这样实现的——`nn.Conv2d(3, D, kernel_size=P, stride=P)`。这里的「线性投影」只是概念性描述，卷积实现效率更高。

### 位置嵌入（Positional embeddings）

Patch 本身没有内在顺序——transformer 看它们就像一个无序集合。早期 ViT 使用可学习的一维位置嵌入（每个位置一个 768 维向量，共 197 个）。可行，但将模型绑定到训练分辨率上：推理时分辨率改变则需插值位置表。

现代视觉主干网络使用 2D-RoPE（Qwen2-VL 的 M-RoPE，SigLIP 2 默认）或分解的二维位置编码。2D-RoPE 通过根据 patch 的（行，列）索引旋转 query 和 key 向量，使模型通过旋转角度推断相对二维位置。无需位置表，可适应任意网格大小推理。

### CLS 令牌、池化输出和寄存令牌

图像级表示有三种并存选择：

1. `[CLS]` 令牌。在 patch 序列前面加一个可学习向量。通过所有 transformer 层后该 CLS 令牌的隐藏状态作为图像表示。继承自 BERT。原始 ViT、CLIP 使用。
2. 均值池化。对所有 patch 令牌的输出隐藏状态求平均。SigLIP、DINOv2 和大部分现代 VLM 使用。
3. 寄存令牌（Register tokens）。Darcet 等人（2023）观察到无显式汇合令牌的 ViT 会生成高范数的「伪影」patch，这些伪影劫持自注意力。添加 4~16 个可学习寄存令牌可以吸收这种负载，提升密集预测质量（分割、深度）。DINOv2 和 SigLIP 2 都配备了寄存令牌。

选择对下游很关键。CLS 适合分类。对于给 LLM 输入 patch 令牌的 VLM，不做池化——每个 patch 都是 LLM 输入令牌。寄存令牌在交接前丢弃（它们是支架，不是内容）。

### 预训练：监督、对比、掩码、自蒸馏

2020 年的 ViT 用 JFT-300M 做监督分类预训练。很快被替代为：

- CLIP (2021)：400M 图文对的对比学习。详见 Lesson 12.02。  
- MAE (2021, He 等)：遮掩 75% patch，重建像素。自监督，纯图像适用。  
- DINO (2021) / DINOv2 (2023)：学生-教师自蒸馏，无标签无字幕。2023 年的 DINOv2 ViT-g/14 是最强纯视觉主干，且是「密集特征」的默认选择。  
- SigLIP / SigLIP 2 (2023, 2025)：带 sigmoid 损失且用 NaFlex 支持原生宽高比的 CLIP。2026 年开源 VLM 的主流视觉塔（Qwen、Idefics2、LLaVA-OneVision）。

你的预训练选择决定主干的适用场景：CLIP/SigLIP 适合语义匹配文本，DINOv2 适合密集视觉特征，MAE 适合作为下游微调的起点。

### 规模定律

ViT 规模定律（Zhai 等人 2022）表明 ViT 的质量遵循模型大小、数据规模和计算量的可预测规律。在固定计算条件下：  
- 模型更大 + 数据更多 → 质量更好。  
- patch 大小是序列长度与保真度的杠杆。Patch 14（DINOv2/SigLIP SO400m 典型值）每张图产生更多令牌，适合 OCR 和密集任务，但速度较慢。  
- 分辨率是另一个重要杠杆。从 224 到 384 再到 512 几乎总是提升性能，计算量 FLOPs 成二次方增长。

ViT-g/14（10 亿参数，patch 14，分辨率 224 → 256 令牌）和 SigLIP SO400m/14（4 亿参数，patch 14）是 2026 年开源 VLM 的两大主力编码器。

### ViT 的参数量

完整计算见 `code/main.py`。举 ViT-B/16 于 224 分辨率例：

```text
patch_embed = 3 * 16 * 16 * 768 + 768  =  591k
cls + pos    = 768 + 197 * 768          =  152k
block        = 4 * 768^2 (QKVO) + 2 * 4 * 768^2 (MLP) + 2 * 2*768 (LN)
             = 12 * 768^2 + 3k          =  7.1M
12 blocks    = 85M
final LN    = 1.5k
total       ≈ 86M
```

加载 checkpoint 之前，先用这种方法大致估算 ViT 参数。主干体量设定你任何下游 VLM 的 VRAM 下限。

### 2026 年生产配置

2026 年多数开源 VLM 使用的编码器是 SigLIP 2 SO400m/14，原生分辨率（NaFlex）。其特征：  
- 4 亿参数。  
- patch 大小 14，默认分辨率 384 → 每图 729 个 patch 令牌。  
- 图像级任务用均值池化；VQA 所有 729 patch 令牌输入 LLM。  
- 4 个寄存令牌，交接前丢弃。  
- 2D-RoPE，图像级缩放以支持原生宽高比。

配置里每个决策都可追溯到对应论文。

## 使用它（Use It）

`code/main.py` 是 patch tokenizer 和几何计算器。输入（图像 H，W，patch P，隐藏维度 D，深度 L），报告：

- 经过 patch 后的网格形状和序列长度。  
- 一个合成 8x8 像素玩具图像的 token 序列（演示展平 + 投影路径）。  
- 参数量按 patch 嵌入、位置嵌入、transformer block 和 head 分类统计。  
- 目标分辨率下的单次前向 FLOPs。  
- ViT-B/16 @ 224、ViT-L/14 @ 336、DINOv2 ViT-g/14 @ 224、SigLIP SO400m/14 @ 384 的对比表。

运行它。核对参数量与公开数据。修改 patch 大小和分辨率，感受 token 数的开销变化。

## 交付（Ship It）

本课产出 `outputs/skill-patch-geometry-reader.md`。给定 ViT 配置（patch 大小，分辨率，隐藏维度，深度），输出 token 计数，参数计数和 VRAM 估算，并附合理理由。选视觉主干做 VLM 时都用这项技能，避免「令牌爆炸，LLM 上下文溢出」的惊吓。

## 练习（Exercises）

1. 计算 Qwen2.5-VL 原生 1280x720 输入、patch 大小 14 的 patch-token 序列长度。与 CLS-only 表示相比如何？

2. 1080p（1920x1080）分辨率下 patch 14 会产生多少令牌？30 FPS、5 分钟视频累计多少视觉令牌？哪种成本节省最多：池化、帧采样还是令牌合并？

3. 纯 Python 实现 patch 令牌的均值池化。验证 DINOv2 输出 196 个令牌的均值池化，与模型要求 pooled embedding 时 `forward` 返回的一致。

4. 阅读《Vision Transformers Need Registers》（arXiv:2309.16588）第 3 节。用两句话描述寄存令牌吸收的伪影及其对下游密集预测的重要性。

5. 修改 `code/main.py` 支持 patch-n’-pack：给定不同分辨率图像列表，产生单个打包序列和块对角自注意力掩码。完成后对照 Lesson 12.06 验证。

## 关键词（Key Terms）

| 术语         | 大家怎么说               | 实际含义                                                     |
|--------------|--------------------------|--------------------------------------------------------------|
| Patch        | “16x16 像素的小方块”    | 输入图像的固定大小不重叠区域，成为一个令牌                     |
| Patch embedding | “线性投影”               | 共享学习矩阵（或步幅为 P 的 Conv2d），将展平的 patch 像素映射到 D 维向量 |
| CLS token    | “分类令牌”               | 序列前置学习向量，最终隐藏状态代表整个图像；2026 年是可选的       |
| Register token | “汇合令牌”              | 额外学习令牌，吸收 ViT 预训练中产生的高范数注意力伪影             |
| Position embedding | “位置信息”             | 每个位置的向量或旋转，让序列感知顺序；2D-RoPE 是现代默认            |
| Grid         | “patch 网格”             | 按分辨率和 patch 大小划分的 `(H/P) x (W/P)` 二维 patch 阵列      |
| NaFlex       | “原生灵活分辨率”         | SigLIP 2 的特性：单模型支持多长宽比和多分辨率，无需重训练           |
| Backbone     | “视觉塔”                 | 预训练图像编码器，输出 patch 令牌供 VLM 中的 LLM 使用               |
| Pooling      | “图像级摘要”             | 将 patch 令牌汇聚为单向量的策略：CLS、均值、注意力池化或寄存基方法       |
| Patch 14 vs 16 | “更细网格 vs 更粗网格”   | Patch 14 每图生产更多令牌，OCR/密集任务表现更好，速度更慢；Patch 16 是经典默认 |

## 拓展阅读

- [Dosovitskiy et al. — An Image is Worth 16x16 Words (arXiv:2010.11929)](https://arxiv.org/abs/2010.11929) — 原始 ViT（视觉Transformer架构）论文。
- [He et al. — Masked Autoencoders Are Scalable Vision Learners (arXiv:2111.06377)](https://arxiv.org/abs/2111.06377) — MAE（掩码自编码器），自监督预训练。
- [Oquab et al. — DINOv2 (arXiv:2304.07193)](https://arxiv.org/abs/2304.07193) — 大规模自蒸馏，无需标签。
- [Darcet et al. — Vision Transformers Need Registers (arXiv:2309.16588)](https://arxiv.org/abs/2309.16588) — 寄存器标记与伪影分析。
- [Tschannen et al. — SigLIP 2 (arXiv:2502.14786)](https://arxiv.org/abs/2502.14786) — 2026 年默认视觉主干网络。
- [Zhai et al. — Scaling Vision Transformers (arXiv:2106.04560)](https://arxiv.org/abs/2106.04560) — 经验性扩展法规。
