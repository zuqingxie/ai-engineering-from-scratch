# Flamingo 与门控跨注意力用于少样本视觉语言模型（Few-Shot VLMs）

> DeepMind 的 Flamingo（2022）率先完成了两件事。它展示了单一模型能够处理任意交错的图像、视频和文本序列。并且展示了视觉语言模型（VLMs）可以进行上下文学习——只要给出带有三个示例（图像，字幕）对的少样本提示，模型便能在没有梯度更新的情况下为新图像生成字幕。其机制是：插入在冻结的语言模型（LLM）现有层之间的门控跨注意力层，配以一个学习的 tanh 门控，初始时门控值为零，从而在初始化阶段保持 LLM 的文本能力。此课程讲解 Flamingo 的 Perceiver 重采样器和门控跨注意力架构——Gemini 的交错输入和 Idefics2 视觉标记的祖先。

**类型：** 学习  
**语言：** Python（标准库，门控跨注意力 + Perceiver 重采样器示例）  
**前置知识：** 阶段 12 · 03（BLIP-2 Q-Former）  
**时长：** 约 120 分钟

## 学习目标

- 解释门控跨注意力如何通过 tanh(gate) = 0 在初始化时保持冻结 LLM 的文本能力。
- 讲解 Perceiver 重采样器：将 N 个图像块转化为 K 个固定的“潜变量”查询，通过跨注意力实现。
- 描述 Flamingo 如何通过尊重图像位置的因果掩码处理交错的图像-文本序列。
- 复现少样本多模态提示结构（3 个图像-字幕示例，后接查询图像）。

## 问题背景

BLIP-2 将 32 个视觉标记输入到冻结的 LLM 输入层，适合单图像场景。但是如果想要输入*多个*交错的图像和文本，比如"这是图像 A，生成字幕；这是图像 B，生成字幕；现在这是图像 C，生成字幕"情况下，LLM 的自注意力需要在一个流中同时处理图像标记和文本标记，且哪些位置能关注哪个图像的问题变得复杂。

Flamingo 的解决方案是：完全不改变 LLM 的输入流。插入额外的跨注意力层到 LLM 的现有层之间。文本标记依然通过 LLM 的因果自注意力正常流动。在每隔几个 LLM 层之间，文本标记也通过新的门控层跨注意力访问图像特征。门控初始化为零意味着在初始步骤新层是无操作的——模型表现完全等同于预训练的 LLM。随着训练进行，门控逐渐开启，视觉信息开始流入。

第二个问题：Flamingo 如何处理每个提示中可变数量的图像（0、1 或多个）？使用 Perceiver 重采样器——一个小型跨注意力模块，将任意数量的图像块转为固定数量的视觉潜变量标记。LLM 的跨注意力层无论提示中图像多少，输入形状保持一致。

## 核心概念

### 冻结的 LLM

Flamingo 使用冻结的 Chinchilla 70B LLM，70B 权重完全不变。文本自注意力和前馈网络（FFN）正常运作。

### Perceiver 重采样器

每个提示中的图像，ViT 产生 N 个图像块标记。Perceiver 重采样器拥有 K 个固定可学习潜变量（Flamingo 中 K=64）。每个重采样模块包含两个子步骤：

1. 跨注意力：K 个潜变量查询（Q）关注 N 个图像块（K/V）。
2. 潜变量自注意力 + 前馈网络。

经过 6 个重采样模块后，输出成为 K=64 个 1024 维视觉标记，无论 ViT 产生多少图像块。一个 224x224 图像（196 块）和一个 480x480 图像（900 块）都输出为 64 个重采样标记。

视频场景下，重采样器沿时间轴应用：每帧的图像块产生 64 个潜变量，加上时间位置编码，使模型区分 t=0 和 t=N。整段视频成为 T * 64 个视觉标记。

### 门控跨注意力

在冻结 LLM 的每 M 层之间插入一个门控跨注意力块（Flamingo 用 M=4）：

```text
x_after_llm_block = llm_block(x_before)
cross = cross_attn(x_after, resampler_output)
gated = tanh(alpha) * cross + x_after
x_before_next_block = gated
```

- `alpha` 是一个可学习的标量，初始化为零。
- `tanh(0) = 0`，在初始时门控分支贡献为零。
- 随着 `alpha` 远离零，跨注意力贡献平滑增加。
- 残差连接保证即使门完全打开，也不会覆盖 LLM 的文本表示，只是添加视觉信息。

这是 Flamingo 最关键的设计：视觉条件输入是加性、门控且初始化时为零。Flamingo 在初始步骤表现如一个完美的只处理文本的 Chinchilla 70B。

### 交错输入的掩码跨注意力

在提示如 "<image A> caption A <image B> caption B <image C> ?" 的情况下，每个文本标记只能看到序列中它之前出现的图像。跨注意力掩码确保位置 `t` 的文本标记只关注其之前最近的图像索引 `i_t` 对应的重采样器标记。选择“只见最近的前一幅图像”或“见所有之前的图像”都能符合逻辑；Flamingo 采用前者。

### 上下文少样本学习

Flamingo 提示示例：

```text
<image1> A photo of a cat. <image2> A photo of a dog. <image3> A photo of a
```

模型看到这种完成模式后输出“bird”（或图像3所显示内容）。无须梯度更新。冻结的 LLM 的上下文学习能力通过门控跨注意力得以保持——这是论文的重点所在及其意义。

### 训练数据

Flamingo 训练涉及三组数据：

1. MultiModal MassiveWeb (M3W)：4300 万网页，图片与文本交错，重构阅读顺序。
2. Image-Text Pairs（ALIGN + LTIP）：44 亿对。
3. Video-Text Pairs（VTP）：2700 万短视频片段。

OBELICS（2023）是交错网页语料库的开源复现，Idefics、Idefics2 及多数“Flamingo 类”开源模型均基于此训练。

### OpenFlamingo 与 Otter

OpenFlamingo（2023）是开源复现，架构相同（Perceiver 重采样器 + 门控跨注意力，基于冻结的 LLaMA 或 MPT）。提供 3B、4B 和 9B 规模模型。品质不及 Flamingo，因基础 LLM 较小且数据不足。

Otter（2023）基于 OpenFlamingo，通过 MIMIC-IT（多模态指令数据集）进行指令微调，展示门控跨注意力同样适用于指令追随。

### 后续模型

- Idefics / Idefics2 / Idefics3：Hugging Face 的门控跨注意力系列，结构逐步简化（Idefics2 用自适应池化替代了重采样器，直接使用块标记）。
- Flamingo 向 Chameleon 过渡：到 2024 年许多团队转向早期融合（详见课程 12.11）；Flamingo 风格的门控跨注意力仍在需要冻结骨干网络的场景中使用。
- Gemini 的交错输入：概念沿用 Flamingo 的交错格式灵活性，具体机制为专有。

### 与 BLIP-2 的对比

|  | BLIP-2 | Flamingo |
|---|---|---|
| 视觉桥接 | 输入处一次 Q-Former | 每 M 层插入门控跨注意力 |
| 视觉标记数 | 每图像 32 个 | 每图像每跨注意力层 64 个 |
| 冻结 LLM | 是 | 是 |
| 少样本上下文能力 | 弱 | 强——论文核心 |
| 交错输入支持 | 无原生支持 | 有，设计目标 |
| 训练数据量 | 1.3 亿对 | 13 亿对 + 4300 万交错网页 |
| 训练参数量 | 1.88 亿 | 约 100 亿（跨注意力层） |
| 训练计算 | 8 个 A100 数日 | 数千 TPUv4 数周 |

预算有限做单图像 VQA 选 BLIP-2。要做交错、多图像或少样本推理用 Flamingo/Idefics2。

## 使用示例

`code/main.py` 演示：

1. 在 36 个假设图像块上使用 Perceiver 重采样器，8 个可学习潜变量（纯 Python 跨注意力）。
2. 门控跨注意力步骤：`alpha=0` 时输出等于输入（LLM 无变化），`alpha=2.0` 时混合视觉贡献。
3. 交错掩码构建，用于生成 "(image 1) (text 1) (image 2) (text 2)" 序列的二维注意力掩码。

## 投产指南

本课生成 `outputs/skill-gated-bridge-diagnostic.md`。给定一个开放 VLM 的配置（是否有重采样器、跨注意力频率、门控策略），它识别 Flamingo 血统元素并解释冻结策略。适用于调试微调后文本性能退化的原因（一般是门控过快打开）。

## 练习

1. 计算 Flamingo-9B 视觉参数量：9B LLM + 14 亿门控跨注意力层 + 6400 万重采样器参数。训练参数占总参数比例是多少？

2. 用 PyTorch 实现门控残差 `y = tanh(alpha) * cross + x`。实验验证初始化时 `alpha=0` 下 `y==x` 精确成立。

3. 阅读 OpenFlamingo 第 3.2 节（arXiv:2308.01390）关于批处理多个图像时，对于不同图像数的提示如何处理填充的策略。

4. 为什么 Flamingo 的跨注意力掩码让文本标记只关注*最近的一个*前置图像，而非所有前置图像？阅读 Flamingo 论文第 2.4 节，解释权衡。

5. 少样本上下文学习：构造一个含有 4 个“图像→主要对象颜色”示例的新 Flamingo 变体提示。描述示例数量从 0 到 8 变化时预期的准确率模式。

## 关键词

| 术语 | 通俗说法 | 实际含义 |
|------|-----------|------------|
| Perceiver 重采样器 | “固定潜变量跨注意力” | 将变长输入图像块转为 K 个固定标记的模块 |
| 门控跨注意力 | “Tanh 门控桥接” | 残差层 `y = tanh(alpha)*cross + x`，alpha 可学习，初始化为 0 |
| 交错输入 | “混合序列” | 图像与文本任意交错排列的提示格式 |
| 冻结 LLM | “无 LLM 梯度更新” | 文本 LLM 权重不变，只有重采样器和跨注意力层训练 |
| 少样本 | “上下文示例” | 在提示中给几个（图像，答案）对，无需微调即可泛化 |
| OBELICS | “交错网页语料库” | 含图文交错的 1.41 亿网页开源数据集 |
| Chinchilla | “70B 冻结基模型” | Flamingo 使用的冻结文本 LLM，来自 DeepMind 的 Chinchilla 论文 |
| 门控调度 | “alpha 变化” | 跨注意力门控在训练中开启速度 |
| 跨注意力频率 | “每 M 层插入” | 门控跨注意力块插入频率，Flamingo 设为 M=4 |
| OpenFlamingo | “开源复现” | MosaicML/LAION 开放的 3-9B 权重，与 Flamingo 架构一致 |

## 延伸阅读

- [Alayrac 等 — Flamingo (arXiv:2204.14198)](https://arxiv.org/abs/2204.14198) — 原始论文。  
- [Awadalla 等 — OpenFlamingo (arXiv:2308.01390)](https://arxiv.org/abs/2308.01390) — 开源复现。  
- [Laurençon 等 — OBELICS (arXiv:2306.16527)](https://arxiv.org/abs/2306.16527) — 交错网页语料库。  
- [Jaegle 等 — Perceiver IO (arXiv:2107.14795)](https://arxiv.org/abs/2107.14795) — 通用 Perceiver 架构。  
- [Li 等 — Otter (arXiv:2305.03726)](https://arxiv.org/abs/2305.03726) — 指令微调 Flamingo 衍生模型。  
- [Laurençon 等 — Idefics2 (arXiv:2405.02246)](https://arxiv.org/abs/2405.02246) — Flamingo 方法的现代简化版。
