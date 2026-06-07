# 条件式 GAN（Conditional GANs）与 Pix2Pix

> 2014-2017 年期间的重要突破之一是控制 GAN 的输出。附加标签、图像或句子。Pix2Pix 实现了图像版，在狭窄的图像到图像任务中，它仍然胜过任何通用的文本到图像模型。

**类型:** 构建  
**语言:** Python  
**先决条件:** 第 8 阶段 · 03（GANs），第 4 阶段 · 06（U-Net），第 3 阶段 · 07（CNNs）  
**时间:** 约 75 分钟

## 问题描述

无条件 GAN 会随机生成任意人脸。对演示有用，但生产环境中没用。你希望做的是：*将素描映射成照片*，*将地图映射为空中照片*，*将白天场景映射到夜晚*，*给灰度图上色*。在所有这些任务中，你给定输入图像 `x`，输出 `y`，且存在语义对应关系。每个 `x` 对应许多可能的 `y`。均方误差（mean-squared error）会将它们模糊成糊状。对抗损失不会，因为“看起来真实”是尖锐的。

条件式 GAN（Mirza & Osindero, 2014）将条件 `c` 作为输入加给 `G` 和 `D`。Pix2Pix（Isola 等, 2017）对此做了专门化：条件是输入的整张图像，生成器是 U-Net，判别器是*基于 patch*的分类器（PatchGAN），损失是对抗损失加 L1。这个方案在狭窄的图像到图像领域中即使到了 2026 年，也依然优于从零开始训练的文本到图像模型，因为它是基于*成对数据*训练——你得到了你所需的精确信号。

## 概念解析

![Pix2Pix: U-Net 生成器，PatchGAN 判别器](../assets/pix2pix.svg)

**条件生成器（Conditional G）**：`G(x, z) → y`。在 Pix2Pix 中，`z` 是生成器内部的 dropout（不输入显式噪声——Isola 发现显式噪声会被忽略）。

**条件判别器（Conditional D）**：`D(x, y) → [0, 1]`。输入是*成对*的（条件，输出）。这是关键区别：D 需要判定 `y` 是否与 `x` 一致，而不仅仅是 `y` 是否看起来真实。

**U-Net 生成器**：编码器-解码器结构，通过瓶颈层带有跳跃连接。对于输入和输出共享低级结构（边缘、轮廓）的任务至关重要。没有跳跃连接，高频细节会消失。

**PatchGAN 判别器**：不输出单一的真假分数，而是输出一个 `N×N` 格子的评分，每个单元负责约 70×70 像素的感受野。最终取平均。这是马尔可夫随机场假设：真实感是局部的。训练更快，参数更少，输出更锐利。

**损失函数**

```text
loss_G = -log D(x, G(x)) + λ · ||y - G(x)||_1
loss_D = -log D(x, y) - log (1 - D(x, G(x)))
```

L1 项稳定训练并推动 G 向真实目标靠拢。L1 较 L2 给出更锐利的边缘（中位数，不是均值）。`λ = 100` 是 Pix2Pix 的默认设置。

## CycleGAN——无配对数据时的解决方案

Pix2Pix 需要成对的 `(x, y)` 数据。CycleGAN（Zhu 等，2017）舍弃了这一要求，代价是增加了一个*循环一致性*损失。两个生成器 `G: X → Y` 和 `F: Y → X`。训练目标是使 `F(G(x)) ≈ x` 和 `G(F(y)) ≈ y`。这使得你可以在没有成对示例的情况下，将马转换成斑马，夏天变冬天等。

到 2026 年，未配对图像到图像任务主要由扩散模型（ControlNet、IP-Adapter）完成而非 CycleGAN，但循环一致性思想在几乎所有未配对域适应的论文中仍然存在。

## 构建步骤

`code/main.py` 实现了一个小型的条件 GAN 用于一维数据。条件 `c` 是分类标签（0 或 1）。任务是从对应类别的条件分布产生样本。

### 步骤 1：将条件附加到 G 和 D 的输入中

```python
def G(z, c, params):
    return mlp(concat([z, one_hot(c)]), params)

def D(x, c, params):
    return mlp(concat([x, one_hot(c)]), params)
```

独热编码（one-hot encoding）是最简单的做法。更大的模型使用学习得到的嵌入、FiLM 调制或交叉注意力。

### 步骤 2：训练条件模型

```python
for step in range(steps):
    x, c = sample_real_conditional()
    noise = sample_noise()
    update_D(x_real=x, x_fake=G(noise, c), c=c)
    update_G(noise, c)
```

生成器必须匹配*给定条件*下的真实分布，而非边缘分布。

### 步骤 3：验证每个类别的输出

```python
for c in [0, 1]:
    samples = [G(noise, c) for noise in batch]
    mean_c = mean(samples)
    assert_near(mean_c, real_mean_for_class_c)
```

## 常见陷阱

- **条件被忽略**。生成器学会边缘化，判别器因为条件信号弱而不惩罚。解决：更早层次上条件判别器输入，不只是后期层，使用带投影的判别器（Miyato & Koyama 2018）。
- **L1 权重太低**。生成器趋势于生成任意的真实样本，而非忠实样本。Pix2Pix 风格任务推荐从 λ≈100 开始。
- **L1 权重太高**。生成器输出模糊，因为 L1 依然是 L_p 范数。训练稳定后逐渐衰减权重。
- **判别器出现真实数据泄露**。将 `(x, y)` 拼接为判别器输入，而非只输入 `y`。否则判别器无法检查一致性。
- **各类独立发生模式崩溃**。每个类可能独立崩溃。进行类别条件多样性检查。

## 使用建议

2026 年图像到图像任务的最佳方法：

| 任务 | 最佳方法 |
|------|----------|
| 素描 → 照片， 同域，成对数据 | Pix2Pix / Pix2PixHD（依旧快速且锐利） |
| 素描 → 照片，未配对 | 使用 Scribble 条件模型的 ControlNet |
| 语义分割 → 照片 | SPADE / GauGAN2 或 SD + ControlNet-Seg |
| 风格迁移 | 使用带 IP-Adapter 或 LoRA 的扩散方法；GAN 方法已成遗留 |
| 深度图 → 照片 | ControlNet-Depth + Stable Diffusion |
| 超分辨率 | Real-ESRGAN（GAN）、ESRGAN-Plus 或 SD-Upscale（扩散） |
| 上色 | ColTran、扩散上色器或 Pix2Pix-color |
| 白天 → 夜晚、季节、天气转换 | CycleGAN 或基于 ControlNet |

只要 (a) 有数千对样本，(b) 任务窄且可重复，且 (c) 需要快速推理，Pix2Pix 依然是合适工具。对通用开放领域任务，扩散模型更胜一筹。

## 部署指南

保存 `outputs/skill-img2img-chooser.md`。该技能根据任务描述、数据可用性（成对/未配对，样本数量）及延迟/质量预算，输出：方法（Pix2Pix、CycleGAN、ControlNet 变体、SDXL + IP-Adapter）、训练数据要求、推理成本及评估协议（LPIPS、FID、任务特定指标）。

## 练习

1. **简单。** 修改 `code/main.py` 增加第三个类别。验证生成器仍能将每个类别的噪声映射到正确模式。
2. **中等。** 把 L1 损失替换成感知损失（如用一个小型冻结的判别器当特征提取器）在一维场景下。它会影响条件分布的锐利度吗？
3. **困难。** 在一维场景下绘制 CycleGAN：两个分布，两个生成器，加上循环损失。展示它在无配对数据情况下能学习映射。

## 关键词

| 术语 | 通俗说法 | 实际含义 |
|------|----------|----------|
| 条件式 GAN | “带标签的 GAN” | G(z, c)、D(x, c)。两个网络都看到条件。 |
| Pix2Pix | “图像到图像 GAN” | 带成对数据的条件 GAN，U-Net 生成器，PatchGAN 判别器 + L1 损失。 |
| U-Net | “编码器-解码器带跳跃” | 对称卷积网络；跳跃连接保留高频信息。 |
| PatchGAN | “局部真实感分类器” | 判别器输出局部 patch 评分而非全局评分。 |
| CycleGAN | “无配对图像转换” | 两个生成器 + 循环一致性损失；无成对数据。 |
| SPADE | “GauGAN” | 用语义图规范中间激活，语义分割到图像。 |
| FiLM | “特征线性调制” | 根据条件对特征进行仿射变换；经济的条件方法。 |

## 生产笔记：Pix2Pix 作为延迟受限基线

当你有成对数据且任务狭窄（素描→渲染，语义图→照片，白天→夜晚），Pix2Pix 的一次性推理延迟比扩散快一个数量级。生产环境通常对比如下：

| 路径 | 步数 | 单张 512² 图像在单块 L4 上的典型延迟 |
|-------|------|--------------------------------------|
| Pix2Pix（U-Net 前向） | 1 | ~30 毫秒 |
| SD-Inpaint 或 SD-Img2Img | 20 | ~1.2 秒 |
| SDXL-Turbo Img2Img | 1-4 | ~0.15-0.35 秒 |
| ControlNet + SDXL base | 20-30 | ~3-5 秒 |

Pix2Pix 在静态批处理的吞吐量上胜出（每个请求 FLOPs 一致）。扩散在质量和泛化上胜出。现代实践常是为狭窄任务发布 Pix2Pix 风格的蒸馏模型，备份扩散模型以应对罕见输入。

## 拓展阅读

- [Mirza & Osindero (2014). Conditional Generative Adversarial Nets](https://arxiv.org/abs/1411.1784) — 条件 GAN 论文。
- [Isola et al. (2017). Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004) — Pix2Pix。
- [Zhu et al. (2017). Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593) — CycleGAN。
- [Wang et al. (2018). High-Resolution Image Synthesis with Conditional GANs](https://arxiv.org/abs/1711.11585) — Pix2PixHD。
- [Park et al. (2019). Semantic Image Synthesis with Spatially-Adaptive Normalization](https://arxiv.org/abs/1903.07291) — SPADE / GauGAN。
- [Miyato & Koyama (2018). cGANs with Projection Discriminator](https://arxiv.org/abs/1802.05637) — 投影判别器。
