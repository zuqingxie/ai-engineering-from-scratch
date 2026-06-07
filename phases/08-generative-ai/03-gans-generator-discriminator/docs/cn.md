# GANs — 生成器（Generator）与判别器（Discriminator）

> Goodfellow 在 2014 年的妙招是完全跳过密度估计。两个网络。一个制造假样本。一个识别它们。它们彼此对抗，直到假样本和真实样本无法区分。理应行不通，但实际常常不行。当确实生效时，样本通常是特定领域中最锐利的。

**类型：** 构建  
**语言：** Python  
**先决条件：** 第 3 阶段 · 02（反向传播（Backprop））、第 3 阶段 · 08（优化器（Optimizers））、第 8 阶段 · 02（变分自编码器（VAE））  
**时间：** ~75 分钟

## 问题

变分自编码器（VAE）生成的样本模糊，因为其使用的均方误差（MSE）解码器损失是平均图像的贝叶斯最优解 — 许多可能数字的均值是模糊的数字。你需要的损失函数奖赏的是*合理性*，而非像素级别紧邻某一具体目标的相似度。合理性的形式没有封闭解，你必须学会它。

Goodfellow 的想法：训练一个分类器 `D(x)` 来区分真实图像和假图像。训练生成器 `G(z)` 欺骗 `D`。`G` 的损失信号来源于 `D` 当前判定图像真实的概率。随着 `G` 的进步，此信号不断调整，追赶不断变化的目标。如果两个网络收敛，`G` 就在没有明确写出 `log p(x)` 的情况下学会了数据分布。

这就是对抗训练（adversarial training）。其数学表现为极小极大游戏：

```text
min_G max_D  E_real[log D(x)] + E_fake[log(1 - D(G(z)))]
```

到 2026 年，GAN 不再是最先进的生成器（扩散模型和流匹配（flow matching）已经超越它）。但 StyleGAN 2/3 依然是最锐利的人脸模型，GAN 判别器被用作扩散训练中的*感知损失*，对抗训练推动了快速一步蒸馏（如 SDXL-Turbo，SD3-Turbo，LCM），实现了实时扩散。

## 概念

![GAN training: generator and discriminator in minimax](../assets/gan.svg)

**生成器 `G(z)`。** 将噪声向量 `z ~ N(0, I)` 映射为样本 `x̂`。通常是一个解码器形状的网络（全连接或转置卷积层）。

**判别器 `D(x)`。** 将样本映射到标量概率（或评分）。真实样本 → 1，假样本 → 0。

**损失。** 两个交替更新：

- **训练 `D`：** `loss_D = -[ log D(x) + log(1 - D(G(z))) ]`。对真实=1，假=0 使用二元交叉熵。
- **训练 `G`：** `loss_G = -log D(G(z))`。这是 Goodfellow 使用的*非饱和形式*（原始的 `log(1 - D(G(z)))` 在 `D` 十分自信时会饱和导致梯度消失）。

**训练循环。** 交替执行一步 `D` 更新，一步 `G` 更新。重复进行。

**为何有效。** 当 `G` 完美匹配数据分布 \(p_{data}\) 时，`D` 无法优于随机猜测，输出全为 0.5，`G` 不再获得梯度。达到均衡。

**为何失败。** 模式崩溃（`G` 找到 `D` 识别不了的一个模式并无限产生）、梯度消失（`D` 学得过快，`log D` 饱和）、训练不稳定（学习率、批次大小等）。

## 使 GAN 工作的变体

| 年份  | 创新 | 解决方案 |
|-------|------|---------|
| 2015  | DCGAN | 卷积/转置卷积，批归一化（batch norm），LeakyReLU — 第一个稳定架构。 |
| 2017  | WGAN, WGAN-GP | 用 Wasserstein 距离+梯度惩罚代替 BCE。修复梯度消失。 |
| 2017  | 谱归一化（Spectral normalization） | 约束判别器的 Lipschitz 常数。2026 年判别器依然使用。 |
| 2018  | Progressive GAN | 先训练低分辨率，逐步加层。首次实现兆像素结果。 |
| 2019  | StyleGAN / StyleGAN2 | 引入映射网络和自适应实例归一化（AdaIN）。固定域光写实最佳。 |
| 2021  | StyleGAN3 | 无别名（alias-free）、平移等变——2026 年人脸领域仍然标杆。 |
| 2022  | StyleGAN-XL | 条件式、类别感知、更大规模。 |
| 2024  | R3GAN | 重新命名，增强正则化；无需技巧即可在 1024² 分辨率下工作。 |

## 构建它

`code/main.py` 在一维数据上训练一个小型 GAN：两个高斯混合。生成器和判别器都是单隐层多层感知机（MLP）。我们手动实现前向、反向和极小极大循环。目的是观察两种关键失败模式（模式崩溃和梯度消失）真正出现时的表现。

### 步骤 1：非饱和损失

vanilla Goodfellow 损失 `log(1 - D(G(z)))` 在 `D` 高度确信 `G(z)` 是假时趋近于 0，此时 `G` 的梯度几乎为零，无法改进。非饱和形式 `-log D(G(z))` 有相反的极限：当 `D` 自信时梯度爆炸，给予 `G` 强烈信号。

```python
def g_loss(d_fake):
    # 最大化 log D(G(z))  <=>  最小化 -log D(G(z))
    return -sum(math.log(max(p, 1e-8)) for p in d_fake) / len(d_fake)
```

### 步骤 2：每个生成器步骤对应一个判别器步骤

```python
for step in range(steps):
    # 训练 D
    real_batch = sample_real(batch_size)
    fake_batch = [G(z) for z in sample_noise(batch_size)]
    update_D(real_batch, fake_batch)

    # 训练 G
    fake_batch = [G(z) for z in sample_noise(batch_size)]  # 新生成的假样本
    update_G(fake_batch)
```

给 `G` 使用新生成的假样本，否则梯度会变陈旧。

### 步骤 3：监测模式崩溃

```python
if step % 200 == 0:
    samples = [G(z) for z in sample_noise(500)]
    mode_a = sum(1 for s in samples if s < 0)
    mode_b = 500 - mode_a
    if min(mode_a, mode_b) < 50:
        print("  [!] 模式崩溃：某一模式样本严重不足")
```

典型症状：真实的两个模式中有一个停止被生成。判别器不再校正，因为它不再将其视作假样本。

## 陷阱

- **判别器过强。** 将 `D` 的学习率降低 2-5 倍，或添加实例/层噪声。如果 `D` 准确率超过 95%，`G` 基本死掉。
- **生成器记忆某模式。** 对判别器输入加噪声，使用小批量判别层，或切换至 WGAN-GP。
- **批归一化泄漏统计。** 真实批次与假批次共用 BN 层时会混淆统计。改用实例归一化或谱归一化。
- **基于 Inception 分数的作弊。** FID 和 IS 在低样本数量时噪声大。评价时使用 ≥10k 样本。
- **条件任务下一次采样是假象。** 你仍需 CFG 比例、截断技巧和重采样以得到可用输出。

## 使用它

2026 年 GAN 技术栈：

| 情境          | 选择                |
|---------------|---------------------|
| 固定姿态光写实人脸 | StyleGAN3（最锐利、最小） |
| 动漫/风格化脸部   | StyleGAN-XL 或 Stable Diffusion LoRA |
| 图像到图像转换  | Pix2Pix / CycleGAN（第 8 阶段 · 04）或 ControlNet（第 8 阶段 · 08） |
| 快速一步文字到图像 | 扩散模型对抗蒸馏（SDXL-Turbo，SD3-Turbo） |
| 扩散训练中感知损失 | 图像裁剪上的小型 GAN 判别器 |
| 任何多模态或开放式 | 不用 GAN — 使用扩散或流匹配 |

GAN 样本锐利但适用范围窄。一旦领域扩展到照片、任意文本提示、视频等，转用扩散模型。对抗训练作为一个组件依然存在（感知损失、蒸馏），但不再是独立生成器。

## 投产

保存 `outputs/skill-gan-debugger.md`。Skill 能读取失败的 GAN 训练（损失曲线、样本网格、数据集大小），输出可能原因排名、一行修复建议和重跑方案。

## 练习

1. **简单。** 用默认设置运行 `code/main.py`。然后设置 `D_LR = 5 * G_LR` 并重跑。`G` 的损失多快收敛到常数？
2. **中等。** 用 WGAN 损失替换 Goodfellow 的 BCE 损失：`loss_D = E[D(fake)] - E[D(real)]`，`loss_G = -E[D(fake)]`，且将 `D` 权重剪裁到 `[-0.01, 0.01]`。训练更稳定吗？比较真实时间收敛速度。
3. **困难。** 将一维示例扩展到二维数据（8 个高斯混合在环上）。记录生成器在 1k、5k、10k 步捕获的模式数量。实现小批量判别并重新测量。

## 关键词

| 术语 | 常见说法 | 实际含义 |
|-------|----------|----------|
| 生成器（Generator） | “G” | 噪声到样本网络，`G: z → x̂`。 |
| 判别器（Discriminator） | “D” | 分类器，`D: x → [0, 1]`，真实对假。 |
| 极小极大（Minimax） | “博弈” | `min_G max_D` 的联合目标。 |
| 非饱和损失（Non-saturating loss） | “修正” | 用 `-log D(G(z))` 代替 `log(1 - D(G(z)))` 给 `G`。 |
| 模式崩溃（Mode collapse） | “G 记住一种” | 生成器产出极少不同样本，数据多样情况下表现单一。 |
| WGAN | “Wasserstein” | 用地球移动者距离和梯度惩罚代替 BCE，梯度更平滑。 |
| 谱归一化（Spectral norm） | “Lipschitz 招数” | 约束判别器权重范数，稳定训练。 |
| StyleGAN | “效果最好的” | 映射网络加 AdaIN；人脸生成领域顶尖，2026 仍然流行。 |

## 生产注意：一次性推理是 GAN 持久优势

GAN 生成质量不再是开放域的冠军，但推理成本仍领先。

在生产推理角度，GAN 具备：

- **无预填充与无解码阶段。** 一个 `G(z)` 前向过程。TTFT（端到端时延）≈ 总时延。
- **无 KV-cache 压力。** 唯一状态为权重。批大小受激活内存限制，不受缓存限制。
- **轻松的连续批处理。** 每个请求计算量相同，服务器目标负载的静态批次通常最优。无需实时调度。

这就是为什么 GAN 蒸馏（SDXL-Turbo、SD3-Turbo、ADD、LCM）是 2026 年快速文字到图像的主流方案：它将 20-50 步扩散流程折叠成 1-4 步 GAN 风格前向，同时保持扩散基底的分布。对抗损失作为训练时的调节手段，助力将慢生成器变快。

## 延伸阅读

- [Goodfellow 等 (2014). Generative Adversarial Nets](https://arxiv.org/abs/1406.2661) — 原始 GAN 论文。
- [Radford 等 (2015). Unsupervised Representation Learning with DCGAN](https://arxiv.org/abs/1511.06434) — 首个稳定架构。
- [Arjovsky, Chintala, Bottou (2017). Wasserstein GAN](https://arxiv.org/abs/1701.07875) — WGAN。
- [Miyato 等 (2018). Spectral Normalization for GANs](https://arxiv.org/abs/1802.05957) — 谱归一化。
- [Karras 等 (2020). Analyzing and Improving the Image Quality of StyleGAN](https://arxiv.org/abs/1912.04958) — StyleGAN2。
- [Karras 等 (2021). Alias-Free Generative Adversarial Networks](https://arxiv.org/abs/2106.12423) — StyleGAN3。
- [Sauer 等 (2023). Adversarial Diffusion Distillation](https://arxiv.org/abs/2311.17042) — SDXL-Turbo。
