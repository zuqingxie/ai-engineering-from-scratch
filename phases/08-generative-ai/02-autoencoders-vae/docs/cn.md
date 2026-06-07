# Autoencoders & Variational Autoencoders (VAE)

> 一个普通的自编码器先压缩再重构。它会记忆，但不会生成。加一个技巧——强制代码呈现高斯分布——你就得到了一个采样器。这个技巧，即 `z = μ + σ·ε` 的重参数化，是为什么你在 2026 年使用的每个潜变量扩散（latent-diffusion）和流匹配（flow-matching）图像模型输入端都有 VAE 的原因。

**类型：** 构建  
**语言：** Python  
**先决条件：** 第3阶段 · 02（反向传播）、第3阶段 · 07（卷积神经网络 CNNs）、第8阶段 · 01（分类法 Taxonomy）  
**时间：** ~75 分钟

## 问题

将784像素的MNIST数字压缩成16维代码，然后重构。普通自编码器能实现极好的重构均方误差（MSE），但代码空间非常杂乱。随机选择代码空间中的一点，解码后只能得到噪声。它没有采样器。只是一个伪装的压缩模型。

你真正想要的是：(a) 代码空间是一个干净、平滑且可采样的分布——比如各向同性高斯分布 `N(0, I)`；(b) 解码任何采样都生成合理的数字；(c) 编码器和解码器仍能良好地压缩。三目标，一个架构，一种损失。

Kingma 2013 年的 VAE 解决了这个问题：训练编码器输出一个*分布* `q(z|x) = N(μ(x), σ(x)²)`，通过 KL 惩罚将该分布拉向先验分布 `N(0, I)`，然后从 `q(z|x)` 采样 `z` 再解码。推断时丢弃编码器，从 `N(0, I)` 采样 `z`，解码。KL 惩罚促使代码空间有结构。

到 2026 年，VAE 罕有单独发布——在原始图像质量方面已被扩散模型超越——但它是每个潜变量扩散模型（Stable Diffusion 1/2/XL/3、Flux、AudioCraft）的首选编码器。学习 VAE 就等于学习你用的每个图像流水线的无形第一层。

## 概念

![Autoencoder vs VAE: the reparameterization trick](../assets/vae.svg)

**自编码器（Autoencoder）。** `z = encoder(x)`，`x̂ = decoder(z)`，损失 = `||x - x̂||²`。编码空间无结构。

**VAE 编码器。** 输出两个向量：`μ(x)` 和 `log σ²(x)`。定义了分布 `q(z|x) = N(μ, diag(σ²))`。

**重参数化技巧（Reparameterization trick）。** 从 `q(z|x)` 采样不可微。重写采样为 `z = μ + σ·ε`，其中 `ε ~ N(0, I)`。现在 `z` 是 `(μ, σ)` 的确定性函数加上非参数噪声 —— 梯度可通过 `μ` 和 `σ` 流动。

**损失。** 证据下界（ELBO），包含两项：

```text
loss = reconstruction + β · KL[q(z|x) || N(0, I)]
     = ||x - x̂||²  + β · Σ_i ( σ_i² + μ_i² - log σ_i² - 1 ) / 2
```

重构项推动 `x̂` 接近 `x`。KL 项推动 `q(z|x)` 接近先验。两者权衡。小 β (<1) = 更锐利样本，代码空间不那么高斯；大 β (>1) = 代码空间更干净但样本更模糊。β-VAE（Higgins 2017）让该调节项成名，激发了因子解耦（disentanglement）研究。

**采样。** 推断时：从 `N(0, I)` 采样 `z`，经解码器前向。一次前向，不像扩散那样需要迭代采样。

## 构建

`code/main.py` 实现了一个无需 numpy 或 torch 的小型 VAE。输入是来自8维2成分高斯混合模型的合成数据。编码器和解码器为单隐藏层 MLP。实现了 tanh 激活、前向、损失和手写的反向传播。非生产级，教学用。

### 第1步：编码器前向

```python
def encode(x, enc):
    h = tanh(add(matmul(enc["W1"], x), enc["b1"]))
    mu = add(matmul(enc["W_mu"], h), enc["b_mu"])
    log_sigma2 = add(matmul(enc["W_sig"], h), enc["b_sig"])
    return mu, log_sigma2
```

使用 `log σ²` 而非 `σ` 使网络输出不受限（σ 的 softplus 是陷阱 —— 当 σ ≈ 0 梯度消失）。

### 第2步：重参数化并解码

```python
def reparameterize(mu, log_sigma2, rng):
    eps = [rng.gauss(0, 1) for _ in mu]
    sigma = [math.exp(0.5 * lv) for lv in log_sigma2]
    return [m + s * e for m, s, e in zip(mu, sigma, eps)]

def decode(z, dec):
    h = tanh(add(matmul(dec["W1"], z), dec["b1"]))
    return add(matmul(dec["W_out"], h), dec["b_out"])
```

### 第3步：ELBO

```python
def elbo(x, x_hat, mu, log_sigma2, beta=1.0):
    recon = sum((a - b) ** 2 for a, b in zip(x, x_hat))
    kl = 0.5 * sum(math.exp(lv) + m * m - lv - 1 for m, lv in zip(mu, log_sigma2))
    return recon + beta * kl, recon, kl
```

因两个分布皆为高斯，KL 有解析闭式，不用数值积分。即使到2026年，有些代码仍用蒙特卡洛估计KL，速度慢3倍且无必要。

### 第4步：生成

```python
def sample(dec, z_dim, rng):
    z = [rng.gauss(0, 1) for _ in range(z_dim)]
    return decode(z, dec)
```

这就是生成模型，只有5行代码。

## 陷阱

- **后验崩溃（Posterior collapse）。** KL 项太强导致 `q(z|x) → N(0, I)`，`z` 不包含 `x` 信息。解决：β 退火（起始β=0，逐渐升到1），free bits，或跳过不活跃维度上的KL。
- **模糊样本。** 高斯解码器似然对应 MSE 重构，是L2（均值）贝叶斯最优——一组合理数字的均值是模糊数字。解决：用离散解码器（VQ-VAE，NVAE）或仅用 VAE 做编码器，在潜空间堆叠扩散（Stable Diffusion 所为）。
- **β 太大太早。** 同后验崩溃，建议起始β≈0.01再升温。
- **潜空间维度太小。** MNIST 用16维，ImageNet 256²用256维，ImageNet 1024²用2048维。Stable Diffusion VAE从512×512×3压缩到64×64×4（空间面积降采样32倍，通道降采样32倍）。

## 使用方法

2026 年的 VAE 堆栈：

| 场景 | 选择 |
|-----------|------|
| 图像潜空间扩散编码器 | Stable Diffusion VAE（`sd-vae-ft-ema`）或 Flux VAE |
| 音频潜空间编码器 | Encodec（Meta）、SoundStream 或 DAC（Descript） |
| 视频潜空间 | Sora 的时空补丁、Latte VAE、WAN VAE |
| 解耦表示学习 | β-VAE、FactorVAE、TCVAE |
| 离散潜空间（用于Transformer建模） | VQ-VAE、RVQ（ResidualVQ） |
| 连续潜空间生成 | 普通 VAE，再在潜空间条件化流或扩散模型 |

潜变量扩散模型本质上是 VAE + 一个处于编码器和解码器之间的扩散模型。VAE做粗压缩，扩散模型做重活。视频（VAE+视频扩散 DiT）和音频（Encodec + MusicGen Transformer）也是同理。

## 部署

保存 `outputs/skill-vae-trainer.md`。

该技能需要：数据集配置 + 潜空间目标维度 + 下游用途（重构、采样、潜变量扩散输入），输出：架构选择（普通/β/VQ/RVQ）、β调度、潜空间维度、解码器似然（高斯或分类）、评估方案（重构 MSE、每维 KL、`q(z|x)` 与 `N(0, I)` 的 Fréchet 距离）。

## 练习

1. **简单。** 将 `code/main.py` 中的 `β` 改为 `0.01`、`0.1`、`1.0`、`5.0`，记录最终重构 MSE 和 KL。哪一个 β 对你的合成数据是帕累托最优？
2. **中等。** 用伯努利似然（交叉熵损失）替换高斯解码器似然。比较同合成数据二值化版本上的样本质量。
3. **困难。** 将 `code/main.py` 扩展为小型 VQ-VAE：用 K=32 的码本最近邻查找替代连续 `z`。比较重构 MSE，并报告多少码本条目被使用（码本坍缩问题确实存在）。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------|---------|
| Autoencoder（自编码器） | 编码-解码网络 | `x → z → x̂`，学习 MSE。非生成模型。 |
| VAE（变分自编码器） | 带采样器的 AE | 编码器输出分布，KL 惩罚塑造代码空间。 |
| ELBO（证据下界） | 证据下界 | `log p(x) ≥ recon - KL[q(z\|x) \|\| p(z)]`；当 `q = p(z\|x)` 时紧。 |
| Reparameterization（重参数化） | `z = μ + σ·ε` | 随机节点重写为确定性+纯噪声。使采样可反向传播。 |
| Prior（先验） | `p(z)` | 潜空间分布目标，通常为 `N(0, I)`。 |
| Posterior collapse（后验坍缩） | “KL 项获胜” | 编码器忽略 `x`，输出先验；解码器需幻觉生成。 |
| β-VAE | 可调 KL 权重 | `loss = recon + β·KL`。β越大，解耦更多但更模糊。 |
| VQ-VAE | 离散潜空间 | 用码本最近邻替换连续 `z`；适合 Transformer 建模。 |

## 生产备注：VAE 是扩散服务的最热路径

在 Stable Diffusion / Flux / SD3 流水线中，每次请求会调用两次 VAE——一次编码（如 img2img/修复），一次解码。在 1024² 时，解码器前传通常是整个流水线中激活内存峰值最大者，因为它把 `128×128×16` 潜变量上采样到 `1024×1024×3`。两个实务影响：

- **分片或平铺解码。** `diffusers` 支持 `pipe.vae.enable_slicing()` 和 `pipe.vae.enable_tiling()`。平铺换来微小缝隙伪影，但内存由 `O(H·W)` 降为 `O(tile²)`。对1024²以上消费者GPU至关重要。
- **bf16解码器，最终变换用fp32数值。** SD 1.x VAE 最初是 fp32 发布，转换到 fp16 时 *默默产出 NaN* 于 1024² 及以上。SDXL 发布了 `madebyollin/sdxl-vae-fp16-fix` —— 推荐始终使用 fp16-fix 版本或 bf16。

## 深入阅读

- [Kingma & Welling (2013). Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) — VAE 原始论文。  
- [Higgins et al. (2017). β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/forum?id=Sy2fzU9gl) — 解耦的 β-VAE。  
- [van den Oord et al. (2017). Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937) — VQ-VAE。  
- [Vahdat & Kautz (2021). NVAE: A Deep Hierarchical Variational Autoencoder](https://arxiv.org/abs/2007.03898) — 最先进的图像 VAE。  
- [Rombach et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) — Stable Diffusion；VAE 作为编码器。  
- [Défossez et al. (2022). High Fidelity Neural Audio Compression](https://arxiv.org/abs/2210.13438) — Encodec，音频 VAE 标准。
