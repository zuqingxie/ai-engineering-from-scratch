# 扩散模型 — 从头实现 DDPM

> Ho、Jain、Abbeel（2020）给这个领域带来了一个戒不掉的配方。在一千个小步骤中用噪声破坏数据。训练一个神经网络去预测噪声。在推理时反向进行这个过程。如今，每个主流的图像、视频、3D 和音乐模型都运行在这个循环上，可能还会加上流匹配（flow matching）或一致性（consistency）技巧。

**类型：** 构建  
**语言：** Python  
**先决条件：** 第3阶段 · 02（反向传播），第8阶段 · 02（变分自编码器 VAE）  
**时间：** ~75 分钟

## 问题

你想要从 `p_data(x)` 中采样。GAN 走的是往往会发散的极大极小游戏。VAE 产生的是来自高斯解码器的模糊样本。你真正想要的是一个训练目标，它是：(a) 单一稳定的损失（无鞍点，无极大极小），(b) `log p(x)` 的下界（因此有似然），以及 (c) 匹配最先进水平（SOTA）质量的样本。

Sohl-Dickstein 等人（2015）给出了理论上的答案：定义一个马尔可夫链 `q(x_t | x_{t-1})`，逐渐加高斯噪声，并训练一个反向链 `p_θ(x_{t-1} | x_t)` 来去噪。Ho、Jain、Abbeel（2020）展示损失可以简化为一句话——预测噪声——并理清了数学。在2020年这还只是个好奇的技术。2021 年给出了最先进的样本。2022 年成为了 Stable Diffusion。到 2026 年，它是通用基础。

## 概念

![DDPM：正向加噪声，反向去噪](../assets/ddpm.svg)

**正向过程 `q`。** 在 `T` 个小步骤中逐步加入高斯噪声。封闭形式 —— 数学可行的关键 —— 是累计步骤仍然是高斯分布：

```text
q(x_t | x_0) = N( sqrt(α̅_t) · x_0,  (1 - α̅_t) · I )
```

其中 `α̅_t = ∏_{s=1..t} (1 - β_s)`，`β_t` 是调度参数。在 T=1000 步中线性选取 `β_t` 介于 1e-4 到 0.02，`x_T` 大致服从 `N(0, I)`。

**反向过程 `p_θ`。** 学习神经网络 `ε_θ(x_t, t)` 来预测加上的噪声。给定 `x_t`，进行去噪：

```text
x_{t-1} = (1 / sqrt(α_t)) · ( x_t - (β_t / sqrt(1 - α̅_t)) · ε_θ(x_t, t) )  +  σ_t · z
```

其中 `σ_t` 是 `sqrt(β_t)` 或一个学习出来的方差。表达式看起来复杂，其实就是代数变形 —— 依据后验 `q(x_{t-1} | x_t, x_0)` 求解 `x_{t-1}`，并用噪声预测的 `x_0` 替代真实值。

**训练损失。**

```text
L_simple = E_{x_0, t, ε} [ || ε - ε_θ( sqrt(α̅_t) · x_0 + sqrt(1 - α̅_t) · ε,  t ) ||² ]
```

从数据中采样 `x_0`，随机选取 `t`，从 `N(0, I)` 中采样 `ε`，用封闭式一次性计算加噪后的 `x_t`，然后回归预测噪声。一条损失，无极大极小，无 KL，没重参数技巧。

**采样。** 从 `x_T ~ N(0, I)` 开始，从 `t = T` 到 `1` 反复逆向执行。完成。

## 为什么有效

三个直觉：

1. **去噪容易；生成难。** 在 `t=T`，数据是纯噪声，网络解决一个简单问题。在 `t=0`，只需要修正几个像素，中间 `t` 很难，但网络对所有噪声级别都有梯度传导。
2. **隐含的分数匹配。** Vincent（2011）证明预测噪声等价于估计 `∇_x log q(x_t | x_0)`，即 *score*。反向随机微分方程用这个分数沿密度梯度上升——引导随机游走去高概率区。
3. **ELBO 简化为简单 MSE。** 完整变分下界对每步有 KL 项。DDPM 参数化下，这些 KL 化简为在噪声预测上的 MSE（带特定系数）；Ho 去掉系数（称为“简单”损失）后质量反而提升。

## 构建它

`code/main.py` 实现了一个一维 DDPM。数据是双模态混合。网络是接受 `(x_t, t)` 输入的微小 MLP，输出噪声预测。训练是一行损失。采样迭代反向链。

### 步骤 1：正向调度（封闭式）

```python
betas = [1e-4 + (0.02 - 1e-4) * t / (T - 1) for t in range(T)]
alphas = [1 - b for b in betas]
alpha_bars = []
cum = 1.0
for a in alphas:
    cum *= a
    alpha_bars.append(cum)
```

### 步骤 2：一次性采样 `x_t`

```python
def forward_sample(x0, t, alpha_bars, rng):
    a_bar = alpha_bars[t]
    eps = rng.gauss(0, 1)
    x_t = math.sqrt(a_bar) * x0 + math.sqrt(1 - a_bar) * eps
    return x_t, eps
```

### 步骤 3：一次训练步骤

```python
def train_step(x0, model, alpha_bars, rng):
    t = rng.randrange(T)
    x_t, eps = forward_sample(x0, t, alpha_bars, rng)
    eps_hat = model_forward(model, x_t, t)
    loss = (eps - eps_hat) ** 2
    return loss, gradient_step(model, ...)
```

### 步骤 4：反向采样

```python
def sample(model, alpha_bars, T, rng):
    x = rng.gauss(0, 1)
    for t in range(T - 1, -1, -1):
        eps_hat = model_forward(model, x, t)
        beta_t = 1 - alphas[t]
        x = (x - beta_t / math.sqrt(1 - alpha_bars[t]) * eps_hat) / math.sqrt(alphas[t])
        if t > 0:
            x += math.sqrt(beta_t) * rng.gauss(0, 1)
    return x
```

对于一维问题，40 步和 24 单元的 MLP，大约 200 个周期能学出双模态分布。

## 时间条件

网络需要知道当前去噪的时间步。有两种标准方式：

- **正弦嵌入。** 类似 Transformer 位置编码。`embed(t) = [sin(t/ω_0), cos(t/ω_0), sin(t/ω_1), ...]`。通过 MLP，广播到网络各层。
- **Film / group-norm 条件。** 将嵌入投影成每通道缩放/偏置（FiLM）参数，应用于每个块。

我们的示例代码用正弦嵌入并拼接。生产环境中的 U-Net 用 FiLM。

## 陷阱

- **调度非常关键。** 线性 `β` 是 DDPM 默认，但余弦调度（Nichol & Dhariwal，2021）用同样计算量可得到更优 FID。若质量停滞建议切换调度。
- **时间步嵌入易出错。** 直接传浮点值 `t` 适合一维玩具，不适合图像；应始终用合适的嵌入。
- **V 预测 vs ε 预测。** 极窄时间步（非常小或非常大 t）时，ε 信噪比差。V 预测（`v = α·ε - σ·x`）更稳定；SDXL、SD3 和 Flux 使用它。
- **无分类器引导。** 推理时同时计算条件和无条件 ε，然后用 `ε_cfg = (1 + w) · ε_cond - w · ε_uncond`，`w ≈ 3-7`。第 08 课覆盖。
- **1000 步很多。** 生产用 DDIM（20-50 步）、DPM-Solver（10-20 步）、蒸馏（1-4 步）。见第 12 课。

## 使用场景

| 角色 | 2026 年典型技术栈 |
|------|---------------------|
| 图像像素空间扩散（小型玩具） | DDPM + U-Net |
| 图像潜变量扩散 | VAE 编码器 + U-Net 或 DiT（第 07 课） |
| 视频潜变量扩散 | 时空 DiT（Sora、Veo、WAN） |
| 音频潜变量扩散 | Encodec + 扩散 Transformer |
| 科学领域（分子、蛋白质、物理） | 等变扩散（EDM、RFdiffusion、AlphaFold3） |

扩散是通用生成骨干。流匹配（第 13 课）则是 2024-2026 年的竞争者，通常在相同性能下推理速度更快。

## 部署它

保存 `outputs/skill-diffusion-trainer.md`。Skill 接收数据集 + 计算预算，输出：调度（线性/余弦/Sigmoid），预测目标（ε/v/x），步骤数，指导强度，采样器家族，评估协议。

## 练习

1. **简单。** 将 `code/main.py` 中的 T 从 40 改为 10。样本质量（输出的可视化直方图）如何退化？两模态结构在哪个 T 下崩溃？
2. **中等。** 切换从 ε 预测到 v 预测。重新推导反向步骤。比较最终样本质量。
3. **困难。** 加入无分类器引导。训练时以 10% 概率丢弃类别标签 `c ∈ {0, 1}`，采样时用 `ε = (1+w)·ε_cond - w·ε_uncond`。测量条件模态命中率，w 取值为 0、1、3、7。

## 关键词

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| 正向过程 | “加噪声” | 固定马尔可夫链 `q(x_t \| x_{t-1})`，破坏数据。 |
| 反向过程 | “去噪” | 学习链 `p_θ(x_{t-1} \| x_t)`，重建数据。 |
| β 调度 | “噪声阶梯” | 每步方差；线性、余弦或 Sigmoid。 |
| α̅ | “Alpha bar” | 累积乘积 `∏(1 - β)`；给出封闭式 `x_t` 从 `x_0`。 |
| 简单损失 | “噪声 MSE” | `\|\|ε - ε_θ(x_t, t)\|\|²`；所有变分推导都简化到这里。 |
| ε 预测 | “预测噪声” | 输出是加的噪声；标准 DDPM。 |
| V 预测 | “预测速度” | 输出 `α·ε - σ·x`；跨 t 更稳定。 |
| DDPM | “论文名” | Ho 等，2020 年；线性 β，1000 步，U-Net。 |
| DDIM | “确定性采样” | 非马尔可夫采样，20-50 步，同训练目标。 |
| 无分类器引导 | “CFG” | 混合条件与无条件噪声预测以增强条件影响。 |

## 生产注意事项：扩散推理是步骤数问题

DDPM 论文跑 T=1000 反向步数。没有人生产用这么多步。真实推理栈选用三种策略之一 —— 每种都能清晰对应“延迟来源”的生产框架：

1. **更快采样器，相同模型。** DDIM（20-50 步）、DPM-Solver++（10-20 步）、UniPC（8-16 步）。反向循环的替代，无需改动训练好的 `ε_θ` 权重。加速 20-50 倍延迟。
2. **蒸馏。** 训练学生模型用更少步数匹配教师：渐进蒸馏（2 → 1 步），一致性模型（任意 → 1-4 步），LCM，SDXL-Turbo，SD3-Turbo。再加速 5-10 倍，需要重新训练。
3. **缓存与编译。** `torch.compile(unet, mode="reduce-overhead")`，TensorRT-LLM 的扩散后端，`xformers` / SDPA 注意力，加 bf16 权重。每步延迟减半。可与 (1) 和 (2) 叠加。

生产扩散服务器的预算讨论跟 LLM 文献相同：延迟是 `num_steps × step_cost + VAE_decode`，吞吐是 `batch_size × (num_steps × step_cost)^-1`。TTFT 很小（一步）；TPOT 等价为完整响应时间，因为图像生成对用户来说是“一次性完成”。

## 深入阅读

- [Sohl-Dickstein et al. (2015). Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https://arxiv.org/abs/1503.03585) — 扩散早期论文，领先时代。
- [Ho, Jain, Abbeel (2020). Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) — DDPM。
- [Song, Meng, Ermon (2021). Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) — DDIM，更少步骤。
- [Nichol & Dhariwal (2021). Improved DDPM](https://arxiv.org/abs/2102.09672) — 余弦调度，学习方差。
- [Dhariwal & Nichol (2021). Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233) — 分类器引导。
- [Ho & Salimans (2022). Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) — 无分类器引导。
- [Karras et al. (2022). Elucidating the Design Space of Diffusion-Based Generative Models (EDM)](https://arxiv.org/abs/2206.00364) — 统一符号，最整洁配方。
