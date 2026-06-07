# Flow Matching 和 Rectified Flows（修正流）

> 扩散模型（Diffusion models）需要 20-50 次采样步骤，因为它们沿着弯曲路径从噪声走向数据。Flow matching（Lipman 等，2023）和 rectified flow（Liu 等，2022）训练的是直线路径。更直的路径意味着更少的步骤，从而推理更快。Stable Diffusion 3、Flux.1 和 AudioCraft 2 都在 2024 年转向了 flow matching。

**类型：** 构建  
**编程语言：** Python  
**前置知识：** Phase 8 · 06 (DDPM)，Phase 1 · 微积分（Calculus）  
**时间：** 约 45 分钟

## 问题

DDPM 的逆过程是从 `N(0, I)` 向数据分布做 1000 步的随机游走。DDIM 折叠为 20-50 步的确定性过程。你希望减少步骤——理想情况下只有一步。阻碍是解决逆过程的 ODE 是刚性的（stiff）；路径是弯曲的。

如果你能训练模型，让从噪声到数据的路径是一条*直线*，那么从 `t=1` 直接 Euler 步进到 `t=0` 就可行。Flow matching 直接构建此方案：定义从 `x_1 ∼ N(0, I)` 到 `x_0 ∼ data` 的直线插值，训练向量场 `v_θ(x, t)` 对其时间导数进行匹配，推理时积分即可。

Rectified flow（Liu 2022）更进一步：通过反复 reflow 过程迭代地拉直路径，得到越来越逼近线性的 ODE。经过两次 reflow 迭代，2 步采样器可以匹配 50 步 DDPM 的质量。

## 概念

![Flow matching: straight-line interpolation between noise and data](../assets/flow-matching.svg)

### 直线流

定义：

```text
x_t = t · x_1 + (1 - t) · x_0,   t ∈ [0, 1]
```

其中 `x_0 ~ data`，`x_1 ~ N(0, I)`。沿这条直线的时间导数是常数：

```text
dx_t / dt = x_1 - x_0
```

定义神经向量场 `v_θ(x_t, t)` 并训练其匹配该导数：

```text
L = E_{x_0, x_1, t} || v_θ(x_t, t) - (x_1 - x_0) ||²
```

这就是**条件流匹配**损失（Lipman 2023）。训练无需仿真：无需展开 ODE，只需采样 `(x_0, x_1, t)` 并做回归。

### 采样

推理时，沿时间*逆向*积分学习到的向量场：

```text
x_{t-Δt} = x_t - Δt · v_θ(x_t, t)
```

从 `x_1 ~ N(0, I)` 开始，用 Euler 步逐步降到 `t=0`。

### Rectified flow（Liu 2022）

直线流虽然理论有效，但学习的路径实际上*不是真正直线*，因为许多不同的 `x_0` 会映射到同一个 `x_1`，路径会弯曲。Rectified flow 的 reflow 步骤：

1. 训练流模型 v_1，基于随机配对。
2. 通过积分 v_1，从 `x_1` 采样 N 对 `(x_1, x_0)`。
3. 在这些配对样本上训练 v_2。因为这些配对已“ODE 匹配”，它们之间的直线插值更加平坦。
4. 重复。

实际中，2 次 reflow 迭代即可达近线性，使推理只需 2-4 步。SDXL-Turbo、SD3-Turbo、LCM 都是基于 flow matching 蒸馏得到的模型。

### 2024 年图像领域为何流匹配获胜

三点原因：

1. **无仿真训练**——训练时无须展开 ODE，简单易实现。  
2. **更好的损失几何**——直线路径信号噪声比（SNR）稳定，而 DDPM 的 ε 损失在时间表边缘 SNR 很差。  
3. **更快推理**——SDXL-Turbo 4-8 步质量；一致性蒸馏可实现一步推理。  

## Flow matching 与 DDPM 的精确联系

带高斯条件路径的 flow matching 实际上是*带特定噪声调度的扩散过程*。选取路径 `x_t = α(t) x_0 + σ(t) x_1`，flow matching 恢复 Stratonovich 重写的扩散，且 `v = α'·x_0 - σ'·x_1`。对高斯路径两者代数等价。

flow matching 的创新：目标更明确（普通速度场），损失更纯粹，并且支持非高斯插值的实验。

## 实现它

`code/main.py` 实现了一维 flow matching，目标是两模态高斯混合。向量场 `v_θ(x, t)` 是个小型 MLP，训练目标是直线目标。推理时分别集成 1、2、4 和 20 步 Euler，比较样本质量。

### 第一步：训练损失

```python
def train_step(x0, net, rng, lr):
    x1 = rng.gauss(0, 1)
    t = rng.random()
    x_t = t * x1 + (1 - t) * x0
    target = x1 - x0
    pred = net_forward(x_t, t)
    loss = (pred - target) ** 2
    # 反向传播 + 参数更新
```

### 第二步：多步采样

```python
def sample(net, num_steps):
    x = rng.gauss(0, 1)
    for i in range(num_steps):
        t = 1.0 - i / num_steps
        dt = 1.0 / num_steps
        x -= dt * net_forward(x, t)
    return x
```

### 第三步：对比步数

预期 4 步采样的质量已能匹配 20 步，显著降低延迟。

## 注意事项

- **时间参数化。** Flow matching 用 `t ∈ [0, 1]`，`t=0` 代表数据，`t=1` 代表噪声。DDPM 用 `t ∈ [0, T]`，`t=0` 数据，`t=T` 噪声。方向相同但尺度不同。论文常搞混。  
- **调度选择。** Rectified flow 的直线流是“标准” flow matching 调度，但可用余弦或 logit-normal `t` 采样（SD3 采用）以覆盖更优尺度。  
- **Reflow 成本。** 生成 reflow 配对数据集需对每个样本做完整推理。只有真正需要 1-2 步推理时才做 reflow。  
- **无分类器引导仍适用。** 直接将 ε 换成 v：`v_cfg = (1+w) v_cond - w v_uncond`。

## 使用场景

| 用例 | 2026 技术栈 |
|----------|-----------|
| 文生图，高质量 | Flow matching：SD3，Flux.1-dev |
| 文生图，1-4 步骤 | 蒸馏流匹配：Flux.1-schnell，SD3-Turbo，SDXL-Turbo |
| 实时推理 | 基于 flow-matched 模型的一致性蒸馏（LCM，PCM） |
| 音频生成 | Flow matching：Stable Audio 2.5，AudioCraft 2 |
| 视频生成 | Flow matching 混合扩散（Sora，Veo，Stable Video） |
| 科学/物理（粒子轨迹，分子） | Flow matching + 等变向量场 |

2025-2026 年几乎所有自称“快过扩散”的论文，实际都是 flow matching + 蒸馏。

## 交付

保存为 `outputs/skill-fm-tuner.md`。Skill 会将扩散模型规范转换为 flow matching 训练配置：调度选择、时间采样分布（均匀 / logit-normal）、优化器、reflow 计划、目标步数、评估协议。

## 练习

1. **简单。** 运行 `code/main.py`，比较 1 步和 20 步 MSE 与真实数据分布。  
2. **中级。** 从均匀采样 `t` 切换到 logit-normal（集中采样于中间 `t`），模型质量是否提升？  
3. **困难。** 实现一次 reflow 迭代：通过第一个模型积分产生 `(x_0, x_1)` 配对，基于配对训练第二个模型，比较 1 步采样质量。

## 关键词

| 术语 | 常说法 | 实际含义 |
|------|--------|----------|
| Flow matching | “直线扩散” | 训练 `v_θ(x, t)` 以匹配插值 `x_1 - x_0`。 |
| Rectified flow | “reflow” | 迭代使学习的流逐渐变直的过程。 |
| Velocity field（速度场） | “v_θ” | 模型输出——移动 `x_t` 的方向。 |
| Straight-line interpolant（直线插值） | “路径” | `x_t = (1-t)·x_0 + t·x_1`；目标导数简单。 |
| Euler sampler | “一阶 ODE 求解器” | 最简单积分器，路径直时效果好。 |
| Logit-normal t | “SD3 采样” | 将 `t` 采样集中在中间梯度最强区域。 |
| Consistency distillation | “一步采样器” | 训练学生网络直接映射任意 `x_t` 到 `x_0`。 |
| CFG with velocity | “v-CFG” | `v_cfg = (1+w) v_cond - w v_uncond`；同策略新变量。 |

## 生产备注：Flux.1-schnell 是最快的 flow matching 实现

Flow matching 的生产优势在于 Flux.1-schnell——这是一个 flow-matched DiT，通过蒸馏降至 1-4 推理步，质量保持 Flux-dev 级别。Niels 的“在 8GB 机器上运行 Flux”笔记是参考部署方案：T5 + CLIP 编码，量化 MMDiT 去噪（schnell 用 4 步，开发版用 50 步），VAE 解码。资源成本对比：

| 变体 | 步数 | 1024² 分辨率 L4 显存延迟 | 总 FLOPs（相对） |
|---------|-------|------------------------|------------------|
| Flux.1-dev（原始） | 50 | ~15 秒 | 1.0× |
| Flux.1-schnell | 4 | ~1.2 秒 | 0.08×（快 12 倍） |
| SDXL-base | 30 | ~4 秒 | 0.25× |
| SDXL-Lightning 2步 | 2 | ~0.3 秒 | 0.03× |

生产准则：**flow-matched 基础 + 蒸馏 = 2026 年快速文生图的默认方案。**所有大厂都采用这套组合：SD3-Turbo（SD3+flow+蒸馏）、Flux-schnell（Flux-dev+rectified-flow 拉直）、CogView-4-Flash。纯扩散基础目前仅用于旧版检查点。

## 延伸阅读

- [Liu, Gong, Liu (2022). Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow](https://arxiv.org/abs/2209.03003) — rectified flow。  
- [Lipman et al. (2023). Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) — flow matching。  
- [Esser et al. (2024). Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206) — SD3、rectified flow 大规模应用。  
- [Albergo, Vanden-Eijnden (2023). Stochastic Interpolants](https://arxiv.org/abs/2303.08797) — 通用框架，涵盖 FM + 扩散。  
- [Song et al. (2023). Consistency Models](https://arxiv.org/abs/2303.01469) — 扩散/流的一步蒸馏。  
- [Sauer et al. (2023). Adversarial Diffusion Distillation (SDXL-Turbo)](https://arxiv.org/abs/2311.17042) — turbo 版本。  
- [Black Forest Labs (2024). Flux.1 models](https://blackforestlabs.ai/announcing-black-forest-labs/) — 生产环境中的 flow matching。
