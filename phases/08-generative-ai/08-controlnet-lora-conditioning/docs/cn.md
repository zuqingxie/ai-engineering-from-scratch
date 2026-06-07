# ControlNet、LoRA 与条件控制

> 仅靠文本是一个笨拙的控制信号。ControlNet 让你克隆一个预训练的扩散模型，并通过深度图、姿态骨架、涂鸦或边缘图来引导它。LoRA 让你通过训练 1000 万参数对一个20亿参数的模型进行微调。它们共同把 Stable Diffusion 从一个玩具变成 2026 年每个机构都部署的图像处理流水线。

**类型：** 构建  
**语言：** Python  
**前置知识：** 阶段 8 · 07（潜在扩散 Latent Diffusion），阶段 10（从头训练大语言模型 LLM — LoRA 基础）  
**时间：** 约 75 分钟

## 问题

一个提示词比如“一位穿红裙的女士在繁忙街道遛狗”并不能告诉模型*狗在哪里*，*女士处于什么姿态*，或者*街道的视角*。文本只能锁定你需要指定的图像内容的大约10%。其余部分是视觉上的，无法高效用语言描述。

为每种信号（姿态、深度、Canny 边缘、分割）从零训练一个新的条件模型是非常昂贵的。你希望保持 26 亿参数的 SDXL 主干网冻结，附加一个读取条件信息的小侧网络，并让它推动主干网的中间特征。这就是 ControlNet。

你还想在不重训练整模型的情况下教会模型新概念（你的脸、你的产品、你的风格）。你想要一个小 100 倍差异的调节层。这就是 LoRA — 插入到已有注意力权重中的低秩适配器。

ControlNet + LoRA + 文本 = 2026 年从业者的工具包。大多数生产图像流水线在 SDXL / SD3 / Flux 基础上叠加 2-5 个 LoRA，1-3 个 ControlNet，以及一个 IP-Adapter。

## 概念

![ControlNet 克隆编码器；LoRA 添加低秩增量](../assets/controlnet-lora.svg)

### ControlNet（Zhang 等，2023）

采用一个预训练的 SD。*克隆* U-Net 的编码器半侧。冻结原始编码器。训练克隆以接受额外的条件输入（边缘、深度、姿态）。使用*零卷积*跳跃连接（1×1 卷积初始化为零——初始时为无操作，学习一个增量）将克隆连接回原始解码器半侧。

```text
SD U-Net 解码器:   ... ← orig_enc_features + zero_conv(controlnet_enc(condition))
```

零卷积初始化意味着 ControlNet 初始为恒等映射——训练前不会损害效果。用 100 万个（三元组提示、条件、图像）对及标准扩散损失训练。

每种模态的 ControlNet 作为小型侧模型发布（SDXL 约 360M，SD 1.5 约 70M参数）。它们在推理时可组合：

```text
features += weight_a * control_a(depth) + weight_b * control_b(pose)
```

### LoRA（Hu 等，2021）

对于模型中任意线性层 `W ∈ R^{d×d}`，冻结 `W` 并添加一个低秩增量：

```text
W' = W + ΔW,  ΔW = B @ A,  A ∈ R^{r×d},  B ∈ R^{d×r}
```

其中 `r << d`。注意力中秩 `4-16` 为标准，重微调中可达 `64-128`。新增参数数量为 `2 · d · r`，而非 `d²`。对于参数维度为 `d=640`， `r=16` 的 SDXL 注意力：每个适配器 20k 参数对比 410k，减少 20 倍。整个模型看，LoRA 通常为 20-200MB，而基础模型为 5GB。

推理时可缩放 LoRA：`W' = W + α · B @ A`，通常 `α = 0.5-1.5`。多个 LoRA 以加法堆叠（需注意它们会以非线性方式互动）。

### IP-Adapter（Ye 等，2023）

一个接受*图像*作为条件（联动文本）的微小适配器。使用 CLIP 图像编码器生成图像 tokens，并将其注入交叉注意力以联动文本 tokens。约 20MB 大小每个基础模型。能够让你“以此参考图像风格生成图”而不需要 LoRA。

## 可组合矩阵

| 工具 | 控制内容 | 大小 | 何时使用 |
|------|----------|------|----------|
| ControlNet | 空间结构（姿态、深度、边缘） | 70-360MB | 精确布局、构图 |
| LoRA | 风格、主体、概念 | 20-200MB | 个性化、风格 |
| IP-Adapter | 参考图像的风格或主体 | 20MB | 无文本描述的视觉外观 |
| Textual Inversion | 单一概念作为新词元 | 10KB | 已落后，大多被 LoRA 替代 |
| DreamBooth | 对特定主体完整微调 | 2-5GB | 强身份信息、高计算资源 |
| T2I-Adapter | 轻量级 ControlNet 替代 | 70MB | 边缘设备、推理预算有限 |

ControlNet ≈ 空间控制，LoRA ≈ 语义控制。两者并用。

## 构建它

`code/main.py` 演示两种机制于 1D 例子：

1. **LoRA。** 冻结一个预训练线性层 `W`，训练一个低秩 `B @ A`，使得 `W + BA` 匹配目标线性层。展示秩为 `r=1` 就足够学习一个秩-1 的完全校正。

2. **ControlNet-lite。** 一个“冻结基础”预测器和一个读取额外信号的“侧网络”。侧网络输出由初始化为零的可学习标量门控（类似零卷积）。训练时观察门值逐渐上升。

### 第 1 步：LoRA 数学

```python
def lora(W, A, B, x, alpha=1.0):
    # W 是冻结的；A、B 是可训练的低秩因子。
    return [W[i][j] * x[j] for i, j in ...] + alpha * (B @ (A @ x))
```

### 第 2 步：零初始化侧网络

```python
side_out = control_net(x, condition)
gated = gate * side_out  # gate 初始为0
h = base(x) + gated
```

第 0 步输出与基础相同。训练初期 gate 缓慢更新——无灾难性漂移。

## 陷阱

- **LoRA 过度缩放。** `α = 2` 或 `α = 3` 是常见的“增强强度”hack，导致过度风格化或坏输出。保持 `α ≤ 1.5`。
- **ControlNet 权重冲突。** 姿态 ControlNet 权重 1.0 + 深度 ControlNet 权重 1.0 通常导致过强。权重和约等于 1 是安全默认。
- **LoRA 使用错误基础模型。** SDXL LoRA 对 SD 1.5 会静默失效，因注意力维度不匹配。Diffusers 0.30+ 会提示警告。
- **Textual Inversion 漂移。** 在不同检查点上训练的词元漂移严重。LoRA 更可移植。
- **LoRA 权重合并与存储。** 可将 LoRA 烘焙进基础模型权重以加速推理（无运行时加法），但失去运行时缩放 `α` 的能力。两版本都保留。

## 使用它

| 目标 | 2026 年流水线 |
|------|---------------|
| 复刻品牌艺术风格 | LoRA 在 ~30 张精选图像上的秩 32 训练 |
| 把我的脸放进生成图 | DreamBooth 或 LoRA + IP-Adapter-FaceID |
| 特定姿态 + 提词 | ControlNet-Openpose + SDXL + 文本 |
| 深度感知组合 | ControlNet-Depth + SD3 |
| 参考图 + 文本提示 | IP-Adapter + 文本 |
| 精确布局 | ControlNet-Scribble 或 ControlNet-Canny |
| 背景替换 | ControlNet-Seg + 修补（第 09 课） |
| 快速一步风格化 | LCM-LoRA on SDXL-Turbo |

## 交付它

保存为 `outputs/skill-sd-toolkit-composer.md`。Skill 接受一个任务（输入资产：提示词，可选参考图，可选姿态，可选深度，可选涂鸦），输出工具栈、权重以及可复现的随机种子协议。

## 练习

1. **简单。** 在 `code/main.py` 中，将 LoRA 秩 `r` 从 1 调到 4。在哪个秩上 LoRA 可以完美匹配秩为2的目标增量？
2. **中等。** 训练两个对不同目标变换的独立 LoRA。加载它们并展示它们的加法交互。交互在哪种情况下破坏线性？
3. **高级。** 用 diffusers 堆叠：SDXL-base + Canny-ControlNet（权重 0.8）+ 一个风格 LoRA（α 0.8）+ IP-Adapter（权重 0.6）。随权重变化测量 FID 与提示一致性之间的权衡。

## 关键词

| 术语 | 人们说 | 实际含义 |
|------|--------|----------|
| ControlNet | “空间控制” | 克隆的编码器 + 零卷积跳跃；读取条件图像。 |
| 零卷积 | “初始化为恒等” | 1×1 卷积初始化为零；ControlNet 起始为无操作。 |
| LoRA | “低秩适配器” | `W + B @ A`, `r << d`；参数比完全微调少 100 倍。 |
| 秩 r | “调节旋钮” | LoRA 压缩；典型 4-16，重个性化用 64+。 |
| α | “LoRA 强度” | LoRA 增量的运行时缩放因子。 |
| IP-Adapter | “参考图像” | 通过 CLIP 图像 tokens 进行图像条件适配器。 |
| DreamBooth | “完整主体微调” | 在 ~30 张主体图像上训练模型全权重。 |
| Textual Inversion | “新词元” | 仅学习新词向量；遗留方法，绝大多数被 LoRA 替代。 |

## 生产笔记：LoRA 热替换，ControlNet 通道，多租户服务

真实文本生成图像 SaaS 在同一基础检查点上服务数百个 LoRA 和若干 ControlNet。服务问题很像 LLM 多租户（相关生产文献涵盖 LLM 用连续批次和 LoRAX / S-LoRA）：

- **热替换 LoRA，不要合并。** 合并 `W' = W + α·B·A` 至基础模型可提升 ~3-5% 推理速度，但会冻结 `α` 和基础层。LoRA 保持热加载在显存，作为秩-r 增量；diffusers 通过 `pipe.load_lora_weights()` 和 `pipe.set_adapters([...], adapter_weights=[...])` 实现按请求激活。切换成本为 `2 · d · r · 层数` 权重，MB 级别，亚秒延迟。
- **ControlNet 作为第二注意力通路。** 克隆编码器与基础并行运行。每个权重为1的 ControlNet 等于每步额外两次前向，不是一次合并。批大小容量二次下降。预算活跃 ControlNet 每个约 1.5 倍步伐成本。
- **量化 LoRA 也适用。** 如你量化了基础（见第 07 课，Flux 8GB 上跑），LoRA 增量也能干净量化为 8-bit 或 4-bit。QLoRA 样加载允许你在4-bit Flux基础上堆叠5-10 个 LoRA 而不爆内存。

Flux 专用：Niels 的 Flux-on-8GB 笔记本将基础量化至 4-bit；叠加一个风格 LoRA（`pipe.load_lora_weights("user/style-lora")`）于此量化基础上（权重名 `"pytorch_lora_weights.safetensors"`）依然有效。这是大多数 SaaS 机构 2026 年发布的方案。

## 拓展阅读

- [Zhang, Rao, Agrawala (2023). Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.05543) — ControlNet。  
- [Hu et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — LoRA（最初应用于 LLM，后移植至扩散）。  
- [Ye et al. (2023). IP-Adapter: Text Compatible Image Prompt Adapter](https://arxiv.org/abs/2308.06721) — IP-Adapter。  
- [Mou et al. (2023). T2I-Adapter: Learning Adapters to Dig Out More Controllable Ability](https://arxiv.org/abs/2302.08453) — ControlNet 的轻量替代。  
- [Ruiz et al. (2023). DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation](https://arxiv.org/abs/2208.12242) — DreamBooth。  
- [HuggingFace Diffusers — ControlNet / LoRA / IP-Adapter 文档](https://huggingface.co/docs/diffusers/training/controlnet) — 参考流水线。
