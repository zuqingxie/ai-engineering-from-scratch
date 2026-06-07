# 潜在扩散与稳定扩散

> 在 512×512 图像上的像素空间扩散是计算上的罪行。Rombach 等人（2022）注意到，生成一张图像并不需要全部 78.6 万维——只需捕捉语义结构的足够维度，剩下的部分由一个单独的解码器处理。在 VAE 的潜在空间中运行扩散。这就是 Stable Diffusion（稳定扩散）的核心。

**类型：** 构建  
**语言：** Python  
**先决条件：** 第8阶段 · 02（VAE）、第8阶段 · 06（DDPM）、第7阶段 · 09（ViT）  
**时间：** 约75分钟  

## 问题

512² 像素空间扩散意味着 U-Net 运行在形状为 `[B, 3, 512, 512]` 的张量上。对于一个5亿参数的 U-Net，每步采样约需100 GFLOPS。50步采样便是每张图像5 TFLOPS。训练十亿张图像，计算资源消耗巨大到荒谬。

大部分 FLOPS 都花在了推动感知上不重要的细节上——那些高频纹理，这是一个有损 VAE 可以压缩掉的内容。Rombach 的想法是：训练一个VAE（*第一阶段*），冻结它，并完全在4通道64×64的潜在空间中运行扩散（*第二阶段*）。同样的U-Net，像素数减少16倍，计算量减少约64倍，质量相当。

这就是 Stable Diffusion 的配方。SD 1.x / 2.x 使用860M参数的U-Net在 `64×64×4` 潜在空间上，SDXL 采用2.6B参数的U-Net在 `128×128×4`，SD3 用扩散 Transformer（Diffusion Transformer, DiT）替代了 U-Net 并结合流匹配。Flux.1-dev（Black Forest Labs，2024）发布了12B参数的DiT-MMDiT。它们都运行在相同的双阶段基础架构上。

## 概念

![潜在扩散：VAE 压缩 + 潜在空间扩散](../assets/latent-diffusion.svg)

**两阶段，分别训练。**

1. **第一阶段——VAE。** 编码器 `E(x) → z`，解码器 `D(z) → x`。目标是压缩：每个空间轴下采样8倍，调整通道使得潜在空间总大小约为像素数的1/16。损失函数=重建误差（L1 + LPIPS 感知）+ KL 散度（权重较小，避免`z`过于高斯化，因为不需要精确采样）。通常还会结合对抗损失以保证解码图像的清晰度。

2. **第二阶段——在 `z` 上扩散。** 把 `z = E(x_real)` 作为数据。训练一个 U-Net（或 DiT）对 `z_t` 进行去噪。推理时：先用扩散采样 `z_0`，然后 `x = D(z_0)`。

**文本条件控制。** 两个额外组件：一个冻结的文本编码器（SD 1.x 用 CLIP-L，SD 2/XL 用 CLIP-L+OpenCLIP-G，SD3 和 Flux 用 T5-XXL），以及一个交叉注意力注入机制：每个 U-Net 模块都会接收 `[Q=图像特征, K=V=文本标记]` 并混合它们。文本通过这些标记唯一影响图像。

**损失函数同第06课相同。** 使用相同的 DDPM / 流匹配均方误差（MSE）噪声目标。仅仅换了数据域。

## 架构变体

| 模型        | 年份  | 主体架构   | 潜在形状       | 文本编码器                | 参数量     |
|-------------|-------|------------|----------------|---------------------------|------------|
| SD 1.5      | 2022  | U-Net      | 64×64×4        | CLIP-L（77 tokens）        | 860M       |
| SD 2.1      | 2022  | U-Net      | 64×64×4        | OpenCLIP-H                | 865M       |
| SDXL        | 2023  | U-Net + 精修器 | 128×128×4    | CLIP-L + OpenCLIP-G       | 2.6B + 6.6B |
| SDXL-Turbo  | 2023  | 蒸馏模型   | 128×128×4      | 同上                      | 1-4步采样   |
| SD3         | 2024  | MMDiT（多模态 DiT） | 128×128×16 | T5-XXL + CLIP-L + CLIP-G  | 2B / 8B    |
| Flux.1-dev  | 2024  | MMDiT      | 128×128×16     | T5-XXL + CLIP-L           | 12B        |
| Flux.1-schnell | 2024 | MMDiT 蒸馏 | 128×128×16    | T5-XXL + CLIP-L           | 12B，1-4步 |

趋势：用 DiT（即潜在块上的 Transformer）代替 U-Net，规模更大的文本编码器（T5 在提示依从性上优于 CLIP），增加潜在通道数（从4到16，提供更多细节空间）。

## 构建

`code/main.py` 堆叠了一个玩具版的一维“VAE”（身份编码器加解码器，仅示范；真实VAE是卷积网络）在第06课的 DDPM 之上，并添加了无分类器引导的分类条件控制。该示例显示扩散损失在原始一维数据和编码数据两者上都适用——这是关键洞察。

### 步骤 1：编码器/解码器

```python
def encode(x):    return x * 0.5          # 玩具“压缩”到更小的尺度
def decode(z):    return z * 2.0
```

真实的 VAE 有训练权重。为了教学，线性映射足够展示扩散在 `z` 上运行，而不关心原始数据空间。

### 步骤 2：在 `z` 空间中扩散

与第06课的 DDPM 相同。网络看到的数据是 `z = E(x)`。采样 `z_0` 后，解码为 `D(z_0)`。

### 步骤 3：无分类器引导（classifier-free guidance）

训练时，10%概率删除类别标签（替换为空标记）。推断中，同时计算 `ε_cond` 和 `ε_uncond`，然后：

```python
eps_cfg = (1 + w) * eps_cond - w * eps_uncond
```

`w = 0` 表示无引导（最大多样性），`w = 3` 默认，`w = 7+` 表示饱和/过度锐利。

### 步骤 4：文本条件（概念，非代码）

用冻结的文本编码器输出替代分类标签。通过交叉注意力将文本嵌入到 U-Net：

```python
h = h + CrossAttention(Q=h, K=text_embed, V=text_embed)
```

这是分类条件扩散模型与稳定扩散之间的唯一区别。

## 注意事项

- **VAE比例不匹配。** SD 1.x VAE 具有一个编码后应用的比例常数（`scaling_factor ≈ 0.18215`）。忘记此步骤会使得 U-Net 训练时潜在的方差严重不匹配。每个检查点均包含该常数。
- **文本编码器悄无声息错误。** SD3 需要使用 T5-XXL 且最大提示长度≥128，回退到只有 CLIP 会严重损失质量。务必保证设置`use_t5=True`，否则提示忠实度大幅下降。
- **潜在空间混用。** SDXL、SD3、Flux 使用不同的 VAE。在 SDXL 潜在上训练的 LoRA 无法用于 SD3。Hugging Face diffusers 0.30+ 版本会拒绝加载不匹配的检查点。
- **CFG 过高。** `w > 10` 会产生饱和油画般的图像，过拟合提示文本，牺牲多样性。最佳范围是 `w = 3-7`。
- **负面提示泄漏。** 空的负向提示变为空标记；非空负向提示用于计算 `ε_uncond`。两者不同，部分流水线默认空负向提示，导致行为异常。

## 使用建议

2026年生产堆栈：

| 目标                           | 推荐主体架构                        |
|--------------------------------|-----------------------------------|
| 狭域，成对数据，从零训练模型     | SDXL 微调（LoRA / 全量）——最快投产    |
| 开域文本到图像，开源权重         | Flux.1-dev（12B，Apache/非商业）或 SD3.5-Large |
| 最快推理，开源权重               | Flux.1-schnell（1-4步，Apache）或 SDXL-Lightning |
| 最佳提示忠实度，托管服务         | GPT-Image / DALL-E 3（仍中），Midjourney v7，Imagen 4 |
| 编辑工作流                     | Flux.1-Kontext（2024年12月）——原生支持图像+文本 |
| 研究、基线                     | SD 1.5——古老但研究充分                |

## 发布指南

保存 `outputs/skill-sd-prompter.md`。Skill 输入一个文本提示+目标风格，输出：模型+检查点，CFG 比例，采样器，负向提示，分辨率，任选的 ControlNet/IP-Adapter 组合，以及每步质量检查清单。

## 练习

1. **简单。** 运行 `code/main.py`，引导权重 `w ∈ {0, 1, 3, 7, 15}`。记录每类的平均样本。在哪个 `w` 下类别均值首次偏离真实数据均值？
2. **中等。** 用 tanh-MLP 编码器/解码器替代玩具线性编码器，附带重建损失。基于新潜变量重新训练扩散。样本质量是否变化？
3. **困难。** 用 diffusers 设置真实的 Stable Diffusion 推理：加载 `sdxl-base`，运行30步 Euler，CFG=7，计时。然后切换至 `sdxl-turbo`，4步，CFG=0。主题相同但质量不同 —— 描述变化及原因。

## 关键词汇

| 术语           | 常见说法          | 实际意义                           |
|----------------|-------------------|----------------------------------|
| 第一阶段       | “VAE”             | 训练好的编码器/解码器对；压缩512²到64² |
| 第二阶段       | “U-Net”           | 在潜在空间上运行的扩散模型          |
| CFG            | “Guidance scale”  | `(1+w)·ε_cond - w·ε_uncond`；调整条件强度 |
| 空标记（Null token） | “空提示嵌入”       | 计算 `ε_uncond` 时使用的无条件嵌入    |
| 交叉注意力     | “文本进入机制”      | U-Net每层对文本标记作为 K、V 进行注意力 |
| DiT            | “扩散 Transformer” | 用 Transformer 替代 U-Net 处理潜在块，扩展性更好 |
| MMDiT          | “多模态 DiT”       | SD3 架构：文本和图像流联合注意力      |
| VAE缩放因子    | “魔法数字”          | 将潜变量约除以5.4，让扩散在单位方差空间操作 |

## 生产注解：在 8GB 消费级 GPU 上运行 Flux-12B

Flux 的参考集成是“我只有消费级 GPU，能投产吗？”的经典配方。诀窍是应用扩散 DiT 的三大可调参数生产推理方案：

1. **分阶段加载。** Flux 有三个网络不会同时驻留显存：T5-XXL 文本编码器（fp32 占约10 GB），CLIP-L（小），12B MMDiT 和 VAE。先编码提示，*删除*编码器，加载 DiT，去噪，*删除* DiT，加载 VAE，解码。消费者级8GB GPU 一次只放一个阶段。
2. **BitsAndBytes 4位量化。** 在 T5 编码器和 DiT 上应用 `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)`。显存减少8倍，根据 Aritra 的基准（笔记本有链接）文本至图像质量损失可忽略。
3. **CPU 卸载。** `pipe.enable_model_cpu_offload()` 自动在每前向传递步骤间交换 CPU 和 GPU 模块。增加10-20%延迟，但使流畅运行成为可能。

内存预算是：10 GB T5 量化后约1.25 GB，12B 参数×0.5 字节约6 GB量化 DiT，加上激活内存。按 stas00 定义，这是 TP=1（无模型并行，最大量化）推理极限。生产环境下建议使用 TP=2 或 TP=4 在 H100 上；开发笔记本则采纳此方案。

## 延伸阅读

- [Rombach et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) — Stable Diffusion（稳定扩散）  
- [Podell et al. (2023). SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis](https://arxiv.org/abs/2307.01952) — SDXL  
- [Peebles & Xie (2023). Scalable Diffusion Models with Transformers (DiT)](https://arxiv.org/abs/2212.09748) — DiT  
- [Esser et al. (2024). Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206) — SD3，MMDiT  
- [Ho & Salimans (2022). Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) — 无分类器引导（CFG）  
- [Labs (2024). Flux.1 — Black Forest Labs 发布](https://blackforestlabs.ai/announcing-black-forest-labs/) — Flux.1 家族  
- [Hugging Face Diffusers 文档](https://huggingface.co/docs/diffusers/index) — 上述所有检查点的参考实现
