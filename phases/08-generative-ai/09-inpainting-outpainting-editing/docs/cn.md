# 修补（Inpainting）、扩画（Outpainting）与图像编辑

> 文本生成图像创造新内容。修补则修复已有内容。在生产中，70%的计费图像工作都是编辑——替换背景、去除水印、扩展画布、重生手部。修补是扩散模型（diffusion）大显身手的领域。

**类型：** 构建  
**语言：** Python  
**前置知识：** 第8阶段 · 07（Latent Diffusion 隐空间扩散模型）、第8阶段 · 08（ControlNet 与 LoRA）  
**时间：** ~75分钟

## 问题描述

客户提供一张完美的产品照片，但背景中有分散注意力的标牌。你想擦除这个标牌，且让其他部分像素完全一致。不能从头运行文本生成图像——结果会有不同的颜色、光照、产品角度。你只想重建*掩码区域*，且希望重建结果尊重周围上下文。

这就是修补。它的变体：

- **修补（Inpainting）。** 在掩码内重建，掩码外保持像素不变。
- **扩画（Outpainting）。** 在掩码外（或画布之外）重建，掩码内保持不变。
- **图像编辑（Image editing）。** 重生成整张图片，但保持语义或结构上的一致性（如 SDEdit、InstructPix2Pix）。

2026年的每个扩散管线都会支持修补模式。Flux.1-Fill、Stable Diffusion Inpaint、SDXL-Inpaint、DALL-E 3 Edit 都是基于同一原理。

## 概念

![修补：带掩码感知的去噪与上下文保持的再注入](../assets/inpainting.svg)

### 朴素方法（为什么行不通）

直接用带掩码的标准文本生成图像方法。每步采样时，用前向扩散的干净图像替换无掩码区域的噪声潜在向量。这个方法……效果很差。边界伪影明显，因为模型对掩码区域内内容毫无信息。

### 正确的修补模型

训练一个改进的 U-Net，它接受9个输入通道而非4个：

```text
input = concat([ noisy_latent (4通道), encoded_image (4通道), mask (1通道) ], dim=channel)
```

额外通道是 VAE 编码的来源图像和单通道掩码。在训练时，随机遮挡图像区域，训练模型只去噪掩码区域，而无掩码区域作为干净的条件信号。推理时，模型能“看到”掩码外的内容，从而生成连贯的补全。

SD-Inpaint、SDXL-Inpaint、Flux-Fill 都用这种9通道（或类似）输入。Diffusers中对应 `StableDiffusionInpaintPipeline`、`FluxFillPipeline`。

### SDEdit（Meng等人，2022）——免费编辑

给源图加噪到中间时间步`t`，然后用新提示从`t`向0反向采样。无需重新训练。选择起始的`t`权衡保持度与创意自由：

- `t/T = 0.3` → 几乎和原图相同，只做细微风格调整
- `t/T = 0.6` → 中度编辑，保留粗结构
- `t/T = 0.9` → 近噪声生成，最大限度地脱离原图

### InstructPix2Pix（Brooks等人，2023）

对扩散模型微调，输入为 `(输入图, 指令, 输出图)` 三元组。推理时同时条件输入图和文本指令（如“做成日落”、“加条龙”）。有两个 CFG 规模参数：图像和文本。

### RePaint（Lugmayr等人，2022）

保持标准的无条件扩散模型。每次反向步骤中偶尔重采样，跳回更高噪声状态再重去噪，避免边界伪影。适合没有训练修补模型时使用。

## 动手构建

`code/main.py`实现了一个玩具的1维修补方法，数据为5维混合分布。训练一个5维DDPM，样本是两个簇中一簇的5个浮点数。推理时“掩码”2维，注入无掩码的三个维度的加噪版本，重建掩码维度。

### 第1步：5维DDPM数据

```python
def sample_data(rng):
    cluster = rng.choice([0, 1])
    center = [-1.0] * 5 if cluster == 0 else [1.0] * 5
    return [c + rng.gauss(0, 0.2) for c in center], cluster
```

### 第2步：训练全部5维的去噪器

标准DDPM。网络输出5维的噪声预测。

### 第3步：推理时的掩码感知反向步骤

```python
def inpaint_step(x_t, mask, clean_image, alpha_bars, t, rng):
    # 用干净图像的新加噪版本替换无掩码维度
    a_bar = alpha_bars[t]
    for i in range(len(x_t)):
        if not mask[i]:
            x_t[i] = math.sqrt(a_bar) * clean_image[i] + math.sqrt(1 - a_bar) * rng.gauss(0, 1)
    # ...然后对x_t执行正常反向采样步骤
```

这是朴素方法，玩具1D数据有效。真实图像修补用9通道输入更重要，因为质感一致性尤为关键。

### 第4步：扩画

扩画是掩码反过来：把原本不存在的新画布区域置掩码，其他区域保持原图。训练目标完全相同。

## 注意事项

- **边界缝隙。** 朴素方法会有明显边界，因为梯度无法跨掩码流动。解决方案：将掩码膨胀8-16像素，或用专业修补模型。
- **掩码泄漏。** 如果无掩码区域质量差或有噪声，会污染掩码内生成。可先去噪或轻微模糊。
- **CFG与掩码大小互动。** 小掩码加大CFG会导致饱和斑块。小编辑应降低CFG。
- **SDEdit保真度断崖。** 从`t/T=0.5`到`t/T=0.6`，主题识别可能骤降。需扫参与存档。
- **提示不匹配。** 提示应描述*整张*图像，而非仅新内容。比如用“坐在椅子上的猫”，而非“猫”。

## 使用示例

| 任务               | 流水线                                             |
|--------------------|----------------------------------------------------|
| 去除物体，小掩码   | SD-Inpaint或Flux-Fill，普通提示                   |
| 替换天空           | SD-Inpaint + “夕阳下的蓝天”                        |
| 扩展画布           | SDXL扩画模式（8像素羽化）或Flux-Fill带扩画掩码    |
| 重生手部/脸部      | SD-Inpaint，提示重述主题 + ControlNet-Openpose    |
| 改变某一区域风格   | SDEdit，掩码区域`t/T=0.5`                          |
| “做成日落”         | InstructPix2Pix或Flux-Kontext                      |
| 背景替换           | SAM掩码 → SD-Inpaint                               |
| 超高保真           | Flux-Fill或GPT-Image（托管），处理最难任务         |

SAM（Meta的 Segment Anything，2023）+扩散修补是2026年背景去除首选方案。SAM 2（2024）支持视频。

## 部署

保存 `outputs/skill-editing-pipeline.md`。技能输入原图 + 编辑描述 + 可选掩码（或SAM提示），输出：掩码生成方法、基础模型、CFG尺度（图像+文本）、SDEdit起始时间或修补模式、QA检查列表。

## 练习

1. **简单。** 在 `code/main.py` 中调整掩码维度比例从0.2到0.8。在哪个比例下，掩码区域的修补误差与无条件生成相当？
2. **中等。** 实现RePaint：每经过第10个反采样步骤，跳回5步（加噪）再重去噪。测量是否减少掩码边界残差。
3. **困难。** 用Hugging Face diffusers对比：SD 1.5 Inpaint + ControlNet-Openpose 和 Flux.1-Fill 对20个脸部重生任务。分别评分姿态遵循度和身份保留。

## 关键词

| 术语               | 常见说法                   | 实际含义                            |
|--------------------|----------------------------|-----------------------------------|
| Inpainting         | “填补空洞”                 | 掩码内重建，掩码外保留像素         |
| Outpainting        | “扩展画布”                 | 掩码外重建，掩码内保持             |
| 9-channel U-Net    | “专业修补模型”             | 输入为 `noisy | encoded-source | mask` 的 U-Net |
| SDEdit             | “带噪版img2img”            | 加噪到时间`t`，用新提示去噪       |
| InstructPix2Pix    | “纯文本编辑”               | 训练于（输入图，指令，输出图）三元组  |
| RePaint             | “无需重训练”               | 反向时周期性加噪减少缝隙           |
| SAM                | “任意分割”                 | 可通过点击或框选生成掩码，与修补配合 |
| Flux-Kontext       | “带上下文编辑”             | Flux变体接受参考图 + 指令进行编辑   |

## 生产笔记：编辑流水线的延迟敏感性

用户期待图像编辑往返小于5秒。1024²分辨率下，30步的SDXL-Inpaint在L4上需3-4秒，再加SAM掩码生成（约200毫秒）和VAE编码/解码（共约500毫秒）。生产环境关注的是首次响应时间（TTFT），非吞吐率——仅批量1，低并发，极限压缩每阶段：

- **SAM-H最慢。** SAM-H 1024²约200毫秒，SAM-ViT-B约40毫秒，质量略有损失。SAM 2（视频版）有额外时序开销，不适用于单图编辑。
- **尽量跳过编码。** `pipe.image_processor.preprocess(img)`会编码成潜变量。迭代编辑中已存在潜变量时，可直接通过 `latents=...` 传入，跳过一次VAE编码。
- **掩码膨胀也影响吞吐。** 小掩码导致大部分U-Net前向计算浪费（非掩码像素被钳制）。`diffusers`的`StableDiffusionInpaintPipeline`会完整运行U-Net，只有9通道专业修补模型利用掩码推理节省计算。
- **Flux-Kontext是2025的答案。** 一次前向运行 `(source_image, instruction)`，无掩码、无SDEdit噪声扫描。在H100上约1.5秒完成编辑。架构教训：合并阶段。

## 延伸阅读

- [Lugmayr et al. (2022). RePaint: Inpainting using Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2201.09865) — 无需训练的修补方法。  
- [Meng et al. (2022). SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations](https://arxiv.org/abs/2108.01073) — SDEdit方法。  
- [Brooks, Holynski, Efros (2023). InstructPix2Pix](https://arxiv.org/abs/2211.09800) — 基于文本指令的编辑。  
- [Kirillov et al. (2023). Segment Anything](https://arxiv.org/abs/2304.02643) — SAM掩码生成器。  
- [Ravi et al. (2024). SAM 2: Segment Anything in Images and Videos](https://arxiv.org/abs/2408.00714) — 支持视频的SAM。  
- [Hertz et al. (2022). Prompt-to-Prompt Image Editing with Cross-Attention Control](https://arxiv.org/abs/2208.01626) — 基于注意力的编辑。  
- [Black Forest Labs (2024). Flux.1-Fill and Flux.1-Kontext](https://blackforestlabs.ai/flux-1-tools/) — 2024年工具链。
