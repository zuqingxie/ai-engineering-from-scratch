# 视觉 Transformer（Vision Transformers, ViT）

> 将图像切成若干补丁（patches），视每个补丁为一个单词（token），运行标准的 Transformer（Transformer architecture）。不回头。

**类型：** 构建  
**语言：** Python  
**前置知识：** 第7阶段第2课（自注意力 self-attention），第4阶段第4课（图像分类 Image Classification）  
**时长：** ~45分钟  

## 学习目标

- 从零实现patch embedding（补丁嵌入）、learned positional embedding（学习位置嵌入）、class token（分类标记）和transformer编码器块，构建一个最小ViT模型  
- 解释为何ViT被认为需要海量预训练数据，直到DeiT和MAE证明了并非如此  
- 比较ViT、Swin和ConvNeXt在架构先验（无、本地窗口注意力、卷积骨干）上的异同  
- 使用`timm`和标准的linear-probe / fine-tune方案，在小型数据集上微调预训练ViT  

## 双语术语速查

| 中文术语 | English term |
|----------|--------------|
| 视觉 Transformer | Vision Transformer, ViT |
| 补丁 | Patch |
| 补丁嵌入 | Patch embedding |
| 图像令牌 | Image token |
| 类别令牌 | Class token, CLS token |
| 位置嵌入 | Positional embedding |
| 自注意力 | Self-attention |
| 多头注意力 | Multi-head attention |
| 查询/键/值 | Query/Key/Value, Q/K/V |
| 前馈网络 | Feed-forward network, FFN |
| 层归一化 | Layer Normalization, LayerNorm |
| 预归一化 | Pre-LN |
| 掩码自编码器 | Masked Autoencoder, MAE |
| DeiT | Data-efficient Image Transformer |
| Swin Transformer | Swin Transformer |
| 窗口注意力 | Window attention |


## 问题背景

十年来，卷积神经网络（CNN）与计算机视觉几乎画上等号。CNN具有强烈的归纳偏置（inductive biases）——局部性（locality）、平移等变性（translation equivariance）——没人认为能被替代。直到Dosovitskiy等人（2020）展示，纯粹对展平的图像patch应用标准Transformer，不使用任何卷积机制，就能在大规模数据上匹敌甚至超越最强CNN。

关键是“大规模”。ViT在ImageNet-1k上不及ResNet。先在ImageNet-21k或JFT-300M上预训练再在ImageNet-1k上微调后，才超过了ResNet。结论是transformer缺乏有用的先验，但可以从足够的数据中学习到。后续工作（DeiT、MAE、DINO）显示，只要训练策略得当——强数据增强、自监督预训练、蒸馏——ViT也能在小数据上训练良好。

到2026年，纯CNN在边缘设备上仍具竞争力（ConvNeXt是最强），但transformer统治了所有其他领域：分割（Mask2Former、SegFormer）、目标检测（DETR、RT-DETR）、多模态（CLIP、SigLIP）、视频（VideoMAE、VJEPA）。ViT的模块结构值得熟知。

## 核心概念

### 关键公式（Key equations）

ViT 先把图像切成补丁并线性投影成 token，再用自注意力混合全局信息：

$$
N = \frac{H W}{P^2}, \qquad X_{\mathrm{patch}} \in \mathbb{R}^{N \times (P^2C)}
$$

$$
\operatorname{Attention}(Q,K,V) = \operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

### 流程图

```mermaid
flowchart LR
    IMG["图像<br/>(3, 224, 224)"] --> PATCH["Patch embedding<br/>conv 16x16 s=16<br/>-> (768, 14, 14)"]
    PATCH --> FLAT["展平成<br/>(196, 768) tokens"]
    FLAT --> CAT["前置<br/>[CLS]标记"]
    CAT --> POS["加上学习的<br/>位置嵌入"]
    POS --> ENC["N个transformer<br/>编码器块"]
    ENC --> CLS["取[CLS]<br/>标记输出"]
    CLS --> HEAD["MLP分类器"]

    style PATCH fill:#dbeafe,stroke:#2563eb
    style ENC fill:#fef3c7,stroke:#d97706
    style HEAD fill:#dcfce7,stroke:#16a34a
```

七个步骤。patches -> tokens -> attention -> 分类器。每个变体（DeiT、Swin、ConvNeXt、MAE预训练）只改动其中一两个，其他保持不变。

### Patch embedding（补丁嵌入）

第一个卷积层是关键。核大小16，步长16，将224×224图像变成14×14个16×16patch，每个映射到768维向量。该卷积既完成切片，也完成线性映射。

```text
输入: (3, 224, 224)
卷积 (3 -> 768, k=16, s=16, 无填充)：
输出: (768, 14, 14)
展平成空间维：(196, 768)
```

196个patch即196个tokens。每个token维度为768（ViT-B）、1024（ViT-L）或1280（ViT-H）。

### Class token（分类标记）

单个学习向量，前置到序列头：

```text
tokens = [CLS; patch_1; patch_2; ...; patch_196]   形状 (197, 768)
```

经过N个transformer块后，`[CLS]`的输出就是整个图像的全局表示。分类头只读取这一向量。

### Positional embedding（位置嵌入）

Transformer本身无空间位置信息。给每个token加上学习得到的位置向量：

```text
tokens = tokens + learned_pos_embedding   （大小也为 (197, 768)）
```

这个嵌入是模型参数，梯度训练中适应二维图像结构。也有基于正弦的二维嵌入方案，但实际中不常用。

### Transformer encoder block（transformer编码器块）

标准模块：多头自注意力、多层感知机（MLP），残差连接，pre-LayerNorm。

```text
x = x + MSA(LN(x))
x = x + MLP(LN(x))

MLP为两层网络，激活GELU：Linear(d -> 4d) -> GELU -> Linear(4d -> d)
```

ViT-B/16堆叠12层，每层12个注意力头，共计8600万参数。

### 为什么用pre-LN

早期transformer用post-LN（`x = LN(x + sublayer(x))`），训练超过6-8层时难以收敛且需预热。pre-LN（`x = x + sublayer(LN(x))`）可以稳定训练更深网络，无需预热。现代ViT和所有先进大语言模型（LLM）都采用pre-LN。

### Patch大小权衡

- 16×16 patch→196 tokens，标准配置。  
- 32×32 patch→49 tokens，推理更快但分辨率较低。  
- 8×8 patch→784 tokens，更细粒度，但注意力计算复杂度为O(n²)上升严重。  

大patch数少，速度快但空间细节损失。SwinV2用4×4 patch分层窗口。

### DeiT训练ViT于ImageNet-1k的方案

原始ViT必须在JFT-300M大数据上预训练才能超越CNN。DeiT（Touvron等，2020）用以下四点，单纯在ImageNet-1k上训练ViT-B，top-1精度81.8%：

1. 强度增强：RandAugment、Mixup、CutMix、Random Erasing。  
2. 随机深度（stochastic depth），训练时随机丢弃整个块。  
3. 重复增强，批次中同一张图片采样3次。  
4. 蒸馏自CNN教师模型（可选，提升准确率）。

现今所有ViT训练策略均源自DeiT。

### Swin vs ConvNeXt

- **Swin**（Liu等人，2021）——基于窗口的注意力。每个块内只关注局部窗口，交替块间窗口滑移，实现金融窗口间信息融合。引入CNN类的局部先验，同时保持注意力机制。  
- **ConvNeXt**（Liu等人，2022）——重新设计的CNN，匹配Swin的架构选择（深度卷积、LayerNorm、GELU、反向瓶颈）。表明差距并非“注意力对比卷积”，而是“现代训练策略及架构”。  

2026年，ConvNeXt-V2和Swin-V2皆为生产级模型，选用哪一个视推理栈（ConvNeXt更适合边缘设备）和预训练语料而定。

### MAE预训练

Masked Autoencoder（He等人，2022）：随机遮蔽75%patch，只用可见25%patch训练encoder；训练小型decoder从encoder输出恢复被遮蔽的patch。预训练后丢弃decoder，微调encoder。

MAE使ViT在ImageNet-1k数据上可训练，达到SOTA，并成为现行默认自监督预训练方案。

## 实践构建

### 步骤1：Patch embedding

```python
import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, dim=192, image_size=64):
        super().__init__()
        assert image_size % patch_size == 0
        self.proj = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (image_size // patch_size) ** 2
        self.num_patches = num_patches

    def forward(self, x):
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)
```

一个卷积，一次展平，一次转置。完成整个图像转token。

### 步骤2：Transformer块

采用pre-LN、多头自注意力、带GELU的MLP，残差连接。

```python
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        a, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), need_weights=False)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x
```

`nn.MultiheadAttention`管理head拆分、缩放点乘和输出映射。`batch_first=True`使形状为`(N, seq, dim)`。

### 步骤3：ViT模型主体

```python
class ViT(nn.Module):
    def __init__(self, image_size=64, patch_size=16, in_channels=3,
                 num_classes=10, dim=192, depth=6, num_heads=3, mlp_ratio=4):
        super().__init__()
        self.patch = PatchEmbedding(in_channels, patch_size, dim, image_size)
        num_patches = self.patch.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, dim))
        self.blocks = nn.ModuleList([
            Block(dim, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        x = self.patch(x)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.ln(x[:, 0])
        return self.head(x)

vit = ViT(image_size=64, patch_size=16, num_classes=10, dim=192, depth=6, num_heads=3)
x = torch.randn(2, 3, 64, 64)
print(f"output: {vit(x).shape}")
print(f"params: {sum(p.numel() for p in vit.parameters()):,}")
```

约280万参数——小型ViT能在CPU上运行。真正ViT-B有8600万参数；同样类定义`dim=768, depth=12, num_heads=12`。

### 步骤4：单张图片推理测试

```python
logits = vit(torch.randn(1, 3, 64, 64))
print(f"logits: {logits}")
print(f"probs:  {logits.softmax(-1)}")
```

应能正常运行。概率和为1。

## 使用指南

`timm`提供所有ViT变体的ImageNet预训练权重。一行代码：

```python
import timm

model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=10)
```

2026年`timm`是视觉transformer的生产默认库。支持ViT、DeiT、Swin、Swin-V2、ConvNeXt、ConvNeXt-V2、MaxViT、MViT、EfficientFormer及其他数十种API统一调用。

多模态（图像+文本）工作可用`transformers`库，内置CLIP、SigLIP、BLIP-2、LLaVA，均用ViT变种作为图像编码器。

## 部署交付

本节产物：

- `outputs/prompt-vit-vs-cnn-picker.md` —— 一个prompt，根据数据集大小、计算量、推理栈选择ViT、ConvNeXt或Swin。  
- `outputs/skill-vit-patch-and-pos-embed-inspector.md` —— 一个技能脚本，验证ViT的patch embedding及positional embedding维度是否与模型序列长度匹配，可捕捉最常见的移植错误。  

## 练习

1. **（简单）** 打印上面小ViT前向过程中的所有中间张量形状。确认输入`(N, 3, 64, 64)`→patch `(N, 16, 192)`→加CLS `(N, 17, 192)`→分类器输入 `(N, 192)`→输出 `(N, num_classes)`。  
2. **（中等）** 使用`timm`预训练的ViT-S/16，在第4课中synthetic-CIFAR数据集上微调。与微调ResNet-18的结果对比，报告训练时间和最终准确率。  
3. **（困难）** 实现小ViT的MAE预训练：遮蔽75% patch，训练编码器+小解码器重建被遮挡的patch。评估预训练前后的synthetic数据线性探测准确率。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| Patch embedding（补丁嵌入） | “第一个卷积层” | 一个卷积，核大小 = 步幅 = 补丁大小；将图像转换为一组令牌嵌入 |
| Class token（类别令牌） | “[CLS]” | 一个学习到的向量，添加在令牌序列前面；其最终输出是全局图像表示 |
| Positional embedding（位置嵌入） | “学习的位置” | 加到每个令牌上的学习向量，使 Transformer（Transformer 架构）知道每个补丁的位置 |
| Pre-LN（预先 LayerNorm（层归一化）） | “子层前的 LayerNorm” | 一种稳定的 Transformer 变体：`x + sublayer(LN(x))`，而不是 `LN(x + sublayer(x))` |
| Multi-head attention（多头注意力） | “并行注意力” | 标准 Transformer 注意力机制被划分为 num_heads 个独立子空间，之后拼接起来 |
| ViT-B/16 | “基础版，补丁大小16” | 标准尺寸：dim=768，depth=12，heads=12，patch_size=16，图像大小=224；约8600万参数 |
| DeiT | “数据高效 ViT” | 仅用 ImageNet-1k 强增强训练的 ViT；证明了大型预训练数据集并非硬性要求 |
| MAE | “掩码自编码器” | 自监督预训练：掩盖75%的补丁，重建它们；主流的 ViT 预训练方案 |

## 进一步阅读

- [An Image is Worth 16x16 Words (Dosovitskiy et al., 2020)](https://arxiv.org/abs/2010.11929) — ViT 论文
- [DeiT: Data-efficient Image Transformers (Touvron et al., 2020)](https://arxiv.org/abs/2012.12877) — 如何仅用 ImageNet-1k 训练 ViT
- [Masked Autoencoders are Scalable Vision Learners (He et al., 2022)](https://arxiv.org/abs/2111.06377) — MAE 预训练
- [timm 文档](https://huggingface.co/docs/timm) — 你将在生产中使用的所有视觉 Transformer 的参考手册
