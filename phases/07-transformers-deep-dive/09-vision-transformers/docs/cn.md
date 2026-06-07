# 视觉Transformer（ViT）

> 图像是一个补丁网格。句子是一个标记网格。相同的transformer处理两者。

**类型：** 构建  
**语言：** Python  
**先决条件：** 第7阶段 · 05（完整Transformer），第4阶段 · 03（卷积神经网络CNN），第4阶段 · 14（视觉Transformer介绍）  
**时间：** 约45分钟

## 问题背景

在2020年前，计算机视觉意味着卷积。ImageNet、COCO及目标检测基准测试的所有最先进方法均使用CNN骨干架构。Transformer是语言的专属。

Dosovitskiy等人（2020年）——“An Image is Worth 16x16 Words”——展示了可以完全摒弃卷积。将图像切成固定大小的补丁，把每个补丁线性映射到嵌入，再将序列输入普通的Transformer编码器。规模足够大时（如ImageNet-21k预训练或更大），ViT在性能上可以匹配甚至优于基于ResNet的模型。

ViT是2026年更广泛模式的起点：一种架构，多种模态。Whisper将音频标记化。ViT将图像标记化。动作标记用于机器人。像素标记用于视频。Transformer不在乎输入是什么——只要输入序列，它就能学习。

到了2026年，ViT及其衍生版本（DeiT、Swin、DINOv2、ViT-22B、SAM 3）占据视觉领域大部分市场。CNN在边缘设备和低延迟任务仍占优势。其他情况下堆栈中总有ViT的身影。

## 基本概念

![Image → patches → tokens → transformer](../assets/vit.svg)

### 第1步 — 分块（patchify）

将一个 `H × W × C` 的图像拆分为一个包含 `N × (P·P·C)` 的扁平补丁序列。典型配置：`224 × 224` 图像，`16 × 16` 补丁 → 得到196个长度为768的向量。

```text
image (224, 224, 3) → 14 × 14 的16x16x3补丁网格 → 196个长度为768的向量
```

补丁大小是关键。补丁越小，标记数越多，分辨率越高，但注意力计算成本呈二次增长。补丁越大，分辨率越低，计算成本越低。

### 第2步 — 线性嵌入（linear embedding）

用一个学习得到的矩阵将每个扁平补丁投影到 `d_model` 维度。等价于大小为 `P`，步长为 `P` 的卷积。PyTorch实现为：`nn.Conv2d(C, d_model, kernel_size=P, stride=P)`，仅两行代码。

### 第3步 — 添加 `[CLS]` 标记并加上位置嵌入

- 在序列前添加一个可学习的 `[CLS]` 标记。其最终隐藏状态作为图像的表示，用于分类。
- 加上可学习的位置嵌入（ViT原始版本）或二维正弦位置编码（后续版本）。
- 2024年以后，RoPE扩展到二维来编码位置，有时不包含显式的位置嵌入。

### 第4步 — 标准Transformer编码器

堆叠L个包含 `LayerNorm → Self-Attention → + → LayerNorm → MLP → +` 的块。与BERT完全相同，无视觉专用层。这是该论文最重要的教学核心。

### 第5步 — 头部（head）

分类任务：取 `[CLS]` 的隐藏状态 → 线性层 → softmax。DINOv2或SAM则舍弃 `[CLS]`，直接使用补丁嵌入。

### 重要变体

| 模型 | 年份 | 变化 |
|-------|------|--------|
| ViT | 2020 | 原版。固定补丁大小，全局注意力。 |
| DeiT | 2021 | 蒸馏；仅在ImageNet-1k上训练。 |
| Swin | 2021 | 分层结构，带偏移窗口。近似亚二次复杂度。 |
| DINOv2 | 2023 | 自监督（无标签）。最优秀的通用视觉特征。 |
| ViT-22B | 2023 | 220亿参数；符合缩放定律。 |
| SigLIP | 2023 | ViT + 语言对，使用sigmoid对比损失。 |
| SAM 3 | 2025 | “Segment Anything”；ViT-Large + 支持提示的掩码解码器。 |

### 为什么耗时较长

ViT缺乏CNN的归纳偏置（平移不变性、局部性），因此比CNN需要更多数据才能达到同等效果。没有超过1亿标注图像或强大的自监督预训练，CNN仍在相同计算资源下占优。DeiT于2021年用蒸馏技巧解决了部分问题；DINOv2于2023年用自监督方法彻底解决。

## 构建

详见`code/main.py`。使用纯标准库实现patchify、线性嵌入和简单检查。无训练——实际规模的ViT需要PyTorch和数小时GPU时间。

### 第1步：伪造图像

24×24 RGB图像，数据结构为行列表，每个像素是 `(R, G, B)` 元组。我们使用6×6补丁，共16个补丁，每个嵌入向量108维。

### 第2步：分块（patchify）

```python
def patchify(image, P):
    H = len(image)
    W = len(image[0])
    patches = []
    for i in range(0, H, P):
        for j in range(0, W, P):
            patch = []
            for di in range(P):
                for dj in range(P):
                    patch.extend(image[i + di][j + dj])
            patches.append(patch)
    return patches
```

光栅顺序：网格行主序。所有ViT均采用此顺序。

### 第3步：线性嵌入

将每个扁平补丁乘以随机的 `(patch_flat_size, d_model)` 的矩阵。添加 `[CLS]` 后，输出形状应为 `(N_patches + 1, d_model)`。

### 第4步：统计现实ViT的参数数量

打印ViT-Base参数数量：12层，12个头，维度768，补丁16。约8600万个参数。ResNet-50约2500万个参数。ViT-Large约3.07亿，ViT-Huge约6.32亿。

## 使用示例

```python
from transformers import ViTImageProcessor, ViTModel
import torch
from PIL import Image

processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

img = Image.open("cat.jpg")
inputs = processor(img, return_tensors="pt")
out = model(**inputs).last_hidden_state   # (1, 197, 768): [CLS] + 196补丁
cls_emb = out[:, 0]                       # 图像表示
```

**DINOv2嵌入是2026年的图像特征默认选择。** 冻结骨干网络，仅训练小型头部。适用于分类、检索、检测、描述。Meta发布的DINOv2检查点在所有非文本视觉任务上优于CLIP。

**补丁大小选择。** 小模型采用16×16（ViT-B/16）。密集预测（分割）使用8×8或14×14（SAM、DINOv2）。很大模型多用14×14。

## 部署

见`outputs/skill-vit-configurator.md`。该技能根据数据集大小、分辨率和计算预算，为新视觉任务选择合适的ViT变体和补丁大小。

## 练习

1. **简单。** 运行`code/main.py`。验证补丁数量等于 `(H/P) * (W/P)`，补丁扁平化维度等于 `P * P * C`。
2. **中级。** 实现二维正弦位置编码——为每个补丁的`行`和`列`分别生成独立的正弦编码拼接。使用PyTorch实现一个小型ViT，比较该编码和可学习位置编码在CIFAR-10上的准确率。
3. **高级。** 构建三层ViT（PyTorch），用4×4补丁在1000张MNIST图片上训练。测试准确率。之后用DINOv2预训练（简化版：训练编码器从掩码补丁预测补丁嵌入），测试准确率是否提升。

## 关键词

| 术语 | 常说法 | 实际含义 |
|------|---------|----------|
| Patch | “视觉transformer的token” | `P × P × C`图像区域的扁平化像素值向量。 |
| Patchify | “切分+扁平化” | 切成不重叠补丁，将每个补丁展平为向量。 |
| `[CLS]` token | “图像摘要” | 序列开头的可学习标记，其最终嵌入是图像的表示。 |
| 归纳偏置（Inductive bias） | “模型的假设” | ViT比CNN先验少，需更多数据弥补差距。 |
| DINOv2 | “自监督的ViT” | 无标签训练，借助图像增强和动量教师。2026年最佳通用图像特征。 |
| SigLIP | “CLIP的继任者” | ViT + 语言编码器，使用sigmoid对比损失；计算量相当情况下优于CLIP。 |
| Swin | “带窗口的ViT” | 分层ViT，局部注意力+偏移窗口，亚二次复杂度。 |
| 注册标记（Register tokens） | “2023年技巧” | 一些额外的可学习标记，用于吸收注意力“汇流点”；提升DINOv2特征。 |

## 拓展阅读

- [Dosovitskiy et al. (2020). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) — ViT论文。  
- [Touvron et al. (2021). Training data-efficient image transformers & distillation through attention](https://arxiv.org/abs/2012.12877) — DeiT。  
- [Liu et al. (2021). Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030) — Swin。  
- [Oquab et al. (2023). DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193) — DINOv2。  
- [Darcet et al. (2023). Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588) — DINOv2的注册标记修正。
