# CLIP 和对比视觉-语言预训练

> OpenAI 的 CLIP（2021）验证了一个足够强大的单一理念，可以驱动未来五年发展：仅使用噪声丰富的网页图像-标题对及对比损失，将图像编码器和文本编码器对齐到同一向量空间。零监督标签，4 亿对标注。由此产生的嵌入空间实现零-shot 分类、图像-文本检索，并作为每个 2026 年视觉语言模型（VLM）的视觉主塔。SigLIP 2（2025）用 sigmoid 代替 softmax，实现更低成本下超越 CLIP。本课涵盖从 InfoNCE 到 sigmoid 对偶损失的数学推导，并以 stdlib Python 构建训练步骤。

**类型:** 构建  
**语言:** Python（stdlib，InfoNCE + sigmoid 损失实现）  
**先决条件:** Phase 12 · 01（ViT patches），Phase 7（Transformers）  
**时间:** ~180 分钟  

## 学习目标

- 从互信息推导 InfoNCE 损失，并实现数值稳定的向量化版本。
- 解释为什么 sigmoid 对偶损失（SigLIP）可以扩展到批量大小 32768+，而 softmax 需要的 all-gather 通信限制了扩展。
- 通过构建文本模板（`a photo of a {class}`）并基于余弦相似度取 argmax，实现零-shot ImageNet 分类。
- 掌握 CLIP / SigLIP 预训练中可调节的四个关键因素：批量大小、温度、提示模板、数据质量。

## 问题背景

CLIP 之前，视觉模型是监督训练。通过收集带标签数据集（ImageNet：120万张图像，1000 类），训练 CNN 并上线。标签昂贵，标签带有标注者的偏见，且标签对新任务迁移有限，需要微调。

网页中的图像-字幕数据有十亿+个松散标签对免费可用。一张金毛猎犬的图片配文本“my dog Max in the park”包含监督信号 —— 文本描述图像。问题是：如何将这些数据转化为有效训练？

CLIP 的答案：将图像-字幕对视为匹配任务。给定批量 N 个图像和 N 个字幕，学习使每幅图像与自身字幕匹配，对抗其他 N-1 个干扰。监督是“这两个属于一组；其他 N-1 个不属于”。无类别标签，无人工注释，仅使用对比损失。

得到的嵌入空间功能远超训练目标。ImageNet 的零-shot 分类成功，因为“a photo of a cat”的嵌入靠近未显式标注为猫的猫的图片。这是 2026 年所有 VLM 的创业赌注。

## 概念讲解

### 双编码器结构

CLIP 由两部分组成：

- 图像编码器 `f`：ViT 或 ResNet，为每张图像输出 D 维向量。
- 文本编码器 `g`：小型 transformer，为每个标题输出 D 维向量。

两个编码器都将输出归一化为单位长度。相似度为 `cos(f(x), g(y)) = f(x)^T g(y)`，因单位范数而直接为内积。

对于一个包含 N 个（图像，标题）对的批次，构建形状为 `(N, N)` 的相似度矩阵 `S`：

```text
S[i, j] = cos(f(x_i), g(y_j)) / tau
```

其中 `tau` 是学习得到的温度参数（CLIP 初始化为 0.07；在对数空间里学习）。

### InfoNCE 损失

CLIP 使用对称交叉熵，分别对行和列操作：

```text
loss_i2t = CE(S, labels=identity)     # 每幅图像的正样本是对应标题
loss_t2i = CE(S^T, labels=identity)   # 每个标题的正样本是对应图像
loss = (loss_i2t + loss_t2i) / 2
```

这就是 InfoNCE。交叉熵中的 softmax 促使每幅图像与自身标题的匹配得分高于批次内其他标题。“负样本”是批次中所有其他元素。批次越大，负样本越多，信号越强。CLIP 训练时批次高达 32k，批次规模至关重要。

### 温度参数

`tau` 控制 softmax 的锐度。低 tau → 分布更尖锐，类似困难负样本挖掘效应。高 tau → 分布平滑，所有样本贡献一致。CLIP 学习 `log(1/tau)`，并进行截断以防坍缩。SigLIP 2 固定初始 tau，改用学习偏置。

### 为什么 sigmoid 扩展性更好（SigLIP）

Softmax 需要所有相似矩阵在训练的每个副本间同步。在分布式训练中，必须 all-gather 所有嵌入到所有副本，再计算 softmax。通信量按世界规模平方增长。

SigLIP 以元素级 sigmoid 代替 softmax：对任意对 `(i, j)`，视为二分类问题——“这对是否匹配？”对角线为正样本，其他皆为负样本。损失为：

```text
L = -1/N sum over (i, j) [ y_ij log sigmoid(S[i,j]) + (1-y_ij) log sigmoid(-S[i,j]) ]
```

`y_ij = 1` 当且仅当 `i == j`，否则为 0。每对损失相互独立，无需 all-gather。各 GPU 计算本地区块并汇总。SigLIP 2 可低成本扩展至大批量（32k-512k），而 CLIP 需要成比例更多通信。

### 零-shot 分类

给定 N 个类名，为每个类别构造文本模板：

```text
"a photo of a {class}"
```

用文本编码器嵌入每个模板。用图像编码器嵌入图像。基于余弦相似度取 argmax 即为预测类别。目标类别无训练。

提示模板非常关键。CLIP 原论文每类使用 80 个模板（普通、艺术、照片、绘画等），嵌入平均，提高 ImageNet 准确度约 3%。现代应用通常选用一两个模板。

### 线性探测器与微调

零-shot 是基线。线性探测器（在冻结 CLIP 特征上训练一层线性层）可超越零-shot 表现。完整微调在领域内任务优于线性探测器，但可能损害零-shot 迁移。三种方案各有取舍。

### SigLIP 2：NaFlex 与密集特征

SigLIP 2（2025）新增功能：

- NaFlex：单模型支持变化长宽比和分辨率的输入。
- 更好密集特征用于分割和深度估计，目标是作为 VLM 中冻结的主干。
- 多语言支持：训练于 100+ 语言，而 CLIP 仅限英语。
- 参数规模 10 亿，超越 CLIP 的 4 亿。

在 2026 年开源 VLM 中，SigLIP 2 SO400m/14 是默认视觉主塔。CLIP 仍是纯图文检索的首选，特别当 LAION-2B 训练分布与查询模式匹配时。

### ALIGN、BASIC、OpenCLIP、EVA-CLIP

ALIGN（Google，2021）：与 CLIP 思路相同，规模达 18 亿对，噪声率约 90%，证明噪声数据能扩展。OpenCLIP（LAION）：CLIP 基于 LAION-400M / 2B 数据的开源复现，支持多规模，是主流开源基线。EVA-CLIP：从掩码图像建模初始化，强大背骨用于 VLM。BASIC：Google 的 CLIP+ALIGN 混合版本。同属一脉，不同数据和调优策略。

### 零-shot 上限

CLIP 类模型在 ImageNet 零-shot 精度大约封顶于 76%（CLIP-G，OpenCLIP-G）。进一步提升需更多数据（SigLIP 2 达 80%+）或架构改进（监督头、更大模型）。该基准趋于饱和，真正价值在于下游 VLM 利用的高质量嵌入空间。

## 使用说明

`code/main.py` 实现：

1. 一个玩具双编码器（基于哈希的图像特征，字符级文本特征），使你无需 numpy 即可理解 InfoNCE 的形状。
2. 纯 Python 实现的 InfoNCE 损失（通过 log-sum-exp 保持数值稳定）。
3. 对比的 sigmoid 对偶损失实现。
4. 零-shot 分类例程：对一组文本提示计算余弦相似度，取 argmax 得预测。

运行它，观察损失曲线。绝对数值是玩具级别，形状符合真实 CLIP 训练器输出。

## 输出说明

本课生成 `outputs/skill-clip-zero-shot.md`。给定图像路径和目标类别列表，构建 CLIP 模板文本提示，用指定 checkpoint（如 `openai/clip-vit-large-patch14`）分别对两侧编码，返回 top-1 / top-5 预测及相似度分数。该技能不会对提示列表外的类别做出预测。

## 练习

1. 手工实现 4 对样本的 InfoNCE。构造 4x4 相似度矩阵，执行 softmax，提取对角线，计算交叉熵。将该计算结果与 Python 实现对比验证。

2. SigLIP 除温度 `tau` 外，还使用偏置参数 `b`：`S'[i,j] = S[i,j]/tau + b`。若批次类别严重不均衡（每行负样本远多于正样本）， `b` 有何作用？阅读 SigLIP 第 3 节（arXiv:2303.15343）。

3. 构建猫狗零-shot 分类器。试用两种提示模板：`a photo of a {class}` 和 `a picture of a {class}`。在 100 张测试图上测评准确率。模板集成是否优于单一模板？

4. 计算在 512 GPU 批量 32k 训练时，softmax InfoNCE 与 sigmoid 对偶的通信成本。哪个是 O(N)，哪个是 O(N²)？引用 SigLIP 第 4 节。

5. 阅读 OpenCLIP 规模定律论文（arXiv:2212.07143, Cherti 等）。通过图表复现其结论：固定模型规模时，ImageNet 零-shot 精度与训练数据量的对数线性关系如何？

## 关键术语

| 术语 | 通俗说法 | 准确定义 |
|------|----------|----------|
| InfoNCE | “对比损失” | 批次相似度矩阵的交叉熵；每个样本的正例是配对样本，负例是所有其他样本 |
| Sigmoid loss | “SigLIP 损失” | 按对的二分类交叉熵；无 softmax，无 all-gather；分布式训练低成本扩展 |
| Temperature | “tau” | softmax/sigmoid 前缩放 logit 的标量；控制分布锐度 |
| Zero-shot | “无微调分类” | 通过文本提示构造类嵌入，基于余弦相似度分类；无目标类别训练 |
| Prompt template | “a photo of a ...” | 类名的文本模板；影响零-shot 准确度 1-5 点 |
| Dual encoder | “双塔结构” | 一个图像编码器 + 一个文本编码器，输出共享 D 维空间 |
| Hard negative | “困难负样本” | 与正样本相似度高，需模型努力区分的负样本 |
| Linear probe | “冻结+一层” | 仅训练冻结特征上的线性分类器；衡量特征质量 |
| NaFlex | “原生灵活分辨率” | SigLIP 2 支持不缩放任意长宽比分辨率输入 |
| Temperature scaling | “以对数参数化 tau” | CLIP 用 `log(1/tau)` 参数化，确保梯度稳定；截断以防 tau 降至接近零 |

## 延伸阅读

- [Radford et al. — Learning Transferable Visual Models From Natural Language Supervision (arXiv:2103.00020)](https://arxiv.org/abs/2103.00020) — CLIP 论文。
- [Zhai et al. — Sigmoid Loss for Language Image Pre-Training (arXiv:2303.15343)](https://arxiv.org/abs/2303.15343) — SigLIP。
- [Tschannen et al. — SigLIP 2 (arXiv:2502.14786)](https://arxiv.org/abs/2502.14786) — 多语言 + NaFlex。
- [Jia et al. — ALIGN (arXiv:2102.05918)](https://arxiv.org/abs/2102.05918) — 利用海量噪声网页数据扩展规模。
- [Cherti et al. — Reproducible scaling laws for contrastive language-image learning (arXiv:2212.07143)](https://arxiv.org/abs/2212.07143) — OpenCLIP 规模定律。
