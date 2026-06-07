# Transfusion：一个 Transformer 同时实现自回归文本与扩散图像

> Chameleon 和 Emu3 完全押注于离散 tokens。它们工作正常，但量化瓶颈明显——图像质量在连续空间扩散模型之下徘徊。Transfusion（Meta，Zhou 等，2024年8月）则走了相反的路：保持图像连续，完全摒弃 VQ-VAE，同时训练一个带两种损失的 Transformer。文本 tokens 用下一个 token 预测（Next-Token-Prediction，NTP），图像 patch 则用流匹配 / 扩散损失。两个目标优化同一权重。Stable Diffusion 3（MMDiT）背后的架构是亲戚。本课将解读 Transfusion 论文，构建一个玩具两损失训练器，并追踪实现一个 Transformer 同时完成两项任务的注意力掩码。

**类型：** 实践  
**语言：** Python（标准库，针对 MNIST 规模玩具的两损失训练器）  
**先决条件：** 第12阶段·11课（Chameleon），第8阶段（生成式 AI）  
**时间：** ~180 分钟

## 学习目标

- 构建一个 Transformer，能在线性骨干网络上运行两种损失（文本 tokens 进行 NTP，图像 patch 进行扩散 MSE）。
- 解释为何对图像 patch 使用双向注意力，对文本 tokens 使用因果注意力是正确的掩码选择。
- 对比 Transfusion 风格（连续图像，扩散损失）和 Chameleon 风格（离散图像，NTP）在计算量、质量和代码复杂度上的差异。
- 描述 MMDiT 的贡献：每层块施加模态特定权重，残差流上的联合注意力。

## 问题背景

离散与连续图像 tokens 的争论早于大型语言模型（LLMs）。连续表示（原始像素，VAE 潜变量）保持细节；离散 tokens（VQ 索引）适应 Transformer 的原生词表，但在量化阶段损失细节。

Chameleon / Emu3 走了离散路线：单一损失，单一架构，但图像保真度受限于分词器质量。

扩散模型采用连续路线：图像质量卓越，但需要不同于 LLM 的单独模型，噪声调度工程复杂，且与文本生成没有清晰整合。

Transfusion 提问：我们可以兼顾两者吗？保持图像连续，仍训练一个模型，同时用两种损失合成到一次梯度更新。

## 概念

### 两损失架构

一个仅解码器 Transformer 处理序列，序列包含：

- 文本 tokens（离散的，来自 BPE 词汇表）。
- 图像 patch（连续的，16x16 像素块，通过线性映射投射到隐藏维度——同 ViT 编码器输入）。
- `<image>` 和 `</image>` 标签标注连续 patch 所在位置。

前向传播仅执行一次。损失针对每 token 选择两个头之一：

- 文本 tokens：标准交叉熵损失，针对词汇 logits 头。
- 图像 patch：针对连续 patch 的扩散损失—预测加在每个 patch 上的噪声。

梯度流经共享 Transformer 主体。两个损失同时改进共享权重。

### 注意力掩码：因果文本 + 双向图像

文本 token 必须是因果的——文本 token 不能关注未来文本，否则教师强制就会失效。图像 patch 表示同一时间快照，应在同一图像块内实现双向关注。

掩码定义：

```text
M[i, j] = 1 如果：
  (i 是文本且 j 是文本且 j <= i)   # 文本因果
  或 (i 是图像且 j 是图像且在同一图像块(i, j))   # 图像内双向
  或 (i 是文本且 j 是图像且 j < i_image_end)   # 文本关注先前图像
  或 (i 是图像且 j 是文本且 j < i_image_start)   # 图像关注之前文本
```

训练和推理时实现为块三角掩码。

### Transformer 内的扩散损失

扩散损失标准做法：对图像 patch 添加噪声，令模型预测噪声（或原始干净 patch，等价）。Transfusion 版本用流匹配——预测从有噪声到干净的速度场。

训练步骤：  
1. 对每个图像 patch x0，采样随机时间步 t。  
2. 采样噪声 ε，计算 xt = (1-t) * x0 + t * ε（使用线性插值以匹配流匹配）。  
3. Transformer 预测 v_theta(xt, t)；损失 = MSE(v_theta(xt, t), ε - x0)。  
4. 与相同序列中的文本 NTP 损失一起反向传播。

推理生成：  
- 文本 tokens：常规自回归采样。  
- 图像 patch：基于前序文本 tokens 条件、执行扩散采样循环（通常 10-30 步）。

### MMDiT：Stable Diffusion 3 的变体

Stable Diffusion 3（Esser 等，2024年3月）同期发布了 MMDiT（Multimodal Diffusion Transformer），两者架构相似。

MMDiT 关键区别：

- 每层块有模态特定权重。Transformer 块对文本 tokens 和图像 patch 分别采用不同的 Q、K、V 和 MLP 权重。注意力是联合的（跨模态），其它部分是模态专属的。
- 修正流训练（Rectified flow training）。一种特定流匹配变体，带有已知采样方法，数值比 DDPM 更简单。
- 规模。MMDiT 是 SD3（2B 与 8B 参数变体）的骨干。Transfusion 论文扩展至 7B 参数。

两者共识：一个 Transformer 同时运行文本 NTP 与图像连续扩散。

### 为什么优于 Chameleon 式

连续扩散与离散 NTP 在图像生成质量上的差距明显。Transfusion 报告：

- 7B 参数规模上，按 FID 指标领先同规模 Chameleon 式 3-5 点。
- 不需训练分词器，图像编码器更简单（线性投影至隐藏维度，等同 ViT 输入层）。
- 推理时图像 patch 去噪可并行，而非离散图像 tokens 的自回归方式。

缺点：Transfusion 是双损失模型，训练过程更复杂，损失权重需调节。NTP 与扩散间调度不匹配易导致某头支配。

### 后续发展

Janus-Pro（第12.15课）对 Transfusion 思路做改进，分离视觉编码器用于理解与生成——SigLIP 处理理解，VQ 用于生成，同时共享 Transformer 主体。Show-o（第12.14课）将扩散替换为离散扩散（掩码预测）。统一生成系列在 Transfusion 之后快速分支。

2026 年发布的生产级视觉语言模型（VLM）如 Gemini 3 Pro、GPT-5 和 Claude Opus 4.7 的图像生成路径，几乎肯定源于该家族某个继承体。细节尚属商业机密。

## 使用方法

`code/main.py` 构建在极小 MNIST 类问题上的 Transfusion 玩具模型：

- 文本标题是简短整数序列，描述数字（0-9）。  
- 图像是 4x4 字节网格。  
- 一对共享权重的线性投影作 Transformer 近似；文本部分用 NTP 损失，噪声图像 patch 用 MSE 损失。  
- 训练循环交替两种损失，注意力掩码显式编写。  
- 生成时一次前向即产出文本标题和 4x4 图像。

Transformer 是玩具版。两损失系统、注意力掩码构建与推理循环是核心成果。

## 交付成果

本课产生 `outputs/skill-two-loss-trainer-designer.md`。给定新多模态训练任务（文本+图像、文本+音频、文本+视频等），设计两损失调度（损失权重、掩码形状、共享与模态特定块），并标注实现风险。

## 练习题

1. 一款 Transfusion 式模型的训练数据中 70% 是文本 tokens，30% 是图像 patches。图像扩散损失幅度约为文本 NTP 损失的 10 倍。应如何设置损失权重实现平衡？

2. 针对序列 `[T, T, <image>, P, P, P, P, </image>, T]` 实现块三角掩码。对每个条目标 0 或 1。

3. MMDiT 采用模态专用 QKV 权重。相较于 Transfusion 完全共享 Transformer，这在参数量上多出多少？7B 参数规模下是否值得？

4. 生成流程：给文本提示，模型生成 50 个文本 token 后遇到 `<image>`，对 256 个图像 patch 用 20 步去噪扩散。共需多少前向传播？

5. 阅读 SD3 论文第3节，描述修正流（Rectified flow）及其为何比 DDPM 推理步数更少即可收敛。

## 关键词

| 术语 | 大众说法 | 实际含义 |
|------|----------|----------|
| 两损失训练 | "NTP + diffusion" | 单个 Transformer 在同一次梯度更新中同时优化文本 tokens 的交叉熵和连续图像 patch 的 MSE |
| 流匹配 | "Rectified flow" | 一种扩散变体，预测从噪声到干净数据的速度场；数学公式比 DDPM 简单 |
| MMDiT | "Multimodal DiT" | Stable Diffusion 3 架构：联合注意力，模态专属 MLP 和归一化层 |
| 块三角掩码 | "因果文本 + 双向图像" | 注意力掩码，对文本因果，对图像区域双向 |
| 连续图像表示 | "无 VQ" | 图像 patch 用实值向量表示，不是整数码本索引 |
| 速度预测 | "v 参数化" | 网络输出是噪声与数据间的速度场，而非噪声本身 |

## 延伸阅读

- [Zhou 等 — Transfusion（arXiv:2408.11039）](https://arxiv.org/abs/2408.11039)  
- [Esser 等 — Stable Diffusion 3 / MMDiT（arXiv:2403.03206）](https://arxiv.org/abs/2403.03206)  
- [Peebles & Xie — DiT（arXiv:2212.09748）](https://arxiv.org/abs/2212.09748)  
- [Zhao 等 — MonoFormer（arXiv:2409.16280）](https://arxiv.org/abs/2409.16280)  
- [Xie 等 — Show-o（arXiv:2408.12528）](https://arxiv.org/abs/2408.12528)
