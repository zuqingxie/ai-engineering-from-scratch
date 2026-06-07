# 评估 — FID、CLIP 分数、人类偏好

> 每个生成模型排行榜都会引用 FID（Fréchet Inception Distance，弗雷谢特距离）、CLIP 分数和人类偏好竞赛的胜率。每个指标都有一个失败模式，足够精明的研究者能利用它进行作弊。如果你不了解这些失败模式，就无法分辨真正的改进与作弊。

**类型：** 构建  
**语言：** Python  
**先决条件：** 第8阶段 · 01（分类法），第2阶段 · 04（评估指标）  
**时间：** 约45分钟

## 问题所在

生成模型的评价依赖于*样本质量*和*条件一致性*。两者都没有封闭式的度量方法。模型必须生成 10,000 张图像；需要有人为它们分配分数；你还必须信任这些分数在不同模型族、分辨率、架构间是通用的。三个指标在2014-2026年间筛选存活：

- **FID（Fréchet Inception Distance，弗雷谢特嵌入距离）**。在 Inception 网络的特征空间中对真实分布和生成分布的距离。数值越低越好。
- **CLIP 分数。** 生成图像的 CLIP-图像嵌入和提示语的 CLIP-文本嵌入的余弦相似度。数值越高越好。衡量提示语遵守程度。
- **人类偏好。** 让两个模型对同一提示语进行对抗，由人类（或 GPT-4 等级模型）选择更佳者，然后聚合成 Elo 分数。

你还会见到：IS（Inception 分数，多已弃用）、KID、CMMD、ImageReward、PickScore、HPSv2、MJHQ-30k。每个都是对前者一种失败模式的纠正。

## 概念介绍

![FID、CLIP 和偏好：三个轴，不同的失败模式](../assets/evaluation.svg)

### FID — 样本质量

Heusel 等人（2017）。步骤：

1. 对 N 张真实图像和 N 张生成图像提取 Inception-v3 特征（2048 维）。
2. 对每个集合拟合高斯分布：计算均值 `μ_r, μ_g` 和协方差 `Σ_r, Σ_g`。
3. FID = `||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2 · (Σ_r · Σ_g)^0.5)`。

解释：在特征空间中两个多元高斯的 Fréchet 距离。数值越小表示分布越相似。

失败模式：  
- **小样本量偏差。** FID 是对特征分布均值平方误差的度量，小样本 N 会低估协方差，导致虚假的低 FID。始终保证 N ≥ 10,000。  
- **依赖 Inception 网络。** Inception-v3 训练于 ImageNet，远离 ImageNet 的领域（人脸、艺术、文本图像）产生无意义的 FID。应使用特定领域的特征提取器。  
- **作弊。** 过度拟合 Inception 先验会产生低 FID 但图像质量无提升。用 CMMD（下文）可以打败它。

### CLIP 分数 — 提示遵守

Radford 等人（2021）。针对生成图像 + 提示语：

```text
clip_score = cos_sim( CLIP_image(x_gen), CLIP_text(prompt) )
```

对 30k 生成图像求平均 → 一个可比较的标量。

失败模式：  
- **CLIP 自身盲点。** CLIP 的组合推理较弱（“一个红色立方体在蓝色球上”经常失败）。模型可以在 CLIP 分数上排名靠前，但未真正遵守复杂提示。  
- **短提示偏差。** 短提示在真实世界中更易匹配 CLIP-图像。长提示在计算上更难达到高分。  
- **提示作弊。** 在提示中加入“高质量，4k，杰作”会人为提升 CLIP 分数，但不增加图像-文本绑定。

CMMD（Jayasumana 等人，2024）修复了一部分问题：使用 CLIP 特征代替 Inception，使用最大均值差异取代 Fréchet，能更好检测微妙质量差异。

### 人类偏好 — 基准真理

选择一组提示语。用模型 A 和模型 B 生成图像。将图像对展示给人类（或强大的大语言模型裁判）。聚合胜利得出 Elo 或 Bradley-Terry 分数。参考基准：

- **PartiPrompts（Google）**：1,600 个多样化提示，12 个类别。  
- **HPSv2**：107k 人类注释，广泛用作自动代理。  
- **ImageReward**：137k 提示-图像偏好对，MIT 许可证。  
- **PickScore**：基于 Pick-a-Pic 2.6M 偏好训练。  
- **聊天机器人赛场式图像赛场**：https://imagearena.ai/ 及其他。

失败模式：  
- **评判者差异。** 非专家与专家偏好不同。两者兼顾。  
- **提示语分布。** 精挑细选的提示可能偏向某个模型族。必须记录。  
- **大模型裁判奖励作弊。** GPT-4 裁判容易被漂亮但错误的输出欺骗。与人类一起验证。

## 结合使用

生产环境的评估报告应包括：

1. 10-30k 样本上的 FID，针对保留的真实分布（样本质量）。  
2. 相同样本对应提示上的 CLIP 分数 / CMMD（遵守度）。  
3. 对上一个版本的盲测竞赛胜率（整体偏好）。  
4. 失败模式分析：随机抽取 50 个输出，标记已知问题（手部解剖、文本渲染、一致的物体数量）。

单一指标都是谎言。三个互相佐证的指标 + 定性审查才是真正结论。

## 构建实现

`code/main.py` 实现了 FID、类似 CLIP 分数和 Elo 聚合，用于合成的“特征向量”（我们用 4 维向量作为 Inception 特征替代）。界面包括：

- 小样本 N 和大样本 N 的 FID 计算 —— 展示偏差。  
- 特征池间的“CLIP 分数”余弦相似度。  
- 基于合成偏好流的 Elo 更新规则。

### 步骤 1：四行代码实现 FID

```python
def fid(real_features, gen_features):
    mu_r, cov_r = mean_and_cov(real_features)
    mu_g, cov_g = mean_and_cov(gen_features)
    mean_diff = sum((a - b) ** 2 for a, b in zip(mu_r, mu_g))
    trace_term = trace(cov_r) + trace(cov_g) - 2 * sqrt_cov_product(cov_r, cov_g)
    return mean_diff + trace_term
```

### 步骤 2：CLIP 式余弦相似度

```python
def clip_like(image_feat, text_feat):
    dot = sum(a * b for a, b in zip(image_feat, text_feat))
    norm = math.sqrt(dot_self(image_feat) * dot_self(text_feat))
    return dot / max(norm, 1e-8)
```

### 步骤 3：Elo 聚合

```python
def elo_update(r_a, r_b, winner, k=32):
    expected_a = 1 / (1 + 10 ** ((r_b - r_a) / 400))
    actual_a = 1.0 if winner == "a" else 0.0
    r_a_new = r_a + k * (actual_a - expected_a)
    r_b_new = r_b - k * (actual_a - expected_a)
    return r_a_new, r_b_new
```

## 陷阱

- **FID 在 N=1000。** 经验法则在 N < 10k 时不可靠。报告低 N FID 的论文可能在作弊。  
- **跨分辨率比较 FID。** Inception 的 299×299 调整会改变特征分布。只在匹配分辨率下比较。  
- **只报告一个随机种子。** 至少跑三个种子，报告标准差。  
- **通过负面提示作弊提升 CLIP 分数。** 一些流程通过过拟合提示提升 CLIP 分数。检查图像是否过饱和。  
- **Elo 受提示重复偏差。** 如果两个模型训练时都见过测试提示，Elo 分数无意义。使用保留提示集。  
- **有人类评测付费人群偏差。** Prolific、MTurk 注释者年龄偏低、技术倾向强。与艺术/设计专家混合使用。

## 实际应用

2026 年生产评估方案：

| 支柱 | 最低要求 | 推荐配置 |
|------|----------|----------|
| 样本质量 | 保留真实集上的 10k FID | + 5k CMMD + 各类别子集 FID |
| 提示遵守 | 30k 上 CLIP 分数 | + HPSv2 + ImageReward + VQA 风格问答 |
| 偏好 | 200 对盲测对阵基线 | + 2000 对人工+LLM 裁判 + 聊天机器人赛场 |
| 失败分析 | 50 手工标注 | 500 手工标注 + 自动安全分类器 |

四项齐全的报告即是结论。任何单一项只是营销。

## 上线

保存 `outputs/skill-eval-report.md`。该技能输入新模型检查点和基线，输出完整评估计划：样本量、指标、失败模式探测、签发标准。

## 练习

1. **简单。** 运行 `code/main.py`，对相同合成分布比较 N=100 和 N=1000 的 FID，报告偏差大小。  
2. **中等。** 从合成 CLIP 式特征实现 CMMD（参见 Jayasumana 等人，2024 的公式），比较它对质量差异的敏感度与 FID。  
3. **困难。** 复现 HPSv2 设置：从 Pick-a-Pic 子集抽取 1000 对图像-提示，微调基于小型 CLIP 的偏好打分器，评测与保留集一致性。

## 关键术语

| 术语 | 常见说法 | 真实含义 |
|------|----------|----------|
| FID | “Fréchet Inception Distance” | 真实与生成 Inception 特征的高斯拟合 Fréchet 距离。 |
| CLIP 分数 | “文本-图像相似度” | CLIP 影像与文本嵌入的余弦相似度。 |
| CMMD | “FID 替代指标” | 基于 CLIP 特征的最大均值差异；较少偏差，无高斯假设。 |
| IS | “Inception 分数” | KL 散度期望值；与现代模型相关性差，已弃用。 |
| HPSv2 / ImageReward / PickScore | “学习型偏好代理” | 基于人类偏好训练的小模型，作为自动判官。 |
| Elo | “国际象棋评分” | Bradley-Terry 模型对配对胜负的综合评分。 |
| PartiPrompts | “基准提示集” | Google 筛选的 1,600 个提示，涵盖 12 类。 |
| FD-DINO | “自监督替代” | 使用 DINOv2 特征的 Fréchet 距离，更适合 ImageNet 之外领域。 |

## 生产注意：评估也是推理负载

对 10k 样本计算 FID 需生成 10k 张图。以单个 L4 GPU 运行 50 步的 SDXL base 1024²，约需 11 小时的单请求推理时间。评估预算是真实成本，其环境就是离线推理场景（最大吞吐，忽略首帧时间）：

- **批量处理，忽略延迟。** 离线评估即在能装入内存的最大批次上静态批处理。80GB H100 运行 `pipe(...).images`，`num_images_per_prompt=8`，墙钟速度比单请求快 4-6 倍。  
- **缓存真实特征。** Inception（FID）或 CLIP（CLIP-score，CMMD）对真实参考集的特征提取仅运行一次，保存为 `.npz` 文件。评估时勿重复计算。

对于 CI / 回归门禁：每个 PR 对 500 样本子集跑 FID + CLIP 分数（约30 分钟）；每天夜间跑完整 10k FID + HPSv2 + Elo。

## 拓展阅读

- [Heusel 等人（2017）。由双时尺度更新规则训练的 GAN 收敛至局部纳什均衡（FID）](https://arxiv.org/abs/1706.08500) — FID 论文。  
- [Jayasumana 等人（2024）。重新思考 FID：迈向更好的图像生成评估指标（CMMD）](https://arxiv.org/abs/2401.09603) — CMMD。  
- [Radford 等人（2021）。从自然语言监督学习可迁移视觉模型（CLIP）](https://arxiv.org/abs/2103.00020) — CLIP。  
- [Wu 等人（2023）。HPSv2：综合性人类偏好评分](https://arxiv.org/abs/2306.09341) — HPSv2。  
- [Xu 等人（2023）。ImageReward：学习并评估文本到图像生成中的人类偏好](https://arxiv.org/abs/2304.05977) — ImageReward。  
- [Yu 等人（2023）。扩大自回归模型用于内容丰富的文本到图像生成（Parti + PartiPrompts）](https://arxiv.org/abs/2206.10789) — PartiPrompts。  
- [Stein 等人（2023）。揭示生成模型评估指标的缺陷](https://arxiv.org/abs/2306.04675) — 失败模式调研。
