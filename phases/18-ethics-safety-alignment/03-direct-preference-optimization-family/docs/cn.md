# 直接偏好优化家族

> Rafailov 等人（2023）展示了 RLHF 的最优解在偏好数据上具有封闭形式，因此可以跳过显式的 reward model（奖励模型）而直接优化 policy（策略）。这一见解催生了一系列方法——IPO、KTO、SimPO、ORPO、BPO——每个方法都修正了 DPO 的一个失效模式。到了 2026 年，直接对齐算法在训练后运行前沿任务中的部署数量超过了 PPO。但第 2 课中的过度优化曲线仍然适用：DAAs（直接对齐算法）并未逃脱 Goodhart 法则，只是将其影响转移到了别处。

**类型：** 学习  
**语言：** Python（标准库，六变体偏好损失比较器）  
**先修知识：** 阶段 18 · 01（InstructGPT），阶段 18 · 02（奖励黑客），阶段 10 · 08（DPO 基础）  
**时长：** 约 75 分钟

## 学习目标

- 推导出 RLHF 带 KL 正则化最优解的 DPO 封闭形式。
- 阐述 IPO、KTO、SimPO、ORPO、BPO 各自修复 DPO 的哪种失败模式。
- 区分“隐式奖励差距”与“偏好强度”，并解释为何 IPO 的恒等映射重要。
- 说明 Rafailov 等人（NeurIPS 2024）如何证明 DAAs 会过度优化，即使没有显式的奖励模型。

## 问题描述

RLHF 目标（第 1 课）：

```text
max_pi E_{x,y~pi} [ r(x, y) ] - beta * KL(pi || pi_ref)
```

其已知最优解为：

```text
pi*(y|x) = (1/Z(x)) * pi_ref(y|x) * exp(r(x, y) / beta)
```

因此奖励隐式由最优策略与参考策略的比率定义：

```text
r(x, y) = beta * log(pi*(y|x) / pi_ref(y|x)) + beta * log Z(x)
```

将其代入 Bradley-Terry 偏好似然中，分区函数 `Z(x)` 被消去，因为它只依赖于 `x`。剩下的损失仅关于策略参数——无需奖励模型，这就是 DPO。

难点在于推导假设最优解可达、偏好数据在分布内且参考策略是真实模式锚点，这些假设都不完全成立。家族中每个方法修正了不同的违背假设。

## 核心概念

### DPO（Rafailov 等，2023）

```text
L_DPO = -log sigmoid(
  beta * log(pi(y_w | x) / pi_ref(y_w | x))
  - beta * log(pi(y_l | x) / pi_ref(y_l | x))
)
```

可能出现的问题：

- 隐式奖励差距 `beta * (log(pi/pi_ref)_w - log(pi/pi_ref)_l)` 无界。极小的偏好也可能产生任意大的差距。
- 损失驱动选择的和拒绝的对数概率朝相反方向调整。这可能导致被选的绝对对数概率下降，只要拒绝的下降更快。这就是“退化的选择响应”现象。
- 分布外偏好（如稀有对稀有）导致任意的隐式奖励。

### IPO（Azar 等，2024）

Identity Preference Optimization 将对数 sigmoid 替换为偏好概率的恒等映射，损失变为一个有界目标的平方误差：

```text
L_IPO = (log(pi(y_w | x) / pi_ref(y_w | x)) - log(pi(y_l | x) / pi_ref(y_l | x)) - 1/(2 beta))^2
```

边界由 `1/(2 beta)` 限制。偏好强度和隐式奖励差距成比例，无爆炸。

### KTO（Ethayarajh 等，2024）

Kahneman-Tversky Optimization 完全抛弃成对结构。给定单个带标签输出和二元的“期望”或“不期望”信号，将其映射到前景理论效用中：

```text
v(x, y) = sigma(beta * log(pi(y|x) / pi_ref(y|x)) - z_ref)
```

对收益与损失采用不同权重（损失厌恶）。优点是可以使用非配对数据，数据量更丰富。

### SimPO（Meng 等，2024）

Simple Preference Optimization 使训练信号与生成保持一致。移除参考政策，按长度标准化对数似然：

```text
L_SimPO = -log sigmoid(
  (beta / |y_w|) * log pi(y_w | x)
  - (beta / |y_l|) * log pi(y_l | x)
  - gamma
)
```

引入边界参数 `gamma` 以稳定训练。长度归一消除了利用 DPO 长度偏差失败模式的动机（更长的 `y_w` 会自然造成更大对数概率差距）。

### ORPO（Hong 等，2024）

Odds-Ratio Preference Optimization 在标准 SFT 负对数似然上增加了偏好项：

```text
L_ORPO = L_NLL(y_w) + lambda * L_OR
L_OR = -log sigmoid(log(odds(y_w) / odds(y_l)))
```

无参考策略 —— SFT 项作为正则项。从基础模型单阶段训练到对齐模型，无需分离的 SFT 检查点。

### BPO（ICLR 2026 投稿，OpenReview id=b97EwMUWu7）

识别出退化选择响应问题：DPO 保持了排名关系 `y_w > y_l`，但 `y_w` 的绝对对数概率会下降。BPO 添加了单行修正，惩罚选择响应的减少。报告在 Llama-3.1-8B-Instruct 的数学推理上比 DPO 准确率提升 +10.1%。

### 通用结论：DAAs 仍会过度优化

Rafailov 等人“Scaling Laws for Reward Model Overoptimization in Direct Alignment Algorithms”（NeurIPS 2024）使用 DPO、IPO、SLiC 在多个数据集和 KL 预算上训练策略。黄金奖励-与-KL 曲线展现与 Gao 等人类似的峰值-崩溃形态。隐式奖励在训练中查询了分布外样本，KL 正则无法稳定该过程。

DAAs 并未摆脱 Goodhart 法则。它们将“奖励模型过度优化”转变为“参考策略比率过度优化”。通用的解决方案——更好的数据、集成、早停——对于两者同样适用。

### 选择建议（2026）

- 拥有大量成对偏好数据时：使用保守 beta 的 DPO，若存在长度偏差则选 SimPO。
- 只有非配对二元反馈时：使用 KTO。
- 想要从基础模型起单阶段管线时：使用 ORPO。
- 看到 DPO 日志中选中响应的对数概率退化时：用 BPO。
- 偏好强度差异大且 DPO 饱和时：用 IPO。

各实验室都会对五种方法全套测试，根据任务选择最佳。数学推理和安全任务的最优解不必相同。

## 使用方法

`code/main.py` 在玩具偏好数据集上比较六种损失（DPO、IPO、KTO、SimPO、ORPO、BPO），真实偏好强度随样本对变化。每种损失都在相同的 500 对样本和一个小软最大策略上优化。绘制最终胜率、选中对数概率漂移和隐式奖励分布等指标。

## 部署建议

本课生成 `outputs/skill-preference-loss-selector.md` 文件。给定数据集统计（成对或非成对，偏好强度均匀或有变，长度分布）和目标（单阶段或先 SFT 后偏好），推荐合适的偏好损失并报告其防护的失效模式。

## 练习

1. 运行 `code/main.py`。报告 DPO 和 BPO 最终的选中对数概率下降。BPO 应保留更高的选中绝对概率，验证此点。

2. 修改偏好数据使所有对的偏好强度相等。哪个方法最稳健？哪个最易退化？解释 IPO 的优势。

3. 使被拒绝响应的平均长度是选择响应的 2 倍。在其他条件不变下，数值展示 DPO 长度利用的失败模式及 SimPO 的修正。

4. Rafailov 等（NeurIPS 2024）称 DAAs 过度优化。重现单点版本：绘制选中与被拒绝的 KL 散度差，观察大 beta 下 DPO 的过拟合。

5. 阅读 BPO 论文摘要（OpenReview b97EwMUWu7）。写出 BPO 对 DPO 的一行修正。与 `code/main.py` 实现确认。

## 关键词

| 术语       | 常见说法                 | 实际含义                                     |
|------------|--------------------------|----------------------------------------------|
| DPO        | “没有奖励模型的 RLHF”    | 源自带 KL 的 RLHF 最优解推导；仅优化策略参数 |
| 隐式奖励   | “对数比率”               | `beta * log(pi(y\|x) / pi_ref(y\|x))` —— DPO 隐含奖励 |
| IPO        | “有界 DPO”               | 将对数 sigmoid 替换为恒等；隐式奖励差距被 `1/(2 beta)` 限制 |
| KTO        | “非配对 DPO”             | 前景理论效用，使用带损失厌恶的单标签数据          |
| SimPO      | “无参考 DPO”             | 长度归一对数似然加边界，无参考策略               |
| ORPO       | “单阶段 DPO”             | NLL 加 odds-ratio 偏好项；基模型单轮训练         |
| BPO        | “选中保留 DPO”           | 在 DPO 上添加惩罚选中响应绝对对数概率下降的项     |
| 退化的选中 | “选中概率下降”           | DPO 允许选中对数概率降低，只要拒绝下降更快         |
| DAA        | “直接对齐算法”           | 任意跳过显式奖励模型的偏好损失方法                |

## 进一步阅读

- [Rafailov 等 — Direct Preference Optimization（NeurIPS 2023，arXiv:2305.18290）](https://arxiv.org/abs/2305.18290)
- [Azar 等 — A General Theoretical Paradigm to Understand Learning from Human Preferences（AISTATS 2024，arXiv:2310.12036）](https://arxiv.org/abs/2310.12036) — IPO
- [Ethayarajh 等 — KTO: Model Alignment as Prospect Theoretic Optimization（arXiv:2402.01306）](https://arxiv.org/abs/2402.01306)
- [Meng, Xia, Chen — SimPO（NeurIPS 2024，arXiv:2405.14734）](https://arxiv.org/abs/2405.14734)
- [Hong, Lee, Thorne — ORPO（EMNLP 2024，arXiv:2403.07691）](https://arxiv.org/abs/2403.07691)
- [BPO — Behavior Preservation Optimization（ICLR 2026 OpenReview b97EwMUWu7）](https://openreview.net/forum?id=b97EwMUWu7)
- [Rafailov 等 — Scaling Laws for RM Overoptimization in DAAs（NeurIPS 2024，arXiv:2406.02900）](https://arxiv.org/abs/2406.02900)
