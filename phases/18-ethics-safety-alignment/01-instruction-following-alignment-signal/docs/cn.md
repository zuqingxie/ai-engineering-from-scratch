# 以指令遵循作为对齐信号

> 后续对 RLHF 的所有批评都针对这个流程。在你研究优化压力如何扭曲代理目标之前，你必须先了解这个代理目标。InstructGPT（Ouyang 等，2022）定义了参考架构：在指令-回应对上进行监督微调（supervised fine-tuning，SFT），训练一个根据偏好排名的奖励模型（reward model，RM），以及针对奖励模型的带 KL 惩罚项的 PPO 优化，惩罚项用于限制策略偏离 SFT 策略。一个 13 亿参数的 InstructGPT 被偏好于一个 1750 亿参数的 GPT-3。这个单一结果是 2026 年每个前沿实验室仍然使用 RLHF 后训练流程的原因。

**类型:** 学习  
**语言:** Python（标准库，玩具三阶段流程）  
**先修:** 阶段 10 · 06（SFT）、阶段 10 · 07（RLHF）、阶段 10 · 08（DPO）  
**时间:** 约 45 分钟

## 学习目标

- 说出 InstructGPT 流程的三个阶段及其各自使用的损失函数。
- 解释为何一个 13 亿参数的指令微调模型能在人工偏好评测中胜过原始的 1750 亿参数 GPT-3。
- 说明阶段 3 中 KL 惩罚项的保护作用，以及为何移除它会导致模式寻求行为崩溃。
- 描述对齐税（alignment tax）的含义及 Ouyang 等人用 PPO-ptx 进行的缓解方法。

## 问题

预训练语言模型会续写文本，而非回答问题。向 GPT-3 提问“写一个反转列表的 Python 函数”，你常会得到另一个提示词，因为训练中的大多数文本分布是网络文本，续写更多网络文本。模型做对了工作——工作目标却错误。

每个严肃团队用来解决此问题的代理目标是人类偏好。将两个生成结果交给标注者，由标注者选出更好的结果；奖励模型学习这一标注。然后用强化学习循环将策略朝着奖励模型得分高的输出调整。三句话总结 InstructGPT 的全部理论，其余就是工程细节。

## 概念

### 阶段 1：监督微调（SFT）

收集提示-回应对，回应是一个善意的人类会写出的答案。Ouyang 等使用了 1.3 万个标注者提示与 OpenAI API 生成的数据。在这些数据上对基础模型进行标准的交叉熵微调。

SFT 带来的效果：模型开始回答问题，而非续写问题。它没有带来的效果：对多个合理答案标注者的偏好信号。

### 阶段 2：奖励模型（RM）

对每个提示，从 SFT 模型抽样 K 个生成结果。标注者对它们进行排序。训练奖励模型，对任意提示-回应对进行评分，满足对于标注偏好 `y_w` 优于 `y_l` 的对：

```text
L_RM = -log sigmoid(r(x, y_w) - r(x, y_l))
```

这是 Bradley-Terry 配对偏好损失。奖励模型通常用 SFT 模型初始化，将语言模型头替换成标量头。

奖励模型较小：对于 1750 亿 InstructGPT，6B 参数就足够。它们也较脆弱——论文第 5 节主要讨论小规模展示出的奖励作弊行为。

### 阶段 3：带 KL 惩罚的 PPO

定义目标函数：

```text
J(pi) = E_{x~D, y~pi(.|x)} [ r(x, y) ] - beta * KL(pi(.|x) || pi_SFT(.|x))
```

用 PPO 最大化。KL 项保持策略 `pi` 不偏离 SFT 策略过远。若无此项，优化器会找到对抗样本——奖励模型未见过但打分很高的字符串，而非人类真的偏好。

KL 系数 `beta` 是最重要的 RLHF 超参数。太低：奖励作弊；太高：无法优于 SFT。

### 对齐税（alignment tax）

RLHF 后模型在人工偏好上优于旧模型，但在标准基准（SQuAD、HellaSwag、DROP）上表现下降。Ouyang 等称之为对齐税，用 PPO-ptx 解决：将预训练梯度混入 RL 目标，避免模型忘记未被奖励的下游任务。

```text
J_ptx(pi) = J(pi) + gamma * E_{x~D_pretrain} [ log pi(x) ]
```

PPO-ptx 现已成为标准，Anthropic、DeepMind 与 Meta 都使用某种变体。

### 结果

13 亿参数的 InstructGPT（SFT + RM + PPO-ptx）在人类标注者中大约 70% 的时间被偏好于 1750 亿参数基础 GPT-3。在来自生产流量的隐藏测试提示上，差距更大。由此可得到两点：

1. 对齐是不同于能力的另一坐标。1750 亿模型能力更强，13 亿模型更对齐；标注者偏好对齐模型。
2. 能力底层由基础模型决定。不能通过 RLHF 让基础模型掌握它未曾见过的事实。

### 为什么这是阶段 18 的参考点

后续课程的所有批评——奖励作弊（课程 2）、DPO（课程 3）、马屁精行为（课程 4）、CAI（课程 5）、内鬼（课程 7）、对齐伪装（课程 9）——都针对这个流程的某部分。奖励作弊攻击阶段 2；DPO 合并阶段 2 与 3；CAI 替代人类标注者；马屁行为暴露标注者信号有偏；对齐伪装表明策略可完全绕过阶段 3。没有先理解该流程，无法理解这些批评。

## 使用方法

`code/main.py` 在玩具偏好数据上模拟三阶段流程。基础“策略”是对动作 {A, B, C} 的有偏硬币。第一阶段 SFT 模拟标注者在 200 条提示上的动作。第二阶段基于 500 次配对排名训练 Bradley-Terry 奖励模型。第三阶段运行简化的 PPO，并带有针对 SFT 策略的 KL 惩罚项。你可以观察奖励上升、KL 散度增加和策略漂移——也可以关闭 KL 项，观察 50 步更新内出现奖励作弊。

留意：

- `beta = 0.1` 和 `beta = 0.0` 的奖励轨迹对比。
- 训练步数下的 KL(pi || pi_SFT) 变化。
- 最终动作分布与标注者偏好对比。

## 实践输出

本课生成 `outputs/skill-instructgpt-explainer.md`。给定一段 RLHF 流程描述或论文摘要，判断哪个阶段被修改，使用的损失，及是否存在 KL 惩罚或等效正则项。

## 练习

1. 运行 `code/main.py`。设置 `beta = 0.0`，报告 200 步 PPO 后的动作分布。用一段话解释模式寻求行为。
2. 修改奖励模型，使动作 B 获得 +0.5 偏置（模拟奖励漏洞）。用 `beta = 0.1` 运行 PPO，KL 惩罚能阻止策略利用漏洞吗？在什么 `beta` 下利用行为显现？
3. 阅读 Ouyang 等人论文（arXiv:2203.02155）图 1。运行 PPO 1、5、20、100 步，测量偏好与 SFT 模型对比，复现标注者偏好曲线。
4. 论文第 4.3 节报告 13 亿 InstructGPT 在约 70% 情况下胜过 1750 亿 GPT-3。为何这一比例在隐藏生产提示上会更高，而非标注者自有提示？
5. 用 DPO（阶段 10 · 08）代替 PPO 损失，保留同一偏好数据。比较最终策略漂移（KL 离 SFT）与最终奖励。哪个方法在相同奖励水平下漂移更远？

## 关键术语

| 术语         | 常见说法       | 实际含义                                  |
|--------------|----------------|-------------------------------------------|
| SFT          | “instruction tuning”（指令微调） | 阶段 1：基于提示-回应对进行交叉熵微调         |
| 奖励模型（RM） | “the RM”       | 以 Bradley-Terry 配对标签训练的提示-回应对标量回归模型 |
| Bradley-Terry | “pairwise preference loss”（配对偏好损失） | -log sigmoid(r_w - r_l)，将配对排序归约为二分类   |
| KL 惩罚      | “the regularizer”（正则项） | `beta * KL(pi \|\| pi_SFT)`，让 RL 策略靠近 SFT 策略   |
| PPO-ptx      | “PPO with pretraining mix”（带预训练混合的 PPO） | 向 PPO 目标加入预训练对数似然，抵消对齐税           |
| 对齐税       | “the RLHF regression”        | RLHF 后在未被奖励的标准基准上的性能下降           |
| 标注者偏好   | “the ground truth”            | 人类偏好样本；奖励模型是其统计代理，不等同于“人类价值” |

## 延伸阅读

- [Ouyang 等 — Training language models to follow instructions with human feedback（arXiv:2203.02155）](https://arxiv.org/abs/2203.02155) — InstructGPT 论文，是随后所有 RLHF 流程的基础  
- [Stiennon 等 — Learning to summarize from human feedback（arXiv:2009.01325）](https://arxiv.org/abs/2009.01325) — RLHF 用于摘要的前驱工作  
- [Christiano 等 — Deep reinforcement learning from human preferences（arXiv:1706.03741）](https://arxiv.org/abs/1706.03741) — 最初的基于偏好的强化学习表述  
- [Bai 等 — Training a Helpful and Harmless Assistant with RLHF（arXiv:2204.05862）](https://arxiv.org/abs/2204.05862) — Anthropic 基于 InstructGPT 流程的 HH 拓展
