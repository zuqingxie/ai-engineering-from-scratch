# Reward Modeling & RLHF（奖励建模与人类反馈强化学习）

> 人类无法为“良好助理回应”编写奖励函数，但他们可以比较两个回应并选择更好的那个。拟合一个奖励模型以符合这些比较，然后使用强化学习（RL）对语言模型进行优化。Christiano 2017。InstructGPT 2022。将 GPT-3 转变为 ChatGPT 的关键方法。2026 年此方法主要被 DPO 取代——但思维模型依旧存在。

**类型:** 构建  
**语言:** Python  
**先决条件:** 第5阶段 · 05（情感分析），第9阶段 · 08（PPO）  
**时间:** 约45分钟

## 问题

你已经使用下一词预测（next-token-prediction）目标训练了语言模型。它可以写出语法正确的英文，但也会撒谎、漫无边际地废话、且拒绝拒绝。无法通过更多预训练解决这个问题——网络文本本身是问题所在，而非解决方案。

你想要一个*标量奖励*，用来表示“对于指令 X，回应 A 比回应 B 更好”。手写这种奖励函数是不可能的。“有帮助性”（helpfulness）不是一个基于标记的封闭形式表达式。但人类能比较两个输出并标记偏好，这种数据便宜且可大规模收集。

RLHF（Christiano 等 2017；Ouyang 等 2022）将偏好转化为奖励模型，然后通过 PPO 优化语言模型以最大化该奖励。三步过程：SFT → RM → PPO。这是实现 ChatGPT、Claude、Gemini 及2023–2025年其他所有对齐大语言模型（aligned-LLM）的秘诀。

2026 年，PPO步骤主要被 DPO（第10阶段 · 08）取代，因为它更廉价且对齐调优效果几乎一样好。但*奖励模型*环节仍是所有 Best-of-N 采样器、所有基于可验证奖励的RL流程，以及所有使用过程奖励模型的推理模型的基础。理解 RLHF，你就理解了整个对齐技术栈。

## 概念

![三阶段 RLHF：SFT、基于成对偏好的 RM 训练、带 KL 惩罚的 PPO](../assets/rlhf.svg)

**阶段 1：监督微调（Supervised Fine-Tuning, SFT）。** 从预训练基础模型开始，微调于人类编写的示范目标行为（指令跟随型回应，有帮助的答复等）。结果：模型 `π_SFT` *偏向于良好行为*，但动作空间仍无限制。

**阶段 2：奖励模型训练。**

- 收集对提示 `x` 的回应对 `(y_+, y_-)`，由人类标注为“y_+优于y_-”。
- 训练奖励模型 `R_φ(x, y)`，使其对 `y_+` 赋予更高的分数。
- 损失函数：**Bradley-Terry 成对逻辑回归**：

  `L(φ) = -E[ log σ(R_φ(x, y_+) - R_φ(x, y_-)) ]`

  σ为 sigmoid 函数。奖励差值对应偏好的对数赔率。BT 模型自1952年（Bradley-Terry）以来一直是标准，且是现代 RLHF 的主流选择。

- `R_φ` 通常从 SFT 模型初始化，顶部加入一个标量头。保持相同 Transformer（Transformer 架构）骨干；单层线性输出奖励。

**阶段 3：基于带 KL 惩罚的 RM 的 PPO。**

- 从 `π_SFT` 初始化可训练策略 `π_θ`。保持冻结的*参考策略* `π_ref = π_SFT`。
- 回应 `y` 结束时的奖励：

  `r_total(x, y) = R_φ(x, y) - β · KL(π_θ(·|x) || π_ref(·|x))`

  KL惩罚防止 `π_θ` 与 `π_SFT` 偏离过远——它是*正则项*，不是硬性信赖区域。β通常取0.01至0.05。
- 用此奖励运行 PPO（第08课）。优势值基于逐标记轨迹计算，但RM只给整条回应打分。

**为什么加 KL？** 无它，PPO将轻松找到奖励作弊策略——RM仅对分布内的完成结果训练。分布外回应可能得分比任何人类写作都高。KL保持 `π_θ` 接近 RM 训练的流形。这是 RLHF 中最重要的调节器。

**2026状态：**

- **DPO**（Rafailov 2023）：闭式代数将阶段2+3合并为单一监督偏好损失。无需RM，无需PPO。对齐基准表现相当，算力消耗大幅减少。覆盖于第10阶段 · 08。
- **GRPO**（DeepSeek 2024–2025）：使用组相对基线替代critic的PPO，奖励来自*验证器*（代码运行结果／数学答案匹配），非人类训练的RM。推理模型主流方法。覆盖于第9阶段 · 12。
- **过程奖励模型（Process Reward Models，PRMs）：** 对部分解（每个推理步骤）评分，既用于 RLHF 又用于 GRPO 变体的推理环节。
- **宪法AI / RLAIF（Reinforcement Learning with AI Feedback）：** 用对齐的大语言模型生成偏好，替代人类标注，增加偏好预算。

## 构建它

本课使用小型合成“提示”和“回应”，均为字符串。RM是基于词袋（bag-of-tokens）表示的线性评分器。没有真实LLM——重要的是流程*形态*，非规模。详见 `code/main.py`。

### 第1步：合成偏好数据

```python
PROMPTS = ["help me", "answer me", "explain this"]
GOOD_WORDS = {"clear", "specific", "kind", "thorough"}
BAD_WORDS = {"vague", "rude", "wrong", "short"}

def make_pair(rng):
    x = rng.choice(PROMPTS)
    y_good = rng.choice(list(GOOD_WORDS)) + " " + rng.choice(list(GOOD_WORDS))
    y_bad = rng.choice(list(BAD_WORDS)) + " " + rng.choice(list(BAD_WORDS))
    return (x, y_good, y_bad)
```

在真实 RLHF 中，此处由人工标注替代。形式——`(prompt, preferred_response, rejected_response)`——完全相同。

### 第2步：Bradley-Terry 奖励模型

线性评分：`R(x, y) = w · bag(y)`。训练使BT成对对数损失最小化：

```python
def rm_train_step(w, x, y_pos, y_neg, lr):
    r_pos = dot(w, bag(y_pos))
    r_neg = dot(w, bag(y_neg))
    p = sigmoid(r_pos - r_neg)
    for tok, cnt in bag(y_pos).items():
        w[tok] += lr * (1 - p) * cnt
    for tok, cnt in bag(y_neg).items():
        w[tok] -= lr * (1 - p) * cnt
```

几百次更新后，`w` 会给好词标记正权重，坏词负权重。

### 第3步：在RM之上的类PPO策略

我们的玩具策略从词汇表中只产出单个标记。对该标记打分，计算 `log π_θ(token | prompt)`，加上KL惩罚，然后执行裁剪PPO代理。

```python
def rlhf_step(theta, ref, w, prompt, rng, eps=0.2, beta=0.1, lr=0.05):
    logits_theta = policy_logits(theta, prompt)
    probs = softmax(logits_theta)
    token = sample(probs, rng)
    logits_ref = policy_logits(ref, prompt)
    probs_ref = softmax(logits_ref)
    reward = dot(w, bag([token])) - beta * kl(probs, probs_ref)
    # ppo-style update on theta, treating reward as the return
    ...
```

### 第4步：监控 KL

每次更新跟踪平均 `KL(π_θ || π_ref)`。若超过约5-10，说明策略与 `π_SFT` 偏离严重——β 太低或奖励作弊开始。此为真实 RLHF 中的首要诊断指标。

### 第5步：使用 TRL 的生产流程

理解玩具流程后，下面是实际库用户代码。Hugging Face 的 [TRL](https://huggingface.co/docs/trl) 是参考实现——阶段2用 `RewardTrainer`，阶段3用带内置KL的 `PPOTrainer`。

```python
# 阶段 2：基于成对偏好的奖励模型训练
from trl import RewardTrainer, RewardConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
rm = AutoModelForSequenceClassification.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct", num_labels=1
)

# 数据集行结构：{"prompt", "chosen", "rejected"} — Bradley-Terry 格式
trainer = RewardTrainer(
    model=rm,
    tokenizer=tok,
    train_dataset=preference_data,
    args=RewardConfig(output_dir="./rm", num_train_epochs=1, learning_rate=1e-5),
)
trainer.train()
```

```python
# 阶段 3：对抗 RM 进行带 KL 惩罚的 PPO，KL 以 SFT 参考为准
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

policy = AutoModelForCausalLMWithValueHead.from_pretrained("./sft-checkpoint")
ref    = AutoModelForCausalLMWithValueHead.from_pretrained("./sft-checkpoint")  # 冻结参数

ppo = PPOTrainer(
    config=PPOConfig(learning_rate=1.41e-5, batch_size=64, init_kl_coef=0.05,
                     target_kl=6.0, adap_kl_ctrl=True),
    model=policy, ref_model=ref, tokenizer=tok,
)

for batch in dataloader:
    responses = ppo.generate(batch["query_ids"], max_new_tokens=128)
    rewards   = rm(torch.cat([batch["query_ids"], responses], dim=-1)).logits[:, 0]
    stats     = ppo.step(batch["query_ids"], responses, rewards)
    # stats 包含：mean_kl, clip_frac, value_loss —— 三个PPO诊断值
```

库自动做三件事。`adap_kl_ctrl=True` 实现自适应β调节：若观测KL超出 `target_kl`，β加倍；低于半数则减半。参考模型默认冻结——不能与 `policy` 误共享参数。价值头与策略共享骨干（`AutoModelForCausalLMWithValueHead`附加标量MLP头），故TRL能分别报告 `policy/kl` 和 `value/loss`。

## 陷阱

- **过度优化 / 奖励作弊。** RM不完美，`π_θ` 会找到得分高但实际糟糕的对抗性回答。表现为奖励无限上升，而人工评估得分停滞或下降。解决：提早停止、升高β、扩大RM训练数据。
- **长度作弊。** RM通常隐式奖励长回应，策略学会填充冗长回复。解决方案：长度归一化奖励，或使用带长度感知RM的RLAIF。
- **RM太小。** RM规模需不小于策略。过小RM无法准确评估策略输出。
- **KL调节。** β太低→漂移及奖励作弊。β太高→策略几乎无变化。标准做法是目标固定KL的*自适应β*。
- **偏好数据噪声。** 人类标签约30%存在噪声或模糊。通过在过滤一致性后的数据上训练RM或在BT模型使用温度校准。
- **离策略问题。** PPO数据在首轮后轻微离策略。监控裁剪比例（clip fraction），详见第08课。

## 使用它

2026年RLHF层级划分：

| 层级                 | 目标                     | 方法                                  |
|----------------------|--------------------------|-------------------------------------|
| 指令跟随、有帮助性、安全无害性 | 对齐（Alignment）          | 优先采用 DPO（第10阶段 · 08），替代RLHF-PPO。  |
| 推理正确性（数学、代码）    | 能力（Capability）         | 使用带验证器奖励的 GRPO（第9阶段 · 12）。     |
| 长周期多步任务           | 行为体（Agentic）          | 使用带过程奖励模型的 PPO / GRPO。           |
| 安全/拒绝行为           | 安全性（Safety）           | 使用带独立安全RM的 RLHF-PPO，或宪法AI。         |
| 推理时Best-of-N选择      | 快速对齐                  | 推理时使用RM，无需训练策略。                |
| 奖励蒸馏               | 推理计算量                | 在冻结LM上训练小型“奖励头”。               |

RLHF 是 2022–2024 年的主流方法。2026年生产管线首选DPO，仅在RM密集或安全关键步骤使用PPO。

## 交付它

保存为 `outputs/skill-rlhf-architect.md`:

```markdown
---
name: rlhf-architect
description: Design an RLHF / DPO / GRPO alignment pipeline for a language model, including RM, KL, and data strategy.
version: 1.0.0
phase: 9
lesson: 9
tags: [rl, rlhf, alignment, llm]
---

Given a base LM, a target behavior (alignment / reasoning / refusal / agent), and a preference or verifier budget, output:

1. Stage. SFT? RM? DPO? GRPO? With justification.
2. Preference or verifier source. Humans, AI feedback, rule-based, unit-test-pass, or reward distillation.
3. KL strategy. Fixed β, adaptive β, or DPO (implicit KL).
4. Diagnostics. Mean KL, reward stability, over-optimization guard (holdout human eval).
5. Safety gate. Red-team set, refusal rate, safety RM separate from helpfulness RM.

Refuse to ship RLHF-PPO without a KL monitor. Refuse to use an RM smaller than the target policy. Refuse length-only rewards. Flag any pipeline that does not hold back a blind human-eval set as lacking over-optimization protection.
```

## 练习

1. **简单。** 在 `code/main.py` 中使用 500 个人工合成偏好对训练 Bradley-Terry 奖励模型。测量保留的 100 对上的成对准确率，应该超过 90%。
2. **中等。** 运行 β ∈ {0.0, 0.1, 1.0} 的玩具版 PPO-RLHF 循环。针对每个 β，绘制 RM 分数与更新过程中 KL-to-reference 的关系图。哪些运行存在 reward-hack（奖励欺骗）？
3. **困难。** 在相同的偏好数据上实现 DPO（闭式偏好似然损失），并与 RLHF-PPO 流水线在计算消耗和最终 RM 分数上进行比较。

## 关键词

| 术语 | 人们如何说 | 实际含义 |
|------|------------|----------|
| RLHF | “对齐强化学习” | 三阶段的 SFT + RM + PPO 流水线（Christiano 2017，Ouyang 2022）。 |
| Reward Model (RM) | “评分网络” | 通过 Bradley-Terry 拟合成对偏好的学习标量函数。 |
| Bradley-Terry | “成对逻辑损失” | `P(y_+ ≻ y_-) = σ(R(y_+) - R(y_-))`；标准 RM 目标函数。 |
| KL penalty | “保持接近参考模型” | 奖励中加入的 `β · KL(π_θ \|\| π_ref)`；防止奖励欺骗的正则项。 |
| Reward hacking | “古德哈特法则” | 策略利用 RM 的缺陷；表现为奖励上升但人工评估无变化。 |
| RLAIF | “AI 标注的偏好” | RLHF，其中偏好标签来自另一个语言模型，而非人工。 |
| PRM | “过程奖励模型” | 评分推理过程中的部分推理步骤；用于推理流水线中。 |
| Constitutional AI | “Anthropic 的方法” | 由 AI 生成且受明确规则引导的偏好。 |

## 延伸阅读

- [Christiano et al. (2017). Deep Reinforcement Learning from Human Preferences](https://arxiv.org/abs/1706.03741) — 启动 RLHF 的开创论文。
- [Ouyang et al. (2022). InstructGPT — Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) — ChatGPT 背后的训练方案。
- [Stiennon et al. (2020). Learning to summarize with human feedback](https://arxiv.org/abs/2009.01325) — 早期基于 RLHF 的摘要训练。
- [Rafailov et al. (2023). Direct Preference Optimization](https://arxiv.org/abs/2305.18290) — DPO；2026 年后的 RLHF 默认方法。
- [Bai et al. (2022). Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) — RLAIF 与自我批判循环。
- [Anthropic RLHF paper (Bai et al. 2022). Training a Helpful and Harmless Assistant](https://arxiv.org/abs/2204.05862) — HH 论文。
- [Hugging Face TRL library](https://huggingface.co/docs/trl) — 生产级 `RewardTrainer` 和 `PPOTrainer`。阅读训练器源码了解自适应 KL 和价值头细节。
- [Hugging Face — Illustrating Reinforcement Learning from Human Feedback](https://huggingface.co/blog/rlhf) by Lambert, Castricato, von Werra, Havrilla — 使用图示详细讲解三阶段流水线。
- [von Werra et al. (2020). TRL: Transformer Reinforcement Learning](https://github.com/huggingface/trl) — 该库；`examples/` 目录包含 Llama、Mistral 和 Qwen 的端到端 RLHF 脚本。
- [Sutton & Barto (2018). Ch. 17.4 — Designing Reward Signals](http://incompleteideas.net/book/RLbook2020.pdf) — 奖励假设视角；理解奖励欺骗的基本必备。
