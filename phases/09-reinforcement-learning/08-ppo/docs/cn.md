# Proximal Policy Optimization (PPO)

> A2C 在每次更新后就丢弃 rollouts。PPO 将策略梯度包裹在一个裁剪的重要性比率中，这样你可以对同一数据进行 10+ 个 epoch 的训练，而不会导致策略发散。Schulman 等人 (2017)。到 2026 年仍是默认的策略梯度算法。

**类型：** 构建  
**语言：** Python  
**先修课程：** 阶段 9 · 06 (REINFORCE), 阶段 9 · 07 (Actor-Critic)  
**时间：** ~75 分钟

## 问题

A2C（第 07 课）是 on-policy（基于当前策略）的：梯度 `E_{π_θ}[A · ∇ log π_θ]` 需要从 *当前* 的 `π_θ` 中采样数据。做一次更新，`π_θ` 就改变了；你之前用的数据就变成 off-policy（离策略）。如果重用，梯度就有偏差。

采样路径代价高。在 Atari 上，一次 rollout 是 8 个环境 × 128 步 = 1024 个转移和几十秒的环境时间。每次更新后丢弃这些数据很浪费。

Trust Region Policy Optimization（TRPO，Schulman 2015）是第一个解决方案：限制每次更新，使新旧策略之间的 KL 散度保持在 `δ` 以下。理论上很严谨，但每次更新都需用共轭梯度求解。到 2026 年没人用 TRPO。

PPO（Schulman 等，2017）用简单的裁剪目标函数代替了硬性信赖域约束。只需一行额外代码。每个 rollout 训练十个 epoch。没有共轭梯度。理论保证足够好。九年后，它仍是 MuJoCo 到 RLHF 等各种任务的默认策略梯度算法。

## 概念

![PPO clipped surrogate objective: ratio clipping at 1 ± ε](../assets/ppo.svg)

**重要性比率（importance ratio）。**

`r_t(θ) = π_θ(a_t | s_t) / π_{θ_old}(a_t | s_t)`

这是新策略相对收集数据的旧策略的似然比。`r_t = 1` 表示无变化。`r_t = 2` 表示新策略执行动作 `a_t` 的概率是旧策略的两倍。

**裁剪代理目标（clipped surrogate）。**

`L^{CLIP}(θ) = E_t [ min( r_t(θ) A_t, clip(r_t(θ), 1-ε, 1+ε) A_t ) ]`

两个部分：

- 如果优势 `A_t > 0`，且比率试图超过 `1 + ε`，裁剪限制梯度——不要让一个好的动作概率比旧策略高出超过 `+ε`。
- 如果优势 `A_t < 0`，且比率试图低于 `1 - ε`（意味着把坏动作概率相对于裁剪后的减少值变大），裁剪限制梯度——不要让坏动作概率降得比 `-ε` 更低。

`min` 函数处理另一个方向：如果比率朝有利方向移动，你仍然获得梯度（不会裁剪对你不利的方向）。

典型 `ε = 0.2`。将目标函数绘制为 `r_t` 的函数，它是分段线性函数，在“好”一侧有扁平顶，在“坏”一侧有扁平底。

**完整 PPO 损失。**

`L(θ, φ) = L^{CLIP}(θ) - c_v · (V_φ(s_t) - V_t^{target})² + c_e · H(π_θ(·|s_t))`

和 A2C 一样的 actor-critic 结构。三个超参通常 `c_v = 0.5`，`c_e = 0.01`，`ε = 0.2`。

**训练循环。**

1. 在 `N` 个并行环境中每个环境跑 `T` 步，收集 `N × T` 转移。
2. 计算优势（GAE），将其固定为常数。
3. 将当前 `π_θ` 的快照冻结为 `π_{θ_old}`。
4. 迭代 `K` 个 epoch，每个小批次 `(s, a, A, V_target, log π_old(a|s))`：
   - 计算 `r_t(θ) = exp(log π_θ(a|s) - log π_old(a|s))`。
   - 应用 `L^{CLIP}` + 价值损失 + 熵。
   - 梯度更新。
5. 丢弃 rollout，返回步骤 1。

`K=10`，小批次大小 64 是标准超参。PPO 很稳健：具体数值通常在 ±50% 范围内都行。

**KL 惩罚变体。** 原论文提出了使用自适应 KL 惩罚的替代方法：`L = L^{PG} - β · KL(π_θ || π_old)`，`β` 基于观测到的 KL 自动调整。裁剪版本后来成主流；KL 版本在 RLHF 中继续存在（因为对参考策略的 KL 总是一个独立约束）。

## 构建它

### 第一步：在 rollout 期间捕获 `log π_old(a | s)`

```python
for step in range(T):
    probs = softmax(logits(theta, state_features(s)))
    a = sample(probs, rng)
    s_next, r, done = env.step(s, a)
    buffer.append({
        "s": s, "a": a, "r": r, "done": done,
        "v_old": value(w, state_features(s)),
        "log_pi_old": log(probs[a] + 1e-12),
    })
    s = s_next
```

快照只在 rollout 时采集一次。更新多个 epoch 期间不变。

### 第二步：计算 GAE 优势（第 07 课）

和 A2C 一样。对整个批次做归一化。

### 第三步：裁剪代理目标更新

```python
for _ in range(K_EPOCHS):
    for mb in minibatches(buffer, size=64):
        for rec in mb:
            x = state_features(rec["s"])
            probs = softmax(logits(theta, x))
            logp = log(probs[rec["a"]] + 1e-12)
            ratio = exp(logp - rec["log_pi_old"])
            adv = rec["advantage"]
            surrogate = min(
                ratio * adv,
                clamp(ratio, 1 - EPS, 1 + EPS) * adv,
            )
            # 反向传播 -surrogate，加价值损失，减熵
            grad_logpi = onehot(rec["a"]) - probs
            if (adv > 0 and ratio >= 1 + EPS) or (adv < 0 and ratio <= 1 - EPS):
                pg_grad = 0.0  # 裁剪
            else:
                pg_grad = ratio * adv
            for i in range(N_ACTIONS):
                for j in range(N_FEAT):
                    theta[i][j] += LR * pg_grad * grad_logpi[i] * x[j]
```

“裁剪 → 梯度为零”模式是 PPO 核心。如果新策略已经在有利方向漂移过远，更新就停止。

### 第四步：价值和熵

添加标准均方误差（MSE）对 critic 目标，以及像 A2C 一样给 actor 一个熵奖励。

### 第五步：诊断

每次更新关注三件事：

- **平均 KL** `E[log π_old - log π_θ]`。应保持在 `[0, 0.02]`。若超出 `0.1`，减小 `K_EPOCHS` 或学习率（LR）。
- **裁剪比例（clip fraction）** —— 比率落在 `[1-ε, 1+ε]` 外的样本比例。应为 `~0.1-0.3`。若接近 0 裁剪从未触发 → 增大学习率或 epoch 数。若 ≥ 0.5 超拟合 rollout → 减少它们。
- **解释方差** `1 - Var(V_target - V_pred) / Var(V_target)`。评判价值网络质量。应逐渐接近 1。

## 陷阱

- **裁剪系数调节不当。** `ε = 0.2` 是事实标准。降到 `0.1` 导致更新过于保守；升到 `0.3+` 会不稳定。
- **epoch 太多。** `K > 20` 容易不稳定，因为策略与旧策略偏离太远。特别是大网络应限制 epoch。
- **无奖励归一化。** 奖励尺度太大侵占裁剪区间。先对奖励做归一化（动态标准差）再计算优势。
- **忘记优势归一化。** 按批次进行零均值单位方差归一化是标准操作，跳过它在多数基准上毁掉 PPO。
- **学习率不衰减。** PPO 受益于线性衰减到零的学习率。恒定学习率通常性能差。
- **重要性比率计算错误。** 始终用 `exp(log_new - log_old)` 保证数值稳定，不用直接除新旧概率。
- **梯度符号反了。** 最大化代理目标 = *最小化* `-L^{CLIP}`。符号翻转是 PPO 最常见的 bug。

## 使用它

PPO 到 2026 年已是诸多领域默认的 RL 算法：

| 用例 | PPO 变体 |
|------|----------|
| MuJoCo / 机器人控制 | 带高斯策略，GAE(0.95) 的 PPO |
| Atari / 离散游戏 | 带分类策略、128 步滚动 rollout 的 PPO |
| LLM 的 RLHF | 以参考模型为 KL 惩罚，结果由奖励模型（RM）在回复末尾提供的 PPO |
| 大规模游戏代理 | IMPALA + PPO （AlphaStar、OpenAI 五号） |
| 推理 LLM | GRPO（第 12 课）— 无价值网络的 PPO 变体 |
| 仅有偏好数据 | DPO — PPO+KL 的闭式折叠，纯离线版本 |

PPO 的 *损失形态*——裁剪代理 + 价值 + 熵——是 DPO，GRPO 以及几乎所有 RLHF 流水线的骨架。

## 发布它

保存为 `outputs/skill-ppo-trainer.md`：

```markdown
---
name: ppo-trainer
description: Produce a PPO training config and a diagnostic plan for a given environment.
version: 1.0.0
phase: 9
lesson: 8
tags: [rl, ppo, policy-gradient]
---

Given an environment and training budget, output:

1. Rollout size. `N` envs × `T` steps.
2. Update schedule. `K` epochs, minibatch size, LR schedule.
3. Surrogate params. `ε` (clip), `c_v`, `c_e`, advantage normalization on.
4. Advantage. GAE(`λ`) with explicit `γ` and `λ`.
5. Diagnostics plan. KL, clip fraction, explained variance thresholds with alerts.

Refuse `K > 30` or `ε > 0.3` (unsafe trust region). Refuse any PPO run without advantage normalization or KL/clip monitoring. Flag clip fraction sustained above 0.4 as drift.
```

## 练习

1. **简单。** 在 4×4 GridWorld 上运行 PPO，`ε=0.2, K=4`。与 A2C（每次 rollout 只训练一个 epoch）的样本效率比较，在相同环境步数下。
2. **中等。** 扫描 `K ∈ {1, 4, 10, 30}`。绘制回报与环境步数，并跟踪每次更新的平均 KL。哪一个 `K` 导致 KL 在此任务中爆炸？
3. **困难。** 用自适应 KL 惩罚替换裁剪代理（当 `KL > 2·target` 时 `β` 翻倍，`KL < target/2` 时减半）。比较最终回报、稳定性以及无裁剪特性。

## 关键词

| 术语 | 人们说 | 实际含义 |
|-------|--------|----------|
| Importance ratio（重要性比率） | “r_t(θ)” | `π_θ(a\|s) / π_old(a\|s)`；相对于收集数据时策略的偏差。 |
| Clipped surrogate（裁剪代理目标） | “PPO 的主要技巧” | `min(r·A, clip(r, 1-ε, 1+ε)·A)`；在有利方向裁剪后梯度变平。 |
| Trust region（信赖域） | “TRPO / PPO 目标” | 限制每次更新的 KL，保证单调改进。 |
| KL penalty（KL 惩罚） | “软信赖域” | PPO 的替代版本：`L - β · KL(π_θ \|\| π_old)`。自适应 `β`。 |
| Clip fraction（裁剪比例） | “裁剪触发频率” | 诊断指标—应为 0.1-0.3；超出表示调参不当。 |
| Multi-epoch training（多 epoch 训练） | “数据重用” | 针对每次 rollout 训练 K 个 epoch；以方差为代价换取样本效率提升。 |
| On-policy-ish（准 on-policy） | “大体上是 on-policy” | PPO 名义上是 on-policy，K>1 epoch 时安全地用略微 off-policy 的数据。 |
| PPO-KL（PPO 的另一种） | “另一种 PPO” | KL 惩罚变体；在 RLHF 中用得多，因为 KL 到参考策略是额外约束。 |

## 延伸阅读

- [Schulman 等 (2017)。Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) — 论文原文。
- [Schulman 等 (2015)。Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477) — TRPO，PPO 的前身。
- [Andrychowicz 等 (2021)。What Matters In On-Policy RL? A Large-Scale Empirical Study](https://arxiv.org/abs/2006.05990) — 大规模 PPO 超参数消融研究。
- [Ouyang 等 (2022)。Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) — InstructGPT；RLHF 中 PPO 的配方。
- [OpenAI Spinning Up — PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html) — 干净现代的 PPO 介绍，基于 PyTorch。
- [CleanRL PPO implementation](https://github.com/vwxyzjn/cleanrl) — 许多论文的参考单文件 PPO 实现。
- [Hugging Face TRL — PPOTrainer](https://huggingface.co/docs/trl/main/en/ppo_trainer) — 语言模型 PPO 生产流水线；配合第 09 课 RLHF 学习。
- [Engstrom 等 (2020)。Implementation Matters in Deep Policy Gradients](https://arxiv.org/abs/2005.12729) — “37 个代码级优化”论文；哪些 PPO 技巧是关键，哪些是民间传说。
