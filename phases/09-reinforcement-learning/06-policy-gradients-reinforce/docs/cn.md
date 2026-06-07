# 策略梯度 — 从零开始实现 REINFORCE

> 不再估计价值。直接参数化策略，计算期望回报的梯度，向上爬坡。Williams（1992）用一个定理阐述了这一点。这也是 PPO、GRPO 以及所有大语言模型（LLM）强化学习循环存在的原因。

**类型：** 实现  
**语言：** Python  
**先决条件：** 阶段 3 · 03（反向传播），阶段 9 · 03（蒙特卡洛），阶段 9 · 04（时序差分学习）  
**时间：** 约 75 分钟

## 问题

Q-learning 和 DQN 参数化的是*价值（value）*函数。你通过 `argmax Q` 来选择动作。这对离散动作和离散状态来说没问题。但动作连续时（对一个10维扭矩做 `argmax`？）或当你想要随机策略时（`argmax` 从定义上就是确定的），这个方法就不适用。

策略梯度则直接参数化*策略（policy）*。`π_θ(a | s)` 是一个神经网络，输出行动的分布。我们从中采样执行动作。计算期望回报相对于 `θ` 的梯度，然后向上更新。没有 `argmax`，没有贝尔曼递归。就是对 `J(θ) = E_{π_θ}[G]` 做梯度上升。

REINFORCE 定理（Williams 1992）告诉你这个梯度是可计算的：  
`∇J(θ) = E_π[ G · ∇_θ log π_θ(a | s) ]`。  
运行一条轨迹，计算回报，乘以每步的 `∇ log π_θ(a | s)`。平均，梯度上升。完成。

2026年所有的 LLM-RL 算法——PPO、DPO、GRPO——都是 REINFORCE 的改进。对它熟练掌握是本阶段其余内容的先决条件，也是阶段 10 · 07（RLHF 实现）和阶段 10 · 08（DPO）的基础。

## 概念

![策略梯度：softmax 策略，log-π 梯度，按回报加权的更新](../assets/policy-gradient.svg)

**策略梯度定理。** 对任意策略 `π_θ`，参数为 `θ`：

`∇J(θ) = E_{τ ~ π_θ}[ Σ_{t=0}^{T} G_t · ∇_θ log π_θ(a_t | s_t) ]`

其中 `G_t = Σ_{k=t}^{T} γ^{k-t} r_{k+1}` 是从时刻 `t` 开始的折扣回报。期望是对从 `π_θ` 采样的完整轨迹 `τ`。

**证明很短。** 对期望中的 `J(θ) = Σ_τ P(τ; θ) G(τ)` 求导。利用 `∇P(τ; θ) = P(τ; θ) ∇ log P(τ; θ)`（对数求导技巧）。分解 `log P(τ; θ) = Σ log π_θ(a_t | s_t) + 不依赖 θ 的环境项`，环境项消失，几行代数推导得出定理。

**方差减少技巧。** 纯 REINFORCE 方差极大——回报很嘈杂，`∇ log π` 也嘈杂，两者乘积更噪声。两个标准改进：

1. **基线减法。** 用 `G_t - b(s_t)` 替代 `G_t`，`b(s_t)` 不依赖于动作 `a_t`。无偏，因为 `E[b(s_t) · ∇ log π(a_t | s_t)] = 0`。典型选择是用一个评论家学习的 `b(s_t) = V̂(s_t)` → 演化成 actor-critic（第07课）。
2. **奖励到达后（Reward-to-go）。** 用 `Σ_t G_t^{from t} · ∇ log π_θ(a_t | s_t)` 替代整体回报乘积。针对某动作，只考虑未来回报，过去奖励增加零均值噪声。

结合后得到：

`∇J ≈ (1/N) Σ_{i=1}^{N} Σ_{t=0}^{T_i} [ G_t^{(i)} - V̂(s_t^{(i)}) ] · ∇_θ log π_θ(a_t^{(i)} | s_t^{(i)})`

这就是带基线的 REINFORCE—A2C（第07课）和 PPO（第08课）的直接祖先。

**Softmax 策略参数化。** 离散动作的常用选择：

`π_θ(a | s) = exp(f_θ(s, a)) / Σ_{a'} exp(f_θ(s, a'))`

`f_θ` 是输出每动作得分的神经网络。梯度形式干净：

`∇_θ log π_θ(a | s) = ∇_θ f_θ(s, a) - Σ_{a'} π_θ(a' | s) ∇_θ f_θ(s, a')`

即采取动作的得分减去在策略下的期望得分。

**连续动作的高斯策略。** `π_θ(a | s) = N(μ_θ(s), σ_θ(s))`。`∇ log N(a; μ, σ)` 有闭式表达。这是阶段 9 · 07 中 SAC 所需的全部。

## 实现步骤

### 第一步：softmax 策略网络

```python
def policy_logits(theta, state_features):
    return [dot(theta[a], state_features) for a in range(N_ACTIONS)]

def softmax(logits):
    m = max(logits)
    exps = [exp(l - m) for l in logits]
    Z = sum(exps)
    return [e / Z for e in exps]
```

用线性策略（每动作一个权重向量）应对表格环境。对于 Atari，替换为 CNN 保留 softmax 头。

### 第二步：采样及对数概率

```python
def sample_action(probs, rng):
    x = rng.random()
    cum = 0
    for a, p in enumerate(probs):
        cum += p
        if x <= cum:
            return a
    return len(probs) - 1

def log_prob(probs, a):
    return log(probs[a] + 1e-12)
```

### 第三步：采样轨迹并记录 log-probs

```python
def rollout(theta, env, rng, gamma):
    trajectory = []
    s = env.reset()
    while not done:
        logits = policy_logits(theta, s)
        probs = softmax(logits)
        a = sample_action(probs, rng)
        s_next, r, done = env.step(s, a)
        trajectory.append((s, a, r, probs))
        s = s_next
    return trajectory
```

### 第四步：REINFORCE 更新

```python
def reinforce_step(theta, trajectory, gamma, lr, baseline=0.0):
    returns = compute_returns(trajectory, gamma)
    for (s, a, _, probs), G in zip(trajectory, returns):
        advantage = G - baseline
        grad_log_pi_a = [-p for p in probs]
        grad_log_pi_a[a] += 1.0
        for i in range(N_ACTIONS):
            for j in range(len(s)):
                theta[i][j] += lr * advantage * grad_log_pi_a[i] * s[j]
```

梯度 `∇ log π(a|s) = e_a - π(·|s)`（动作的 one-hot 减去概率向量）是 softmax 策略梯度的核心。熟记于心。

### 第五步：基线

用最近多条轨迹的 `G` 运行均值作为基线，能将方差降低到使 4×4 GridWorld 能跑起来；大约 500 条轨迹收敛。基线升级成学习的 `V̂(s)` 即变成 actor-critic。

## 陷阱

- **梯度爆炸。** 回报可能极大。乘之前一定归一化 `G` 到批次的标准正态 `~N(0,1)`。
- **熵崩溃。** 策略过早陷入近确定动作，停止探索，陷入死局。解决方法：把熵奖励 `β · H(π(·|s))` 加入目标。
- **高方差。** 纯 REINFORCE 需要几千条轨迹。常用解决方案是评论家基线（第07课）或 TRPO/PPO 的信赖域方法（第08课）。
- **样本效率低。** On-policy 意味着每条转换只能用一次。通过重要性采样做 off-policy 校正可回收数据，但带来方差（PPO 的 ratio 是裁剪后的重要性权重）。
- **非平稳梯度。** 一百条轨迹前的梯度用的还是旧策略。On-policy 方法因此每隔数次 rollout 更新一次。
- **归因识别问题。** 不用奖励到达时，过去奖励增添噪声。务必用奖励到达。

## 应用

2026 年，REINFORCE 自身很少直接运行，但其梯度公式广泛应用：

| 使用场景         | 派生方法                        |
|------------------|------------------------------|
| 连续控制         | 基于高斯策略的 PPO / SAC      |
| 大语言模型 RLHF  | 带KL惩罚的 PPO，基于 token 级策略 |
| 大语言模型推理（DeepSeek） | GRPO — 具群体相关基线无评论家版本 REINFORCE |
| 多智能体         | 中央评论家 REINFORCE（MADDPG, COMA）  |
| 离散动作机器人   | A2C、A3C、PPO                 |
| 仅偏好设置       | DPO — 将 REINFORCE 重写为偏好似然损失，无需采样 |

当你在 2026 年的训练脚本里看到 `loss = -advantage * log_prob`，那就是带基线的 REINFORCE。整篇论文（DPO、GRPO、RLOO）都是基于这行代码的方差降低技巧。

## 交付

保存为 `outputs/skill-policy-gradient-trainer.md`：

```markdown
---
name: policy-gradient-trainer
description: Produce a REINFORCE / actor-critic / PPO training config for a given task and diagnose variance issues.
version: 1.0.0
phase: 9
lesson: 6
tags: [rl, policy-gradient, reinforce]
---

给定环境（离散/连续动作，时长，回报统计），输出：

1. 策略头。离散用 softmax，连续用高斯，列出参数数量。
2. 基线。无（纯 REINFORCE）、运行均值、学习的 `V̂(s)` 或 A2C 评论家。
3. 方差控制。默认开启奖励到达，回报归一化，梯度裁剪值。
4. 熵奖励。系数 β 和衰减方案。
5. 批大小。每次更新的轨迹数；on-policy 更新频率保证数据新鲜。

拒绝对步长 > 500 的任务用无基线 REINFORCE。拒绝对连续动作用 softmax 策略头。对任何实验检测到 `β = 0` 且策略熵 < 0.1，标记为熵崩溃。
```

## 练习

1. **简单。** 在 4×4 GridWorld 实现带线性 softmax 策略的 REINFORCE。无基线，训练 1000 条轨迹。绘制学习曲线；测量回报方差（标准差）。
2. **中等。** 加入运行均值基线。重新训练。比较样本效率和方差与纯 REINFORCE。基线减少多少步收敛？
3. **困难。** 加入熵奖励 `β · H(π)`。对 β 取 `{0, 0.01, 0.1, 1.0}` 进行调参。绘制最终回报和策略熵。这个任务中最佳 β 取值在哪？

## 关键词

| 术语              | 通俗说法           | 实际含义                                  |
|-------------------|--------------------|-------------------------------------------|
| 策略梯度（Policy gradient） | “直接训练策略”    | `∇J(θ) = E[G · ∇ log π_θ(a\|s)]`，由对数求导技巧导出。 |
| REINFORCE          | “原始 PG 算法”     | Williams (1992)，蒙特卡洛回报乘以对数策略梯度。         |
| 对数求导技巧（Log-derivative trick） | “得分函数估计器” | `∇P(τ;θ) = P(τ;θ) · ∇ log P(τ;θ)`，使期望梯度计算变易。     |
| 基线（Baseline）   | “减少方差”         | 任何从 `G` 减去的 `b(s)`；无偏因 `E[b · ∇ log π] = 0`。   |
| 奖励到达（Reward-to-go） | “只算未来回报” | `G_t^{from t}` 替代完整回报；正确且方差更低。              |
| 熵奖励（Entropy bonus） | “鼓励探索”       | 增加项 `+β · H(π(·\|s))`，防止策略坍缩。                  |
| On-policy          | “用当前策略数据训练” | 梯度期望相对于当前策略，不能直接重复使用旧数据。              |
| 优势（Advantage）  | “比平均好多少”       | `A(s, a) = G(s, a) - V(s)`；REINFORCE 带基线乘的签名量。     |

## 延伸阅读

- [Williams (1992). 简单的连接主义强化学习梯度跟踪算法](https://link.springer.com/article/10.1007/BF00992696) — REINFORCE 原始论文。
- [Sutton et al. (2000). 有函数逼近的策略梯度方法](https://papers.nips.cc/paper_files/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html) — 现代策略梯度定理及函数逼近。
- [Sutton & Barto (2018). 第13章 — 策略梯度方法](http://incompleteideas.net/book/RLbook2020.pdf) — 教科书阐述。
- [OpenAI Spinning Up — VPG / REINFORCE](https://spinningup.openai.com/en/latest/algorithms/vpg.html) — 教学清晰，包含 PyTorch 代码。
- [Peters & Schaal (2008). 用策略梯度强化学习运动技能](https://homes.cs.washington.edu/~todorov/courses/amath579/reading/PolicyGradient.pdf) — 方差减少及自然梯度连接信赖域家族（TRPO、PPO）视角。
