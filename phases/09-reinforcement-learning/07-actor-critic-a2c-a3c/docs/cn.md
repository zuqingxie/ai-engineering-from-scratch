# Actor-Critic — A2C 和 A3C

> REINFORCE 很嘈杂。添加一个学习 `V̂(s)` 的 critic（评论家），从回报中减去它，就得到一个期望相同但方差更小的优势（advantage）。这就是 actor-critic（行动者-评论家）方法。A2C 同步运行它；A3C 跨线程异步运行。它们是所有现代深度强化学习方法的思维模型。

**类型:** 构建  
**语言:** Python  
**先决条件:** 阶段 9 · 04（时序差分学习 TD Learning）、阶段 9 · 06（REINFORCE）  
**时长:** ~75 分钟

## 问题

vanilla（基础）REINFORCE 可用，但方差非常大。蒙特卡洛回报 `G_t` 在不同回合间可能波动超过 10 倍。将这种噪声乘以 `∇ log π` 并平均，会产生一个梯度估计器，需要数千个回合才能使策略移动到与少量 DQN 更新相同的距离。

方差来自于使用原始回报。如果减去基线 `b(s_t)`——任意状态函数，包括学习得到的值——期望不变而方差下降。最佳可行基线是 `V̂(s_t)`。现在乘以 `∇ log π` 的量是**优势**：

`A(s, a) = G - V̂(s)`

一个动作好的时候，意味着它产生了高于平均的回报；坏的时候低于平均。带有学习 critic 的 REINFORCE 就是*actor-critic*。critic 给 actor 一个低方差的“老师”。这就是 2015 年后所有深度策略方法的基础（A2C、A3C、PPO、SAC、IMPALA）。

## 概念

![Actor-critic：policy net 加 value net，TD 残差作为优势](../assets/actor-critic.svg)

**两个网络，共用一个损失：**

- **Actor** `π_θ(a | s)`：策略网络。用来采样动作。通过策略梯度训练。
- **Critic** `V_φ(s)`：估计从状态得到的期望回报。通过最小化 `(V_φ(s) - 目标)²` 训练。

**优势（advantage）。两种标准形式：**

- *蒙特卡洛优势（MC advantage）：* `A_t = G_t - V_φ(s_t)`。无偏，高方差。
- *时序差分优势（TD advantage）：* `A_t = r_{t+1} + γ V_φ(s_{t+1}) - V_φ(s_t)`。有偏（使用 `V_φ`），方差远低。也叫*TD 残差* `δ_t`。

**n 步优势。** 两者之间的插值：

`A_t^{(n)} = r_{t+1} + γ r_{t+2} + … + γ^{n-1} r_{t+n} + γ^n V_φ(s_{t+n}) - V_φ(s_t)`

`n = 1` 是纯 TD。`n = ∞` 是 MC。大多数实现中，Atari 用 `n=5`，PPO 在 MuJoCo 用 `n=2048`。

**广义优势估计（GAE）。** Schulman 等（2016）提出对所有 n 步优势做指数加权平均：

`A_t^{GAE} = Σ_{l=0}^{∞} (γλ)^l δ_{t+l}`

其中 `λ ∈ [0, 1]`。`λ=0` 是 TD （低方差，高偏差）。`λ=1` 是 MC （高方差，无偏）。`λ=0.95` 是 2026 年默认——根据偏差/方差权衡调节。

**A2C：同步优势行动者-评论家。** 在 `N` 个并行环境中采集 `T` 步。计算每步优势。在合并批次上更新 actor 和 critic。重复。A3C 更简单、更可扩展的兄弟。

**A3C：异步优势行动者-评论家。** Mnih 等（2016）。启动 `N` 个线程，每个运行一个环境。各线程本地计算梯度，异步推送到共享参数服务器。无重放缓存，线程通过跑不同轨迹去相关。A3C 证明了可以用 CPU 大规模训练。2026 年，GPU 版本的 A2C（批处理并行环境）占主导，因为 GPU 需要大批量。

**联合损失：**

`L(θ, φ) = -E[ A_t · log π_θ(a_t | s_t) ]  +  c_v · E[(V_φ(s_t) - G_t)²]  -  c_e · E[H(π_θ(·|s_t))]`

三项：策略梯度损失，价值回归，熵奖励。`c_v ~ 0.5`，`c_e ~ 0.01` 是经典起点。

## 构建步骤

### 步骤 1：一个 critic

线性 critic `V_φ(s) = w · features(s)` 用均方误差更新：

```python
def critic_update(w, x, target, lr):
    v_hat = dot(w, x)
    err = target - v_hat
    for j in range(len(w)):
        w[j] += lr * err * x[j]
    return v_hat
```

在表格环境（tabular env）中，critic 几百集就会收敛。Atari 环境中用共享 CNN 主干加价值头替代线性 critic。

### 步骤 2：n 步优势

给出长度为 `T` 的 rollout 和 bootstrapped 的末尾 `V(s_T)`：

```python
def compute_advantages(rewards, values, gamma=0.99, lam=0.95, last_value=0.0):
    advantages = [0.0] * len(rewards)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        next_v = values[t + 1] if t + 1 < len(values) else last_value
        delta = rewards[t] + gamma * next_v - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
    returns = [a + v for a, v in zip(advantages, values)]
    return advantages, returns
```

`returns` 是 critic 的目标。`advantages` 是乘以 `∇ log π` 的量。

### 步骤 3：联合更新

```python
for step_i, (x, a, _r, probs) in enumerate(traj):
    adv = advantages[step_i]
    target_v = returns[step_i]

    # critic
    critic_update(w, x, target_v, lr_v)

    # actor
    for i in range(N_ACTIONS):
        grad_logpi = (1.0 if i == a else 0.0) - probs[i]
        for j in range(N_FEAT):
            theta[i][j] += lr_a * adv * grad_logpi * x[j]
```

在线策略，一次 rollout 对应一次更新，actor 和 critic 用不同学习率。

### 步骤 4：并行化（A3C vs A2C）

- **A3C：** 启动 `N` 个线程，各自运行环境和前向推理。周期性将梯度更新推送到共享主服务器。主服务器不加锁——竞态条件没关系，只是增加噪声。
- **A2C：** 在单进程运行 `N` 个环境实例，将观察打包成 `[N, obs_dim]` 批次，批量做前向和反向传播。GPU 利用率高，确定性强，更容易调试。2026 年默认选择。

我们的示例代码为单线程以保持清晰；改写为批量 A2C 只需三行 numpy 代码。

## 陷阱

- **Critic 在 actor 梯度前的偏差。** 如果 critic 是随机的，基线无用，你就在用纯噪声训练。预热 critic 几百步再开启策略梯度，或使用较慢的 actor 学习率。
- **优势归一化。** 每批次将优势归一化为零均值单位方差。极大稳定训练，几乎无成本。
- **共享主干。** 在图像输入时，actor 和 critic 用共享特征提取层，独立头。共享特征同时受两损失驱动。
- **在线策略的限制。** A2C 每条数据只用一次更新。多用导致梯度偏差（PPO 通过重要性采样校正解决）。
- **熵崩塌。** 没有 `c_e > 0`，策略在几百次更新内变得近似确定，停止探索。
- **奖励尺度。** 优势幅度依赖奖励尺度。归一化奖励（如用运行标准差除）保证不同任务梯度幅度一致。

## 使用场景

2026 年，A2C/A3C 很少是最终方法，但它们构成后续所有改进方法的架构基础：

| 方法 | 与 A2C 关系 |
|--------|----------------|
| PPO | A2C + 剪裁的重要性比率多轮更新 |
| IMPALA | A3C + V-trace off-policy 校正 |
| SAC（阶段 9 · 07） | 带软价值 critic 的 off-policy A2C（下一课） |
| GRPO（阶段 9 · 12） | 无 critic 的 A2C——群体相对优势 |
| DPO | A2C 簇集为偏好排序损失，无采样 |
| AlphaStar / OpenAI Five | A2C 加联盟训练 + 模仿预训练 |

2026 年若论文出现“优势”关键词，想想 actor-critic。

## 交付

保存为 `outputs/skill-actor-critic-trainer.md`：

```markdown
---
name: actor-critic-trainer
description: Produce an A2C / A3C / GAE configuration for a given environment, with advantage estimation and loss weights specified.
version: 1.0.0
phase: 9
lesson: 7
tags: [rl, actor-critic, gae]
---

Given an environment and compute budget, output:

1. Parallelism. A2C (GPU batched) vs A3C (CPU async) and the number of workers.
2. Rollout length T. Steps per env per update.
3. Advantage estimator. n-step or GAE(λ); specify λ.
4. Loss weights. `c_v` (value), `c_e` (entropy), gradient clip.
5. Learning rates. Actor and critic (separate if using).

Refuse single-worker A2C on environments with horizon > 1000 (too on-policy, too slow). Refuse to ship without advantage normalization. Flag any run with `c_e = 0` and observed entropy < 0.1 as entropy-collapsed.
```

## 练习

1. **简单。** 用 MC 优势（`G_t - V(s_t)`）训练 actor-critic，在 4×4 GridWorld 上。对比第 06 课带运行均值基线的 REINFORCE 的样本效率。
2. **中等。** 改用 TD 残差优势（`r + γ V(s') - V(s)`）。测量优势批次方差。降低了多少？
3. **困难。** 实现 GAE(λ)。对 `λ ∈ {0, 0.5, 0.9, 0.95, 1.0}` 做扫参。绘制最终回报与样本效率关系。这个任务的偏差/方差最佳点在哪？

## 关键词

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| Actor | “策略网络” | `π_θ(a\|s)`，通过策略梯度更新。 |
| Critic | “价值网络” | `V_φ(s)`，通过均方差回归回报或 TD 目标更新。 |
| Advantage | “比平均好多少” | `A(s, a) = Q(s, a) - V(s)` 或其估计量。`∇ log π` 的乘数。 |
| TD residual | “δ” | `δ_t = r + γ V(s') - V(s)`；一步优势估计。 |
| GAE | “插值旋钮” | 所有 n 步优势的指数加权和，参数为 `λ`。 |
| A2C | “同步 actor-critic” | 在多个环境批处理，一轮 rollout 对应一步梯度。 |
| A3C | “异步 actor-critic” | 工作线程推梯度到共享参数服务器。原创论文；2026 年较少用。 |
| Bootstrap | “用 V 截断” | 截断 rollout，加 `γ^n V(s_{t+n})` 收尾。 |

## 延伸阅读

- [Mnih et al. (2016). Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783) — A3C，异步 actor-critic 原始论文。
- [Schulman et al. (2016). High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438) — GAE。
- [Sutton & Barto (2018). Ch. 13 — Actor-Critic Methods](http://incompleteideas.net/book/RLbook2020.pdf) — 理论基础；与 Ch. 9 函数逼近（critic 为神经网时）结合阅读。
- [Espeholt et al. (2018). IMPALA](https://arxiv.org/abs/1802.01561) — 基于 V-trace 校正的可扩展分布式 actor-critic。
- [OpenAI Baselines / Stable-Baselines3](https://stable-baselines3.readthedocs.io/) — 生产级的 A2C/PPO 实现，值得阅读。
- [Konda & Tsitsiklis (2000). Actor-Critic Algorithms](https://papers.nips.cc/paper/1786-actor-critic-algorithms) — 两时尺度 actor-critic 分解的基础收敛结果。
