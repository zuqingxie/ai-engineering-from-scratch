# Deep Q-Networks（深度 Q 网络，DQN）

> 2013 年：Mnih 在原始像素上训练了一个 Q-learning 网络，在七个雅达利游戏中击败了所有传统 RL 智能体。2015 年：扩展到 49 个游戏，发表在《Nature》杂志，掀开了深度强化学习时代的序幕。DQN 是 Q-learning 加上三个使函数逼近稳定的技巧。

**类型：** 实践  
**语言：** Python  
**先决条件：** 第 3 阶段 · 03 （反向传播），第 9 阶段 · 04 （Q-learning，SARSA）  
**时间：** 约 75 分钟

## 问题

表格式 Q-learning 需要为每个（状态，动作）对分别存一个 Q 值。象棋棋盘有约 10⁴³ 个状态。雅达利一帧是 210×160×3 = 100,800 个特征。表格式 RL 在几千状态时就崩溃，更别说数十亿状态了。

事后看来解决方案显而易见：用神经网络代替 Q 表，`Q(s, a; θ)`。但这“事后诸葛亮”花了几十年才实现。用 Q-learning 进行简单函数逼近会因为“致命三角”——函数逼近 + 自举 + 离策略学习——而发散。Mnih 等人（2013、2015）识别出三个稳定学习的工程技巧：

1. **经验回放（Experience replay）** 去相关性转换。
2. **目标网络（Target network）** 固定自举目标。
3. **奖励裁剪（Reward clipping）** 归一化梯度幅度。

DQN 在雅达利上是第一次，一个单一架构和一套超参数解决了几十个原始像素控制问题。此后所有“深度强化学习”——DDQN，Rainbow，对决网络，分布式，R2D2，Agent57——都是基于这三个技巧叠加的。

## 概念

![DQN 训练循环：环境、回放缓冲区、在线网络、目标网络、Bellman TD损失](../assets/dqn.svg)

**目标。** DQN 在神经 Q 函数上最小化一步 TD 损失：

`L(θ) = E_{(s,a,r,s')~D} [ (r + γ max_{a'} Q(s', a'; θ^-) - Q(s, a; θ))² ]`

`θ` = 在线网络，通过梯度下降每步更新。`θ^-` = 目标网络，定期从 `θ` 复制（约每 10,000 步）。`D` = 过去转换的回放缓冲区。

**三个技巧，按重要性顺序：**

**经验回放。** 大约包含 `~10⁶` 转换的循环缓冲区。每次训练随机均匀抽样一个小批量。这打破了时间相关（连续帧几乎相同），让网络多次学习稀有奖励转换，并去相关连续梯度更新。否则神经网上的 on-policy TD 在雅达利上会发散。

**目标网络。** 用同一个网络 `Q(·; θ)` 计算 Bellman 方程两边会让目标每次更新都动——“追自己的尾巴”。解决方案：保留第二个网络 `Q(·; θ^-)`，权重冻结。每隔 `C` 步复制一次 `θ → θ^-`。这让数千梯度步回归目标稳定。软更新 `θ^- ← τ θ + (1-τ) θ^-`（DDPG、SAC 用）是更平滑的变体。

**奖励裁剪。** 雅达利奖励幅度从 1 到 1000+ 不等。裁剪为 `{-1, 0, +1}` 阻止任何单一游戏主导梯度。奖励幅度重要时不适用；雅达利只关心符号，故可行。

**双重 DQN（Double DQN）。** Hasselt（2016）修正最大化偏差：用在线网络 *选择* 动作，用目标网络 *评估* 动作。

`target = r + γ Q(s', argmax_{a'} Q(s', a'; θ); θ^-)`

可即插即用，表现持续更好。默认使用。

**其他改进（Rainbow，2017）：** 优先回放（高 TD 错误转采样更频繁），对决架构（分离 `V(s)` 和优势头），噪声网络（学习探索），多步回报，分布式 Q（C51/QR-DQN），多步自举。每项提升几个百分点，总体增益近似叠加。

## 实现

此处代码只用标准库，无 numpy —— 我们在一个极小的连续网格世界上手写单隐层 MLP，每步训练耗时微秒级。算法与大规模雅达利 DQN 一致。

### 第一步：回放缓冲区

```python
class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = []
        self.capacity = capacity
    def push(self, s, a, r, s_next, done):
        if len(self.buf) == self.capacity:
            self.buf.pop(0)
        self.buf.append((s, a, r, s_next, done))
    def sample(self, batch, rng):
        return rng.sample(self.buf, batch)
```

雅达利用 ~50,000 容量；我们的玩具环境 5,000 容量足够。

### 第二步：一个极小的 Q 网络（手写 MLP）

```python
class QNet:
    def __init__(self, n_in, n_hidden, n_actions, rng):
        self.W1 = [[rng.gauss(0, 0.3) for _ in range(n_in)] for _ in range(n_hidden)]
        self.b1 = [0.0] * n_hidden
        self.W2 = [[rng.gauss(0, 0.3) for _ in range(n_hidden)] for _ in range(n_actions)]
        self.b2 = [0.0] * n_actions
    def forward(self, x):
        h = [max(0.0, sum(w * xi for w, xi in zip(row, x)) + b) for row, b in zip(self.W1, self.b1)]
        q = [sum(w * hi for w, hi in zip(row, h)) + b for row, b in zip(self.W2, self.b2)]
        return q, h
```

前向传递：线性 → ReLU → 线性。这就是整个网络。

### 第三步：DQN 更新

```python
def train_step(online, target, batch, gamma, lr):
    grads = zeros_like(online)
    for s, a, r, s_next, done in batch:
        q, h = online.forward(s)
        if done:
            y = r
        else:
            q_next, _ = target.forward(s_next)
            y = r + gamma * max(q_next)
        td_error = q[a] - y
        accumulate_grads(grads, online, s, h, a, td_error)
    apply_sgd(online, grads, lr / len(batch))
```

形态是第 04 课的 Q-learning，区别有两：  
(a) 反向传播通过可微 `Q(·; θ)`，而不是索引表；  
(b) 目标使用 `Q(·; θ^-)`。

### 第四步：外部循环

每集，基于 `Q(·; θ)` 采取 ε-greedy 策略，存转换进缓冲，每采样小批量，梯度更新，定期同步 `θ^- ← θ`。模式如：

```python
for episode in range(N):
    s = env.reset()
    while not done:
        a = epsilon_greedy(online, s, epsilon)
        s_next, r, done = env.step(s, a)
        buffer.push(s, a, r, s_next, done)
        if len(buffer) >= batch:
            train_step(online, target, buffer.sample(batch), gamma, lr)
        if steps % sync_every == 0:
            target = copy(online)
        s = s_next
```

在我们的 16 维独热状态的小网格世界中，智能体约 500 集学出接近最优策略。雅达利上，将此扩展到 2 亿帧，追加 CNN 特征提取器。

## 陷阱

- **致命三角（Deadly triad）。** 函数逼近 + 离策略 + 自举可能发散。DQN 用目标网 + 回放缓解；两者均不可删。
- **探索。** ε 必须衰减，通常从 1.0 下降到 0.01，约占训练前 10%。早期探索不足，Q网会陷入局部最优。
- **过估计（Overestimation）。** 带噪 Q 的 max 操作有上倾偏差。生产环境总用双重 DQN。
- **奖励尺度。** 裁剪或归一化奖励；梯度幅度与奖励幅度成比例。
- **回放缓冲冷启动。** 缓冲区积累数千转换前不训练。20 个样本早期梯度会过拟合。
- **目标同步频率。** 同步太频繁 ≈ 无目标网；太少 ≈ 目标过时。雅达利用 10,000 环境步。经验法则：同步频率约为训练步数的 1%。
- **观察预处理。** 雅达利堆叠 4 帧成状态满足马尔可夫。任何含速度信息的环境需帧堆叠或递归状态。

## 应用

2026 年，DQN 很少是最先进，但仍是参考离策略算法：

| 任务 | 首选方法 | 为什么不用 DQN？ |
|------|----------|-----------------|
| 离散动作雅达利类 | Rainbow DQN 或 Muesli | 同框架，更多技巧。 |
| 连续控制 | SAC / TD3（第 9 阶段 · 07） | DQN 无策略网络。 |
| On-policy / 高吞吐 | PPO（第 9 阶段 · 08） | 无回放缓冲，易扩展。 |
| 离线 RL | CQL / IQL / 决策 Transformer | 保守 Q 目标，无自举崩溃。 |
| 大规模离散动作（推荐系统） | 带动作嵌入的 DQN 或 IMPALA | 好用，装饰很重要。 |
| 大型语言模型 RL | PPO / GRPO | 序列级，不是步级；损失不同。 |

经验回放和目标网络存在于 SAC、TD3、DDPG、SAC-X、AlphaZero 的自我对弈缓冲，以及所有离线 RL 方法中。奖励裁剪延续为 PPO 的优势归一。架构是蓝图。

## 部署

保存为 `outputs/skill-dqn-trainer.md`：

```markdown
---
name: dqn-trainer
description: 生成离散动作 RL 任务的 DQN 训练配置（缓冲区，目标同步，ε 调度，奖励裁剪）。
version: 1.0.0
phase: 9
lesson: 5
tags: [rl, dqn, deep-rl]
---

给定离散动作环境（观察形状、动作数、地平线、奖励尺度），输出：

1. 网络。架构（MLP / CNN / Transformer）、特征维度、深度。
2. 回放缓冲区。容量、小批量大小、热身大小。
3. 目标网络。同步策略（硬同步每 C 步或软更新 τ）。
4. 探索。ε 起始 / 终止 / 调度长度。
5. 损失。Huber 或 MSE，梯度裁剪值，奖励裁剪规则。
6. 双重 DQN。默认开启，除非有禁用理由。

拒绝部署无目标网络、无回放缓冲或 ε 保持在 1 的 DQN。拒绝连续动作任务（推荐用 SAC / TD3）。若奖励范围 > 每步均值 10 倍，标记需裁剪或归一化。
```

## 练习

1. **简单。** 运行 `code/main.py`，画每集回报曲线。几集后运行平均超过 -10？
2. **中等。** 禁用目标网络（两边 Bellman 目标都用在线网）。测训练稳定性——回报是否振荡或发散？
3. **困难。** 加双重 DQN：用在线网选 `argmax a'`，用目标网评估。对比 1,000 集后有无双重 DQN 时，带噪奖励网格世界中 `Q(s_0, best_a)` 与真实 `V*(s_0)` 偏差。

## 术语

| 术语 | 通常说法 | 实际含义 |
|-------|-----------|----------|
| DQN | “深度 Q 学习” | 带神经 Q 函数、回放缓冲和目标网络的 Q-learning。 |
| 经验回放 | “洗牌转换” | 每次梯度步均匀采样循环缓冲区；去相关数据。 |
| 目标网络 | “冻结自举” | Bellman 目标中定期复制的 Q 网络，稳定训练。 |
| 致命三角 | “为何 RL 发散” | 函数逼近 + 自举 + 离策略 = 无收敛保证。 |
| 双重 DQN | “最大化偏差修正” | 在线网选动作，目标网评估动作。 |
| 对决网络 | “V 和 A 头” | Q 分解为 V + A - 均值(A)；输出相同，梯度更好。 |
| Rainbow | “所有技巧汇聚” | 集成 DDQN + 优先经验回放 + 对决 + n 步 + 噪声 + 分布式。 |
| 优先经验回放（PER） | “以 TD 错误采样” | 以 TD 错误大小为采样权重。 |

## 参考文献

- [Mnih 等人（2013）。Playing Atari with Deep Reinforcement Learning（用深度强化学习玩雅达利）](https://arxiv.org/abs/1312.5602) — 2013 NeurIPS 研讨会论文，深度 RL 起点。
- [Mnih 等人（2015）。Human-level control through deep reinforcement learning（通过深度强化学习实现人类水平控制）](https://www.nature.com/articles/nature14236) — Nature 论文，49 游戏 DQN。
- [Hasselt, Guez, Silver（2016）。Deep Reinforcement Learning with Double Q-learning（带双 Q-learning 的深度强化学习）](https://arxiv.org/abs/1509.06461) — DDQN。
- [Wang 等人（2016）。Dueling Network Architectures（对决网络架构）](https://arxiv.org/abs/1511.06581) — 对决 DQN。
- [Hessel 等人（2018）。Rainbow: Combining Improvements in Deep RL（Rainbow：结合深度 RL 改进）](https://arxiv.org/abs/1710.02298) — 堆叠技巧论文。
- [OpenAI Spinning Up — DQN](https://spinningup.openai.com/en/latest/algorithms/dqn.html) — 清晰现代讲解。
- [Sutton & Barto（2018）。第 9 章 — 带函数逼近的 on-policy 预测](http://incompleteideas.net/book/RLbook2020.pdf) — “致命三角”（函数逼近 + 自举 + 离策略）教科书讲解，DQN 的目标网络和回放缓冲就是为此设计。
- [CleanRL DQN 实现](https://docs.cleanrl.dev/rl-algorithms/dqn/) — 参考单文件 DQN，适合与本课手写版本一同阅读。
