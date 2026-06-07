# Temporal Difference — Q-Learning & SARSA

> Monte Carlo（蒙特卡罗）等待直到回合结束。TD（时序差分）通过对下一状态价值估计进行引导，步步更新。Q-learning（Q学习）是离策略且乐观的；SARSA 是在策略且谨慎的。两者都只是一行代码。它们是本阶段每个深度强化学习方法的基础。

**类型：** 构建  
**语言：** Python  
**先决条件：** 第9阶段 · 01（MDPs 马尔可夫决策过程）、第9阶段 · 02（动态规划）、第9阶段 · 03（蒙特卡罗）  
**时间：** ~75分钟

## 问题

蒙特卡罗方法可用，但有两个昂贵的要求。需要回合终止，而且只在最终回报后才更新。如果回合有1000步，蒙特卡罗要等1000步才更新任何内容。它是高方差、低偏差且在实践中速度慢。

动态规划正好相反——零方差的自举备份——但要求环境模型已知。

时序差分（TD）学习折中处理。根据单个转移 `(s, a, r, s')`，形成一步目标 `r + γ V(s')`，然后将 `V(s)` 向该目标靠拢。无需模型，无需完整回合。右边使用近似的 `V` 导致偏差，但方差远低于蒙特卡罗，且从第一步起在线更新。

这正是现代强化学习——DQN、A2C、PPO、SAC——的核心。第9阶段剩余部分是构建在你将在本课中编写的一步TD更新之上的函数逼近和技巧层。

## 概念

![Q-learning vs SARSA: off-policy max vs on-policy Q(s', a')](../assets/td.svg)

**V的TD(0)更新公式：**

`V(s) ← V(s) + α [r + γ V(s') - V(s)]`

括号中的量是TD误差 `δ = r + γ V(s') - V(s)`。它是蒙特卡罗中 `G_t - V(s_t)` 的在线对应。收敛需要学习率 `α` 满足 Robbins-Monro 条件（`Σ α = ∞`，`Σ α² < ∞`），且所有状态被无限次访问。

**Q-learning。** 一种离策略TD方法用于控制：

`Q(s, a) ← Q(s, a) + α [r + γ max_{a'} Q(s', a') - Q(s, a)]`

`max` 假设从 `s'` 开始后续完全按照贪心策略执行，不管智能体实际采取哪种动作。这种解耦使得Q-learning在采用ε-greedy探索时也能学习到最优Q值 `Q*`。Mnih 等人（2015）将此转为深度Q学习应用于Atari（第05课）。

**SARSA。** 一种在策略TD方法：

`Q(s, a) ← Q(s, a) + α [r + γ Q(s', a') - Q(s, a)]`

名称来自元组 `(s, a, r, s', a')`。SARSA使用智能体*实际*采取的下一个动作 `a'`，而不是贪心的 `argmax`。其收敛到当前ε-greedy策略 `π` 的Q值 `Q^π`，在ε趋近0时变为 `Q*`。

**悬崖行走差异。** 在经典的悬崖行走任务（跌落悬崖 = 奖励 -100）中，Q-learning学会沿悬崖边的最优路径，但探索时偶尔会触发惩罚。SARSA因为考虑了探索噪声，更倾向于选择离悬崖一步远的更安全路径。训练充分后，二者在ε→0时均能达到最优。实际上，这很重要：实际部署时仍在探索，SARSA表现更保守。

**期望SARSA（Expected SARSA）。** 用策略 `π` 下`Q(s', a')`的期望替代采样动作的Q值：

`Q(s, a) ← Q(s, a) + α [r + γ Σ_{a'} π(a'|s') Q(s', a') - Q(s, a)]`

比SARSA方差低（无采样动作的随机性），目标仍是策略内的。常作为现代教材默认。

**n步TD和TD(λ)。** 通过等待n步再引导，将TD(0)和蒙特卡罗插值。`n=1`为TD，`n=∞`为蒙特卡罗。TD(λ)对所有n以几何权重 `(1-λ) λ^{n-1}`加权平均。多数深度强化学习用n介于3到20。

## 构建实践

### 步骤1：基于ε-greedy策略的SARSA

```python
def sarsa(env, episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})

    def choose(s):
        if random() < epsilon:
            return choice(ACTIONS)
        return max(Q[s], key=Q[s].get)

    for _ in range(episodes):
        s = env.reset()
        a = choose(s)
        while True:
            s_next, r, done = env.step(s, a)
            a_next = choose(s_next) if not done else None
            target = r + (gamma * Q[s_next][a_next] if not done else 0.0)
            Q[s][a] += alpha * (target - Q[s][a])
            if done:
                break
            s, a = s_next, a_next
    return Q
```

代码仅8行。与Q-learning唯一不同的是目标这一行。

### 步骤2：Q-learning

```python
def q_learning(env, episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})
    for _ in range(episodes):
        s = env.reset()
        while True:
            a = choose(s, Q, epsilon)
            s_next, r, done = env.step(s, a)
            target = r + (gamma * max(Q[s_next].values()) if not done else 0.0)
            Q[s][a] += alpha * (target - Q[s][a])
            if done:
                break
            s = s_next
    return Q
```

`max` 使得目标行为与实际策略解耦。该操作是离策略和在策略的区别所在。

### 步骤3：学习曲线

记录每100回合的平均回报。Q-learning在简单确定性GridWorld中收敛更快；SARSA在悬崖行走中行为更保守。在 `code/main.py`的4×4 GridWorld中，二者在~2000回合后以 `α=0.1, ε=0.1` 均接近最优。

### 步骤4：对比DP真值

运行值迭代（第02课）得到最优Q `Q*`。检查误差 `max_{s,a} |Q_learned(s,a) - Q*(s,a)|`。一个训练良好的表格TD智能体在4×4 GridWorld 10000回合后误差约为0.5。

## 陷阱

- **初始Q值关键。** 乐观初始化（如负奖励任务中设Q=0）可鼓励探索。悲观初始化会让贪心策略永久陷入局部。
- **学习率α调度。** 常数α对非平稳问题可用。衰减α_n=1/n理论收敛但实际太慢。建议α固定在[0.05, 0.3]内，监控学习曲线。
- **探索率ε调度。** 起始高（ε=1.0），逐渐降低到0.05。GLIE（无限探索极限上的贪心）是收敛条件。
- **Q-learning最大偏差。** `max`在噪声Q值上有向上偏差。导致过估计——Hasselt提出的Double Q-learning（第05课的DDQN）通过双Q表修正。
- **非终止回合。** TD可在无终止情况下学习，但需限制步数或正确处理截断时的引导。通常：将截断视为非终止状态，继续引导。
- **状态哈希。** 若状态为元组或张量，使用可哈希键（元组非列表，浮点数四舍五入的元组非原始值）。

## 用途

2026年TD方法应用场景：

| 任务             | 方法              | 理由                            |
|-----------------|-----------------|-------------------------------|
| 小规模表格环境         | Q-learning      | 直接学习最优策略                   |
| 在策略安全关键任务       | SARSA / Expected SARSA | 探索期间更保守                     |
| 高维状态            | DQN (第9阶段 · 05)   | 带重放和目标网络的神经网络Q函数        |
| 连续动作            | SAC / TD3 (第9阶段 · 07) | 对Q网做TD更新，策略网输出动作            |
| LLM强化学习（奖励模型） | PPO / GRPO (第9阶段 · 08, 12) | 使用GAE的带TD优势估计的演员-评论家方法       |
| 离线强化学习          | CQL / IQL (第9阶段 · 08) | 采用保守正则化的Q学习                 |

你在2026年读到的百分之九十的“强化学习”论文，都是围绕Q-learning或SARSA的各种改进。在深入阅读之前，务必要手动操练清楚表格更新。

## 发布它

保存为 `outputs/skill-td-agent.md`：

```markdown
---
name: td-agent
description: Pick between Q-learning, SARSA, Expected SARSA for a tabular or small-feature RL task.
version: 1.0.0
phase: 9
lesson: 4
tags: [rl, td-learning, q-learning, sarsa]
---

Given a tabular or small-feature environment, output:

1. Algorithm. Q-learning / SARSA / Expected SARSA / n-step variant. One-sentence reason tied to on-policy vs off-policy and variance.
2. Hyperparameters. α, γ, ε, decay schedule.
3. Initialization. Q_0 value (optimistic vs zero) and justification.
4. Convergence diagnostic. Target learning curve, `|Q - Q*|` check if DP is possible.
5. Deployment caveat. How will exploration behave at inference? Is SARSA's conservatism needed?

Refuse to apply tabular TD to state spaces > 10⁶. Refuse to ship a Q-learning agent without a max-bias caveat. Flag any agent trained with ε held at 1.0 throughout (no exploitation phase).
```

## 练习

1. **简单。** 实现在4×4 GridWorld上的Q-learning和SARSA。绘制2000回合每100回合平均回报的学习曲线。谁收敛更快？
2. **中等难度。** 构建悬崖行走环境（4×12，最底行是悬崖，奖励为-100并重置到起点）。比较Q-learning和SARSA最终策略。截图它们各自路径。哪个更靠近悬崖？
3. **困难。** 实现双Q学习。在噪声奖励GridWorld（每步奖励加高斯噪声σ=5）中，展示Q-learning显著高估 `V*(0,0)`，而双Q学习无此问题。

## 关键词

| 术语           | 常说法           | 实际含义                                      |
|--------------|------------------|--------------------------------------------|
| TD误差        | “更新信号”        | `δ = r + γ V(s') - V(s)`，引导的残差。           |
| TD(0)        | “一步TD”          | 每次转移后用下一状态估计更新。                       |
| Q-learning   | “离策略强化学习101” | 使用下状态动作的`max`做TD更新；无视行为策略学习最优Q函数。|
| SARSA        | “在策略Q学习”      | 用实际采样动作做TD更新；学当前ε-greedy策略的Q值。           |
| 期望SARSA      | “低方差SARSA”      | 用策略期望替代采样动作Q值。                           |
| GLIE         | “正确的探索调度”    | 无限探索极限上的贪心，保证Q-learning收敛。                |
| 引导          | “利用当前估计做目标” | TD区别于MC的关键。偏差来源，但大幅降低方差。                 |
| 最大化偏差      | “Q-learning过估计”  | 噪声估计的`max`有向上偏差；由双Q学习修正。                   |

## 拓展阅读

- [Watkins & Dayan (1992). Q-learning](https://link.springer.com/article/10.1007/BF00992698) — 原始论文及收敛证明。
- [Sutton & Barto (2018). 第6章 — 时序差分学习（Temporal-Difference Learning）](http://incompleteideas.net/book/RLbook2020.pdf) — TD(0)、SARSA、Q-learning、期望SARSA。
- [Hasselt (2010). 双Q学习（Double Q-learning）](https://papers.nips.cc/paper_files/paper/2010/hash/091d584fced301b442654dd8c23b3fc9-Abstract.html) — 最大化偏差的修正。
- [Seijen, Hasselt, Whiteson, Wiering (2009). Expected SARSA的理论与经验分析](https://ieeexplore.ieee.org/document/4927542) — 期望SARSA动机。
- [Rummery & Niranjan (1994). 在线Q学习使用连接主义系统](https://www.researchgate.net/publication/2500611_On-Line_Q-Learning_Using_Connectionist_Systems) — 首次提出SARSA（当时称“修改连接主义Q学习”）的论文。
- [Sutton & Barto (2018). 第7章 — n步引导](http://incompleteideas.net/book/RLbook2020.pdf) — 将TD(0)推广至TD(n)，Q-learning通向资格迹与后续PPO中GAE的路径。
