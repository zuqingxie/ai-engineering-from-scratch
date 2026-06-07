# MDPs, 状态（States）, 动作（Actions）与奖励（Rewards）

> 马尔可夫决策过程（Markov Decision Process）由五部分组成：状态（states）、动作（actions）、转移（transitions）、奖励（rewards）、折扣（discount）。强化学习（RL）的所有方法——Q学习（Q-learning）、PPO、DPO、GRPO——都在这个框架内进行优化。学会它，你就免费读懂了后续所有强化学习内容。

**类型：** 学习  
**编程语言：** Python  
**先修知识：** 第一阶段 · 06（概率与分布）、第二阶段 · 01（机器学习分类）  
**时间：** 约45分钟

## 问题描述

你正在写一个国际象棋机器人。或者库存规划器。或者交易代理。又或者训练推理模型的PPO循环。四个领域，惊人的事实是：这四者都归结为同一个数学对象。

监督学习提供`(x, y)`对，要求你拟合一个函数。强化学习没有标签，只有一连串的状态、你采取的动作，以及一个标量奖励。这个动作赢了比赛吗？补货决策省钱了吗？交易盈利了吗？LLM刚产生的token带来了更高的评判奖励吗？

在你把它形式化之前，是学不到东西的。“我看到了什么”，“我做了什么”，“接下来发生了什么”，“结果好不好”——这四个都必须变成你可以推理的对象。这个形式化就是马尔可夫决策过程。任何这个阶段的强化学习算法，包括最后的RLHF与GRPO循环，都是基于这个模型进行优化的。

## 概念介绍

![Markov decision process: states, actions, transitions, rewards, discount](../assets/mdp.svg)

**五个组成部分：**

- **状态（States）** `S`。代理做决策所需的一切。在GridWorld中是格子；在国际象棋中是棋盘；在大型语言模型（LLM）中是上下文窗口加任何记忆。
- **动作（Actions）** `A`。选择项。上/下/左/右移动，下一步棋，产生一个token。
- **转移（Transitions）** `P(s' | s, a)`。在状态`s`下执行动作`a`，下一状态的分布。国际象棋确定性，库存随机，LLM解码近似确定性。
- **奖励（Rewards）** `R(s, a, s')`。标量信号。胜利=+1，失败=-1。收入减去成本。GRPO中的对数似然比项。
- **折扣（Discount）** `γ ∈ [0, 1)`。未来奖励相较当前的权重。`γ=0.99`对应大约100步的时间视野；`γ=0.9`对应约10步。

**马尔可夫性质（Markov property）**：`P(s_{t+1} | s_t, a_t) = P(s_{t+1} | s_0, a_0, …, s_t, a_t)`，未来只依赖当前状态。如果不满足，说明状态表示不完整——不是方法失败，而是状态设计失败。

**策略（Policies）与回报（Returns）**。策略`π(a|s)`将状态映射到动作分布。回报`G_t = r_t + γ r_{t+1} + γ² r_{t+2} + …`为折扣后的未来奖励总和。状态值`V^π(s) = E[G_t | s_t = s]`表示从状态`s`开始在策略`π`下的期望回报。动作值`Q^π(s,a) = E[G_t | s_t = s, a_t = a]`为从状态`s`采取动作`a`开始的期望回报。每个强化学习算法会估计其中一个值，并据此改进`π`。

**贝尔曼方程（Bellman equations）**。本阶段一切的固定点方程：

`V^π(s) = Σ_a π(a|s) Σ_{s', r} P(s', r | s, a) [r + γ V^π(s')]`  
`Q^π(s, a) = Σ_{s', r} P(s', r | s, a) [r + γ Σ_{a'} π(a'|s') Q^π(s', a')]`

将期望回报拆分成“本步骤奖励”加“折扣后的后续状态值”。递归定义。阶段9的所有算法要么迭代此方程达到收敛（动态规划），要么采样估计（蒙特卡洛），要么单步更新（时序差分）。

## 实践演练

### 第1步：一个小的确定性MDP

一个4×4的GridWorld。代理初始于左上角，终点为右下角。每走一步奖励-1，动作集合 `{up, down, left, right}`。参见 `code/main.py`。

```python
GRID = 4
TERMINAL = (3, 3)
ACTIONS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

def step(state, action):
    if state == TERMINAL:
        return state, 0.0, True
    dr, dc = ACTIONS[action]
    r, c = state
    nr = min(max(r + dr, 0), GRID - 1)
    nc = min(max(c + dc, 0), GRID - 1)
    return (nr, nc), -1.0, (nr, nc) == TERMINAL
```

五行代码，整个环境。确定性转移，恒定步长惩罚，终止状态吸收。

### 第2步：执行策略

策略是一个从状态到动作分布的函数。最简单的是均匀随机。

```python
def uniform_policy(state):
    return {a: 0.25 for a in ACTIONS}

def rollout(policy, max_steps=200):
    s, total, steps = (0, 0), 0.0, 0
    for _ in range(max_steps):
        a = sample(policy(s))
        s, r, done = step(s, a)
        total += r
        steps += 1
        if done:
            break
    return total, steps
```

用随机策略运行1000次。在4×4棋盘上平均回报约在-60到-80之间。最优回报为-6（右下角直线路径）。收敛到这个差距即为第9阶段的目标。

### 第3步：通过贝尔曼方程精确计算`V^π`

对于小MDP，贝尔曼方程为线性系统。列举所有状态，计算期望，迭代直到值收敛。

```python
def policy_evaluation(policy, gamma=0.99, tol=1e-6):
    V = {s: 0.0 for s in all_states()}
    while True:
        delta = 0.0
        for s in all_states():
            if s == TERMINAL:
                continue
            v = 0.0
            for a, pi_a in policy(s).items():
                s_next, r, _ = step(s, a)
                v += pi_a * (r + gamma * V[s_next])
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            return V
```

这就是迭代策略评估。是Sutton与Barto的首个算法，也是所有后续强化学习方法的理论基础。

### 第4步：`γ`是有物理意义的超参数

有效视野约为 `1 / (1 - γ)`。`γ=0.9`对应10步，`γ=0.99`对应100步，`γ=0.999`对应1000步。

太低策略短视，太高信用归因变得嘈杂，因为前面许多动作都负有对远期奖励的责任。LLM RLHF一般用`γ=1` 因为回合短且有限；控制任务用`0.95–0.99`；长视野策略游戏用`0.999`。

## 常见陷阱

- **非马尔可夫状态。** 如果要用过去三次观测决策，状态不只是当前观测。解决：堆叠帧（Atari上的DQN堆叠4帧）或递归状态（对观测使用LSTM/GRU）。
- **稀疏奖励。** 仅胜利奖励使得大状态空间学习几乎不可能。设计中间奖励信号（shaped rewards）或用模仿学习引导（第9阶段 · 09）。
- **奖励欺骗。** 优化代理奖励可能导致异常行为。OpenAI游艇竞速代理一直绕圈收集奖励而不完成比赛。奖励必须定义为目标结果，不是代理指标。
- **折扣参数设错。** 无限回合任务设置`γ=1`使得价值无限大。必须设定有限回合或`γ<1`。
- **奖励尺度。** {+100，-100} 与 {+1，-1} 最优策略一致，但梯度幅度相差巨大。在PPO/DQN等算法中使用前请归一化至`[-1, 1]`附近。

## 应用场景

2026年技术栈中，每条RL流水线在写代码之前都先归约到MDP：

| 场景                     | 状态                          | 动作            | 奖励                | γ                |
|--------------------------|------------------------------|-----------------|---------------------|------------------|
| 控制（运动学，操作）     | 关节角度和速度               | 连续扭矩        | 任务特定的中间奖励  | 0.99             |
| 游戏（国际象棋，围棋，扑克）| 棋盘和历史                  | 合法走法        | 胜=+1 / 负=-1       | 1.0（有限）       |
| 库存 / 定价               | 库存和需求                   | 下订单数量      | 收入减成本          | 0.95             |
| LLM的RLHF                | 上下文token                  | 生成下一个token | 终局奖励模型评分    | 1.0（约200 token）  |
| 推理的GRPO               | 提示和部分回复               | 生成下一token   | 终局验证器打分0/1   | 1.0              |

写完五元组再写训练循环。大多数“RL不工作”基本都是MDP建模错误导致。

## 部署样例

保存为 `outputs/skill-mdp-modeler.md`：

```markdown
---
name: mdp-modeler
description: Given a task description, produce a Markov Decision Process spec and flag formulation risks before training.
version: 1.0.0
phase: 9
lesson: 1
tags: [rl, mdp, modeling]
---

Given a task (control / game / recommendation / LLM fine-tuning), output:

1. State. Exact feature vector or tensor spec. Justify Markov property.
2. Action. Discrete set or continuous range. Dimensionality.
3. Transition. Deterministic, stochastic-with-known-model, or sample-only.
4. Reward. Function and source. Sparse vs shaped. Terminal vs per-step.
5. Discount. Value and horizon justification.

Refuse to ship any MDP where the state is non-Markovian without explicit mention of frame-stacking or recurrent state. Refuse any reward that was not defined in terms of the target outcome. Flag any `γ ≥ 1.0` on an infinite-horizon task. Flag any reward range >100x the typical step reward as a likely gradient-explosion source.
```

## 练习题

1. **简单。** 在 `code/main.py` 实现4×4的GridWorld和随机策略执行。跑10,000回合。报告均值与方差。与最优回报（-6）比较。
2. **中等。** 用`γ ∈ {0.5, 0.9, 0.99}`对均匀随机策略运行`policy_evaluation`。打印4×4网格格式的状态值。解释终止状态附近值随γ变大而增长更快的原因。
3. **困难。** 使GridWorld变为随机：每个动作有`p=0.1`概率滑向相邻方向。重新评估均匀策略下的`V[起点]`变好还是变差？为什么？

## 关键词汇

| 术语        | 通俗说法                   | 实际含义                                               |
|-------------|---------------------------|--------------------------------------------------------|
| MDP         | “强化学习设置”            | 满足马尔可夫性质的元组 `(S, A, P, R, γ)`              |
| 状态（State）| “代理看到的东西”          | 在所选策略空间下对未来动态充分的统计量                  |
| 策略（Policy）| “代理行为”                | 条件分布 `π(a | s)` 或确定性映射 `s → a`               |
| 回报（Return）| “总体奖励”                | 当前步起的折扣奖励和 `Σ γ^t r_t`                        |
| 状态值（Value）|“状态的好坏程度”          | 策略`π`下从状态`s`开始的期望回报                        |
| 动作值（Q-value）|“动作的好坏程度”         | 策略`π`下从状态`s`采取动作`a`开始的期望回报             |
| 贝尔曼方程   | “动态规划递归”             | 将值函数/动作值分解为一步奖励加折扣的后继状态值的固定点方程 |
| 折扣`γ`      | “未来与现在的权重”          | 对远期奖励的几何递减权重，有效视野约为`~1/(1-γ)`         |

## 拓展阅读

- [Sutton & Barto (2018). 《强化学习：导论》，第2版](http://incompleteideas.net/book/RLbook2020.pdf) — 教科书。第3章讲MDP与贝尔曼方程；第1章阐述奖励假说，贯穿后续课程。
- [Bellman (1957). 动态规划](https://press.princeton.edu/books/paperback/9780691146683/dynamic-programming) — 贝尔曼方程的起源。
- [OpenAI Spinning Up — 第一部分：关键概念](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html) — 深度强化学习视角的简明MDP入门。
- [Puterman (2005). 马尔可夫决策过程](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470316887) — 运营研究角度关于MDP和精确解法的权威资料。
- [Littman (1996). 连续决策制定算法（博士论文）](https://www.cs.rutgers.edu/~mlittman/papers/thesis-main.pdf) — 对MDP作为动态规划特化的最清晰推导。
