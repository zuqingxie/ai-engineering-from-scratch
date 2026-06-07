# 蒙特卡洛方法 — 从完整回合中学习

> 动态规划（Dynamic programming）需要模型。蒙特卡洛（Monte Carlo）只需要回合。运行策略，观察回报，取平均。在强化学习（RL）中最简单的想法——也是开启后续所有内容的关键。

**类型：** 构建  
**语言：** Python  
**先决条件：** 第9阶段 · 01（MDPs），第9阶段 · 02（动态规划）  
**时间：** ~75分钟  

## 问题

动态规划优雅，但假设你能查询每个状态和动作的转移概率分布 `P(s' | s, a)`。现实世界几乎没有什么能这样工作的。机器人无法解析计算一个关节扭矩后摄像头像素的分布。定价算法无法整合所有可能顾客反应。大型语言模型（LLM）也无法枚举一个词元后的所有可能续写。

你需要一种仅依赖于*环境采样*能力的方法。运行策略，得到轨迹 `s_0, a_0, r_1, s_1, a_1, r_2, …, s_T`，用来估计价值函数。这就是蒙特卡洛。

从动态规划到蒙特卡洛的转变在哲学上很重要：我们从*已知模型+精确备份*转向*采样回合+平均回报*。方差变大，但适用范围急剧扩大。此课后，所有强化学习算法——TD、Q学习、REINFORCE、PPO、GRPO——本质上都是蒙特卡洛估计器，有时加上了自举（bootstrapping）。

## 概念

![蒙特卡洛：展开，计算回报，求平均；首次访问与每次访问](../assets/monte-carlo.svg)

**核心观点，一句话概括：**  
`V^π(s) = E_π[G_t | s_t = s] ≈ (1/N) Σ_i G^{(i)}(s)` ，其中 `G^{(i)}(s)` 是策略 `π` 访问状态 `s` 后的观察到的回报。

**首次访问与每次访问蒙特卡洛。** 给定一个回合多次访问状态 `s`，首次访问蒙特卡洛只计算第一次访问的回报；每次访问蒙特卡洛计入所有访问。二者在极限下均为无偏估计。首次访问更容易分析（独立同分布样本），每次访问在每个回合用更多数据，常常收敛更快。

**增量平均。** 不存储所有回报，而是更新运行均值：

`V_n(s) = V_{n-1}(s) + (1/n) [G_n - V_{n-1}(s)]`

变形为：`V_new = V_old + α · (target - V_old)`，其中 `α = 1/n`。用固定步长 `α ∈ (0, 1)` 代替 `1/n`，你就得到了一个非平稳的蒙特卡洛估计器，能跟踪策略 `π` 的变化。这个转变就是蒙特卡洛到时序差分（TD）再到现代强化学习算法的根本跳跃。

**探索问题由此产生。** 动态规划通过枚举遍历所有状态。蒙特卡洛只能看到策略访问的状态。如果策略 `π` 是确定性的，状态空间有些区域永远不会被采样，其价值估计永远为零。历史上三种解决方法：

1. **探索起始（Exploring starts）。** 每个回合从随机的 `(s, a)` 对开始。保证覆盖；实际中不现实（你无法将机器人重置到任意状态）。
2. **ε-贪婪（ε-greedy）。** 大多数时间按当前Q值贪婪选动作，概率 `ε` 时随机选动作。所有状态动作对最终都会被采样。
3. **离策略蒙特卡洛（Off-policy MC）。** 在行为策略 `μ` 下采样，用重要性采样（importance sampling）来学习目标策略 `π`。方差高，但它是重放缓存（replay-buffer）方法如DQN的桥梁。

**蒙特卡洛控制。** 评估 → 改进 → 评估，和策略迭代一样，只是评估基于采样：

1. 运行策略 π，采集回合。
2. 根据回报更新 `Q(s, a)`。
3. 将 π 修改为相对于 Q 的 ε-贪婪策略。
4. 重复。

在温和条件下（每对状态动作无限访问，学习率 α 满足Robbins-Monro条件）以概率1收敛到最优 `Q*` 和 `π*`。

## 构建步骤

### 第1步：展开轨迹 → (s, a, r) 列表

```python
def rollout(env, policy, max_steps=200):
    trajectory = []
    s = env.reset()
    for _ in range(max_steps):
        a = policy(s)
        s_next, r, done = env.step(s, a)
        trajectory.append((s, a, r))
        s = s_next
        if done:
            break
    return trajectory
```

无模型，仅使用 `env.reset()` 和 `env.step(s, a)`。接口类似gym环境但更简化。

### 第2步：计算回报（反向遍历）

```python
def returns_from(trajectory, gamma):
    returns = []
    G = 0.0
    for _, _, r in reversed(trajectory):
        G = r + gamma * G
        returns.append(G)
    return list(reversed(returns))
```

单次遍历，时间复杂度为 `O(T)`。反向递推公式 `G_t = r_{t+1} + γ G_{t+1}`避免重复求和。

### 第3步：首次访问蒙特卡洛评估

```python
def mc_policy_evaluation(env, policy, episodes, gamma=0.99):
    V = defaultdict(float)
    counts = defaultdict(int)
    for _ in range(episodes):
        trajectory = rollout(env, policy)
        returns = returns_from(trajectory, gamma)
        seen = set()
        for t, ((s, _, _), G) in enumerate(zip(trajectory, returns)):
            if s in seen:
                continue
            seen.add(s)
            counts[s] += 1
            V[s] += (G - V[s]) / counts[s]
    return V
```

三行代码完成工作：首次访问标记状态，计数自增，更新运行均值。

### 第4步：ε-贪婪蒙特卡洛控制（在策略）

```python
def mc_control(env, episodes, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})
    counts = defaultdict(lambda: {a: 0 for a in ACTIONS})

    def policy(s):
        if random() < epsilon:
            return choice(ACTIONS)
        return max(Q[s], key=Q[s].get)

    for _ in range(episodes):
        trajectory = rollout(env, policy)
        returns = returns_from(trajectory, gamma)
        seen = set()
        for (s, a, _), G in zip(trajectory, returns):
            if (s, a) in seen:
                continue
            seen.add((s, a))
            counts[s][a] += 1
            Q[s][a] += (G - Q[s][a]) / counts[s][a]
    return Q, policy
```

### 第5步：与动态规划黄金标准比较

你的蒙特卡洛估计 `V^π` 应该随着回合数趋向无穷，与第02课动态规划结果一致。实际中：在4×4 GridWorld中运行50,000回合可使估计误差在 `~0.1` 以内。

## 陷阱

- **无限回合。** 蒙特卡洛要求回合*终止*。若你的策略能无限循环，请限制 `max_steps` 并将其视为隐式失败。GridWorld中使用随机策略经常超时——这是正常的，只要正确统计即可。
- **方差。** 蒙特卡洛用全回报。长回合时方差巨大——尾部一次不利奖励能等幅度影响 `V(s_0)`。TD方法（第04课）通过自举减少方差。
- **状态覆盖。** 贪婪蒙特卡洛在初始Q相等时只尝试一动作。你*必须*要探索（ε-贪婪、探索起始、UCB）。
- **非平稳策略。** 如果策略 `π` 变化（如蒙特卡洛控制），旧的回报来自不同策略。固定α蒙特卡洛处理该问题，采样均值蒙特卡洛不行。
- **离策略重要性采样。** 权重 `π(a|s)/μ(a|s)` 在轨迹上连乘，方差随时间步数爆炸。可用逐决策加权重要性采样或转向时序差分减少问题。

## 应用场景

2026年蒙特卡洛方法的作用：

| 应用场景 | 为什么用蒙特卡洛 |
|----------|-------------------|
| 短回合游戏（黑杰克、扑克） | 回合自然终止，回报干净。 |
| 离线评价日志策略 | 计算存储轨迹的平均折扣回报。 |
| 蒙特卡洛树搜索（AlphaZero） | 从树叶展开蒙特卡洛回合指导选择。 |
| LLM强化学习评价 | 统计给定策略采样续写的平均奖励。 |
| PPO中的基线估计 | 优势目标 `A_t = G_t - V(s_t)` 采用蒙特卡洛回报 `G_t`。 |
| 强化学习教学 | 最简单且实际有效的算法——剥离自举看核心。 |

现代深度强化学习算法（PPO、SAC）介于纯蒙特卡洛（全回报）和纯TD（一步自举）之间，通过`n`步回报或广义优势估计（GAE）。两端都属于同一估计家族。

## 交付文档

保存为 `outputs/skill-mc-evaluator.md`：

```markdown
---
name: mc-evaluator
description: 通过蒙特卡洛回合评估策略，并如有则生成包含动态规划比较的收敛报告。
version: 1.0.0
phase: 9
lesson: 3
tags: [rl, monte-carlo, evaluation]
---

给定一个环境（episodic，带reset+step API）和策略，输出：

1. 方法。首次访问 vs 每次访问蒙特卡洛。理由。
2. 回合预算。目标数量、方差诊断、预期标准误。
3. 探索计划。ε策略（如需）或探索起始。
4. 黄金标准比较。若为表格环境则DP最优值 `V*`；否则基于Q学习 / PPO基线的界限。
5. 终止检测。最大步数限制、超时、非终止轨迹处理。

拒绝对非回合任务无有限步长限制运行蒙特卡洛。拒绝在表格任务中每状态回合数不足100时报策略估计。将零方差动作的策略标记为探索风险。
```

## 习题

1. **简单。** 实现4×4 GridWorld中均匀随机策略的首次访问蒙特卡洛评估。运行10,000回合。绘制状态 `(0,0)` 的价值函数随回合数变化与动态规划结果对比曲线。
2. **中等。** 实现ε-贪婪蒙特卡洛控制，ε取 `{0.01, 0.1, 0.3}`。比较20,000回合后的平均回报。曲线形状如何？偏差-方差权衡在哪里？
3. **困难。** 实现*离策略*蒙特卡洛带重要性采样：在均匀随机策略 μ 下采样，估计确定性最优策略 π 的 `V^π`。比较普通重要性采样、逐决策重要性采样与加权重要性采样。哪个方差最低？

## 关键词

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| Monte Carlo | “随机采样” | 通过对分布的 iid 样本平均估计期望。 |
| 回报 `G_t` | “未来奖励” | 从时间步 t 开始累积折扣奖励：`Σ_{k≥0} γ^k r_{t+k+1}`。 |
| 首次访问蒙特卡洛 | “每个状态只算一次” | 每回合只计算首次访问的回报。 |
| 每次访问蒙特卡洛 | “用所有访问” | 每次访问均计入，略有偏差但样本效率更高。 |
| ε-贪婪 | “探索噪声” | 以概率 `1-ε` 选择贪婪动作，概率 `ε` 随机动作。 |
| 重要性采样 | “修正错误分布采样” | 通过加权 `π(a|s)/μ(a|s)` 重构目标策略期望。 |
| 在策略 | “用自己的数据学习” | 目标策略等于行为策略。经典蒙特卡洛、PPO、SARSA。 |
| 离策略 | “用别人的数据学习” | 目标策略不等同于行为策略。重要性采样蒙特卡洛、Q学习、DQN。 |

## 延伸阅读

- [Sutton & Barto (2018). 第5章 — 蒙特卡洛方法](http://incompleteideas.net/book/RLbook2020.pdf) — 权威教材。  
- [Singh & Sutton (1996). 带替换资格迹的强化学习](https://link.springer.com/article/10.1007/BF00114726) — 首次访问与每次访问比较。  
- [Precup, Sutton, Singh (2000). 离策略资格迹策略评估](http://incompleteideas.net/papers/PSS-00.pdf) — 离策略蒙特卡洛与方差控制。  
- [Mahmood 等 (2014). 离策略学习的加权重要性采样](https://arxiv.org/abs/1404.6362) — 现代低方差重要性采样估计器。  
- [Tesauro (1995). TD-Gammon，自我教学的西洋双陆棋程序](https://dl.acm.org/doi/10.1145/203330.203343) — 首个大规模实证展示蒙特卡洛/TD自玩达到超人水平；是本阶段后半部分所有课程的概念前身。
