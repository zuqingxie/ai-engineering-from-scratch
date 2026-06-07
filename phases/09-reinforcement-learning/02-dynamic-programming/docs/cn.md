# 动态规划 — 策略迭代 & 价值迭代

> 动态规划是作弊版的强化学习（RL）。你已经知道转移函数和奖励函数；只需迭代贝尔曼方程，直到 `V` 或 `π` 不再变化。它是每个基于采样方法努力接近的基准。

**类型:** 构建  
**语言:** Python  
**先决条件:** 第9阶段 · 01（MDPs）  
**时间:** 约75分钟

## 问题描述

你有一个已知模型的马尔可夫决策过程（MDP）：你可以查询任意状态动作对的 `P(s' | s, a)` 和 `R(s, a, s')`。库存管理员知道需求分布。棋盘游戏有确定性转移。格子世界是几行 Python 代码。你有一个*模型*。

无模型的强化学习（Q-learning、PPO、REINFORCE）是为没有模型情况发明的——只能从环境中采样。但当你有模型时，有更快更好的方法：动态规划。贝尔曼1957年设计了它们。它们仍定义了正确性标准：当人们说“此MDP的最优策略”，他们指的是动态规划会返回的策略。

到了2026年，你需要它们的三个原因。第一，强化学习研究中每个表格环境（GridWorld、FrozenLake、CliffWalking）都用动态规划求解，以生成标准策略。第二，精确的值可以用来*调试*采样方法：如果Q-learning对 `V*(s_0)` 的估计与动态规划答案相差30%，说明你的Q-learning有bug。第三，现代离线RL和规划方法（蒙特卡洛树搜索（MCTS）、AlphaZero搜索、第9阶段 · 10中的基于模型的RL）都在已知或学习到的模型上迭代贝尔曼备份。

## 概念

![策略迭代和价值迭代并列示意](../assets/dp.svg)

**两个算法，都是贝尔曼方程的固定点迭代。**

**策略迭代。** 交替进行两步，直到策略不再改变。

1. *评估:* 给定策略 `π`，通过反复应用 `V(s) ← Σ_a π(a|s) Σ_{s',r} P(s',r|s,a) [r + γ V(s')]` 计算 `V^π`，直到收敛。
2. *改进:* 给定 `V^π`，使 `π` 对 `V^π` 贪婪：`π(s) ← argmax_a Σ_{s',r} P(s',r|s,a) [r + γ V(s')]`。

收敛是有保障的，因为 (a) 每次改进步骤要么保持 `π` 不变，要么至少使某个状态的 `V^π` 严格增加，(b) 确定性策略空间是有限的。通常即使状态空间较大，也能在~5–20个外部迭代内收敛。

**价值迭代。** 将评估和改进合并为一次遍历。应用贝尔曼*最优性*方程：

`V(s) ← max_a Σ_{s',r} P(s',r|s,a) [r + γ V(s')]`

重复直到 `max_s |V_{new}(s) - V(s)| < ε`。最后通过贪婪选取动作提取策略。每次迭代严格更快——没有内部评估循环——但通常需要更多迭代次数才能收敛。

**广义策略迭代（GPI）。** 统一的框架。值函数和策略在双向改进循环中相互锁定；任何使两者朝相互一致方向前进的方法（异步价值迭代、修改策略迭代、Q-learning、actor-critic、PPO）都是 GPI 的实例。

**为什么 `γ < 1` 很重要。** 贝尔曼算子是超范数下的 `γ`-收缩映射：`||T V - T V'||_∞ ≤ γ ||V - V'||_∞`。收缩性质保证唯一不动点和几何收敛。去掉 `γ < 1` 条件就失去保证——需要有限时间步长或吸收终端状态。

## 实现步骤

### 步骤1：构建 GridWorld MDP 模型

使用第01课中相同的4×4格子世界。添加随机变体：以概率 `0.1` 代理会滑向随机的垂直方向。

```python
SLIP = 0.1

def transitions(state, action):
    if state == TERMINAL:
        return [(state, 0.0, 1.0)]
    outcomes = []
    for direction, prob in action_probs(action):
        outcomes.append((apply_move(state, direction), -1.0, prob))
    return outcomes
```

`transitions(s, a)` 返回一个 `(s', r, p)` 元组列表。这就是整个模型。

### 步骤2：策略评估

给定策略 `π(s) = {action: prob}`，迭代贝尔曼方程直到 `V` 不再变化：

```python
def policy_evaluation(policy, gamma=0.99, tol=1e-6):
    V = {s: 0.0 for s in states()}
    while True:
        delta = 0.0
        for s in states():
            v = sum(pi_a * sum(p * (r + gamma * V[s_prime])
                              for s_prime, r, p in transitions(s, a))
                   for a, pi_a in policy(s).items())
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            return V
```

### 步骤3：策略改进

用 `V` 对应的贪婪策略替代 `π`。若 `π` 不变，则返回——表示达到最优。

```python
def policy_improvement(V, gamma=0.99):
    new_policy = {}
    for s in states():
        best_a = max(
            ACTIONS,
            key=lambda a: sum(p * (r + gamma * V[s_prime])
                              for s_prime, r, p in transitions(s, a)),
        )
        new_policy[s] = best_a
    return new_policy
```

### 步骤4：组合

```python
def policy_iteration(gamma=0.99):
    policy = {s: "up" for s in states()}   # 任意初始策略
    for _ in range(100):
        V = policy_evaluation(lambda s: {policy[s]: 1.0}, gamma)
        new_policy = policy_improvement(V, gamma)
        if new_policy == policy:
            return V, policy
        policy = new_policy
```

4×4格子世界典型收敛所需外部迭代4–6次。结果输出 `V*(0,0) ≈ -6` 和显著减少步骤数的策略。

### 步骤5：价值迭代（一循环版本）

```python
def value_iteration(gamma=0.99, tol=1e-6):
    V = {s: 0.0 for s in states()}
    while True:
        delta = 0.0
        for s in states():
            v = max(sum(p * (r + gamma * V[s_prime])
                       for s_prime, r, p in transitions(s, a))
                   for a in ACTIONS)
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            break
    policy = policy_improvement(V, gamma)
    return V, policy
```

相同不动点，代码更精简。

## 注意事项

- **不要忘了处理终端状态。** 对吸收状态应用贝尔曼方程仍然会选出一个“最佳动作”，但不会改变价值。用 `if s == terminal: V[s] = 0` 保护。
- **使用超范数而非 L2 收敛指标。** 用 `max |V_new - V|`，而非平均值。理论保证基于超范数。
- **原地更新 vs 同步更新。** 原地更新（高斯-赛德尔）比分离的 `V_new` 字典（雅可比）收敛更快。生产代码用原地更新。
- **策略平分情况。** 如果两个动作Q值相等，`argmax`可能每次迭代打破平局不同，导致“策略稳定”判定振荡。使用稳定的平局规则（固定顺序中第一个动作）。
- **状态空间爆炸。** 动态规划每次遍历的复杂度为 `O(|S| · |A|)`。可处理约10⁷个状态。更大需用函数近似（第9阶段 · 05之后）。

## 适用场景

2026年，动态规划是正确性基准与规划器内循环：

| 使用场景             | 方法                                 |
|--------------------|------------------------------------|
| 精确求解小型表格 MDP       | 价值迭代（更简洁）或策略迭代（外循环更少）          |
| 验证 Q-learning / PPO 实现 | 与动态规划最优 `V*` 比较（玩具环境）                 |
| 基于模型的强化学习（第9阶段 · 10） | 对学习到的转移模型做贝尔曼备份                      |
| AlphaZero / MuZero 规划  | 蒙特卡洛树搜索 = 异步贝尔曼备份                         |
| 离线RL（CQL、IQL）       | 保守Q迭代——加对OOD动作惩罚的动态规划                        |

每次有人说“最优值函数”，他们意指“动态规划不动点”。论文中出现 `V*` 或 `Q*`，就想象这一循环。

## 交付成果

保存至 `outputs/skill-dp-solver.md`：

```markdown
---
name: dp-solver
description: 通过策略迭代或价值迭代精确求解小型表格MDP。报告收敛表现。
version: 1.0.0
phase: 9
lesson: 2
tags: [rl, dynamic-programming, bellman]
---

给定已知模型的MDP，输出：

1. 选择。策略迭代 vs 价值迭代。理由关联 |S|、|A|、γ。
2. 初始化。V_0，起始策略。收敛敏感性。
3. 停止条件。超范数容忍度 ε。预期遍历次数。
4. 验证。准确计算 `V*(s_0)`。贪婪策略提取。
5. 应用。该基准如何用于调试/评估基于采样的方法。

拒绝处理状态空间>10⁷的DP。拒绝无超范数检验的收敛声明。无限时域任务若 γ≥1 警告为违保证。
```

## 练习

1. **简单。** 在4×4格子世界上运行价值迭代，`γ ∈ {0.9, 0.99}`。扫多少次直到 `max |ΔV| < 1e-6`？打印 `V*` 为4×4网格。
2. **中等。** 在*随机*格子世界（滑动概率0.1）上比较策略迭代与价值迭代。计数：遍历次数、实际耗时、最终 `V*(0,0)`。哪种迭代次数更少？耗时更短？
3. **困难。** 构建修改后的策略迭代：评估步骤只运行 `k` 次扫描而非收敛。绘制 `V*(0,0)` 误差对 `k` 的关系曲线，`k ∈ {1, 2, 5, 10, 50}`。该曲线说明了评估/改进的权衡？

## 关键词

| 术语           | 人们怎么说           | 实际含义                            |
|----------------|----------------------|-----------------------------------|
| 策略迭代       | “动态规划算法”       | 交替评估（`V^π`）和改进（相对于 `V^π` 贪婪的 `π`），直到策略不变。 |
| 价值迭代       | “更快的动态规划”     | 贝尔曼最优性备份一次完成；几何收敛到 `V*`。      |
| 贝尔曼算子     | “递归计算”           | `(T V)(s) = max_a Σ P (r + γ V(s'))`；超范数下`γ`-收缩映射。 |
| 收缩映射       | “动态规划为什么收敛” | 任意满足 `\|\|T x - T y\|\| ≤ γ \|\|x - y\|\|` 的算子具有唯一不动点。 |
| 广义策略迭代（GPI） | “一切都是动态规划”   | Generalized Policy Iteration：任何驱动 `V` 和 `π` 互相一致的方法。 |
| 同步更新       | “雅可比式”           | 整个遍历中使用旧 `V`；分析简单但较慢。          |
| 原地更新       | “高斯-赛德尔式”     | 更新时立即使用 `V`，通常收敛更快。                 |

## 相关阅读

- [Sutton & Barto (2018). 第4章 — 动态规划](http://incompleteideas.net/book/RLbook2020.pdf) — 策略迭代和价值迭代的权威讲解。
- [Bertsekas (2019). Reinforcement Learning and Optimal Control](http://www.athenasc.com/rlbook.html) — 收缩映射论证的严谨处理。
- [Puterman (2005). Markov Decision Processes](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470316887) — 修改策略迭代及其收敛分析。
- [Howard (1960). Dynamic Programming and Markov Processes](https://mitpress.mit.edu/9780262582300/dynamic-programming-and-markov-processes/) — 最初的策略迭代论文。
- [Bertsekas & Tsitsiklis (1996). Neuro-Dynamic Programming](http://www.athenasc.com/ndpbook.html) — 动态规划到近似DP/深度强化学习的桥梁，是后续课程的基础。
