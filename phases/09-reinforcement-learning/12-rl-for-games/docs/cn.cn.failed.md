# 游戏中的强化学习（RL）—— AlphaZero、MuZero 与大语言模型推理时代

> 1992 年：TD-Gammon 使用纯 TD（时间差分学习）击败了人类西洋双陆棋冠军。2016 年：AlphaGo 战胜李世乭。2017 年：AlphaZero 从零开始称霸国际象棋、将棋和围棋。2024 年：DeepSeek-R1 证明了用 GRPO 代替 PPO 的相同流程可在推理任务中奏效。游戏是这一阶段推动所有突破的基准。

**类型：** 构建  
**语言：** Python  
**先决条件：** 第 9 阶段 · 05（DQN）、第 9 阶段 · 08（PPO）、第 9 阶段 · 09（RLHF）、第 9 阶段 · 10（MARL）  
**时长：** 约 120 分钟

## 问题概述

游戏满足了强化学习所需的一切条件。明确的奖励（胜负）。无限回合（自我对弈可重置）。完美模拟（游戏即模拟器）。离散或小规模连续动作空间。多智能体结构促进了对抗性鲁棒性。

游戏也是每个重大 RL 突破的测试场。TD-Gammon（西洋双陆棋，1992），Atari-DQN（2013），AlphaGo（2016），AlphaZero（2017），OpenAI Five（Dota 2，2019），AlphaStar（星际争霸 II，2019），MuZero（学得模型，2019），AlphaTensor（矩阵乘法，2022），AlphaDev（排序算法，2023），DeepSeek-R1（数学推理，2025）——最新证明游戏 RL 技术可应用于文本推理。

本结业项目通过一个统一视角回顾三大标志性架构——AlphaZero、MuZero 和 GRPO：**自我对弈 + 搜索 + 策略改进**。每个架构都泛化了前一个；GRPO 特别是将 AlphaZero 的流程应用于大语言模型推理，以 token 作为动作，数学验证作为胜利信号。

## 核心概念

![AlphaZero ↔ MuZero ↔ GRPO：相同循环，不同环境](../assets/rl-games.svg)

**统一循环。**

```text
while True:
    trajectory = self_play(current_policy, search)     # 自我对弈
    policy_target = search.improved_policy(trajectory) # 搜索改进策略
    policy_net.update(policy_target, value_target)     # 以搜索输出为监督信号更新网络
```

**AlphaZero（2017）。** Silver 等人提出。针对规则已知的游戏（国际象棋、将棋、围棋）：

- 策略-价值网络：单塔架构 `f_θ(s) → (p, v)`，其中 `p` 是合法动作的先验分布，`v` 是预期游戏结果。
- 蒙特卡洛树搜索（MCTS）：每一步扩展可能的后续树。利用 `(p, v)` 作为先验并做引导。节点选择通过 UCB（PUCT）实现：`a* = argmax Q(s, a) + c · p(a|s) · √N(s) / (1 + N(s, a))`。
- 自我对弈：智能体自我对弈。在第 `t` 步，MCTS 访问分布 `π_t` 作为策略训练目标。
- 损失函数：`L = (v - z)² - π · log p + c · ||θ||²`，`z` 是实际游戏结果（+1 / 0 / -1）。

零人类知识，零手工启发式。一套流程称霸国际象棋、将棋和围棋，经过数千万局自我对弈。

**MuZero（2019）。** Schrittwieser 等人提出。取消了规则已知的要求。

- 不依赖固定环境，学习一个*潜变量动态模型* `(h, g, f)`：
  - `h(s)`：将观测编码为潜在状态。
  - `g(s_latent, a)`：预测下一个潜在状态和奖励。
  - `f(s_latent)`：预测策略先验和价值。
- MCTS 在*潜在空间*中运行，搜索和训练逻辑相同。
- 适用围棋、国际象棋、将棋，甚至 Atari——单一算法，无需规则知识。

**随机 MuZero（2022）。** 增加了随机动态和机会节点；扩展到类似西洋双陆棋的游戏。

**Muesli，Gumbel MuZero（2022-2024）。** 提升采样效率和确定性搜索。

**GRPO（2024-2025）。** DeepSeek-R1 流程。与 AlphaZero 相似的循环，但应用于语言模型推理：

- “游戏”：回答数学/编码/推理题目。“胜利”指验证器判定（测试用例通过、数值答案匹配）返回 1。
- 策略：LLM。动作：tokens。状态：提示加已生成内容。
- 无价值函数（PPO 式的 V_φ）。对每个提示，策略采样 `G` 个完成。计算各自奖励。用**群体相对优势**`A_i = (r_i - mean_r) / std_r`作为 REINFORCE 更新信号。
- KL 惩罚锚定基准策略，防止策略漂移（类似 RLHF）。
- 完整损失：

  `L_GRPO(θ) = -E_{q, {o_i}} [ (1/G) Σ_i A_i · log π_θ(o_i | q) ] + β · KL(π_θ || π_ref)`

无奖励模型、无价值估计器、无 MCTS。群体相对基线代替了全部三个。以更低计算开销达到甚至超越 PPO-RLHF 在推理基准上的效果。

**完整的 R1 流程。** DeepSeek-R1（DeepSeek 2025）是一篇含两个模型的论文：

- **R1-Zero。** 从 DeepSeek-V3 基础模型开始，无监督微调（SFT）。直接用包含“准确性奖励”（基于规则——最终答案是否解析成正确数字/代码是否通过单元测试）和“格式奖励”（结果链思维是否用 `<think>…</think>` 标签包裹）的 GRPO 训练。训练数千步后，平均响应长度从约 100 增至近 10,000 token，数学基准成绩提升至 o1-preview 水平，模型学会零基础推理。缺点在于思维链常难以读懂，混杂语言，且缺少文风修饰。
- **R1。** 通过四阶段流水线修正 R1-Zero 的可读性：
  1. **冷启动 SFT。** 收集数千条格式规范的长链思维示范，对基础模型进行监督微调，获得一个可读的起点。
  2. **面向推理的 GRPO。** 用准确性+格式奖励加语言一致性奖励进行 GRPO，防止语言切换。
  3. **拒绝采样 + 第二轮 SFT。** 从 RL 检查点抽取约 60 万条推理轨迹，仅保留最终答案正确且思路可读的，联同约 20 万条非推理 SFT 例子（写作、问答、自认知）共同再微调基础模型。
  4. **全谱 GRPO。** 进行涵盖推理（基于规则奖励）和通用对齐（基于偏好奖励的有帮助/无害）的最后一轮 RL。

结果在 AIME 和 MATH-500 上开放权重可匹敌 o1，并规模适合蒸馏。同一篇论文发布了六个蒸馏后的密集模型（Qwen-1.5B 至 Llama-70B），通过 SFT 在 R1 的推理轨迹上训练学生模型——学生不使用 RL。强 RL 教师的蒸馏版本在学生规模上持续胜过从零 RL。

**为何推理用 GRPO 而非 PPO。** DeepSeekMath 论文（2024 年 2 月）给出三点原因：（1）无价值网络训练，节省一半内存；（2）群体基线自然应对推理任务罕见的轨迹末奖励；（3）逐提示归一化让优势值在难度差异极大的问题间可比，PPO 单一价值网络则不行。

**无搜索与基于搜索。** 游戏领域已分叉：

- *完美信息、长规划游戏*（围棋、国际象棋）：仍使用基于搜索的方法。AlphaZero / MuZero 统治。
- *LLM 推理*：迄今无生产环境中 MCTS；用 GRPO 基于完整 rollout，推理时选 best-of-N。处理奖励模型（PRM）暗示将来可能加回逐步搜索。

## 构建步骤

`code/main.py` 实现简化版 **GRPO**——多群体样本的多臂赌博机问题。算法与 LLM 相同，仅策略和环境更简单。讲解 *损失* 和 *群体相对优势*，即 2025 年革新。

### 第 1 步：一个微型验证器环境

```python
QUESTIONS = [
    {"prompt": "q1", "correct": 3},
    {"prompt": "q2", "correct": 1},
]

def verify(prompt_idx, answer_token):
    return 1.0 if answer_token == QUESTIONS[prompt_idx]["correct"] else 0.0
```

真实 GRPO 中，验证器执行单元测试或检查数学等式。

### 第 2 步：策略：对每提示的 K 个答案 token 做 softmax

```python
def policy_probs(theta, p_idx):
    return softmax(theta[p_idx])
```

等同于 LLM 在给定提示时的最终层输出。

### 第 3 步：群体采样与群体相对优势

```python
def grpo_step(theta, p_idx, G=8, beta=0.01, lr=0.1, rng=None):
    probs = policy_probs(theta, p_idx)
    samples = [sample(probs, rng) for _ in range(G)]
    rewards = [verify(p_idx, s) for s in samples]
    mean_r = sum(rewards) / G
    std_r = stddev(rewards) + 1e-8
    advs = [(r - mean_r) / std_r for r in rewards]

    for a, A in zip(samples, advs):
        grad = onehot(a) - probs
        for i in range(len(probs)):
            theta[p_idx][i] += lr * A * grad[i]
    # KL 惩罚：将 theta 拉向参考分布
    for i in range(len(probs)):
        theta[p_idx][i] -= beta * (theta[p_idx][i] - reference[p_idx][i])
```

群体相对优势是 2024 年 DeepSeek 的技巧。无需价值函数。“基线”是群体均值，归一化用群体标准差。

### 第 4 步：对比无价值函数的 REINFORCE 基线

同样设置，同样计算，普通 REINFORCE。GRPO 收敛更快、更稳定。

### 第 5 步：监控熵与 KL

参考 RLHF 的诊断指标：平均 KL 与参考的距离，策略熵，奖励随时间变化。指标稳定后训练完成。

## 陷阱提醒

- **通过验证器漏洞篡改奖励。** GRPO 继承了 RLHF 的风险：如果验证器错误或易被利用，LLM 会找到漏洞。需要稳健的验证器（多测试用例、形式证明）。
- **群体规模过小。** 群体基线方差约为 `1/√G`。低于 `G=4` 时优势信号噪声大，常见选用 `G=8` 到 `64`。
- **长度偏差。** 不同长度完成的对数概率不同。需按 token 计数归一，或用序列级对数概率，或截断最大长度。
- **纯自我对弈循环。** AlphaZero 风格训练在一般非零和博弈中可能陷入主导循环。可通过多对手池（联赛机制，教程第 10 课）缓解。
- **搜索与策略不匹配。** AlphaZero 训练策略模仿搜索输出。若策略网络容量不足以表示搜索分布，训练停滞。
- **计算下限。** MuZero / AlphaZero 需巨大算力。单个消融实验往往数百 GPU 小时。存在微缩示例（如 AlphaZero 下的四子棋）供学习。
- **验证器覆盖不足。** 通过的单元测试若对有缺陷解无效，则强化了缺陷。设计捕捉边缘情况的验证器。

## 应用场景

2026 年游戏 RL 领域按应用分布：

| 领域 | 主导方法 |
|--------|-----------------|
| 双人零和棋盘游戏（围棋、国际象棋、将棋） | AlphaZero / MuZero / KataGo |
| 不完全信息纸牌游戏（扑克） | CFR + 深度学习（DeepStack、Libratus、Pluribus） |
| Atari / 像素游戏 | Muesli / MuZero / IMPALA-PPO |
| 大规模多人策略游戏（Dota、星际争霸） | PPO + 自我对弈 + 联赛（OpenAI Five, AlphaStar） |
| LLM 数学/代码推理 | GRPO（DeepSeek-R1、Qwen-RL、开源复现） |
| LLM 对齐 | DPO / RLHF-PPO（非 GRPO；验证器是偏好无法验证） |
| 机器人 | PPO + DR（非游戏 RL，但使用相同策略梯度工具） |
| 组合优化问题 | AlphaZero 变体（AlphaTensor、AlphaDev） |

*流程*——自我对弈、搜索增强改进、策略蒸馏——横跨文本、像素及物理控制。GRPO 是其中最年轻的实例，更多玩法即将到来。

## 交付成品

保存为 `outputs/skill-game-rl-designer.md`：

```markdown
---
name: game-rl-designer
description: 为给定领域设计游戏强化学习（game-RL）或推理强化学习（reasoning-RL）训练管线（AlphaZero / MuZero / GRPO）。
version: 1.0.0
phase: 9
lesson: 12
tags: [rl, alphazero, muzero, grpo, self-play]
---

给定一个目标（完全信息游戏 / 不完全信息 / Atari / 大语言模型推理 / 组合优化），输出：

1. 环境适配。已知规则？马尔可夫？随机？多智能体？决定使用 AlphaZero 还是 MuZero 还是 GRPO。
2. 搜索策略。MCTS（带学习先验的 PUCT）、Gumbel 采样、N 选最佳，或无搜索。
3. 自我对弈计划。对称自我对弈 / 联赛制 / 离线数据 / 验证器生成。
4. 目标信号。比赛结果 / 验证器奖励 / 偏好 / 学习模型。包含鲁棒性方案。
5. 诊断。相对于基线的胜率、ELO 曲线、验证器通过率、与参考分布的 KL 散度。

拒绝在不完全信息游戏使用 AlphaZero（应转向 CFR）。拒绝无可信验证器的 GRPO。拒绝无固定基线对手集合的任何游戏-RL管线（否则自我对弈 ELO 不具备校准性）。
```

## 练习

1. **简单。** 在 `code/main.py` 实现 GRPO 多臂老虎机算法。针对 2 条提示 × 各 4 个答案标记训练。用 `G=8`，在不到 1,000 次更新时收敛。
2. **中等。** 接入 PPO（裁剪版）和基础 REINFORCE。比较与 GRPO 在同一多臂老虎机上的采样效率和奖励方差。
3. **困难。** 扩展到长度为 2 的“推理链”：智能体输出两个标记，验证器给出对这对标记的奖励。测量 GRPO 如何处理跨两步序列的信用分配。（提示：按*完整序列*计算组优势，传递给两个标记位置。）

## 关键词

| 词汇 | 常见说法 | 实际含义 |
|------|----------|----------|
| MCTS | “带学习网络的树搜索” | 蒙特卡洛树搜索；使用学习得出的 `(p, v)` 先验执行 UCB1/PUCT 选择。 |
| AlphaZero | “自我对弈 + MCTS” | 策略-价值网络，训练以匹配 MCTS 访问次数和游戏最终结果。 |
| MuZero | “学习模型版 AlphaZero” | 使用学习的隐状态动力学，在潜在空间中执行相同的循环。 |
| GRPO | “无评论家 PPO” | 组相对策略优化；使用基于组均值的基线和 KL 的 REINFORCE 算法。 |
| PUCT | “AlphaZero 的 UCB” | `Q + c · p · √N / (1 + N_a)` — 在价值估计和先验之间平衡。 |
| 自我对弈 | “智能体与历史自己对战” | 零和博弈标准，提供对称训练信号。 |
| 联赛制 | “基于群体的自我对弈” | 从历史、当前和利用者中采样对手。 |
| 验证器奖励 | “可验证强化学习” | 奖励来源于确定性验证器（测试是否通过，答案是否匹配）。 |
| 过程奖励 | “过程奖励模型（PRM）” | 对每个推理步骤进行打分，而非仅对最终答案打分。 |

## 深入阅读

- [Silver et al. (2017). Mastering the game of Go without human knowledge (AlphaGo Zero)](https://www.nature.com/articles/nature24270)。
- [Silver et al. (2018). A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play (AlphaZero)](https://www.science.org/doi/10.1126/science.aar6404)。
- [Schrittwieser et al. (2020). Mastering Atari, Go, chess and shogi by planning with a learned model (MuZero)](https://www.nature.com/articles/s41586-020-03051-4)。
- [Vinyals et al. (2019). Grandmaster level in StarCraft II (AlphaStar)](https://www.nature.com/articles/s41586-019-1724-z)。
- [DeepSeek-AI (2024). DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (GRPO)](https://arxiv.org/abs/2402.03300) — 引入 GRPO 和组相对基线的论文。
- [DeepSeek-AI (2025). DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948) — 全套四阶段 R1 配方及 R1-Zero 消融实验。
- [Brown et al. (2019). Superhuman AI for multiplayer poker (Pluribus)](https://www.science.org/doi/10.1126/science.aay2400) — 结合 CFR 和深度学习的大规模扑克 AI。
- [Tesauro (1995). Temporal Difference Learning and TD-Gammon](https://dl.acm.org/doi/10.1145/203330.203343) — 开创之作。
- [Hugging Face TRL — GRPOTrainer](https://huggingface.co/docs/trl/main/en/grpo_trainer) — 使用自定义奖励函数应用 GRPO 的生产级参考。
- [Qwen Team (2024). Qwen2.5-Math — GRPO replication](https://github.com/QwenLM/Qwen2.5-Math) — 多尺度开放复现 R1 配方。
- [Sutton & Barto (2018). Ch. 17 — Frontiers of Reinforcement Learning](http://incompleteideas.net/book/RLbook2020.pdf) — 教科书中的自我对弈、搜索和“设计奖励”框架，R1 在 LLM 规模上的实例化。
```text
