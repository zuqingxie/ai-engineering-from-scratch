# 推测解码与EAGLE

> 前沿的大型语言模型（Large Language Model，LLM）生成一个token需要对数十亿参数进行完整的前向传播。这次前向传播资源远远超过实际需求：大多数时候，一个更小的模型可以准确猜出接下来的3-5个token，大模型只需要*验证*这个猜测而已。当猜测正确时，你用生成一个token的成本获得了5个token。推测解码（Leviathan等，2023）实现了这一点的精确化，而EAGLE-3（2025）将接受率推升至约每次验证产生4.5个token —— 在匹配输出分布的情况下，实现了4-5倍的加速。

**类型：** 实现  
**语言：** Python（含numpy）  
**先修课程：** 第10阶段第12课（推理优化），第10阶段第4课（预训练Mini-GPT）  
**时长：** ~75分钟

## 问题描述

在H100上，70B级模型的解码吞吐量通常为40-80 tokens/秒。每个token都需要全模型权重的完整前向传播。不能在不改变输出的情况下缩小模型，也不能突破内存限制增加batch大小。你陷入了瓶颈 —— 除非你能让模型每次前向传播输出超过一个token。

自回归生成看似本质上是串行的：`x_{t+1} = sample(p(· | x_{1:t}))`。但这里存在并发机会。如果你有一个廉价的预测器说“接下来的4个token大概率是[a, b, c, d]”，你可以用**大模型的一次前向传播**同时验证所有5个位置，并接受最长的匹配前缀。

Leviathan、Kalai和Matias（2023年，“Fast Inference from Transformers via Speculative Decoding”）通过巧妙的接受/拒绝规则实现了这一点，保证了目标模型的采样分布不变。输出分布相同，加速2-4倍。

## 概念解析

### 双模型架构

- **目标模型** `M_p`：你实际想从中采样的大而慢、高质量模型。分布：`p(x)`。
- **草稿模型** `M_q`：一个小而快、但质量较低的模型。分布：`q(x)`。大小是目标模型的1/5到1/30。

每一步：

1. 草稿模型自回归地提出`K`个token：`x_1, x_2, ..., x_K ~ q`。
2. 目标模型对所有`K+1`个位置并行运行一次前向传播，计算每个提出的token的`p(x_k)`。
3. 按顺序对每个token执行下述修正拒绝采样规则，接受最长匹配的前缀。
4. 若有token被拒绝，从修正分布中采样替代token并停止；否则从`p(· | x_1...x_K)`采样一个额外的奖励token。

如果草稿和目标完全匹配，则每次目标模型前向传播能产出`K+1`个token；如果草稿在第1个token就错了，则只能产出1个token。

### 精确性规则

推测解码**在分布意义上等价于从p采样**。拒绝采样规则：

```text
对每个草稿token x_t:
    r ~ Uniform(0, 1)
    若 r < p(x_t) / q(x_t):
        接受 x_t
    否则:
        从残差分布 (p - q)+ / ||(p - q)+||_1 中采样替代token
        停止
```

其中 `(p - q)+` 表示逐点差值的正部分。当草稿与目标一致（`p ≈ q`）时，接受率几乎为1；当两者不一致时，残差分布被构造以保证整体采样依然精确为`p`。

**贪婪情况**。温度为0时，直接检查 `argmax(p) == x_t`。相等则接受，不等则输出 `argmax(p)` 并停止。

### 期待加速

如果草稿模型的单个token接受率为`α`，那么每次目标前向传播产出的期望token数为：

```text
E[tokens] = (1 - α^{K+1}) / (1 - α)        # K为草稿长度，α ∈ [0, 1]
```

在`α = 0.8, K = 4`时：`(1 - 0.8^5)/(1 - 0.8) = 3.36`，即每次前向传播生成3.36个token。目标模型前向传播的成本大致为 `cost_q * K + cost_p` （K步草稿加1步目标验证）。若 `cost_p >> cost_q * K`，带来的吞吐量加速比是`3.36×`。

唯一关键参数是`α`，完全依赖于草稿-目标的对齐。一个良好的草稿模型是成功的关键。

### 草稿训练：蒸馏

随机的小模型草稿效果差。标准方法是从目标蒸馏：

1. 选一个小架构（70B目标约1B，7B目标约500M）。
2. 在大文本语料上运行目标模型，存储其下一token分布。
3. 用KL散度以目标分布为监督训练草稿（非监督真实token）。

结果：`α`通常在编程任务中为0.6-0.8，自然语言聊天中为0.7-0.85。实际应用中能提升2-3倍速度。

### EAGLE：树状草稿与特征复用

Li、Wei、Zhang 等（2024，“EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty”）发现传统推测解码有两大低效：

1. 草稿进行K步串行，每步都完整decoder一次。但草稿其实可以复用目标模型最近一次验证时计算的特征（隐藏态），因为目标已经计算了丰富的表示，而草稿是重新从头计算一遍。
2. 草稿输出线性链。若改为输出一个*树状候选集合*（每个节点多个猜测），目标模型一次前向传播利用树状注意力掩码即可并行验证多条路径，选择最长被接受路径。

EAGLE-1改进点：
- 草稿输入改为目标在位置t的最终隐藏态，而非原始tokens。
- 草稿架构是1层transformer decoder（非独立小模型）。
- 输出是深度为4-6、每层4-8候选的树。

EAGLE-2（2024）引入动态树拓扑：树在草稿不确定处变宽，确定处保持窄，提升有效接受率`α_effective`且不增加验证成本。

EAGLE-3（Li等，2025，“EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test”）取消了固定顶层特征依赖，采用新的“测试时模拟”损失训练草稿 —— 草稿在符合目标测试时分布的输出上训练，而非教师强制训练分布。接受率由EAGLE-2的0.75提升到0.82，平均tokens/验证由3.0升至4.5。

### 树状注意力验证

草稿输出树后，目标模型用**树状注意力掩码**一次前向传播验证——这是一种因果掩码，编码树形拓扑结构而非纯线性。每个token只关联树上的祖先节点。验证过程仍是一次前向、一次矩阵乘法；拓扑掩码仅带来少量额外KV条目。

```text
        root
       /    \
      a      b
     / \    / \
    c  d   e   f
```

若`a, b`为第一步的候选token，`c, d, e, f`为第二步的候选，6个位置都能在一次前向验证。输出是任意被接受路径中最长的前缀。

### 适用与限制

**适用场景：**
- 预测性强的聊天/补全任务（代码、常见英语、结构化输出）。`α`高。
- 解码时GPU计算资源未完全使用（如内存受限阶段）。树状草稿利用剩余FLOPs。

**不适用/无加速：**
- 高度随机输出（高温度创作写作）。`α`降低至约 `1/词汇表大小`。
- 超高并发批量服务 —— 批量已满载FLOPs，树验证空间有限。
- 目标模型很小，草稿模型几乎无缩减。

实测中，生产环境中聊天加速2-3倍，代码生成3-5倍，创意写作无明显加速。

## 实现它

`code/main.py`：

- 一个参考实现函数 `speculative_decode(target, draft, prompt, K, temperature)` ，实现精确拒绝采样规则，并验证其保持目标分布（经验KL < 0.01，相对于普通目标采样）。
- 一个EAGLE风格树形草稿器，实现带top-p分支的深度K树构建。
- 一个树状注意力掩码生成器，输出正确的因果模式供验证器使用。
- 一个接受率测试框架，在微小语言模型（使用GPT-2-medium目标蒸馏出GPT-2-small）上运行。

```python
def speculative_step(p_target, q_draft, K, temperature=1.0):
    """一轮推测解码。返回接受的token列表。"""
    # 1. 草稿生成K个token
    draft_tokens = []
    q_probs = []
    state = draft_state_init()
    for _ in range(K):
        probs = softmax(q_draft(state) / temperature)
        t = np.random.choice(len(probs), p=probs)
        draft_tokens.append(t)
        q_probs.append(probs[t])
        state = draft_step(state, t)

    # 2. 目标计算所有草稿token及1个额外位置的概率
    p_probs_all = target_forward_batched(p_target, draft_tokens, temperature)

    # 3. 按顺序接受或拒绝
    accepted = []
    for k, tok in enumerate(draft_tokens):
        r = np.random.uniform()
        if r < p_probs_all[k][tok] / q_probs[k]:
            accepted.append(tok)
        else:
            residual = np.maximum(p_probs_all[k] - q_probs[k], 0)
            residual /= residual.sum()
            accepted.append(np.random.choice(len(residual), p=residual))
            return accepted
    # 4. 全部K个都接受 → 从目标分布采样一个奖励token
    accepted.append(np.random.choice(len(p_probs_all[-1]), p=p_probs_all[-1]))
    return accepted
```

## 使用它

- **vLLM** 和 **SGLang** 已内置高质量推测解码。参数：`--speculative_model`，`--num_speculative_tokens`。EAGLE-2/3支持`--spec_decoding_algorithm eagle`。
- **NVIDIA TensorRT-LLM**原生支持Medusa和EAGLE树形结构。
- **参考草稿模型**：`Qwen/Qwen3-0.6B-spec`（Qwen3-32B的草稿）、`meta-llama/Llama-3.2-1B-Instruct-spec`（70B的草稿）。
- **Medusa heads**（Cai等，2024年，“Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads”）：不靠外部草稿模型，而是在目标模型本身添加K个并行预测头。部署更简单，但接受率略低于EAGLE。

## 上线部署

本课输出`outputs/skill-speculative-tuning.md` —— 一项技能，用于剖析目标模型的具体工作负载，从而选择：草稿模型、K（草稿长度）、树宽、温度、以及何时回退到普通解码。

## 练习

1. 实现精确拒绝规则并做经验验证。运行1万次`speculative_decode`和普通目标采样，计算两者输出分布的总变差距离（Total Variation Distance）。应小于0.01。

2. 计算加速公式。固定`α`和`K`，绘制预期每次目标前向生成的token数。对`α ∈ {0.5, 0.7, 0.9}`寻找最优K。

3. 训练微小草稿模型。选124M GPT-2目标，使用1亿tokens，基于KL散度蒸馏30M GPT-2草稿。测量验证文本上的`α`，预期0.6-0.7。

4. 实现EAGLE风格树形草稿。让草稿在每层输出top-3分支。构建树状注意力掩码。验证目标能接受最长正确分支。

5. 测量失败情况。温度1.5运行推测解码（高随机性）。观察`α`崩溃，算法因草稿开销而比普通解码更慢。

## 关键词

| 术语             | 易混淆说法         | 实际含义                                    |
|------------------|--------------------|---------------------------------------------|
| 目标模型         | “大模型”           | 慢且高质量的采样模型（p分布）              |
| 草稿模型         | “猜测器”           | 小型快速预测器（q分布）；体积缩减5-30倍    |
| K / 草稿长度     | “前瞻长度”         | 每次验证中推测的token数量                    |
| α / 接受率       | “命中率”           | 草稿所提token被接受的概率                    |
| 精确拒绝规则     | “接受测试”         | r < p/q 比较，保证样本分布精确为目标分布     |
| 残差分布         | “修正p-q”          | (p - q)+ / ||(p - q)+||_1，拒绝时采样分布  |
| 树形推测         | “分支猜测”         | 草稿输出树形候选，单次通过树状注意力验证    |
| 树状注意力掩码   | “拓扑掩码”         | 编码树结构的因果掩码，每节点只关注祖先        |
| Medusa heads     | “并行预测头”       | 目标自身带K个额外预测头，无需独立草稿模型    |
| EAGLE特征复用    | “隐藏态草稿”       | 草稿输入为目标的最终隐藏态，非原始tokens     |
| 测试时模拟损失   | “EAGLE-3训练”      | 草稿针对目标测试时分布训练，而非教师强制     |

## 进一步阅读

- [Leviathan, Kalai, Matias, 2023 — "Fast Inference from Transformers via Speculative Decoding"](https://arxiv.org/abs/2211.17192) — 精确拒绝规则和理论加速分析
- [Chen, Borgeaud, Irving et al., 2023 — "Accelerating Large Language Model Decoding with Speculative Sampling"](https://arxiv.org/abs/2302.01318) — DeepMind 并行的推测采样论文
- [Cai, Li, Geng, Wang, Wang, Zhu, Dao, 2024 — "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads"](https://arxiv.org/abs/2401.10774) — 多解码头并行的模型草稿替代方案
- [Li, Wei, Zhang, Zhang, 2024 — "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty"](https://arxiv.org/abs/2401.15077) — 特征重用与树状草稿
- [Li et al., 2024 — "EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees"](https://arxiv.org/abs/2406.16858) — 动态树形拓扑
- [Li et al., 2025 — "EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test"](https://arxiv.org/abs/2503.01840) — 训练时测试匹配
- [Fu, Haotian, Peng et al., 2024 — "Break the Sequential Dependency of LLM Inference Using Lookahead Decoding"](https://arxiv.org/abs/2402.02057) — Jacobi/前瞻解码，一种无推测器的替代方法
