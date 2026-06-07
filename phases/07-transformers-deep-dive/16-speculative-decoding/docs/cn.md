# 推测解码 — 草稿、验证、重复

> 自回归（Autoregressive decoding）解码是串行的。每个标记都要等待前一个标记完成。推测解码打破了这个链条：一个廉价模型草拟 N 个标记，然后昂贵模型一次前向传递验证所有 N 个标记。当草稿正确时，你只为 N 次生成付出一次大的前向计算代价。

**类型：** 构建  
**语言：** Python  
**先决条件：** Phase 7 · 07（GPT 因果语言模型 GPT Causal LM），Phase 7 · 12（KV 缓存 & Flash Attention）  
**时间：** ~60 分钟

## 问题

一个 70B 大型语言模型（LLM）在 H100 上采样一个标记大约需要 30 毫秒。一个 3B 草稿模型大约需要 3 毫秒。如果让 3B 草稿模型提前草拟 5 个标记，然后让 70B *一次* 验证这 5 个标记，总代价是 `5×3 + 30 = 45 毫秒`，可接受最多 5 个标记 —— 相比之下，直线生成需要 `5×30 = 150 毫秒`。这就是完整的推测解码方案：以少量额外的 GPU 内存（草稿模型）换取 2–4 倍更低的解码延迟。

关键是要保持分布不变。由 Leviathan 等人（2023）和 Chen 等人同期提出的推测采样（Speculative sampling），保证输出序列的**分布与大模型独立采样的分布完全一致**。无任何质量折中。仅仅更快。

2026 年推理中，四大类草稿-验证对占主导地位：

1. **原味推测（Vanilla speculative，Leviathan 2023）。** 独立的草稿模型（例如 Llama 3 1B）+ 验证器（例如 Llama 3 70B）。
2. **Medusa（Cai 2024）。** 验证器上多个解码头并行预测位置 `t+1..t+k`，无独立草稿模型。
3. **EAGLE 系列（Li 2024、2025）。** 轻量草稿模型复用验证器的隐藏状态，接受率更接近原味，典型提升 3–4 倍。
4. **向前看解码（Lookahead decoding，Fu 2024）。** Jacobi 迭代方式；根本不需要草稿模型。自我推测。小众但无依赖。

2026 年，几乎所有生产推理栈默认出厂支持推测解码。vLLM、TensorRT-LLM、SGLang 和 llama.cpp 至少支持原味和 EAGLE-2。

## 概念

### 核心算法

给定一个验证器 `M_q` 和一个更便宜的草稿模型 `M_p`：

1. 令 `x_1..x_k` 为已经解码的前缀。
2. **草稿**：用 `M_p` 自回归提出 `d_{k+1}, d_{k+2}, ..., d_{k+N}`，对应草稿概率 `p_1..p_N`。
3. **并行验证**：用 `M_q` 运行一次，输入序列为 `x_1..x_k, d_{k+1}, ..., d_{k+N}`，得到验证概率 `q_1..q_{N+1}`，对应位置 `k+1..k+N+1`。
4. **从左到右接受/拒绝草稿标记**：对每个 `i`，以概率 `min(1, q_i(d_i) / p_i(d_i))` 接受。
5. 第一次拒绝发生在位置 `j` 时：从“残差”分布 `(q_j - p_j)_+` 归一化中采样一个标记 `t_j`，丢弃位置 `j` 之后的所有草稿。
6. 如果全部 `N` 个标记全部接受，则从 `q_{N+1}` 中采样一个额外标记 `t_{N+1}`（免费奖励标记）。

残差分布技巧是数学上的关键洞见，保证输出分布完全等同于 `M_q` 从零采样的分布。

### 决定加速的因素

设 `α` = 每个草稿标记的期望接受率。设 `c` = 草稿模型相对于验证器的计算成本比。每一步：

- 常规生成每采样一个标记调用 1 次大模型。
- 推测法对于很高的 `α` 值，每采样 `(1 - α^{N+1}) / (1 - α) ≈ 1/(1-α)` 个标记调用 1 次大模型。

经验法则：当 `α = 0.75`，`N = 5` 时，调用大模型次数减小 3 倍。草稿成本是大模型的 1/5。总耗时减少约 2.5 倍。

**α 取决于：**

- 草稿模型对验证器的拟合程度。相同系列/相同训练数据会显著提升 α。
- 解码策略。贪心草稿对贪心验证器：高 α。温度采样：更难匹配，接受率下降。
- 任务类型。代码和结构化输出更可预测，接受率高；自由创作写作接受率低。

### Medusa — 无需草稿模型的草稿

Medusa 用额外的输出头替代草稿模型。在位置 `t`：

```text
共享主体 → 隐藏态 h_t
    ├── head_0：预测位置 t+1 的标记（标准语言模型头）
    ├── head_1：预测位置 t+2 的标记
    ├── head_2：预测位置 t+3 的标记
    ├── head_3：预测位置 t+4 的标记
```

每个头输出自己的 logits。推理时从每个头采样得到候选序列，然后用一次前向传递的树状注意力（tree-attention）机制同时验证所有候选续写。

优点：无第二模型。缺点：增加可训练参数；需要监督微调阶段（约 10 亿标记）；接受率略低于带有良好草稿模型的原味推测。

### EAGLE — 利用隐藏状态实现更好的草稿

EAGLE-1/2/3（Li 等，2024–2025）让草稿模型成为一个非常小的 Transformer（通常 1 层），输入为验证器的最后一层隐藏状态。由于草稿模型看到的是验证器的特征表示，其预测与验证器的输出分布高度相关。接受率从大约 0.6（原味）提升到超过 0.85。

EAGLE-3（2025）加入了候选续写的树搜索。vLLM 和 SGLang 将 EAGLE-2/3 作为 Llama 3/4 和 Qwen 3 的默认推测路径。

### KV 缓存管理

验证一次前向传递时，将 `N` 个草稿标记输入验证器，这会将验证器的 KV 缓存扩展 `N` 条目。如果部分草稿被拒绝，必须将缓存回滚到被接受的前缀长度。

生产实现（如 vLLM 的 `--speculative-model`，TensorRT-LLM 的 LookaheadDecoder）使用临时 KV 缓冲区；先写入，接受后提交。概念上不难，但实现细节复杂。

## 构建它

参见 `code/main.py`。我们实现核心推测采样算法（拒绝步骤 + 残差分布），具体包括：

- 使用“大模型”为基于手写分布的确定性 softmax（方便解析验证接受数学）。
- 使用“草稿模型”为大模型的扰动版本。
- 采用接受/拒绝循环来生成和直接采样等价的边际分布。

### 步骤 1：拒绝步骤

```python
def accept_or_reject(q_prob, p_prob, draft_token, u):
    ratio = q_prob / p_prob if p_prob > 0 else float("inf")
    return u < min(1.0, ratio)
```

`u` 是均匀随机数。`q_prob` 是验证器对草稿标记的概率。`p_prob` 是草稿模型的概率。Leviathan 定理表明，这个伯努利决策及拒绝时从残差分布采样，精确保持验证器分布。

### 步骤 2：残差分布

```python
def residual_dist(q, p):
    raw = [max(0.0, qi - pi) for qi, pi in zip(q, p)]
    s = sum(raw)
    return [r / s for r in raw]
```

逐元素计算 `q - p`，负数截为零，归一化。拒绝时从该分布采样。

### 步骤 3：一次推测步骤

```python
def spec_step(prefix, q_model, p_model, N, rng):
    drafts = []
    p_probs = []
    ctx = list(prefix)
    for _ in range(N):
        p_dist = p_model(ctx)
        d = sample(p_dist, rng)
        drafts.append(d)
        p_probs.append(p_dist[d])
        ctx.append(d)

    q_dists = [q_model(prefix + drafts[:i]) for i in range(N + 1)]

    for i, d in enumerate(drafts):
        u = rng.random()
        q_prob = q_dists[i][d]
        p_prob = p_probs[i]
        if u < min(1.0, q_prob / p_prob if p_prob > 0 else float("inf")):
            prefix = prefix + [d]
        else:
            res = residual_dist(q_dists[i], p_model(prefix))
            prefix = prefix + [sample(res, rng)]
            return prefix
    prefix = prefix + [sample(q_dists[N], rng)]
    return prefix
```

五个草稿通过 → 额外一个奖励标记 → 一次验证器前向得到六个标记。

### 步骤 4：测量接受率

在不同草稿质量下运行 10,000 次推测步骤。绘制接受率与草稿和验证分布间 KL 散度的关系。应见到单调清晰的关系。

### 步骤 5：验证分布等价性

经验上，推测循环生成的标记直方图应与直接从验证器采样的直方图匹配。这即是 Leviathan 定理的实践验证。卡方检验表明误差在采样误差范围内。

## 使用它

生产环境：

```bash
# vLLM 使用 EAGLE
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --speculative-model /models/llama-3.1-eagle-70b \
    --speculative-draft-tensor-parallel-size 1 \
    --num-speculative-tokens 5

# vLLM 使用原味草稿模型
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --speculative-model meta-llama/Llama-3.2-1B-Instruct \
    --num-speculative-tokens 5
```

TensorRT-LLM 在 2026 年中期拥有最快的 Medusa 路径。`faster-whisper` 将推测解码封装到 Whisper-large 任务的小型草稿中。

**选择草稿模型：**

| 策略                 | 何时选择                 | 加速比    |
|----------------------|--------------------------|-----------|
| 原味草稿（1B/3B Llama 系列） | 快速原型，无需训练          | 1.8–2.3×  |
| Medusa 头             | 可以微调验证器             | 2–3×      |
| EAGLE-2 / 3          | 生产环境，最大速度          | 3–4×      |
| 向前看（Lookahead）   | 无草稿，无训练，无额外参数 | 1.3–1.6×  |

**何时不使用推测解码：**

- 单序列生成 1–5 个标记，开销主导。
- 高自由度创造性/高温度采样（α 降低）。
- 内存紧张的部署（草稿模型增加显存需求）。

## 部署它

参见 `outputs/skill-spec-decode-picker.md`。该技能为新推理工作负载自动选择推测解码策略（原味 / Medusa / EAGLE / 向前看）及调优参数（N、草稿温度）。

## 练习

1. **简单。** 运行 `code/main.py`。确认 50,000 标记的推测标记分布与验证器直接采样分布在卡方检验 p > 0.05 范围内匹配。
2. **中等。** 绘制不同 `N` 下 `α = 0.5, 0.7, 0.85` 时的加速比（大模型前向数/标记数）。确定每个 α 的最优 `N`。提示：期望每次验证调用采样标记数 = `(1 - α^{N+1}) / (1 - α)`。
3. **困难。** 实现一个极简 Medusa：从课程 14 的顶点 GPT 修改，增加 3 个预测 t+2、t+3、t+4 的 LM 头。用 tinyshakespeare 训练联合多头损失。比较接受率与截断同模型的原味草稿。
4. **困难。** 实现回滚机制：从 10 标记前缀 KV 缓存开始，输入 5 个草稿标记，模拟第 3 个标记拒绝。验证下次迭代缓存读取正确匹配“前缀 + 前 2 个接受草稿”。

## 关键术语

| 术语           | 人们说                       | 实际含义                                                         |
|----------------|------------------------------|------------------------------------------------------------------|
| 草稿模型       | “便宜那个”                   | 一个较小模型，提出候选标记，通常比验证器便宜 10–50 倍。           |
| 验证器         | “大模型”                    | 目标模型，要保持分布一致性；每次推测步骤调用一次。                 |
| 接受率（α）    | “草稿正确的概率”             | 验证器接受草稿标记的概率，通常 0.7–0.9。                         |
| 残差分布       | “拒绝时备用分布”             | `(q - p)_+` 归一化；拒绝时从该分布采样，保持验证器分布。           |
| 奖励标记       | “免费的那个”                  | 全部 N 个草稿都接受时，从验证器下一步分布额外采样一个标记。         |
| Medusa         | “无草稿推测”                 | 验证器上多个 LM 头并行预测位置 t+1..t+k。                         |
| EAGLE          | “隐藏状态草稿”               | 小型 Transformer 草稿模型，条件为验证器最后一层隐藏状态。         |
| 向前看解码     | “Jacobi 迭代”                | 利用不动点迭代自我推测；无草稿模型。                             |
| 树状注意力     | “同时验证多个候选”           | 使用分支验证同时考虑多个草稿续写。                               |
| KV 回滚        | “撤销被拒绝的草稿”           | 临时 KV 缓冲区；接受时提交，拒绝时丢弃。                         |

## 延伸阅读

- [Leviathan, Kalman, Matias (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — 核心算法和等价定理。
- [Chen et al. (2023). Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318) — 同期介绍；清晰的伯努利拒绝证明。
- [Cai et al. (2024). Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774) — Medusa 论文；树形注意力验证。
- [Li et al. (2024). EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077) — EAGLE-1；隐藏状态条件草稿。
- [Li et al. (2024). EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees](https://arxiv.org/abs/2406.16858) — EAGLE-2；动态树深度。
- [Li et al. (2025). EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test](https://arxiv.org/abs/2503.01840) — EAGLE-3。
- [Fu et al. (2024). Break the Sequential Dependency of LLM Inference Using Lookahead Decoding](https://arxiv.org/abs/2402.02057) — 前瞻解码，无草稿方法。
- [vLLM 文档 — Speculative Decoding（推测解码）](https://docs.vllm.ai/en/latest/features/spec_decode.html) — 带有所有四种策略的规范生产参考。
- [SafeAILab / EAGLE 参考实现](https://github.com/SafeAILab/EAGLE) — EAGLE-1/2/3 的参考代码。
