# GPT — 因果语言模型（Causal Language Modeling）

> BERT 看到的是双向。GPT 只看到过去。三角形掩码是现代 AI 中最关键的一行代码。

**类型：** 构建  
**语言：** Python  
**先决条件：** 阶段 7 · 02（自注意力 Self-Attention）、阶段 7 · 05（完整 Transformer）、阶段 7 · 06（BERT）  
**时间：** ~75 分钟

## 问题

语言模型回答一个问题：给定前 `t-1` 个 token，第 `t` 个 token 的概率分布是什么？基于这个信号训练——下一个 token 预测（next-token prediction）——你就能得到一个一次生成一个 token 的任意文本的模型。

要端到端并行训练整个序列，需要每个位置的预测只依赖于之前的位置。否则，模型会简单粗暴地作弊，直接看答案。

因果掩码（causal mask）实现了这一点。它是一个在 softmax 之前加到注意力得分上的单个上三角矩阵，元素值为 `-inf`。softmax 后这些位置的权重变为 0。每个位置只能关注自己和更早的位置。因为你对整个序列只应用一次掩码，所以一次前向传播可以得到 N 个并行的下一个 token 预测。

GPT-1（2018）、GPT-2（2019）、GPT-3（2020）、GPT-4（2023）、GPT-5（2024）、Claude、Llama、Qwen、Mistral、DeepSeek、Kimi——它们都是仅解码器的因果 Transformer，核心循环相同，只是规模更大、数据更好，强化学习（RLHF）更优。

## 概念

![Causal mask creates a triangular attention matrix](../assets/causal-attention.svg)

### 掩码

给定长度为 `N` 的序列，构建一个 `N × N` 矩阵：

```text
M[i, j] = 0       如果 j <= i
M[i, j] = -inf    如果 j > i
```

将 `M` 加到原始注意力得分上，随后进行 softmax。`exp(-inf) = 0`，所以被掩码的位置权重为零。注意力矩阵的每一行都是只针对之前位置的概率分布。

实现开销：一次 `torch.tril()` 调用。计算时间：纳秒级。对领域影响：巨大全面。

### 并行训练，串行推理

训练时：一次前向传播整个 `(N, d_model)` 序列，计算 N 个交叉熵损失（每个位置一个），相加，反向传播。序列上并行。这就是 GPT 训练能扩展的原因——可以在一个 GPU 批次处理 100 万 token。

推理时：逐 token 生成。输入 `[t1, t2, t3]` 得到 `t4`。输入 `[t1, t2, t3, t4]` 得到 `t5`。输入 `[t1, t2, t3, t4, t5]` 得到 `t6`。键值缓存（KV cache，见第 12 课）保存了 `t1…tn` 的隐藏状态，这样每步不必重新计算它们。但推理时的串行深度等于输出长度。这就是自回归（autoregressive）的成本，也是每个大型语言模型解码时延瓶颈的原因。

### 损失 — 向后错位一位

给定 token `[t1, t2, t3, t4]`：

- 输入：`[t1, t2, t3]`
- 目标：`[t2, t3, t4]`

对每个位置 `i`，计算 `-log P(target_i | inputs[:i+1])`。求和。这就是整个序列的交叉熵损失。

你听过的每个 Transformer 语言模型都用这个损失训练。预训练、微调（fine-tuning）、监督微调（SFT）——相同损失，不同数据。

### 解码策略

训练完成后，采样方案比人们想象的重要。

| 方法           | 功能                   | 适用场景                 |
|----------------|------------------------|--------------------------|
| 贪婪（Greedy） | 每步选最大概率         | 确定性任务、代码补全     |
| 温度（Temperature） | 对 logits 除以温度采样   | 创意任务，温度越高多样性越大 |
| Top-k          | 从 top-k token 采样    | 剔除低概率尾部           |
| Top-p（核采样） | 从累计概率≥p的最小集合采样 | 2020+ 默认；自适应分布形状 |
| Min-p          | 保留 `p > min_p * max_p` 的token | 2024+；比 top-p 更好剔除长尾 |
| 推测解码（Speculative decoding） | 草稿模型预测 N 个 token，大模型验证 | 同质质量下降低 2-3 倍延迟 |

2026 年，min-p + 温度0.7 是对公开权重模型的合理默认组合。推测解码是任何生产推理栈的必备技术。

### “GPT 配方”成功的原因

1. **仅解码器。** 无编码器开销。每层仅一轮注意力 + 前馈网络（FFN）。
2. **扩展性。** 从 1.24 亿→15 亿→1750 亿→万亿规模。Chinchilla 规模定律（第 13 课）告诉你如何合理分配计算资源。
3. **上下文学习。** 6B–13B 规模左右出现。模型能无微调跟随示例做少样本学习。
4. **人类反馈强化学习（RLHF）。** 训练后基于人类偏好，转变预训练文本为聊天助手。
5. **前归一化 + RoPE + SwiGLU。** 实现大规模训练的稳定性。

核心架构自 GPT-2 以来变化不大。有趣的全部是数据、规模和后期训练。

## 构建

### 第一步：因果掩码

见 `code/main.py`。一行代码实现：

```python
def causal_mask(n):
    return [[0.0 if j <= i else float("-inf") for j in range(n)] for i in range(n)]
```

加到注意力得分前，softmax 前。这就是整个机制。

### 第二步：两层 GPT 风格模型

堆叠两个解码器模块（掩码自注意力 + FFN，无交叉注意力）。加上 token embedding、位置编码、和解嵌入（与 token embedding 矩阵权重共享——GPT-2 起常用技巧）。

### 第三步：端到端下一个 token 预测

在一个 20 token 小型词表上，每个位置产生 logits。针对错位一位的目标计算交叉熵损失。不求梯度——仅做前向传播检查。

### 第四步：采样

实现贪婪、温度、top-k、top-p、min-p。分别在固定提示上测试输出。采样函数约十行代码。

## 使用

PyTorch，2026 年习惯写法：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

prompt = "Attention is all you need because"
inputs = tok(prompt, return_tensors="pt")
out = model.generate(
    **inputs,
    max_new_tokens=64,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
)
print(tok.decode(out[0]))
```

底层，`generate()` 运行前向传播，提取最终位置 logits，采样下一个 token，追加到输入，如此循环。每个生产级 LLM 推理栈（vLLM、TensorRT-LLM、llama.cpp、Ollama、MLX）都实现了同样逻辑，但有大量优化——批量预填充、连续批处理、KV 缓存分页、推测解码。

**一句话 GPT vs BERT：** GPT 预测 `P(x_t | x_{<t})`。BERT 预测 `P(x_masked | x_unmasked)`。损失决定了模型能否生成。

## 部署

见 `outputs/skill-sampling-tuner.md`。该能力模块为新的生成任务选择采样参数，并标记何时需要确定性解码。

## 练习

1. **简单。** 运行 `code/main.py`，验证 softmax 后的因果注意力矩阵是下三角矩阵。抽查第 3 行应只在第 0–3 列有权重。
2. **中等。** 实现宽度为4的 beam search。比较 beam-4 与贪婪在 10 个短提示上的困惑度（perplexity）。beam 是否总赢？（提示：通常是翻译任务中赢，开放式聊天任务则不一定。）
3. **困难。** 实现推测解码：用一个微小的 2 层模型作草稿模型，6 层模型作验证模型。测量 100 次长度 64 的生成时钟时间加速。确认结果与验证模型的贪婪采样匹配。

## 关键词

| 术语               | 常说是什么                  | 实际含义                                               |
|--------------------|----------------------------|------------------------------------------------------|
| 因果掩码（Causal mask）  | “三角形”                    | 加到注意力得分上的上三角 `-inf` 矩阵，位置 `i` 只看 `≤ i` 之前的 |
| 下一个 token 预测（Next-token prediction） | “损失”                      | 每位置模型分布与真实下一个 token 的交叉熵                     |
| 自回归（Autoregressive） | “一次生成一个”               | 输出反馈为输入；训练时并行，生成时串行                           |
| logits             | “softmax 前的得分”            | 语言模型头的原始输出，采样基于此                                |
| 温度（Temperature） | “创造力调节”                 | logits 除以温度 T；T→0 贪婪，T→∞ 均匀采样                      |
| Top-p              | “核采样”                    | 截断分布到累积概率 ≥ p 的最小集合，从中采样                     |
| Min-p              | “比 top-p 更好”              | 保留概率满足 `p ≥ min_p × max_p` 的 token，自适应分布陡峭度        |
| 推测解码（Speculative decoding） | “草稿加验证”                 | 轻量模型提议 N 个 token，大模型并行验证                            |
| 教师强制（Teacher forcing） | “训练技巧”                  | 训练时输入真实前一 token 而非模型预测，所有 seq2seq LM 标准做法       |

## 延伸阅读

- [Radford et al. (2018). 通过生成预训练提升语言理解](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) — GPT-1。  
- [Radford et al. (2019). 语言模型是无监督多任务学习者](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — GPT-2。  
- [Brown et al. (2020). 语言模型是少样本学习者](https://arxiv.org/abs/2005.14165) — GPT-3 及上下文学习。  
- [Leviathan, Kalman, Matias (2023). 基于推测解码的快速 Transformer 推理](https://arxiv.org/abs/2211.17192) — 推测解码论文。  
- [HuggingFace `modeling_llama.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) — 标准因果语言模型参考代码。
