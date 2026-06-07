# Transformer（Transformer 架构）之前的文本生成 — N-gram 语言模型（语言模型）

> 如果一个词令人惊讶，说明模型不好。困惑度（perplexity）将惊讶程度量化成数字。平滑方法（smoothing）确保概率不为零。

**类型：** 构建  
**语言：** Python  
**先决条件：** 阶段 5 · 01（文本处理），阶段 2 · 14（朴素贝叶斯）  
**时长：** 约 45 分钟

## 问题描述

在 Transformer（Transformer 架构）、RNN 和词嵌入出现之前，语言模型通过计算一个词在前面 `n-1` 个词之后出现的频率来预测下一个词。比如统计“the cat” → “sat”出现了 47 次，“the cat” → “jumped”出现了 12 次，“the cat” → “refrigerator”出现了 0 次。然后归一化得到一个概率分布。

这就是 n-gram 语言模型。它驱动了从 1980 年到 2015 年的每个语音识别器、拼写检查器和基于短语的机器翻译系统。当你需要廉价的本地设备语言模型时，它仍然在运行。

核心问题是如何处理未见过的 n-gram。基于原始计数的模型会给未见过的 n-gram 分配零概率，这是灾难性的，因为句子很长，几乎每个长句都包含至少一个未见的序列。五十年的平滑研究解决了这个问题。Kneser-Ney 平滑是其成果，现代深度学习继承了这种经验传统。

## 概念

![N-gram 模型：计数、平滑、生成](../assets/ngram.svg)

**N-gram 概率：** `P(w_i | w_{i-n+1}, ..., w_{i-1})`。固定 `n`（通常三元组为 3，四元组为 4）。由计数计算：

```text
P(w | context) = count(context, w) / count(context)
```

**零计数问题。** 训练中未出现的 n-gram 概率为零。2007 年对布朗语料库的研究发现，即便是 4-gram 模型，也有 30% 的保留集 4-gram 未在训练中出现。没有平滑，无法对真实文本做评估。

**平滑方法，按复杂度排序：**

1. **Laplace（加一平滑）。** 每个计数加 1。简单但对稀有事件表现很差。  
2. **Good-Turing。** 根据频率的频率重新分配概率质量，从高频事件转移到未见事件。  
3. **插值（Interpolation）。** 结合 n-gram、(n-1)-gram 等估计值，赋予可调权重。  
4. **回退（Backoff）。** 如果 n-gram 计数为零，则退回到 (n-1)-gram。Katz 回退方法规范化这一过程。  
5. **绝对折扣（Absolute discounting）。** 从所有计数中减去固定折扣 `D`，将减去的概率分配给未见事件。  
6. **Kneser-Ney。** 绝对折扣加上对低阶模型的巧妙选择：使用*延续概率*（词出现的上下文数量）代替原始频率。

Kneser-Ney 的洞察深刻。“San Francisco”是常见二元组。单词“Francisco”几乎只出现在“San”之后。简单的绝对折扣会给“Francisco”很高的单词概率（因为计数高）。而 Kneser-Ney 注意到“Francisco”只出现在唯一的上下文中，相应降低其延续概率。结果：一个新的以“Francisco”结尾的二元组会分配到合适的较低概率。

**评估指标：困惑度（perplexity）。** 在保留测试集上，每词负对数似然的指数。值越低越好。困惑度为 100 意味着模型挑词时的困惑程度相当于从 100 个词中均匀选一个。

```text
perplexity = exp(- (1/N) * Σ log P(w_i | context_i))
```

## 实现步骤

### 第 1 步：三元组计数

```python
from collections import Counter, defaultdict


def train_ngram(corpus_tokens, n=3):
    ngrams = Counter()
    contexts = Counter()
    for sentence in corpus_tokens:
        padded = ["<s>"] * (n - 1) + sentence + ["</s>"]
        for i in range(len(padded) - n + 1):
            ctx = tuple(padded[i:i + n - 1])
            word = padded[i + n - 1]
            ngrams[ctx + (word,)] += 1
            contexts[ctx] += 1
    return ngrams, contexts


def raw_probability(ngrams, contexts, context, word):
    ctx = tuple(context)
    if contexts.get(ctx, 0) == 0:
        return 0.0
    return ngrams.get(ctx + (word,), 0) / contexts[ctx]
```

输入为分词后的句子列表。输出为 n-gram 计数和上下文计数。`<s>` 和 `</s>` 表示句子边界。

### 第 2 步：Laplace 平滑

```python
def laplace_probability(ngrams, contexts, vocab_size, context, word):
    ctx = tuple(context)
    numerator = ngrams.get(ctx + (word,), 0) + 1
    denominator = contexts.get(ctx, 0) + vocab_size
    return numerator / denominator
```

每个计数加一。虽然平滑，但会过度分配概率给未见事件，也会影响罕见已知事件。

### 第 3 步：Kneser-Ney（bigram，插值）

```python
def kneser_ney_bigram_model(corpus_tokens, discount=0.75):
    unigrams = Counter()
    bigrams = Counter()
    unigram_contexts = defaultdict(set)

    for sentence in corpus_tokens:
        padded = ["<s>"] + sentence + ["</s>"]
        for i, w in enumerate(padded):
            unigrams[w] += 1
            if i > 0:
                prev = padded[i - 1]
                bigrams[(prev, w)] += 1
                unigram_contexts[w].add(prev)

    total_unique_bigrams = sum(len(ctx_set) for ctx_set in unigram_contexts.values())
    continuation_prob = {
        w: len(ctx_set) / total_unique_bigrams for w, ctx_set in unigram_contexts.items()
    }

    context_totals = Counter()
    for (prev, w), count in bigrams.items():
        context_totals[prev] += count

    unique_follow = defaultdict(set)
    for (prev, w) in bigrams:
        unique_follow[prev].add(w)

    def prob(prev, w):
        count = bigrams.get((prev, w), 0)
        denom = context_totals.get(prev, 0)
        if denom == 0:
            return continuation_prob.get(w, 1e-9)
        first_term = max(count - discount, 0) / denom
        lambda_prev = discount * len(unique_follow[prev]) / denom
        return first_term + lambda_prev * continuation_prob.get(w, 1e-9)

    return prob
```

三部分核心。`continuation_prob` 捕捉“这个词出现了多少不同上下文？”（Kneser-Ney 的创新）。`lambda_prev` 是折扣释放的概率质量，用于加权回退概率。最终概率是折扣后的主要概率加上加权延续概率。

### 第 4 步：按概率采样生成文本

```python
import random


def generate(prob_fn, vocab, prefix, max_len=30, seed=0):
    rng = random.Random(seed)
    tokens = list(prefix)
    for _ in range(max_len):
        candidates = [(w, prob_fn(tokens[-1], w)) for w in vocab]
        total = sum(p for _, p in candidates)
        r = rng.random() * total
        acc = 0.0
        for w, p in candidates:
            acc += p
            if r <= acc:
                tokens.append(w)
                break
        if tokens[-1] == "</s>":
            break
    return tokens
```

按概率采样。每个种子都产生不同结果。若想要类似束搜索的输出，则每步选最大概率词（贪婪），加一点随机性调整（temperature）。

### 第 5 步：困惑度计算

```python
import math


def perplexity(prob_fn, sentences):
    total_log_prob = 0.0
    total_tokens = 0
    for sentence in sentences:
        padded = ["<s>"] + sentence + ["</s>"]
        for i in range(1, len(padded)):
            p = prob_fn(padded[i - 1], padded[i])
            total_log_prob += math.log(max(p, 1e-12))
            total_tokens += 1
    return math.exp(-total_log_prob / total_tokens)
```

值越低越好。布朗语料库上，经过调优的 4-gram KN 模型困惑度约为 140。Transformer LM 在相同测试集上为 15-30，差距约 10 倍。这就是领域转向的原因。

## 应用场景

- **经典 NLP 教学。** 最透彻的平滑、极大似然估计（MLE）和困惑度讲解。  
- **KenLM。** 生产级 n-gram 库。在对低延迟敏感的语音和机器翻译系统中用作重评分器。  
- **设备端自动补全。** 键盘中的三元组模型仍在使用。  
- **基线模型。** 任何神经语言模型的性能声明之前，应该计算 n-gram 语言模型困惑度。如果你的 Transformer 没有大幅优于 KN，必定有问题。

## 发布模板

保存为 `outputs/prompt-lm-baseline.md`:

```markdown
---
name: lm-baseline
description: Build a reproducible n-gram language model baseline before training a neural LM.
phase: 5
lesson: 16
---

Given a corpus and target use (next-word prediction, rescoring, perplexity baseline), output:

1. N-gram order. Trigram for general English, 4-gram if corpus is large, 5-gram for speech rescoring.
2. Smoothing. Modified Kneser-Ney is the default; Laplace only for teaching.
3. Library. `kenlm` for production, `nltk.lm` for teaching, roll your own only to learn.
4. Evaluation. Held-out perplexity with consistent tokenization between train and test sets.

Refuse to report perplexity computed with different tokenization between systems being compared — perplexity numbers are comparable only under identical tokenization. Flag OOV rate in test set; KN handles OOV poorly unless you reserve a special <UNK> token during training.
```

## 练习

1. **简单。** 在包含 1000 句莎士比亚语料库上训练三元组语言模型。生成 20 句文本。它们局部合理但整体无序。这是经典演示。  
2. **中等。** 在保留的莎士比亚语料分割上实现 Kneser-Ney 模型的困惑度计算。与 Laplace 结果对比，应该看到 KN 降低困惑度约 30-50%。  
3. **困难。** 构建三元组拼写纠正器：给定一个拼写错误词及其上下文，生成候选纠正词，并根据语言模型上下文概率排序。使用 Birkbeck 拼写语料库（公开）评估。

## 关键词

| 术语 | 常用称呼 | 实际含义 |
|------|-----------------|-----------------------|
| N-gram | 词序列 | 连续 `n` 个词的序列。 |
| Smoothing | 避免零概率 | 重新分配概率质量，确保未见事件有非零概率。 |
| Perplexity | 语言模型质量指标 | 保留集上的 `exp(-平均 log-概率)`，越低越好。 |
| Backoff | 退回到更短上下文 | 三元组计数为零时使用二元组。Katz 回退给出形式化定义。 |
| Kneser-Ney | 最佳 n-gram 平滑 | 绝对折扣加低阶模型的延续概率。 |
| Continuation probability | KN 特有 | 根据词出现的上下文数量加权的概率，不是原始计数。 |

## 拓展阅读

- [Jurafsky 和 Martin — 《Speech and Language Processing》, 第 3 章（2026 草稿）](https://web.stanford.edu/~jurafsky/slp3/3.pdf) — 经典的 n-gram 语言模型和平滑处理章节。  
- [Chen 和 Goodman (1998). 语言建模中的平滑技术实证研究](https://dash.harvard.edu/handle/1/25104739) — 确立了 Kneser-Ney 为最佳 n-gram 平滑方法的论文。  
- [Kneser 和 Ney (1995). M-gram 语言建模的改进回退方法](https://ieeexplore.ieee.org/document/479394) — Kneser-Ney 原始论文。  
- [KenLM](https://kheafield.com/code/kenlm/) — 高速生产级 n-gram 语言模型，至 2026 年仍用于延迟敏感应用。
