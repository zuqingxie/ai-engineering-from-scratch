# GloVe、FastText 和子词嵌入

> Word2Vec 为每个单词训练一个嵌入。GloVe 对共现矩阵进行分解。FastText 嵌入词素。BPE 则连接到了 Transformer（Transformer 架构）。

**类型：** 构建  
**语言：** Python  
**先决条件：** 阶段 5 · 03（Word2Vec 从零开始）  
**时间：** 约 45 分钟

## 问题

Word2Vec 留下了两个未解之谜。

首先，有一条并行的研究路线是直接对共现矩阵进行分解（LSA，HAL），而不是做在线的 skip-gram 更新。Word2Vec 的迭代方法是否根本更优，还是两种方法处理计数的方式不同导致的差异？**GloVe** 给出了答案：使用精心设计的损失函数进行矩阵分解能够匹配或优于 Word2Vec，且训练成本更低。

其次，两种方法都没有处理未见过词（Out-Of-Vocabulary，OOV）的问题。像 `Zoomer-approved`、`dogecoin`、上周新创词的专有名词，以及罕见词根的所有屈折形式都不在词汇表内。**FastText** 通过嵌入字符 n 元组来解决这个问题：单词是其组成部分（包括词素）的和，因此即使是 OOV 词也能得到合理的向量表示。

第三，随着 Transformer（Transformer 架构）的出现，问题又发生了变化。基于词的词汇表最多约百万级；而真实语言远远超过这个规模。**字节对编码（Byte-pair encoding，BPE）** 及其变种通过学习频繁子词单位的词汇表解决了这个问题，覆盖了所有内容。每个现代大语言模型（LLM）都是用子词分词器进行分词的。

本课时将介绍这三者，并说明何时选用。

## 概念

**GloVe（全局向量）。** 构建词与词的共现矩阵 `X`，其中 `X[i][j]` 表示词 `j` 在词 `i` 的上下文中出现的次数。训练向量使得 `v_i · v_j + b_i + b_j ≈ log(X[i][j])`。对损失加权以避免高频对占据主导。至此完成。

**FastText。** 一个单词等于其字符 n-gram 和单词自身向量的总和。`where` 会变成 `<wh, whe, her, ere, re>, <where>`。单词向量是这些组成向量的和。训练方式与 Word2Vec 相同。优点是未知词（如 `whereupon`）可以通过已知 n-gram 组合得到向量。

**BPE（字节对编码）。** 从单个字节（或字符）组成的词汇表开始。计算语料中每对相邻符号的频率。合并最频繁的相邻对成为新标记。重复 `k` 次迭代。结果是一个包含 `k + 256` 个标记的词汇表，其中频繁的序列（`ing`、`tion`、`the`）成为单一标记，罕见词则被拆分为常见片段。每句文本都能被分词。

## 构建

### GloVe：分解共现矩阵

```python
import numpy as np
from collections import Counter


def build_cooccurrence(docs, window=5):
    pair_counts = Counter()
    vocab = {}
    for doc in docs:
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)
    for doc in docs:
        indexed = [vocab[t] for t in doc]
        for i, center in enumerate(indexed):
            for j in range(max(0, i - window), min(len(indexed), i + window + 1)):
                if i != j:
                    distance = abs(i - j)
                    pair_counts[(center, indexed[j])] += 1.0 / distance
    return vocab, pair_counts


def glove_train(vocab, pair_counts, dim=16, epochs=100, lr=0.05, x_max=100, alpha=0.75, seed=0):
    n = len(vocab)
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, size=(n, dim))
    W_tilde = rng.normal(0, 0.1, size=(n, dim))
    b = np.zeros(n)
    b_tilde = np.zeros(n)

    for epoch in range(epochs):
        for (i, j), x_ij in pair_counts.items():
            weight = (x_ij / x_max) ** alpha if x_ij < x_max else 1.0
            diff = W[i] @ W_tilde[j] + b[i] + b_tilde[j] - np.log(x_ij)
            coef = weight * diff

            grad_W_i = coef * W_tilde[j]
            grad_W_tilde_j = coef * W[i]
            W[i] -= lr * grad_W_i
            W_tilde[j] -= lr * grad_W_tilde_j
            b[i] -= lr * coef
            b_tilde[j] -= lr * coef

    return W + W_tilde
```

有两个关键点值得说明。加权函数 `f(x) = (x/x_max)^alpha` 对频率极高的词对（如 `(the, and)`）降低权重，避免其支配损失。最终嵌入是 `W`（中心词向量）和 `W_tilde`（上下文词向量）表的和。将两者相加是公开发表的技巧，通常优于只用其中之一。

### FastText：子词感知嵌入

```python
def char_ngrams(word, n_min=3, n_max=6):
    wrapped = f"<{word}>"
    grams = {wrapped}
    for n in range(n_min, n_max + 1):
        for i in range(len(wrapped) - n + 1):
            grams.add(wrapped[i:i + n])
    return grams
```

```python
>>> char_ngrams("where")
{'<where>', '<wh', 'whe', 'her', 'ere', 're>', '<whe', 'wher', 'here', 'ere>', '<wher', 'where', 'here>'}
```

每个单词用其 n-gram 集合（通常为 3 到 6 个字符）表示。词向量是其所有 n-gram 向量的和。在 skip-gram 训练中，将此方法嵌入 Word2Vec 使用单向量的地方。

```python
def fasttext_vector(word, ngram_table):
    grams = char_ngrams(word)
    vecs = [ngram_table[g] for g in grams if g in ngram_table]
    if not vecs:
        return None
    return np.sum(vecs, axis=0)
```

对未见过的词，只要其部分 n-gram 已知，也能得到向量。比如 `whereupon` 与 `where` 共享 `<wh`, `her`, `ere` 和 `<where`，因此两者在向量空间中相近。

### BPE：学习到的子词词汇

```python
def learn_bpe(corpus, k_merges):
    vocab = Counter()
    for word, freq in corpus.items():
        tokens = tuple(word) + ("</w>",)
        vocab[tokens] = freq

    merges = []
    for _ in range(k_merges):
        pair_freq = Counter()
        for tokens, freq in vocab.items():
            for a, b in zip(tokens, tokens[1:]):
                pair_freq[(a, b)] += freq
        if not pair_freq:
            break
        best = pair_freq.most_common(1)[0][0]
        merges.append(best)

        new_vocab = Counter()
        for tokens, freq in vocab.items():
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == best:
                    new_tokens.append(tokens[i] + tokens[i + 1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            new_vocab[tuple(new_tokens)] = freq
        vocab = new_vocab
    return merges


def apply_bpe(word, merges):
    tokens = list(word) + ["</w>"]
    for a, b in merges:
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i + 1 < len(tokens) and tokens[i] == a and tokens[i + 1] == b:
                new_tokens.append(a + b)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens
    return tokens
```

```python
>>> corpus = Counter({"low": 5, "lower": 2, "newest": 6, "widest": 3})
>>> merges = learn_bpe(corpus, k_merges=10)
>>> apply_bpe("lowest", merges)
['low', 'est</w>']
```

第一轮迭代合并出现频率最高的相邻对。经过足够多迭代，常见子串（`low`、`est`、`tion`）成为单个标记，罕见词则被干净拆分。

真实的 GPT / BERT / T5 分词器学习 3 万到 10 万轮合并。结果是任何文本都能分词成固定长度范围的已知 ID 序列，永远无 OOV。

## 使用

实际上，你很少自己训练这些。通常加载预训练检查点。

```python
import fasttext.util
fasttext.util.download_model("en", if_exists="ignore")
ft = fasttext.load_model("cc.en.300.bin")
print(ft.get_word_vector("whereupon").shape)
print(ft.get_word_vector("zoomerapproved").shape)
```

Transformer（Transformer 架构）时代的 BPE 样子：

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("gpt2")
print(tok.tokenize("unbelievably tokenized"))
```

```text
['un', 'bel', 'iev', 'ably', 'Ġtoken', 'ized']
```

`Ġ` 前缀标记单词边界（GPT-2 约定）。每个现代分词器都是 BPE 变种、WordPiece（BERT），或 SentencePiece（T5，LLaMA）。

### 何时选择何种方法

| 场景 | 选择 |
|-------|------|
| 预训练通用词向量，不需 OOV 容忍 | GloVe 300d |
| 预训练通用词向量，需处理错拼、新词、形态丰富语言 | FastText |
| 任何用于 Transformer（训练或推理）的模型 | 使用模型附带的分词器，绝不更换 |
| 自己从头训练语言模型 | 首先训练 BPE 或 SentencePiece 分词器 |
| 生产环境使用线性模型的文本分类 | 仍然使用 TF-IDF。课时 02。 |

## 发布

保存为 `outputs/skill-embeddings-picker.md`：

```markdown
---
name: tokenizer-picker
description: 为新语言模型或文本流水线选择分词方案。
version: 1.0.0
phase: 5
lesson: 04
tags: [nlp, tokenization, embeddings]
---

鉴于任务和数据集描述，请输出：

1. 分词策略（基于词、BPE、WordPiece、SentencePiece、字节级）。一句话说明理由。  
2. 词汇表大小目标（如只英文LM为 32k，多语为 64k-100k）。  
3. 具体训练命令和所用库名称。列出参数。  
4. 一个可复现性陷阱。分词器与模型不匹配是最常见且难察觉的生产缺陷，指出必须配对使用哪一组合。

若用户在微调预训练大语言模型时，拒绝推荐训练自定义分词器。针对任何生产推理模型，拒绝推荐基于词的分词。多语或非英文、多脚本语料必须使用带字节后备机制的 SentencePiece。
```

## 练习

1. **简单。** 运行 `char_ngrams("playing")` 和 `char_ngrams("played")`。计算两集合的 Jaccard 重叠率。你应该看到显著共享片段（`pla`，`lay`，`play`），这解释了 FastText 为何能较好地转移形态变体。
2. **中等。** 扩展 `learn_bpe`，记录词汇大小变化。绘制每个合并数对应的每语料字符标记数。你应看到初期快速压缩，最终趋近于每标记约 2-3 字符。
3. **困难。** 用莎士比亚全集训练一个 1k 轮合并 BPE。比较常用词与罕见专有名词的分词。衡量分词前后平均每词标记数。写下令你惊讶的发现。

## 关键词

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| 共现矩阵 | 词-词频率表 | `X[i][j]` 表示词 `j` 在词 `i` 的窗口内出现的频率。 |
| 子词 | 单词的部分 | 字符 n-gram（FastText）或学习得到的标记（BPE/WordPiece/SentencePiece）。 |
| BPE | 字节对编码 | 反复合并出现频率最高的相邻对，直到词汇大小达到目标。 |
| OOV | 词汇外词 | 模型从未见过的词。Word2Vec/GloVe 失败，FastText 和 BPE 能处理。 |
| 字节级 BPE | 在原始字节上做 BPE | GPT-2 的方案。词汇表起始是 256 字节，永不出现 OOV。 |

## 深入阅读

- [Pennington, Socher, Manning (2014). GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf) — GloVe 论文，七页，依然是损失函数推导的最佳来源。
- [Bojanowski et al. (2017). Enriching Word Vectors with Subword Information](https://arxiv.org/abs/1607.04606) — FastText。
- [Sennrich, Haddow, Birch (2016). Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) — 引入 BPE 到现代自然语言处理（NLP）的论文。
- [Hugging Face tokenizer summary](https://huggingface.co/docs/transformers/tokenizer_summary) — 介绍 BPE、WordPiece 和 SentencePiece 在实际应用中的不同。
