# 词向量 — 从零实现 Word2Vec

> 一个词的意义由它的共现环境决定。基于这个思想训练一个浅层网络，几何结构自然显现。

**类型：** 实践构建  
**语言：** Python  
**前置知识：** 第5阶段·02（BoW + TF-IDF），第3阶段·03（从零实现反向传播）  
**时长：** 约75分钟

## 问题

TF-IDF 知道 `dog` 和 `puppy` 是不同的词，但不知道它们几乎是同义词。训练得到的分类器只识别 `dog`，不能泛化到关于 `puppy` 的评论上。你可以通过列同义词缓解这个问题，但对于罕见词、行业术语和未预料到的语言这方法行不通。

你需要一种表示，使得 `dog` 和 `puppy` 在空间上靠得很近。使 `king - man + woman` 的结果接近 `queen`。使得训练在 `dog` 上的模型能免费向 `puppy` 传递信号。

Word2Vec 给了我们这个空间。两层神经网络，万亿级别的训练数据，2013年发表。结构几乎简单到令人羞愧。结果重塑了十年的自然语言处理（NLP）。

## 概念

**分布假说（distributional hypothesis）**（Firth, 1957）：“你可以通过一个词的共现环境来了解这个词。”两词若出现在类似环境，极可能意义相近。

Word2Vec 有两种实现，均利用该思想。

- **Skip-gram Skip-gram方法。** 给定中心词，预测上下文词。窗口大小2时如 `cat -> (the, sat, on)`。
- **CBOW（连续词袋模型Continuous Bag of Words）方法。** 给定上下文词，预测中心词。比如 `(the, sat, on) -> cat`。

Skip-gram 训练较慢，但对罕见词更友好，成为默认选择。

网络有一层无非线性的隐藏层。输入是词汇表的 one-hot 向量。输出是词汇表上的 softmax。训练完丢弃输出层，隐藏层权重即为词向量。

```text
one-hot(center) ── W ──▶ hidden (d-维度) ── W' ──▶ softmax(词汇表)
                          ^
                          这就是词向量
```

难点是：对超大词汇表做 softmax 极其昂贵。Word2Vec 用**负采样（negative sampling）**将其转化为二分类任务。判断“这个上下文词是否出现在中心词附近， 是或否”。每个正样本只采样几个负样本，替代全词汇表计算 softmax。

## 实现它

### 第1步：从语料生成训练对

```python
def skipgram_pairs(docs, window=2):
    pairs = []
    for doc in docs:
        for i, center in enumerate(doc):
            for j in range(max(0, i - window), min(len(doc), i + window + 1)):
                if i == j:
                    continue
                pairs.append((center, doc[j]))
    return pairs
```

```python
>>> skipgram_pairs([["the", "cat", "sat", "on", "mat"]], window=2)
[('the', 'cat'), ('the', 'sat'),
 ('cat', 'the'), ('cat', 'sat'), ('cat', 'on'),
 ('sat', 'the'), ('sat', 'cat'), ('sat', 'on'), ('sat', 'mat'),
 ...]
```

每个窗口内的（中心词，上下文词）都是正样本。

### 第2步：Embedding 表

两张矩阵：`W` 是中心词向量表（最终保留的），`W'` 是上下文词向量表（经常丢弃，有时会和 `W` 取平均）。

```python
import numpy as np


def init_embeddings(vocab_size, dim, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, size=(vocab_size, dim))
    W_prime = rng.normal(0, 0.1, size=(vocab_size, dim))
    return W, W_prime
```

用小方差高斯初始化。词汇规模1万，维度100是现实场景；教学时50词汇×16维即可看出几何效果。

### 第3步：负采样目标函数

给每个正样本 `(center, context)` 采样 `k` 个随机负例，训练模型使得 `W[center] · W'[context]` 对正例接近1，对负例接近0。

```python
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def train_pair(W, W_prime, center_idx, context_idx, negative_indices, lr):
    v_c = W[center_idx]
    u_pos = W_prime[context_idx]
    u_negs = W_prime[negative_indices]

    pos_score = sigmoid(v_c @ u_pos)
    neg_scores = sigmoid(u_negs @ v_c)

    grad_center = (pos_score - 1) * u_pos
    for i, u in enumerate(u_negs):
        grad_center += neg_scores[i] * u

    W[context_idx] = W[context_idx]
    W_prime[context_idx] -= lr * (pos_score - 1) * v_c
    for i, neg_idx in enumerate(negative_indices):
        W_prime[neg_idx] -= lr * neg_scores[i] * v_c
    W[center_idx] -= lr * grad_center
```

核心公式是对正样本使用逻辑损失（sigmoid 期望接近1），对负样本使用逻辑损失（sigmoid 期望接近0）。梯度同时更新两张表。完整推导见原论文，建议动笔推导一遍印象深刻。

### 第4步：在玩具语料上训练

```python
def train(docs, dim=16, window=2, k_neg=5, epochs=100, lr=0.05, seed=0):
    vocab = build_vocab(docs)
    vocab_size = len(vocab)
    rng = np.random.default_rng(seed)
    W, W_prime = init_embeddings(vocab_size, dim, seed=seed)
    pairs = skipgram_pairs(docs, window=window)

    for epoch in range(epochs):
        rng.shuffle(pairs)
        for center, context in pairs:
            c_idx = vocab[center]
            ctx_idx = vocab[context]
            negs = rng.integers(0, vocab_size, size=k_neg)
            negs = [n for n in negs if n != ctx_idx and n != c_idx]
            train_pair(W, W_prime, c_idx, ctx_idx, negs, lr)
    return vocab, W
```

当训练轮数足够大且语料庞大时，共现环境相似的词中心向量会相似。在玩具语料上这种效果较弱；在十亿级的语料上效果显著。

### 第5步：类比技巧

```python
def nearest(vocab, W, target_vec, topk=5, exclude=None):
    exclude = exclude or set()
    inv_vocab = {i: w for w, i in vocab.items()}
    norms = np.linalg.norm(W, axis=1, keepdims=True) + 1e-9
    W_norm = W / norms
    target = target_vec / (np.linalg.norm(target_vec) + 1e-9)
    sims = W_norm @ target
    order = np.argsort(-sims)
    out = []
    for i in order:
        if i in exclude:
            continue
        out.append((inv_vocab[i], float(sims[i])))
        if len(out) == topk:
            break
    return out


def analogy(vocab, W, a, b, c, topk=5):
    v = W[vocab[b]] - W[vocab[a]] + W[vocab[c]]
    return nearest(vocab, W, v, topk=topk, exclude={vocab[a], vocab[b], vocab[c]})
```

在预训练的 300 维 Google News 向量上：

```python
>>> analogy(vocab, W, "man", "king", "woman")
[('queen', 0.71), ('monarch', 0.62), ('princess', 0.59), ...]
```

`king - man + woman = queen`。模型并不真正“知道”皇室意义，而是向量 `(king - man)` 捕捉了类似“皇权”的向量意义，加上 `woman` 后落入女性皇族的语义区域。

## 使用它

从零写 Word2Vec 可学到很多，生产环境中用 `gensim`。

```python
from gensim.models import Word2Vec

sentences = [
    ["the", "cat", "sat", "on", "the", "mat"],
    ["the", "dog", "ran", "across", "the", "room"],
]

model = Word2Vec(
    sentences,
    vector_size=100,
    window=5,
    min_count=1,
    sg=1,
    negative=5,
    workers=4,
    epochs=30,
)

print(model.wv["cat"])
print(model.wv.most_similar("cat", topn=3))
```

实际工作中几乎不亲自训练 Word2Vec，通常下载预训练向量。

- **GloVe** — 斯坦福的共现矩阵分解方法。50维，100维，200维，300维模型都有。覆盖面广。第04课专门讲GloVe。
- **fastText** — Facebook 的 Word2Vec 拓展，嵌入字符 n-gram，实现子词组合。可以处理未登录词。第04课。
- **预训练的 Google News Word2Vec** — 300维，包含300万词汇，2013年发布，至今依旧常用。

### Word2Vec 在2026年依然有优势的时候

- 轻量级领域专用检索。比如在笔记本上用医学摘要一小时即可训练出专门词向量，普通通用模型捕获不到的专业信息。
- 类比特征工程。计算 `gender_vector = mean(man - woman 对)`。从词向量中减去该向量即得到性别中性轴。仍在公平性研究中使用。
- 可解释性。100维较低，可以用 PCA 或 t-SNE 可视化，能看出聚类结构。
- 任何需要设备端无GPU推理的场景。Word2Vec 查询就是单行向量读取，非常快。

### Word2Vec 的不足

多义性问题。`bank` 只有一个向量，`river bank` 和 `financial bank` 混在一起。`table`（电子表格 vs 家具）也混用。下游分类器无法根据词向量区分词义。

上下文向量（ELMo，BERT，以及所有后来Transformer模型）解决了这个问题，针对同一词的不同出现有不同向量。Word2Vec 到 BERT 的飞跃即是：从静态向量（static embedding）到上下文向量（contextual embedding）。第7阶段会详细讲 Transformer。

未登录词问题是另一大不足。训练集中没有出现的词 Word2Vec 无法产生词向量。fastText 用子词组合解决了此问题（第04课）。

## 部署方案

保存为 `outputs/skill-embedding-probe.md`：

```markdown
---
name: embedding-probe
description: Inspect a word2vec model. Run analogies, find neighbors, diagnose quality.
version: 1.0.0
phase: 5
lesson: 03
tags: [nlp, embeddings, debugging]
---

你可以探测训练好的词向量，确认其有效性。给定一个 `gensim.models.KeyedVectors` 对象和词汇，运行：

1. 三个标准类比测试。`king : man :: queen : woman`，`paris : france :: tokyo : japan`，`walking : walked :: swimming : ?`。报告 top-1 结果及余弦相似度。
2. 五个用户提供的领域特定词的近邻测试。打印 top-5 邻居及相似度。
3. 一个对称性检查。确认 `similarity(a, b) == similarity(b, a)`，容许浮点误差。
4. 一个异常检查。若任意词向量范数 < 0.01 或 > 100，表示模型训练异常，请标记。

仅凭类比准确率不能断定模型优劣，类比测试容易作弊且和下游任务相关性差。建议同时做内在和下游任务评估。
```

## 练习

1. **简单。** 在一个小语料（20句关于猫和狗的句子）上训练模型。200轮后，验证 `nearest(vocab, W, W[vocab["cat"]])` 是否在前三名出现 `dog`。否则增加训练轮数或扩大词汇表。
2. **中等。** 实现高频词下采样。高于 `10^-5` 频率的词以比例概率从训练对中剔除。观察对罕见词相似度的影响。
3. **困难。** 在 20 Newsgroups 语料上训练模型。计算两条偏置轴：`he - she` 和 `doctor - nurse`。把职业词映射到这两轴，报告偏置最大差异的职业。这是公平性研究中的探针方法。

## 关键词语

| 术语 | 常用说法 | 实际含义 |
|------|-----------------|-----------------------|
| Word embedding（词向量） | 词作为向量 | 从上下文中学习到的稠密低维（通常100~300维）表示。 |
| Skip-gram | Word2Vec 的技巧 | 由中心词预测上下文词。较CBOW慢，但对罕见词更好。 |
| Negative sampling（负采样） | 训练捷径 | 用 k 个随机词替代全词表做 softmax，转二分类任务。 |
| Static embedding（静态词向量） | 词固定向量 | 一个词一个向量，不随上下文变化。不能区分多义。 |
| Contextual embedding（上下文词向量） | 上下文敏感向量 | 每次出现的词根据上下文产生不同向量。Transformer产物。 |
| OOV（未登录词） | 词表外词 | 训练时没见过的词，Word2Vec 无法给出向量。 |

## 进一步阅读

- [Mikolov et al. (2013). Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546) — 负采样（negative-sampling）论文。简短且易读。
- [Rong, X. (2014). word2vec Parameter Learning Explained](https://arxiv.org/abs/1411.2738) — 如果原始论文的数学推导感觉晦涩，这是梯度推导最清晰的解释。
- [gensim Word2Vec 教程](https://radimrehurek.com/gensim/models/word2vec.html) — 实际可用的生产环境训练设置。
