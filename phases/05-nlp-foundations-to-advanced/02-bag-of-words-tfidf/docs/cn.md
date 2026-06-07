# 词袋（Bag of Words）、TF-IDF 及文本表示

> 先计数，后思考。到2026年，TF-IDF 在定义明确的任务上仍然胜过嵌入（embedding）。

**类别：** 构建  
**语言：** Python  
**前置知识：** 阶段5 · 01（文本处理）、阶段2 · 02（从头实现线性回归）  
**时间：** 约75分钟

## 问题

模型需要数字，你有字符串。

每个自然语言处理（NLP）流水线都要回答同一个问题。如何将变长的标记（token）序列转换成一个分类器可用的固定大小向量。领域内的第一个答案是最简单有效的。数词。造向量。

这个向量承载了比任何嵌入模型更多的线上NLP应用。垃圾邮件过滤、主题分类、日志异常检测、搜索排序（BM25之前）、第一波情感分析、学术NLP基准的最初十年。2026年的实践者在狭义分类任务上仍首先选择它。它快速、可解释，在只关心词是否出现的任务上，常与4亿参数嵌入模型效果无异。

本课构建词袋模型和TF-IDF，从零开始实现。然后展示scikit-learn用三行代码完成相同功能。最后指出让你转向嵌入模型的失败模式。

## 概念

**词袋（Bag of Words, BoW）** 丢弃顺序。对每个文档，统计每个词汇的出现次数。向量长度为词汇表大小。位置 `i` 存储词 `i` 的计数。

**TF-IDF** 对词袋重新加权。一个词若出现在所有文档中无信息量，则降低它的权重。一个词在语料库中稀有但在某文档中频繁则是信号，提升其权重。

```text
TF-IDF(w, d) = TF(w, d) * IDF(w)
             = count(w in d) / |d| * log(N / df(w))
```

其中 `TF` 是文档中词频，`df` 是文档频率（含此词的文档数），`N` 是文档总数。`log` 保持常见词权重有限。

关键性质：均产生稀疏向量且轴可解释。你可以查看训练分类器的权重，读出哪些词推动文档向某类别倾斜。BERT的768维嵌入无法做到这一点。

## 逐步构建

### 第1步：构建词汇表

```python
def build_vocab(docs):
    vocab = {}
    for doc in docs:
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab
```

输入：分词后的文档列表（任意词级分词器均可；本课的 `code/main.py` 使用简化小写版本）。输出：`{word: index}` 字典。稳定插入顺序意味着词索引0是第一个文档中出现的第一个词。约定不同，scikit-learn按字母排序。

### 第2步：词袋表示

```python
def bag_of_words(docs, vocab):
    matrix = [[0] * len(vocab) for _ in docs]
    for i, doc in enumerate(docs):
        for token in doc:
            if token in vocab:
                matrix[i][vocab[token]] += 1
    return matrix
```

```python
>>> docs = [["cat", "sat", "on", "mat"], ["cat", "cat", "ran"]]
>>> vocab = build_vocab(docs)
>>> bag_of_words(docs, vocab)
[[1, 1, 1, 1, 0], [2, 0, 0, 0, 1]]
```

行表示文档，列表示词汇索引。元素 `[i][j]` 表示第 `i` 个文档中第 `j` 个词出现的次数。文档1有两个 `cat`，文档0没有 `ran`。

### 第3步：词频（term frequency）与文档频率（document frequency）

```python
import math


def term_frequency(doc_bow, doc_length):
    return [c / doc_length if doc_length else 0 for c in doc_bow]


def document_frequency(bow_matrix):
    df = [0] * len(bow_matrix[0])
    for row in bow_matrix:
        for j, count in enumerate(row):
            if count > 0:
                df[j] += 1
    return df


def inverse_document_frequency(df, n_docs):
    return [math.log((n_docs + 1) / (d + 1)) + 1 for d in df]
```

两个平滑技巧值得说明。`(n+1)/(d+1)` 防止出现 `log(x/0)`。尾部 `+1` 使即使词语出现在全部文档，IDF也为1（非0），匹配scikit-learn默认行为。其它实现用原始的 `log(N/df)`。两者均可，平滑版更友好。

### 第4步：计算TF-IDF

```python
def tfidf(bow_matrix):
    n_docs = len(bow_matrix)
    df = document_frequency(bow_matrix)
    idf = inverse_document_frequency(df, n_docs)
    out = []
    for row in bow_matrix:
        length = sum(row)
        tf = term_frequency(row, length)
        out.append([tf_j * idf_j for tf_j, idf_j in zip(tf, idf)])
    return out
```

```python
>>> docs = [
...     ["the", "cat", "sat"],
...     ["the", "dog", "sat"],
...     ["the", "cat", "ran"],
... ]
>>> vocab = build_vocab(docs)
>>> bow = bag_of_words(docs, vocab)
>>> tfidf(bow)
```

三个文档，五个词汇（`the`、`cat`、`sat`、`dog`、`ran`）。`the` 出现在所有三文档，IDF低；`dog`只出现在一篇，IDF高。向量稀疏（大多数值小），区分词突出。

### 第5步：L2归一化行向量

```python
def l2_normalize(matrix):
    out = []
    for row in matrix:
        norm = math.sqrt(sum(x * x for x in row))
        out.append([x / norm if norm else 0 for x in row])
    return out
```

不归一化时，长文档向量大，主导相似度分数。L2归一化使所有文档在单位超球面上，行间余弦相似度变点积。

## 使用

scikit-learn 提供了生产级实现。

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

docs = ["the cat sat on the mat", "the dog sat on the mat", "the cat ran"]

bow_vectorizer = CountVectorizer()
bow = bow_vectorizer.fit_transform(docs)
print(bow_vectorizer.get_feature_names_out())
print(bow.toarray())

tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(docs)
print(tfidf.toarray().round(3))
```

`CountVectorizer` 负责分词、词汇表、词袋一步到位。`TfidfVectorizer` 额外做IDF加权和L2归一化。两者均返回稀疏矩阵。100k文档以上，密集矩阵无法装入内存，保持稀疏直到分类器要求密集。

以下参数影响深远：

| 参数               | 作用                                                            |
|--------------------|-----------------------------------------------------------------|
| `ngram_range=(1, 2)` | 包含二元组（bigram）。通常提高分类效果。                         |
| `min_df=2`          | 丢弃少于2个文档中出现的词。用于噪声数据缩减词汇量。              |
| `max_df=0.95`       | 丢弃出现在95%以上文档中的词。近似去除停用词，无需硬编码列表。      |
| `stop_words="english"` | scikit-learn 内置停用词列表。依任务而定——情感分析**不要**删除否定词。 |
| `sublinear_tf=True` | 用 `1 + log(tf)` 替代原始词频。对单文档高频词更友好。               |

### TF-IDF 仍胜（截至2026年）

- 垃圾邮件检测、主题标注、日志异常检测。关注词出现与否，非语义细节。
- 小数据场景（数百标注样本）。TF-IDF配逻辑回归，无需预训练成本。
- 低延迟要求场景。TF-IDF加线性模型毫秒级响应，Transformer嵌入至少10-100毫秒。
- 需解释预测结果的系统。可查看分类器系数，正向词即解释。

### TF-IDF失败时

语义盲点。看这两句：

- “The movie was not good at all.”  
- “The movie was excellent.”

一条负评，一条正评。TF-IDF交集是 `{the, movie, was}`。词袋分类器必须记住 `not` 附近的 `good` 翻转标签。足够数据能学会，但不如理解句法的模型优雅。

另一失败：推断时遇到词汇外（out-of-vocabulary）词。训练IMDb影评的BoW模型不会处理 `Zoomer-approved` ，若培训时没见过此词。子词嵌入（第四课）能解决，TF-IDF不行。

### 混合：TF-IDF加权嵌入

2026中等规模分类实务默认：用TF-IDF权重作为对词嵌入的注意力权重。

```python
def tfidf_weighted_embedding(doc, tfidf_scores, embedding_table, dim):
    vec = [0.0] * dim
    total_weight = 0.0
    for token in doc:
        if token not in embedding_table or token not in tfidf_scores:
            continue
        weight = tfidf_scores[token]
        emb = embedding_table[token]
        for i in range(dim):
            vec[i] += weight * emb[i]
        total_weight += weight
    if total_weight == 0:
        return vec
    return [v / total_weight for v in vec]
```

从嵌入获得语义能力，从TF-IDF获得稀有词强调。分类器基于该池化向量训练。对50k标注以下的情感、主题和意图分类，优于单独TF-IDF或单独均值嵌入。

## 交付

保存为 `outputs/prompt-vectorization-picker.md`:

```markdown
---
name: vectorization-picker
description: 给定文本分类任务，推荐词袋（BoW）、TF-IDF、嵌入或混合方案。
phase: 5
lesson: 02
---

你给出文本向量化方案。针对任务描述，输出：

1. 表示法（BoW、TF-IDF、Transformer嵌入，或混合）。用一句话说明理由。
2. 具体向量器配置。说明库名。列参数（`ngram_range`、`min_df`、`max_df`、`sublinear_tf`、`stop_words`）。
3. 上线前需测试的失败模式。

用户标注样本不足500时，除非TF-IDF基线明显语义失败，拒绝推荐嵌入。对情感分析拒绝去除停用词（否定词含信号）。提示类别不平衡需超出向量器调整。

示例输入：  
“将3万条客服工单分12类。大部分工单2-3句英文。需审计日志可解释。”

示例输出：

- 表示法：TF-IDF。3万样本不算少；需解释性，排除稠密嵌入。
- 配置：`TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.95, sublinear_tf=True, stop_words=None)`。保留停用词，因为类别关键词有时是停用词（“not working” vs “working”）。
- 测试失败点：确认 `min_df=3` 未丢弃少见类别关键词。用分类过滤 `get_feature_names_out` 目视检查。
```

## 练习

1. **简单。** 基于L2归一化TF-IDF，实现 `cosine_similarity(doc_vec_a, doc_vec_b)`。验证完全相同文档得1.0分，词汇不重合文档得0.0分。
2. **中等。** 给 `bag_of_words` 添加`n-gram`支持。参数 `n` 控制统计 `n`-gram。测试输入 `["the", "cat", "sat"]` 且 `n=2` 应输出二元组 `["the cat", "cat sat"]` 的计数。
3. **困难。** 实现上述基于GloVe 100维向量的TF-IDF加权嵌入混合。下载一次缓存。用20 Newsgroups数据集比较该方法与纯TF-IDF、纯均值嵌入的分类准确率。报告不同场景哪种方法更优。

## 关键术语

| 术语 | 人们说的 | 实际意义 |
|------|-----------------|-----------------------|
| BoW | 词频向量（Word frequency vector） | 一个文档中词汇的计数。忽略词序。 |
| TF | 词频（Term frequency） | 一个文档中某词的计数，可选地按文档长度归一化。 |
| DF | 文档频率（Document frequency） | 包含该词至少一次的文档数。 |
| IDF | 逆文档频率（Inverse document frequency） | `log(N / df)` 平滑处理。减弱在所有文档中都出现的词权重。 |
| 稀疏向量（Sparse vector） | 主要是零 | 词汇量通常是 1 万至 10 万词；大多数词在任何给定文档中都不存在。 |
| 余弦相似度（Cosine similarity） | 向量夹角 | L2 归一化向量的点积。1 表示完全相同，0 表示正交。 |

## 延伸阅读

- [scikit-learn — 从文本提取特征](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) — 官方 API 参考以及所有参数说明。
- [Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval](https://www.sciencedirect.com/science/article/pii/0306457388900210) — 使 TF-IDF 成为十年主流方法的论文。
- ["Why TF-IDF Still Beats Embeddings" — Ashfaque Thonikkadavan (Medium)](https://medium.com/@cmtwskb/why-tf-idf-still-beats-embeddings-ad85c123e1b2) — 2026 年关于旧方法在何时为何胜出的分析。
