# 主题建模 — LDA 和 BERTopic

> LDA：文档是主题的混合体，主题是单词的分布。BERTopic：文档在嵌入空间中聚类，聚类即主题。目标相同，分解方式不同。

**类型：** 学习  
**语言：** Python  
**先决条件：** 第五阶段 · 02（BoW + TF-IDF），第五阶段 · 03（Word2Vec）  
**时长：** 约 45 分钟

## 问题描述

你有 10,000 个客户支持工单、50,000 篇新闻文章或 200,000 条推文。你需要在不阅读每篇文档的情况下了解整个集合的主题。没有标记的类别，甚至不知道存在多少类别。

主题建模（topic modeling）无需监督即可解决这个问题。给它一个语料库，返回一小组连贯的主题，并且对每个文档给出这些主题的分布。

两大算法家族占主导。LDA（2003）将每篇文档视为潜在主题的混合体，每个主题是单词上的分布。推断采用贝叶斯方法。它仍在生产环境中使用，适用于需要混合成员主题分配以及可解释词级概率分布的场景。

BERTopic（2020）用 BERT 编码文档，采用 UMAP 降维，使用 HDBSCAN 聚类，并通过基于类别的 TF-IDF 提取主题词。它在短文本、社交媒体以及语义相似性比词汇重叠更重要的场景中表现优异。每个文档只分配一个主题，这在长文本内容中是其限制。

本课旨在构建两者的直观理解，并指出在给定语料库时选用哪一个。

## 概念介绍

![LDA 混合模型 vs BERTopic 聚类](../assets/topic-modeling.svg)

**LDA 生成过程。** 每个主题是单词的分布。每个文档是主题的混合体。生成文档中的一个词时，先从文档的主题混合体中采样一个主题，再从该主题的词分布中采样一个词。推断过程是反向的：给定观测词，推断每个文档的主题分布和每个主题的词分布。数学计算采用折叠 Gibbs 采样或变分贝叶斯。

LDA 的关键输出：

- `doc_topic`：矩阵 `(n_docs, n_topics)`，每行和为 1（文档的主题混合分布）。
- `topic_word`：矩阵 `(n_topics, vocab_size)`，每行和为 1（主题的词分布）。

**BERTopic 流程。**

1. 用句子变换器（sentence transformer，例如 `all-MiniLM-L6-v2`）对每个文档编码，得到 384 维向量。
2. 用 UMAP 降维至约 5 维。BERT 嵌入维度过高，不适合直接聚类。
3. 用 HDBSCAN 聚类。基于密度，产出大小不一的簇和“离群点”标签。
4. 对每个簇，计算该簇中文档的基于类别的 TF-IDF，提取主题关键词。

输出为每个文档一个主题（加一个-1的离群标签），可选地，利用 HDBSCAN 的概率向量获得软成员权重。

## 实现步骤

### 第一步：用 scikit-learn 实现 LDA

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np


def fit_lda(documents, n_topics=5, max_features=1000):
    cv = CountVectorizer(
        max_features=max_features,
        stop_words="english",
        min_df=2,
        max_df=0.9,
    )
    X = cv.fit_transform(documents)
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        max_iter=50,
        learning_method="online",
    )
    doc_topic = lda.fit_transform(X)
    feature_names = cv.get_feature_names_out()
    return lda, cv, doc_topic, feature_names


def print_top_words(lda, feature_names, n_top=10):
    for idx, topic in enumerate(lda.components_):
        top_idx = np.argsort(-topic)[:n_top]
        words = [feature_names[i] for i in top_idx]
        print(f"topic {idx}: {' '.join(words)}")
```

注意：去除了停用词，`min_df` 和 `max_df` 过滤了罕见和过于常见的词，使用 `CountVectorizer`（而非 `TfidfVectorizer`）是因为 LDA 期待词频计数。

### 第二步：BERTopic（生产环境使用）

```python
from bertopic import BERTopic

topic_model = BERTopic(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    min_topic_size=15,
    verbose=True,
)

topics, probs = topic_model.fit_transform(documents)
info = topic_model.get_topic_info()
print(info.head(20))
valid_topics = info[info["Topic"] != -1]["Topic"].tolist()
for topic_id in valid_topics[:5]:
    print(f"topic {topic_id}: {topic_model.get_topic(topic_id)[:10]}")
```

`Topic != -1` 的过滤去掉了 BERTopic 把无法聚类的文档放入的离群簇。`min_topic_size` 控制 HDBSCAN 的最小簇大小；BERTopic 默认 10，本例中为本课规模显式设为 15。面对 10,000 以上文档的语料库，应增至 50 或 100。

### 第三步：评估

两种方法都会输出主题词。问题是这些词是否连贯。

- **主题一致性（c_v）。** 结合顶级词对的归一点互信息（NPMI），基于滑动窗口上下文，汇聚分数为主题向量，并通过余弦相似度比较。分数越高越好。使用 `gensim.models.CoherenceModel`，参数 `coherence="c_v"`。
- **主题多样性。** 所有主题顶词中的唯一词占比。越高越好（主题不重叠）。
- **定性检验。** 阅读每个主题的顶词，是否能命名一个真实的概念？人工判断仍是最终防线。

## 何时选择哪种算法

| 情景           | 选择         |
| -------------- | ------------ |
| 短文本（推文、评论、标题） | BERTopic    |
| 长文档且含主题混合   | LDA         |
| 无 GPU 或计算资源有限 | LDA 或 NMF  |
| 需要文档级多主题分布 | LDA         |
| 需要集成大语言模型（LLM）做主题标注 | BERTopic（支持直接集成） |
| 资源受限的边缘部署   | LDA         |
| 要求最高的语义一致性 | BERTopic    |

最大实际考量是文档长度。BERT 嵌入会截断；LDA 统计词频对任意长度工作正常。对长于嵌入模型上下文长度的文档，建议分块后汇总或使用 LDA。

## 使用方案

2026 年技术栈：

- **BERTopic。** 适用于短文本及语义较重场景的默认选择。
- **`gensim.models.LdaModel`。** 经典 LDA，生产级，成熟可靠。
- **`sklearn.decomposition.LatentDirichletAllocation`。** 简单易用，适合实验。
- **NMF。** 非负矩阵分解，LDA 快速替代方案，在短文本上质量相当。
- **Top2Vec。** 与 BERTopic 设计类似，社区较小，但某些基准表现良好。
- **FASTopic。** 较新，处理超大语料比 BERTopic 更快。
- **基于 LLM 的标注。** 先任意聚类，再用模型为每个簇命名。

## 部署示例

保存为 `outputs/skill-topic-picker.md`：

```markdown
---
name: topic-picker
description: 选择 LDA 或 BERTopic 处理语料。指定库、参数、评估方法。
version: 1.0.0
phase: 5
lesson: 15
tags: [nlp, topic-modeling]
---

给定语料描述（文档数、平均长度、领域、语言、计算预算），输出：

1. 算法选择。LDA / NMF / BERTopic / Top2Vec / FASTopic。一句话说明选择理由。
2. 配置。主题数：`recommended = max(5, round(sqrt(n_docs)))`，对小于 40,000 文档的语料限制为最大 200；仅对超大语料库（>40k）允许超过 200，并注释计算成本增加。`min_df` / `max_df` 过滤参数和神经模型的嵌入选择也包含在内。
3. 评估。通过 `gensim.models.CoherenceModel` 计算主题一致性（c_v），主题多样性，以及 20 样本人类读验。
4. 需重点检测的失败模式。LDA 可能有“垃圾主题”，吸纳停用词和高频词；BERTopic 的 -1 离群簇可能吞噬模糊文档。

拒绝对长度大于嵌入模型上下文窗口且无分块策略的文档使用 BERTopic。对极短文本（推文、评论少于 10 词）拒绝用 LDA，因一致性低下。任何主题数低于 5 警告为错误选择；对小于 40k 文档且主题数超过 200 警告过度划分。
```

## 习题

1. **简单。** 在 20 Newsgroups 数据集上用 5 个主题训练 LDA。打印每个主题前 10 个词。手工给主题贴标签。算法是否找到真实类别？
2. **中等。** 在相同的 20 Newsgroups 子集上训练 BERTopic。对比发现的主题数、顶词和定性一致性。哪一个更清晰呈现真实类别？
3. **困难。** 计算你语料中 LDA 与 BERTopic 的 c_v 一致性。分别用 5、10、20、50 个主题运行。绘制一致性与主题数的关系图。报告哪种方法在主题数变化下更稳定。

## 关键词

| 术语       | 常用说法                  | 实际含义                                                |
|------------|---------------------------|---------------------------------------------------------|
| 主题       | 语料所谈论的内容          | 一个单词的概率分布（LDA）或一个相似文档的簇（BERTopic）。      |
| 混合成员   | 文档属于多个主题          | LDA 为每个文档分配所有主题的分布。                             |
| UMAP       | 降维方法                  | 保持局部结构的流形学习，被 BERTopic 用于降维。                |
| HDBSCAN    | 基于密度的聚类            | 发现大小可变的簇，产生“噪声”标签（-1）表示离群点。            |
| c_v 一致性 | 主题质量指标              | 主题顶词在滑动窗口中的平均点互信息。                          |

## 延伸阅读

- [Blei, Ng, Jordan (2003). Latent Dirichlet Allocation](https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf) — LDA 论文。
- [Grootendorst (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure](https://arxiv.org/abs/2203.05794) — BERTopic 论文。
- [Röder, Both, Hinneburg (2015). Exploring the Space of Topic Coherence Measures](https://svn.aksw.org/papers/2015/WSDM_Topic_Evaluation/public.pdf) — 引入 c_v 等一致性指标的论文。
- [BERTopic 文档](https://maartengr.github.io/BERTopic/) — 生产环境参考，示例丰富。
