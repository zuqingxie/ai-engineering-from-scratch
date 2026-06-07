# 信息检索与搜索

> BM25 精准但脆弱。Dense（稠密检索）覆盖广泛但漏掉关键词。Hybrid（混合）是 2026 年默认。其他全是调优。

**类型：** 构建  
**语言：** Python  
**前提：** 第5阶段 · 02（词袋模型（BoW） + TF-IDF）、第5阶段 · 04（GloVe、FastText、子词）  
**时间：** ~75 分钟

## 问题

用户输入“what happens if someone lies to get money”，期望找到真正涵盖该内容的法规："Section 420 IPC"。关键词搜索完全找不到它（没有共享词汇）。如果嵌入没有以法律文本训练，语义搜索也找不到。真正的搜索系统必须同时处理这两者。

信息检索（IR）是所有 RAG 系统、搜索框以及文档网站模糊查找的基础流水线。2026 年生产环境可用的架构不是单一方法，而是一系列互补方法的组合，每种都补足之前方法的不足。

本课时构建每个部分，并指出每个部分解决的失败点。

## 概念

![Hybrid retrieval: BM25 + dense + RRF + cross-encoder rerank](../assets/retrieval.svg)

四层结构。按需选择。

1. **稀疏检索（BM25）**。快速，精确匹配准确，语义表现差。基于倒排索引。对百万文档每次查询低于10毫秒。准确获取法规引用、产品编码、错误信息、命名实体。
2. **稠密检索**。将查询和文档编码成向量。最近邻搜索。捕捉同义表达和语义相似性。会漏掉只差一个字符的精确关键词匹配。用 FAISS 或向量数据库查询，耗时 50-200 毫秒。
3. **融合**。合并稀疏和稠密的排名列表。互惠排名融合（RRF）为默认简单方法，因为它忽略不同尺度的分数，仅依据排名位置。加权融合适合确定某信号主导领域。
4. **交叉编码器再排序**。取融合的前30条。运行交叉编码器（查询+文档一起，给每对打分）。保留前5。交叉编码器比双编码器慢，但更准确。通过只对前30条运行做摊销。

三路检索（BM25 + 稠密 + 学习稀疏如 SPLADE）在 2026 基准测试中优于两路，但需要学习稀疏索引的基础设施。对大多数团队，两路加交叉编码器再排序是最佳点。

## 构建它

### 第1步：从零实现 BM25

```python
import math
import re
from collections import Counter

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


class BM25:
    def __init__(self, corpus, k1=1.5, b=0.75):
        if not corpus:
            raise ValueError("corpus must not be empty")
        self.corpus = [tokenize(d) for d in corpus]
        self.k1 = k1
        self.b = b
        self.n_docs = len(self.corpus)
        self.avg_dl = sum(len(d) for d in self.corpus) / self.n_docs
        self.df = Counter()
        for doc in self.corpus:
            for term in set(doc):
                self.df[term] += 1

    def idf(self, term):
        n = self.df.get(term, 0)
        return math.log(1 + (self.n_docs - n + 0.5) / (n + 0.5))

    def score(self, query, doc_idx):
        q_tokens = tokenize(query)
        doc = self.corpus[doc_idx]
        dl = len(doc)
        freq = Counter(doc)
        score = 0.0
        for term in q_tokens:
            f = freq.get(term, 0)
            if f == 0:
                continue
            numerator = f * (self.k1 + 1)
            denominator = f + self.k1 * (1 - self.b + self.b * dl / self.avg_dl)
            score += self.idf(term) * numerator / denominator
        return score

    def rank(self, query, top_k=10):
        scored = [(self.score(query, i), i) for i in range(self.n_docs)]
        scored.sort(reverse=True)
        return scored[:top_k]
```

两个重要参数。`k1=1.5` 控制词频饱和度，越高表示更强调词的重复。`b=0.75` 控制长度归一化，0 表示忽略文档长度，1 表示完全归一化。默认值是 Robertson 在原始论文中的建议，很少需要调优。

### 第2步：使用双编码器构建稠密检索

```python
from sentence_transformers import SentenceTransformer
import numpy as np


def build_dense_index(corpus, model_id="sentence-transformers/all-MiniLM-L6-v2"):
    encoder = SentenceTransformer(model_id)
    embeddings = encoder.encode(corpus, normalize_embeddings=True)
    return encoder, embeddings


def dense_search(encoder, embeddings, query, top_k=10):
    q_emb = encoder.encode([query], normalize_embeddings=True)
    sims = (embeddings @ q_emb.T).flatten()
    order = np.argsort(-sims)[:top_k]
    return [(float(sims[i]), int(i)) for i in order]
```

对嵌入进行 L2 归一化，使点积等同余弦相似度。`all-MiniLM-L6-v2` 是 384 维、速度快且适合大多数英语检索。多语言任务用 `paraphrase-multilingual-MiniLM-L12-v2`。最高准确度用 `bge-large-en-v1.5` 或 `e5-large-v2`。

### 第3步：互惠排名融合

```python
def reciprocal_rank_fusion(rankings, k=60):
    scores = {}
    for ranking in rankings:
        for rank, (_, doc_idx) in enumerate(ranking):
            scores[doc_idx] = scores.get(doc_idx, 0.0) + 1.0 / (k + rank + 1)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(score, doc_idx) for doc_idx, score in fused]
```

`k=60` 是原始 RRF 论文的常数。更高的 `k` 会使排名差异贡献更平缓；更低 `k` 使顶级排名主导。60 是默认值，通常无需调优。

### 第4步：混合搜索 + 再排序

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def hybrid_search(query, bm25, encoder, dense_embeddings, corpus, top_k=5, pool_size=30, reranker=reranker):
    sparse_ranking = bm25.rank(query, top_k=pool_size)
    dense_ranking = dense_search(encoder, dense_embeddings, query, top_k=pool_size)
    fused = reciprocal_rank_fusion([sparse_ranking, dense_ranking])[:pool_size]

    pairs = [(query, corpus[doc_idx]) for _, doc_idx in fused]
    scores = reranker.predict(pairs)
    reranked = sorted(zip(scores, [doc_idx for _, doc_idx in fused]), reverse=True)
    return reranked[:top_k]
```

三个阶段组合。BM25 找到字面匹配，稠密检索找到语义匹配。RRF 合并排名，无需分数校准。交叉编码器用查询-文档对重新评分前30，捕获双编码器遗漏的细粒度相关性。保持前5。

### 第5步：评估

| 指标        | 含义                                   |
|-------------|--------------------------------------|
| Recall@k    | 在存在正确文档的查询中，前k条中出现的比例。           |
| MRR（平均倒数排名） | 第一个相关文档排名倒数的平均值。                        |
| nDCG@k      | 考虑相关度等级，不仅是相关或不相关的二值区分。            |

对于 RAG，检索器的 **Recall@k** 是最重要指标。没有检索到正确片段，阅读器无法给出答案。

调试提示：失败查询时，比较稀疏和稠密排名。如果一方找到正确文档另一方没有，就是词汇不匹配（解决方案：补齐缺失部分）或语义歧义（解决方案：更优嵌入或再排序器）。

## 使用它

2026 年栈：

| 规模             | 技术栈                                               |
|------------------|-----------------------------------------------------|
| 1k-100k 文档     | 内存中 BM25 + `all-MiniLM-L6-v2` 嵌入 + RRF。无独立数据库。    |
| 100k-10M 文档    | 稠密用 FAISS 或 pgvector，BM25 用 Elasticsearch / OpenSearch 并行运行。 |
| 10M+ 文档        | 使用支持混合检索的 Qdrant / Weaviate / Vespa / Milvus，前30再排序。     |
| 最高质量前沿     | 三路（BM25 + 稠密 + SPLADE） + ColBERT 后期交互再排序              |

无论选择什么，都要预算评估环节。先基准测试检索召回，再评估完整 RAG 准确度。阅读器无法修正检索器漏掉的内容。

### 2026 年生产级 RAG 的宝贵经验

- **80% 的 RAG 失败归因于摄取和分块，而非模型。** 团队花几周换大模型、调提示，实际检索每三次返回一次错误上下文。先修正分块。
- **分块策略比分块大小更重要。** 固定大小切分破坏表格、代码和多级标题。句子感知是默认，技术文档和产品手册用语义或基于 LLM 的分块更有效。
- **父文档模式。** 检索小“子”块提高精度。当同一父节的多个子块出现时，换成父块保持上下文。无需重训即可稳定提升答案质量。
- **`k_rerank=3` 通常最优。** 多余分块增加令牌成本和延迟，不提升答案质量。若 k=8 明显优于 k=3，表明再排序器性能不足。
- **HyDE / 查询扩展。** 根据查询生成假设答案，嵌入后检索。填补短问句与长文档的表达差距。无需训练，直接提升精度。
- **上下文预算低于 8K 令牌。** 频繁命中说明再排序阈值过松。
- **所有东西都要版本管理。** 提示、分块规则、嵌入模型、再排序器。漂移会悄悄降低答案质量。可信度、上下文准确率和未回答率 CI 关卡防止回归用户可见。
- **三路（BM25 + 稠密 + 学习稀疏如 SPLADE）优于两路** 在 2026 基准，特别是包含专有名词和语义混合查询时。基础设施支持时上线。

合理的检索设计根据 2026 行业测算可降低 70-90% 幻觉。大多数 RAG 性能提升来自于更好检索，而非模型微调。

## 交付它

保存为 `outputs/skill-retrieval-picker.md`：

```markdown
---
name: retrieval-picker
description: 为给定语料和查询模式选择检索栈。
version: 1.0.0
phase: 5
lesson: 14
tags: [nlp, retrieval, rag, search]
---

根据要求（语料大小、查询模式、延迟预算、质量标准、基础设施约束），输出：

1. 检索栈。仅 BM25，仅稠密，混合（BM25 + 稠密 + RRF），混合 + 交叉编码器再排序，或三路（BM25 + 稠密 + 学习稀疏）。
2. 稠密编码器。具体模型名。匹配语言、领域和上下文长度。
3. 再排序器。如使用，给出交叉编码器模型名。提示再排序对前30条加30-100毫秒延迟。
4. 评估方案。Recall@10 是检索主指标，多答案用 MRR。先测基线，再测增量提升。

对于包含命名实体、错误码、产品 SKU 的语料，除非用户有证据稠密能处理精确匹配，否则拒绝推荐仅稠密。对于法律、医疗等关乎结果的高风险检索场景，拒绝不做再排序。
```

## 练习

1. **简单。** 在500篇文档语料上实现上面的 `hybrid_search`。测20个查询。比较 BM25 单独、稠密单独和混合的 Recall@5。
2. **中等。** 添加 MRR 计算。对每个测试查询找正确文档在 BM25、稠密和混合排名中的位置。报告每个方法的 MRR。
3. **困难。** 在你领域用 MultipleNegativesRankingLoss（Sentence Transformers）微调稠密编码器。用 500 个查询-文档对构建训练集。比较微调前后召回率。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| BM25 | 关键词搜索 | Okapi BM25。通过词频、逆文档频率（IDF）和文档长度评分文档。 |
| Dense retrieval（稠密检索） | 向量搜索 | 将查询和文档编码成向量，寻找最近邻。 |
| Bi-encoder（双编码器） | 嵌入模型 | 独立编码查询和文档。查询时速度快。 |
| Cross-encoder（交叉编码器） | 重排序模型 | 一起编码查询和文档。虽然慢但准确。 |
| RRF（Rank Fusion，排名融合） | 排名融合 | 通过求和 `1/(k + rank)` 合并两个排名。 |
| Recall@k（召回率@k） | 检索指标 | 在前k个结果中包含相关文档的查询比例。 |

## 进一步阅读

- [Robertson 和 Zaragoza (2009). The Probabilistic Relevance Framework: BM25 and Beyond](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf) — BM25的权威解析。
- [Karpukhin 等人 (2020). Dense Passage Retrieval for Open-Domain QA](https://arxiv.org/abs/2004.04906) — DPR，标准双编码器。
- [Formal 等人 (2021). SPLADE: Sparse Lexical and Expansion Model](https://arxiv.org/abs/2107.05720) — 通过学习得到的稀疏检索器，缩小了与稠密方法的差距。
- [Cormack, Clarke, Büttcher (2009). Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) — RRF论文。
- [Khattab 和 Zaharia (2020). ColBERT: Efficient and Effective Passage Search](https://arxiv.org/abs/2004.12832) — 晚期交互检索方法。
