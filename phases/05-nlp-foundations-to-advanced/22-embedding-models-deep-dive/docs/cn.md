# 嵌入模型 — 2026 深度探讨

> Word2Vec（词向量模型）给你每个词一个向量。现代嵌入模型给你每段文本一个向量，支持跨语言，提供稀疏、稠密和多向量视图，大小可调以适配你的索引。选错了，RAG（检索增强生成）就会检索错误内容。

**类型：** 学习  
**语言：** Python  
**先决条件：** 第5阶段 · 03（Word2Vec），第5阶段 · 14（信息检索）  
**时长：** ~60 分钟

## 问题

你的 RAG 系统检索到错误段落的概率高达 40%。罪魁祸首很少是向量数据库或提示词（prompt），而是嵌入模型。

在 2026 年选择嵌入模型意味着需要考虑五个维度：

1. **稠密 vs 稀疏 vs 多向量。** 每段文本一个向量，还是每个词一个向量，或是加权稀疏词袋。
2. **语言覆盖。** 单语言英文模型在英文任务仍占优势。多语言模型在混合语料中表现更佳。
3. **上下文长度。** 512 词 vs 8,192 词 vs 32,768 词——且实际有效容量通常是标称最大值的 60-70%。
4. **维度预算。** 3,072 维浮点数，全精度下每向量占 12 KB。1 亿向量，存储费用约 $1,300/月。Matryoshka 截断可节省约4倍空间。
5. **开源 vs 云端托管。** 开源权重意味着你控制整个栈和数据，托管服务则以控制权换取总是最新版本。

本课列举这些权衡，帮助你基于证据而非一时流行做选择。

## 概念介绍

![稠密、稀疏和多向量嵌入](../assets/embedding-modes.svg)

**稠密嵌入。** 每段文本一个向量（通常 384-3,072 维）。用余弦相似度排序，判断语义接近度。OpenAI `text-embedding-3-large`、BGE-M3 稠密模式、Voyage-3。默认选择。

**稀疏嵌入。** 类似 SPLADE。Transformer为每个词汇表词预测一个权重，然后置零大部分。结果是大小为 |vocab| 的稀疏向量。捕捉词汇匹配（类似 BM25），但权重是学出来的。关键词密集查询表现强。

**多向量（后期交互）。** ColBERTv2、Jina-ColBERT。每个词一个向量。采用 MaxSim 评分：对每个查询词，找到最相似的文档词得分相加。存储和评分更昂贵，但在长查询和领域特定语料中表现优异。

**BGE-M3：三种模式合一。** 单模型同时输出稠密、稀疏和多向量表示。各自可独立查询；分数通过加权求和融合。2026 年默认选项，适合需要灵活性的场景。

**Matryoshka 表示学习。** 训练使得向量的前 N 维形成一个有用且独立的嵌入。将 1,536 维向量截断到 256 维，仅损失约 1% 精度，但节省约 6 倍存储。OpenAI text-3、Cohere v4、Voyage-4、Jina v5、Gemini Embedding 2、Nomic v1.5+支持。

### MTEB 排行榜的局限

Massive Text Embedding Benchmark（大规模文本嵌入基准）— 初版涵盖 56 个任务，8 类任务（2022），MTEB v2 扩展至 100+ 任务。2026 年初，Gemini Embedding 2 以 67.71 MTEB-R 蝉联检索榜首，Cohere embed-v4 领先通用类别（65.2 MTEB），BGE-M3 引领开源多语种（63.0）。排行榜是必要的，但不是充分的——务必在你领域内做基准测试。

### 三层模式

| 用例        | 模式                                      |
|-------------|------------------------------------------|
| 快速初筛    | 稠密双编码器（BGE-M3，text-3-small）     |
| 提升召回    | 稀疏（SPLADE、BGE-M3 稀疏） + RRF 融合    |
| 精确排序前50 | 多向量（ColBERTv2）或跨编码器重排序器     |

绝大多数生产堆栈采用三者结合。

## 实践构建

### 步骤 1：基线 — 使用 Sentence-BERT 生成稠密嵌入

```python
from sentence_transformers import SentenceTransformer
import numpy as np

encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")
corpus = [
    "The first iPhone launched in 2007.",
    "Apple released the iPod in 2001.",
    "Android is an operating system from Google.",
]
emb = encoder.encode(corpus, normalize_embeddings=True)

query = "When was the iPhone released?"
q_emb = encoder.encode([query], normalize_embeddings=True)[0]
scores = emb @ q_emb
print(sorted(enumerate(scores), key=lambda x: -x[1]))
```

`normalize_embeddings=True` 使点积等价于余弦相似度，务必设置。

### 步骤 2：Matryoshka 截断

```python
def truncate(vectors, dim):
    out = vectors[:, :dim]
    return out / np.linalg.norm(out, axis=1, keepdims=True)

emb_256 = truncate(emb, 256)
emb_128 = truncate(emb, 128)
```

截断后重新归一化。Nomic v1.5、OpenAI text-3 和 Voyage-4 训练时确保了前几层截断无损，非 Matryoshka 模型（原始 Sentence-BERT）截断则精度骤降。

### 步骤 3：BGE-M3 多功能

```python
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

output = model.encode(
    corpus,
    return_dense=True,
    return_sparse=True,
    return_colbert_vecs=True,
)
# output["dense_vecs"]:    (n_docs, 1024)
# output["lexical_weights"]: list of dict {token_id: weight}
# output["colbert_vecs"]:  list of (n_tokens, 1024) arrays
```

三种索引，一次推理调用。分数融合示例：

```python
dense_score = ... # 计算稠密向量余弦相似度
sparse_score = model.compute_lexical_matching_score(q_lex, d_lex)
colbert_score = model.colbert_score(q_col, d_col)
final = 0.4 * dense_score + 0.2 * sparse_score + 0.4 * colbert_score
```

根据你的领域调节权重。

### 步骤 4：在自定义任务上进行 MTEB 评测

```python
from mteb import MTEB

tasks = ["ArguAna", "SciFact", "NFCorpus"]
evaluation = MTEB(tasks=tasks)
results = evaluation.run(encoder, output_folder="./mteb-results")
```

针对领域的*代表性*子集运行测试。不要盲目信任排行榜排名——领域差异很重要。

### 步骤 5：手动实现余弦相似度

详见 `code/main.py`。采用平均哈希技巧生成嵌入（仅标准库实现）。表现不及 Transformer 嵌入，但展示流程：分词 → 向量化 → 归一化 → 点积。

## 常见陷阱

- **查询和文档用同一模型。** 部分模型（Voyage, Jina-ColBERT）采用不对称编码——查询和文档经过不同路径。务必查看模型卡。
- **缺少前缀。** `bge-*` 模型要求查询前加 `"Represent this sentence for searching relevant passages: "`。忘记会导致召回率差 3-5 点。
- **Matryoshka 截断过度。** 1,536 维截断到 256 维通常安全，截到 64 维则风险大。务必在评测集上验证。
- **上下文截断。** 多数模型会默默截断超过最大长度的输入。长文档需分块（详见课程 23）。
- **忽视尾部延迟。** MTEB 评分隐藏了 p99 延迟。600M 参数模型可能比 335M 参数好 2 分，但单次查询成本高出 3 倍。

## 选型指南

2026 年典型堆栈：

| 情况             | 选择                                 |
|------------------|------------------------------------|
| 仅英文，快速响应 | `text-embedding-3-large` 或 `voyage-3-large`  |
| 开源权重，英文   | `BAAI/bge-large-en-v1.5`            |
| 开源权重，多语种 | `BAAI/bge-m3` 或 `Qwen3-Embedding-8B` |
| 长上下文（32k+） | Voyage-3-large、Cohere embed-v4、Qwen3-Embedding-8B |
| 仅 CPU 部署      | Nomic Embed v2（137M 参数，MoE）   |
| 存储受限         | Matryoshka 截断 + int8 量化        |
| 关键词密集查询   | 增加 SPLADE 稀疏，RRF 融合稠密结果 |

2026 年模式：从 BGE-M3 或 text-3-large 开始，使用 MTEB 在你领域评估，若领域特定模型领先 3 分以上，则切换。

## 部署示例

保存为 `outputs/skill-embedding-picker.md`：

```markdown
---
name: embedding-picker
description: 选择嵌入模型、维度和检索模式，适用于给定语料和部署环境。
version: 1.0.0
phase: 5
lesson: 22
tags: [nlp, embeddings, retrieval]
---

给定一个语料库（大小、语言、领域、平均长度）、部署目标（云端 / 边缘 / 本地）、延迟预算和存储预算，输出：

1. 模型。命名的检查点或API。简述理由。
2. 维度。全维 / Matryoshka 截断 / int8 量化。基于存储预算的理由。
3. 模式。稠密 / 稀疏 / 多向量 / 混合。理由。
4. 查询前缀 / 模板（若模型卡需要）。
5. 评测方案。与领域相关的 MTEB 任务 + 保留领域的 nDCG@10 评测。

拒绝未经领域验证直接截断 Matryoshka 到 <64 维的推荐。拒绝文档数少于 1 万的语料选用 ColBERTv2（开销不合理）。标记长文档语料（超过 8k 词）应使用 512 词窗口模型。
```

## 练习

1. **简单。** 用 `bge-small-en-v1.5` 生成 100 句子全维（384）和 Matryoshka 128 维的嵌入。测量 10 个查询的 MRR（平均相关率）下降。
2. **中等。** 在你领域的 500 个段落，比较 BGE-M3 稠密、稀疏和 ColBERT 模式。哪个在 recall@10 上最好？RRF 融合结果比单独最优模式好多少？
3. **困难。** 在你的两个主领域任务上运行三款候选模型的 MTEB，报告 MTEB 分数、100 条查询批的 p99 延迟和每百万查询成本。选出帕累托最优模型。

## 关键词汇

| 术语          | 常见说法                | 实际含义                           |
|---------------|-------------------------|-----------------------------------|
| Dense embedding（稠密嵌入）   | 向量                     | 每段文本一个固定长度向量，按余弦相似度排序。    |
| Sparse embedding（稀疏嵌入）   | 学习版 BM25               | 每词汇表词一个权重，大多数为零，端到端训练。    |
| Multi-vector（多向量）        | ColBERT 风格               | 每词一个向量，MaxSim 评分，索引更大，召回更好。   |
| Matryoshka（套娃）             | 俄罗斯套娃技巧             | 向量前 N 维可单独作为有效小嵌入。                 |
| MTEB（大规模文本嵌入基准）  | 基准                     | 启动时 56 任务，v2 时扩充至 100+ 任务。           |
| BEIR（检索基准）               | 检索基准                   | 18 个零样本检索任务，常用于跨域健壮性评测。         |
| Asymmetric encoding（不对称编码） | 查询 ≠ 文档路径             | 模型采用不同的投影路径分别编码查询和文档。            |

## 拓展阅读

- [Reimers, Gurevych (2019). Sentence-BERT](https://arxiv.org/abs/1908.10084) — 双编码器论文。  
- [Muennighoff et al. (2022). MTEB: Massive Text Embedding Benchmark](https://arxiv.org/abs/2210.07316) — 排行榜论文。  
- [Chen et al. (2024). BGE-M3: Multi-lingual, Multi-functionality, Multi-granularity](https://arxiv.org/abs/2402.03216) — 三模式统一模型。  
- [Kusupati et al. (2022). Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147) — 维度阶梯训练目标。  
- [Santhanam et al. (2022). ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction](https://arxiv.org/abs/2112.01488) — 生产环境中的后期交互。  
- [Hugging Face MTEB 排行榜](https://huggingface.co/spaces/mteb/leaderboard) — 实时排行页面。
