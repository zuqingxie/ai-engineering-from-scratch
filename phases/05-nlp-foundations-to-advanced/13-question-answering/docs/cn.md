# 问答系统（Question Answering Systems）

> 三种系统塑造了现代问答。Extractive（抽取式）找到答案片段。Retrieval-augmented（检索增强）使答案有靠文档支撑。Generative（生成式）生成答案。每个现代 AI 助手都是这三者的混合。

**类型：** 构建  
**语言：** Python  
**先修知识：** 第5阶段 · 11（机器翻译），第5阶段 · 10（注意力机制）  
**时间：** ~75 分钟

## 问题

用户输入“第一代 iPhone 何时发布？”，期待得到“2007年6月29日”，而不是“苹果公司的历史悠久多样”，也不是孤立的“2007”，而是一个直接、有依据、正确的答案。

过去十年来，问答领域主要由三种架构主导：

- **抽取式问答（Extractive QA）**：给定问题和已知包含答案的段落，寻找答案在段落中的起止位置。SQuAD 是典型基准。
- **开放域问答（Open-domain QA）**：未给定段落，先检索相关段落，再抽取或生成答案。这是目前所有 RAG 流水线的基础。
- **生成式/闭卷问答（Generative / Closed-book QA）**：大型语言模型（LLM）根据参数记忆回答，无需检索。推理最快但事实准确性最低。

2026 年趋势是混合：先检索最佳的几段上下文，再用生成模型基于这些上下文回答。这就是 RAG，本课第14课详细讲述检索部分。本课构建问答部分。

## 概念

![QA architectures: extractive, retrieval-augmented, generative](../assets/qa.svg)

**抽取式（Extractive）**：使用 Transformer（BERT 家族）共同编码问题和段落。训练两个头分别预测答案的起始和结束 token 索引。损失是对有效位置的交叉熵。输出是段落中的一个片段。绝不幻想（根据设计），也不处理段落无法回答的问题（设计如此）。

**检索增强（Retrieval-augmented, RAG）**：两阶段。第一阶段检索器从语料库找到 top-`k` 相关段落；第二阶段阅读器（抽取或生成）基于这些段落生成答案。检索器和阅读器可独立训练和评估。现代 RAG 通常还在两者之间加个重排序器。

**生成式（Generative）**：仅解码器的大型语言模型（GPT、Claude、Llama）依靠已学权重生成答案，无检索步骤。对常识表现很好，但对罕见或最新事实灾难性出错。幻想率与预训练数据中事实频率成反比。

## 构建它

### 第1步：用预训练模型实现抽取式问答

```python
from transformers import pipeline

qa = pipeline("question-answering", model="deepset/roberta-base-squad2")

passage = (
    "Apple Inc. released the first iPhone on June 29, 2007. "
    "The device was announced by Steve Jobs at Macworld in January 2007."
)
question = "When was the first iPhone released?"

answer = qa(question=question, context=passage)
print(answer)
```

```python
{'score': 0.98, 'start': 57, 'end': 70, 'answer': 'June 29, 2007'}
```

`deepset/roberta-base-squad2` 训练于 SQuAD 2.0，包含无答案问题。默认情况下，`question-answering` pipeline 即使模型的空答案分数更高，也返回最高分的片段——*不会*自动返回空答案。若需显式“无答案”行为，调用时传入 `handle_impossible_answer=True`：当空答案分数超过所有片段分数时才返回空答案。无论如何请始终检查 `score` 字段。

### 第2步：检索增强流水线（框架）

```python
from sentence_transformers import SentenceTransformer
import numpy as np

encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

corpus = [
    "Apple Inc. released the first iPhone on June 29, 2007.",
    "Macworld 2007 featured the iPhone announcement by Steve Jobs.",
    "Android launched in 2008 as Google's mobile operating system.",
    "The first iPod was released in 2001.",
]
corpus_embeddings = encoder.encode(corpus, normalize_embeddings=True)


def retrieve(question, top_k=2):
    q_emb = encoder.encode([question], normalize_embeddings=True)
    sims = (corpus_embeddings @ q_emb.T).squeeze()
    order = np.argsort(-sims)[:top_k]
    return [corpus[i] for i in order]


def answer(question):
    passages = retrieve(question, top_k=2)
    combined = " ".join(passages)
    return qa(question=question, context=combined)


print(answer("When was the first iPhone released?"))
```

两阶段流水线。稠密检索器（Sentence-BERT）依语义相似度找到相关段落。抽取式阅读器（RoBERTa-SQuAD）从拼接的前几个段落中提取答案片段。适用于小语料库。百万文档级别的语料请用 FAISS 或向量数据库。

### 第3步：基于 RAG 的生成

```python
def rag_generate(question, llm):
    passages = retrieve(question, top_k=3)
    prompt = f"""Context:
{chr(10).join('- ' + p for p in passages)}

Question: {question}

Answer using only the context above. If the context does not contain the answer, say "I don't know."
"""
    return llm(prompt)
```

提示设计至关重要。明确告诉模型基于上下文作答，若上下文不包含答案则回答“我不知道”，相比简单提示减少 40-60% 的幻想率。更复杂的模板会带上引用、置信度和结构化抽取。

### 第4步：反映真实世界的评估

SQuAD 使用**Exact Match（EM）** 和**token-level F1**。EM 是严格匹配（忽略大小写、去标点和冠词之后）——预测与参考完全匹配得分1，否则0。F1 对预测和参考的 token 重叠计算，给部分得分。两者都对同义表达打分不足：比如“June 29, 2007” vs “June 29th, 2007”，因序数词不同通常 EM 为0，但 F1 依重叠 token 获得较高分。

生产环境问答关注：

- **答案准确率**（由 LLM 判定或人工判定，因为指标无法捕捉语义等价）。
- **引用准确率。** 引用的段落是否确实支持答案？可通过生成引用字符串与检索段落字符串匹配自动检查。
- **拒绝校准。** 当答案不存在于检索段落中时，系统是否正确回答“我不知道”？需测假信心率。
- **检索召回率。** 在评估阅读器前，测检索器是否返回了正确段落。漏检的段落阅读器无法补救。

### RAGAS：2026 年生产环境评估框架

`RAGAS` 是专为 RAG 系统设计，2026 年默认在用。自动对四个维度打分，无需黄金参考：

- **忠实度（Faithfulness）**。答案中的每个陈述是否源自检索上下文？用基于自然语言推理（NLI）的蕴涵度量。主要幻想指标。
- **答案相关性（Answer relevance）**。答案是否答到了问题？通过从答案生成假设问题，并与实际问题比较。
- **上下文准确率（Context precision）**。检索的段块中有多少是相关的？准确率低意味着提示中噪声多。
- **上下文召回率（Context recall）**。检索集合是否包含所有必要信息？召回率低说明阅读器无法成功。

无参考打分方便直接在生产流量上评估。对开放式问题可在此基础上叠加 LLM 作为评判。

`pip install ragas`。连接您的检索器 + 阅读器。每查询获得四个标量。出现性能回退即可告警。

## 使用它

2026 年栈推荐。

| 用例 | 推荐方案 |
|---------|-------------|
| 给定段落，找答案片段 | `deepset/roberta-base-squad2` |
| 固定语料库，闭卷不可接受 | RAG：稠密检索器 + LLM 阅读器 |
| 基于文档库的实时问答 | RAG 使用混合（BM25 + 稠密）检索器 + 重排序器（第14课） |
| 会话式问答（跟进问题） | 带对话历史的 LLM + 每轮 RAG |
| 高度事实性、受监管领域 | 针对权威语料库的抽取式，不单独用生成式 |

2026 年抽取式问答已不时髦，因为 RAG + LLM 适用场景更多，但仍应用于需逐字引用的场景：法律研究、合规审计工具。

## 部署它

保存为 `outputs/skill-qa-architect.md`：

```markdown
---
name: qa-architect
description: Choose QA architecture, retrieval strategy, and evaluation plan.
version: 1.0.0
phase: 5
lesson: 13
tags: [nlp, qa, rag]
---

Given requirements (corpus size, question type, factuality constraint, latency budget), output:

1. Architecture. Extractive, RAG with extractive reader, RAG with generative reader, or closed-book LLM. One-sentence reason.
2. Retriever. None, BM25, dense (name the encoder), or hybrid.
3. Reader. SQuAD-tuned model, LLM by name, or "domain-fine-tuned DistilBERT."
4. Evaluation. EM + F1 for extractive benchmarks; answer accuracy + citation accuracy + refusal calibration for production. Name what you are measuring and how you are measuring it.

Refuse closed-book LLM answers for regulatory or compliance-sensitive questions. Refuse any QA system without a retrieval-recall baseline (you cannot evaluate the reader without knowing the retriever surfaced the right passage). Flag questions that require multi-hop reasoning as needing specialized multi-hop retrievers like HotpotQA-trained systems.
```

## 练习

1. **简单。** 在 10 个维基百科段落上搭建上述 SQuAD 抽取式流水线。手写 10 个问题。测量正确答案出现频率。段落和问题清晰时正确率应为7-9个。
2. **中等。** 添加拒绝分类器。当最高检索分数低于阈值（如余弦相似度0.3）时，返回“我不知道”，而非调用阅读器。在留出集合上调参。
3. **困难。** 构建一个覆盖您选定的 10,000 文档语料的 RAG 流水线。实现基于 RRF 融合的混合检索（BM25 + 稠密，参见第14课）。测量有无混合检索步骤的答案准确率差异。记录哪类问题获益最多。

## 关键词

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| 抽取式问答（Extractive QA） | 找答案片段 | 预测答案在给定段落中的起止索引。 |
| 开放域问答（Open-domain QA） | 语料库问答 | 无段落给定，需先检索再回答。 |
| RAG | 先检索后生成 | 检索增强生成。检索器 + 阅读器流水线。 |
| SQuAD | 经典基准 | 斯坦福问答数据集。用 EM + F1 指标。 |
| 幻想（Hallucination） | 杜撰的答案 | 阅读器输出与检索上下文不符。 |
| 拒绝校准（Refusal calibration） | 知道何时保持沉默 | 无答案时系统正确回答“我不知道”。 |

## 进一步阅读

- [Rajpurkar et al. (2016). SQuAD: 100,000+ Questions for Machine Comprehension of Text](https://arxiv.org/abs/1606.05250) — 基准论文。  
- [Karpukhin et al. (2020). Dense Passage Retrieval for Open-Domain QA](https://arxiv.org/abs/2004.04906) — DPR，开放域问答的稠密检索经典。  
- [Lewis et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) — RAG 命名论文。  
- [Gao et al. (2023). Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997) — 详尽 RAG 综述。
