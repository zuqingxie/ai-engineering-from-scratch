# 文本摘要

> 抽取式系统告诉你文档说了什么。生成式系统告诉你作者的意图。不同任务，不同陷阱。

**类型：** 实践  
**语言：** Python  
**先修要求：** 第5阶段 · 02（BoW + TF-IDF）、第5阶段 · 11（机器翻译）  
**时间：** 约75分钟

## 问题

一篇2000字的新闻文章出现在你的信息流中。你需要120字捕捉它的核心。你可以选取文章中最重要的三句话（抽取式），或者用自己的话改写内容（生成式）。两者都是摘要，但完全不同的问题。

抽取式摘要是一个排序问题。给每句话评分，返回排名前k的句子。输出总是语法正确，因为直接摘录。风险是遗漏散布于全文的关键信息。

生成式摘要是一个生成问题。Transformer（Transformer 架构）根据输入生成新文本。输出流畅且压缩，但可能“编造”不在源文本中的事实。风险是自信地虚构内容。

本课将构建两者，并展示各自的失败模式。

## 概念

![抽取式TextRank vs 生成式Transformer](../assets/summarization.svg)

**抽取式。** 将文章看作一个图，节点为句子，边为相似度。对图运行PageRank（或类似算法），根据句子与其他句子的连接度评分。得分最高的句子组成摘要。经典实现是**TextRank**（Mihalcea和Tarau，2004）。

**生成式。** 微调Transformer编码器-解码器（BART、T5、Pegasus）在文档-摘要对上。推理时，模型读取文档，通过交叉注意力逐字生成摘要。Pegasus 尤其使用缺句预训练目标，少量微调即可表现优异。

用**ROUGE**（回忆导向简要评估）评价。ROUGE-1和ROUGE-2评分一元和二元词组重叠。ROUGE-L评分最长公共子序列。分数越高越好，40 ROUGE-L算“好”，50算“优秀”。所有论文都会报告这三项。使用`rouge-score`包。

## 构建它

### 步骤1：TextRank（抽取式）

```python
import math
import re
from collections import Counter


def sentence_split(text):
    return re.split(r"(?<=[.!?])\s+", text.strip())


def similarity(s1, s2):
    w1 = Counter(s1.lower().split())
    w2 = Counter(s2.lower().split())
    intersection = sum((w1 & w2).values())
    denom = math.log(len(w1) + 1) + math.log(len(w2) + 1)
    if denom == 0:
        return 0.0
    return intersection / denom


def textrank(text, top_k=3, damping=0.85, iterations=50, epsilon=1e-4):
    sentences = sentence_split(text)
    n = len(sentences)
    if n <= top_k:
        return sentences

    sim = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                sim[i][j] = similarity(sentences[i], sentences[j])

    scores = [1.0] * n
    for _ in range(iterations):
        new_scores = [1 - damping] * n
        for i in range(n):
            total_out = sum(sim[i]) or 1e-9
            for j in range(n):
                if sim[i][j] > 0:
                    new_scores[j] += damping * sim[i][j] / total_out * scores[i]
        if max(abs(s - ns) for s, ns in zip(scores, new_scores)) < epsilon:
            scores = new_scores
            break
        scores = new_scores

    ranked = sorted(range(n), key=lambda k: scores[k], reverse=True)[:top_k]
    ranked.sort()
    return [sentences[i] for i in ranked]
```

值得说明的两点。相似度函数采用对数归一化的词语重叠，这是原始TextRank的版本。也可以用TF-IDF向量的余弦距离。阻尼因子0.85和迭代次数为PageRank默认值。

### 步骤2：用BART进行生成式摘要

```python
from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

article = """(长篇新闻文章文本)"""

summary = summarizer(article, max_length=120, min_length=60, do_sample=False)
print(summary[0]["summary_text"])
```

BART-large-CNN在CNN/DailyMail语料上微调。它能直接生成新闻风格的摘要。对于其他领域（科学论文、对话、法律）使用对应的Pegasus检查点或在目标数据上微调T5。

### 步骤3：ROUGE评估

```python
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
scores = scorer.score(reference_summary, generated_summary)
print({k: round(v.fmeasure, 3) for k, v in scores.items()})
```

务必使用词干化（stemming）。不使用时，“running”和“run”被算作不同的词，ROUGE得分会偏低。

### 超越ROUGE（2026年摘要评测）

ROUGE主导了二十年摘要评测，但到了2026年单独使用已不够。一项大规模自然语言生成元分析显示：

- **BERTScore**（上下文嵌入相似度）到2023年日渐流行，大多数摘要论文同时报告它和ROUGE。
- **BARTScore**将评估看作生成过程：用预训练BART条件概率对摘要进行打分。
- **MoverScore**（上下文嵌入距离的地球移动距离）2025年在摘要基准奠定顶尖地位，因比ROUGE更好捕捉语义重合。
- **FactCC**和基于问答的真实性评估在2021-2023年常用，现常被使用GPT-4链式推理的**G-Eval**取代，用于评分连贯性、一致性、流畅度、相关度。
- **G-Eval**及类似的大型语言模型评委技术在精心设计的评分量表上，约80%评判符合人工判断。

生产环境推荐：报告ROUGE-L用于历史比较，BERTScore用于语义重合，G-Eval用于连贯性和真实性。并针对50-100个人工标注摘要进行校准。

### 步骤4：真实性问题

生成式摘要易产生幻觉。抽取式摘要因直接摘录源句，幻觉风险低，但在脱离语境、数据过时或断句错位时仍会误导。这是工业界合规相关内容常偏好抽取式的最重要原因。

常见幻觉类型：

- **实体替换。** 源文说“John Smith”，摘要说“John Brown”。
- **数字偏移。** 源文说“25,000”，摘要说“25百万”。
- **极性翻转。** 源文说“拒绝了报价”，摘要说“接受了报价”。
- **事实虚构。** 源文未提CEO，摘要断言CEO批准了。

有效评测方式：

- **FactCC。** 一个二分类器，判断摘要句是否从源句蕴涵而来。预测真实或不真实。
- **基于问答的真实性。** 向问答模型提问源文本中的问题，若摘要支持不同答案则标记。
- **实体级F1。** 比较源文和摘要中的命名实体。仅在摘要中出现的实体被怀疑。

凡是涉及用户体验且真实性重要（新闻、医疗、法律、金融）的场景，抽取式是较安全默认。生成式必须在线加入真实性检测。

## 使用建议

2026年技术栈：

| 用例 | 推荐模型 |
|---------|-------------|
| 新闻，3-5句话摘要，英文 | `facebook/bart-large-cnn` |
| 科研论文 | `google/pegasus-pubmed` 或 微调过的 T5 |
| 多文档，长篇摘要 | 任何带32k+上下文的LLM，提示式使用 |
| 对话摘要 | `philschmid/bart-large-cnn-samsum` |
| 抽取式，内在低幻觉风险 | TextRank 或 `sumy` 中的 LSA / LexRank |

长上下文LLM在2026年计算资源充足时常胜过专用模型，但成本和稳定性是权衡因素；专用模型输出更一致。

## 部署方案

保存为 `outputs/skill-summary-picker.md`：

```markdown
---
name: summary-picker
description: 选择抽取式或生成式，指定库，真实性检查。
version: 1.0.0
phase: 5
lesson: 12
tags: [nlp, summarization]
---

给定任务（文档类型、合规要求、长度、计算预算），输出：

1. 方案。抽取式或生成式。一句话说明原因。
2. 起始模型/库。命名它。`sumy.TextRankSummarizer`、`facebook/bart-large-cnn`、`google/pegasus-pubmed`或LLM提示。
3. 评估计划。ROUGE-1、ROUGE-2、ROUGE-L（使用带词干的rouge-score）。生成式则附加真实性检测。
4. 一个待检测的失败模式。实体替换是生成式新闻摘要最常见；标注源实体未出现在摘要的样本。

医疗、法律、金融或受监管内容若无真实性门控，拒用生成式摘要。超出模型上下文窗口的输入需采用分块的映射-归约摘要（非简单截断）。
```

## 练习

1. **简单。** 在5篇新闻文章上运行TextRank。将前三句与参考摘要对比，测ROUGE-L。CNN/DailyMail类文章应看到30-45 ROUGE-L。
2. **中等。** 实现实体级真实性评估：用spaCy抽取源和摘要实体，计算摘要中源实体的召回率及摘要实体中源实体的准确率。高准确率且召回低表示安全但简略；低准确率表示幻觉实体。
3. **困难。** 比较BART-large-CNN和LLM（Claude或GPT-4）在50篇CNN/DailyMail文章上的表现。报告ROUGE-L、真实性（通过实体F1）、摘要成本。记录各自优势领域。

## 关键词

| 术语 | 常用说法 | 实际含义 |
|------|---------|---------|
| Extractive | 选句子 | 直接摘录源句。不会幻觉。 |
| Abstractive | 改写 | 基于源文生成新文本。可能幻觉。 |
| ROUGE | 摘要指标 | 系统输出与参考的N-gram/LCS重叠。 |
| TextRank | 基于图的抽取 | 在句子相似度图上运行PageRank。 |
| Factuality | 准确性 | 摘要内容是否被源文支持。 |
| Hallucination | 虚构内容 | 摘要中源文不支持的内容。 |

## 相关阅读

- [Mihalcea 和 Tarau (2004). TextRank: Bringing Order into Texts](https://aclanthology.org/W04-3252/) — 抽取式经典论文。
- [Lewis 等 (2019). BART: Denoising Sequence-to-Sequence Pre-training](https://arxiv.org/abs/1910.13461) — BART论文。
- [Zhang 等 (2019). PEGASUS: Pre-training with Extracted Gap-sentences](https://arxiv.org/abs/1912.08777) — Pegasus与缺句目标。
- [Lin (2004). ROUGE: A Package for Automatic Evaluation of Summaries](https://aclanthology.org/W04-1013/) — ROUGE论文。
- [Maynez 等 (2020). On Faithfulness and Factuality in Abstractive Summarization](https://arxiv.org/abs/2005.00661) — 真实性综述论文。
