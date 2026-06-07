# 词性标注（POS Tagging）与句法分析（Syntactic Parsing）

> 语法曾一度不流行。然后每个大型语言模型（LLM）管道都需要验证结构化提取，语法又回来了。

**类型：** 构建  
**语言：** Python  
**先修课程：** 第5阶段·01（文本处理），第2阶段·14（朴素贝叶斯）  
**时间：** 约45分钟

## 问题

第一课承诺词形还原（lemmatization）需要词性标注。不了解 `running` 是动词，一个词形还原器无法将其还原为 `run`。不了解 `better` 是形容词，也无法还原为 `good`。

这个承诺背后隐藏着一个完整的子领域。词性标注（POS tagging）给每个词标注语法类别。句法分析（syntactic parsing）恢复句子的树结构：哪个词修饰哪个，哪个动词支配哪些成分。传统自然语言处理（NLP）花了二十年精炼这两者。然后深度学习将它们融合为基于预训练Transformer的分词分类任务，研究界便转向其他方向。

但应用界并未如此。每个结构化提取管道底层仍然使用词性和依存树。大型语言模型生成的JSON会根据语法约束进行验证。问答系统利用依存解析分解查询。机器翻译质量评估检查解析树的对齐。

非常值得了解。本课介绍标记集、基线方法，以及何时停止从零实现而调用spaCy。

## 概念

**词性标注（POS tagging）** 为每个词标注语法类别。英语默认是**宾夕法尼亚树库（Penn Treebank，PTB）**标记集。36个标签，区别细致，非专业读者会觉得麻烦：`NN` 单数名词，`NNS` 复数名词，`NNP` 专有名词单数，`VBD` 动词过去式，`VBZ` 动词第三人称单数现在时，等等。**通用依存（Universal Dependencies，UD）**标记集更粗糙（17个标签），且语言无关，成为跨语言任务的默认标记集。

```text
The/DET cats/NOUN were/AUX running/VERB at/ADP 3pm/NOUN ./PUNCT
```

**句法分析（Syntactic parsing）** 产生一棵树。两大主流风格：

- **成分句法（Constituency parsing）**。名词短语、动词短语、介词短语彼此嵌套。输出是非终结符类别（NP，VP，PP）构成的树，单词作为叶子节点。
- **依存句法（Dependency parsing）**。每个词依赖唯一父节点，用语法关系标注。输出是边为（父词，子词，关系）三元组的树。

2010年代依存句法胜出，因为它能在多语言（尤其是自由语序语言）间干净地泛化。

```text
running is ROOT
cats is nsubj of running
were is aux of running
at is prep of running
3pm is pobj of at
```

## 构建

### 第1步：最频繁标记基线（most-frequent-tag baseline）

最笨但有效的词性标注器。对每个词预测其训练中出现次数最多的标记。

```python
from collections import Counter, defaultdict


def train_mft(train_examples):
    word_tag_counts = defaultdict(Counter)
    all_tags = Counter()
    for tokens, tags in train_examples:
        for token, tag in zip(tokens, tags):
            word_tag_counts[token.lower()][tag] += 1
            all_tags[tag] += 1
    word_best = {w: c.most_common(1)[0][0] for w, c in word_tag_counts.items()}
    default_tag = all_tags.most_common(1)[0][0]
    return word_best, default_tag


def predict_mft(tokens, word_best, default_tag):
    return [word_best.get(t.lower(), default_tag) for t in tokens]
```

在Brown语料库上，这个基线准确率约85%。不算好，但这是任何严肃模型的底线。

### 第2步：二元组隐马尔可夫模型（bigram HMM）标注器

建模序列联合概率：

```text
P(tags, words) = prod P(tag_i | tag_{i-1}) * P(word_i | tag_i)
```

两张表：转移概率（前一个标签给出当前标签），发射概率（标签给出词）。用带拉普拉斯平滑的计数估计。用Viterbi算法（标签图上的动态规划）解码。

```python
import math


def train_hmm(train_examples, alpha=0.01):
    transitions = defaultdict(Counter)
    emissions = defaultdict(Counter)
    tags = set()
    vocab = set()

    for tokens, ts in train_examples:
        prev = "<BOS>"
        for token, tag in zip(tokens, ts):
            transitions[prev][tag] += 1
            emissions[tag][token.lower()] += 1
            tags.add(tag)
            vocab.add(token.lower())
            prev = tag
        transitions[prev]["<EOS>"] += 1

    return transitions, emissions, tags, vocab


def log_prob(table, given, key, smooth_denom, alpha):
    return math.log((table[given].get(key, 0) + alpha) / smooth_denom)


def viterbi(tokens, transitions, emissions, tags, vocab, alpha=0.01):
    tags_list = list(tags)
    n = len(tokens)
    V = [[0.0] * len(tags_list) for _ in range(n)]
    back = [[0] * len(tags_list) for _ in range(n)]

    for j, tag in enumerate(tags_list):
        em_denom = sum(emissions[tag].values()) + alpha * (len(vocab) + 1)
        tr_denom = sum(transitions["<BOS>"].values()) + alpha * (len(tags_list) + 1)
        tr = log_prob(transitions, "<BOS>", tag, tr_denom, alpha)
        em = log_prob(emissions, tag, tokens[0].lower(), em_denom, alpha)
        V[0][j] = tr + em
        back[0][j] = 0

    for i in range(1, n):
        for j, tag in enumerate(tags_list):
            em_denom = sum(emissions[tag].values()) + alpha * (len(vocab) + 1)
            em = log_prob(emissions, tag, tokens[i].lower(), em_denom, alpha)
            best_prev = 0
            best_score = -1e30
            for k, prev_tag in enumerate(tags_list):
                tr_denom = sum(transitions[prev_tag].values()) + alpha * (len(tags_list) + 1)
                tr = log_prob(transitions, prev_tag, tag, tr_denom, alpha)
                score = V[i - 1][k] + tr + em
                if score > best_score:
                    best_score = score
                    best_prev = k
            V[i][j] = best_score
            back[i][j] = best_prev

    last_best = max(range(len(tags_list)), key=lambda j: V[n - 1][j])
    path = [last_best]
    for i in range(n - 1, 0, -1):
        path.append(back[i][path[-1]])
    return [tags_list[j] for j in reversed(path)]
```

Brown语料库上的二元组HMM标注准确率约93%。从85%提升到93%主要得益于转移概率——模型学会了 `DET NOUN` 常见而 `NOUN DET` 罕见。

### 第3步：为什么现代标注器优于此

转移+发射概率是局部的。它们无法捕捉 `saw` 在“I bought a saw”中是名词，而在“I saw the movie”中是动词的区别。带有任意特征（词尾、词形、前后词、本身）的条件随机场（CRF）可达到约97%。双向LSTM-CRF或Transformer能达到98%以上。

此任务的上限由标注者分歧决定。宾夕法尼亚树库上的人工标注者一致率约为97%。超过98%的模型很可能已经过拟合测试集。

### 第4步：依存句法分析概述

全套依存句法从零构建超出本课范围，权威经典教材在Jurafsky和Martin著作中。两个传统家族：

- **基于转换（Transition-based）** 解析器（arc-eager, arc-standard）类似移进-归约解析器：读入词元，移进栈，应用归约动作构建边。贪心解码速度快。经典实现是MaltParser。现代神经版本是Chen和Manning的基于转换解析器。
- **基于图（Graph-based）** 解析器（Eisner算法，Dozat-Manning的双仿射）为每条可能头依赖边打分，找最大生成树。慢但准确度高。

生产环境多数直接调用spaCy：

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The cats were running at 3pm.")
for token in doc:
    print(f"{token.text:10s} tag={token.tag_:5s} pos={token.pos_:6s} dep={token.dep_:10s} head={token.head.text}")
```

```text
The        tag=DT    pos=DET    dep=det        head=cats
cats       tag=NNS   pos=NOUN   dep=nsubj      head=running
were       tag=VBD   pos=AUX    dep=aux        head=running
running    tag=VBG   pos=VERB   dep=ROOT       head=running
at         tag=IN    pos=ADP    dep=prep       head=running
3pm        tag=NN    pos=NOUN   dep=pobj       head=at
.          tag=.     pos=PUNCT  dep=punct      head=running
```

自下而上阅读 `dep` 列，即能获悉句子的语法结构。

## 使用它

每个生产级NLP库都集成了词性和依存解析器作为标准管道部分。

- **spaCy**（`en_core_web_sm` / `md` / `lg` / `trf`）。快速准确，集成了分词、命名实体识别（NER）、词形还原。使用 `token.tag_`（Penn标注），`token.pos_`（UD标注），`token.dep_`（依存关系）。
- **Stanford NLP（stanza）**。斯坦福继CoreNLP之后的工具包。支持60多种语言，属于学术级最新水平。
- **trankit**。基于Transformer，UD准确率高。
- **NLTK**。`pos_tag` 方法。可用但慢，较旧。适合教学。

### 2026年此技能仍重要的场景

- **词形还原。** 第一课表明词形还原必须用到词性标注。始终如此。
- **LLM输出结构化抽取。** 验证生成句子是否满足语法约束（如主谓一致、必需修饰成分）。
- **基于方面的情感分析。** 依存解析告诉你哪个形容词修饰哪个名词。
- **查询理解。** “movies directed by Wes Anderson starring Bill Murray”通过解析转化为结构化约束。
- **跨语言迁移。** UD标签和依存关系无语言依赖，支持对新语言的零样本结构分析。
- **低算力管道。** 无法部署Transformer时，词性+依存解析+地名词典依然能支持很远的应用。

## 部署

保存为 `outputs/skill-grammar-pipeline.md`：

```markdown
---
name: grammar-pipeline
description: Design a classical POS + dependency pipeline for a downstream NLP task.
version: 1.0.0
phase: 5
lesson: 07
tags: [nlp, pos, parsing]
---

Given a downstream task (information extraction, rewrite validation, query decomposition, lemmatization), you output:

1. Tagset to use. Penn Treebank for English-only legacy pipelines, Universal Dependencies for multilingual or cross-lingual.
2. Library. spaCy for most production, stanza for academic-grade multilingual, trankit for highest UD accuracy. Name the specific model ID.
3. Integration pattern. Show the 3-5 lines that call the library and consume the needed attributes (`.pos_`, `.dep_`, `.head`).
4. Failure mode to test. Noun-verb ambiguity (`saw`, `book`, `can`) and PP-attachment ambiguity are the classical traps. Sample 20 outputs and eyeball.

Refuse to recommend rolling your own parser. Building parsers from scratch is a research project, not an application task. Flag any pipeline that consumes POS tags without handling lowercase/uppercase variants as fragile.
```

## 练习

1. **简单。** 在一个小的带标注语料库（例如NLTK的Brown子集）上，用最频繁标记基线测量准确率。验证约85%的结果。
2. **中等。** 训练上述二元组HMM，报告每个标签的精确率和召回率。HMM最容易混淆哪些标签？
3. **困难。** 用spaCy依存解析从1000句样本中提取主谓宾三元组。在50个人工标注的三元组上评估。记录抽取失败的情况（常见于被动语态、并列结构和省略主语）。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| POS tag（词性标注） | 词的类型 | 语法类别。PTB 有 36 种；UD 有 17 种。 |
| Penn Treebank（宾州树库） | 标准标签集 | 英语专用。细粒度的动词时态和名词数。 |
| Universal Dependencies（通用依存） | 多语言标签集 | 比 PTB 粗糙；语言中立；跨语言工作时的默认选择。 |
| Dependency parse（依存句法分析） | 句子树 | 每个词有一个中心词，每条边都有语法关系。 |
| Viterbi（维特比算法） | 动态规划 | 给定发射概率和转移概率，找到概率最高的标签序列。 |

## 推荐阅读

- [Jurafsky and Martin — Speech and Language Processing，第 8 与 18 章](https://web.stanford.edu/~jurafsky/slp3/) — 词性标注和句法分析的权威教材。
- [Universal Dependencies 项目](https://universaldependencies.org/) — 每个多语言解析器使用的跨语言标签集和句库合集。
- [spaCy 语言特征指南](https://spacy.io/usage/linguistic-features) — `Token` 对象公开的每个属性的实用参考。
- [Chen and Manning (2014). A Fast and Accurate Dependency Parser using Neural Networks](https://nlp.stanford.edu/pubs/emnlp2014-depparser.pdf) — 将神经网络解析器推向主流的论文。
