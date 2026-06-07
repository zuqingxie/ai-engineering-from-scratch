# 命名实体识别（Named Entity Recognition）

> 抽取名字。看起来简单，直到你遇到模糊边界、嵌套实体和领域术语。

**类型：** 构建  
**语言：** Python  
**先决条件：** 第5阶段 · 02（BoW + TF-IDF），第5阶段 · 03（词向量）  
**时间：** 约75分钟

## 问题

"Apple sued Google over its iPhone search deal in the US." 有五个实体：Apple（ORG），Google（ORG），iPhone（PRODUCT），search deal（可选），US（GPE）。一个好的NER系统能正确提取所有实体及其类型。一个差的系统会漏掉iPhone，把苹果水果和苹果公司混淆，把“US”标为PERSON。

NER是每个结构化抽取流水线的主力。简历解析、合规日志扫描、医疗记录匿名化、搜索查询理解、聊天机器人响应的基础、法律合同抽取。你从没真正见过它，但一直依赖它。

本课将走经典路径（基于规则、隐马尔可夫模型、条件随机场）到现代方法（BiLSTM-CRF，再到Transformer架构）。每一步都解决前一步的特定局限。这个模式是重点。

## 概念

**BIO标注法**（或BILOU）将实体抽取转为序列标注问题。给每个token标注 `B-TYPE`（实体开头）、`I-TYPE`（实体内部）或`O`（非实体）。

```text
Apple    B-ORG
sued     O
Google   B-ORG
over     O
its      O
iPhone   B-PRODUCT
search   O
deal     O
in       O
the      O
US       B-GPE
.        O
```

多词实体链：`New B-GPE`，`York I-GPE`，`City I-GPE`。理解BIO的模型可以抽取任意跨度实体。

架构进展：

- **基于规则。** 正则表达式 + 词典查找。已知实体高精度，新实体覆盖率为零。  
- **HMM。** 隐马尔可夫模型。给定标签有发射概率，标签之间有转移概率。Viterbi解码。基于标注数据训练。  
- **CRF。** 条件随机场。类似HMM，但判别式，可以混合任意特征（词形、大小写、邻近词）。2026年仍是低资源部署的经典生产利器。  
- **BiLSTM-CRF。** 用神经特征代替手工特征。LSTM双向读句子，CRF层确保标签序列一致性。  
- **基于Transformer。** 微调带token分类头的BERT。准确率最高。计算成本最大。  

## 构建步骤

### 步骤1：BIO标注辅助函数

```python
def spans_to_bio(tokens, spans):
    labels = ["O"] * len(tokens)
    for start, end, label in spans:
        labels[start] = f"B-{label}"
        for i in range(start + 1, end):
            labels[i] = f"I-{label}"
    return labels


def bio_to_spans(tokens, labels):
    spans = []
    current = None
    for i, label in enumerate(labels):
        if label.startswith("B-"):
            if current:
                spans.append(current)
            current = (i, i + 1, label[2:])
        elif label.startswith("I-") and current and current[2] == label[2:]:
            current = (current[0], i + 1, current[2])
        else:
            if current:
                spans.append(current)
                current = None
    if current:
        spans.append(current)
    return spans
```

```python
>>> tokens = ["Apple", "sued", "Google", "over", "iPhone", "sales", "."]
>>> labels = ["B-ORG", "O", "B-ORG", "O", "B-PRODUCT", "O", "O"]
>>> bio_to_spans(tokens, labels)
[(0, 1, 'ORG'), (2, 3, 'ORG'), (4, 5, 'PRODUCT')]
```

### 步骤2：手工特征

对于经典（非神经）NER，特征是关键。有用的特征包括：

```python
def token_features(token, prev_token, next_token):
    return {
        "lower": token.lower(),
        "is_upper": token.isupper(),
        "is_title": token.istitle(),
        "has_digit": any(c.isdigit() for c in token),
        "suffix_3": token[-3:].lower(),
        "shape": word_shape(token),
        "prev_lower": prev_token.lower() if prev_token else "<BOS>",
        "next_lower": next_token.lower() if next_token else "<EOS>",
    }


def word_shape(word):
    out = []
    for c in word:
        if c.isupper():
            out.append("X")
        elif c.islower():
            out.append("x")
        elif c.isdigit():
            out.append("d")
        else:
            out.append(c)
    return "".join(out)
```

`word_shape("iPhone")`返回`xXxxxx`，`word_shape("USA-2024")`返回`XXX-dddd`。大小写模式对专有名词识别非常有用。

### 步骤3：简单基于规则+词典的基线

```python
ORG_GAZETTEER = {"Apple", "Google", "Microsoft", "OpenAI", "Meta", "Amazon", "Netflix"}
GPE_GAZETTEER = {"US", "USA", "UK", "India", "Germany", "France"}
PRODUCT_GAZETTEER = {"iPhone", "Android", "Windows", "ChatGPT", "Claude"}


def rule_based_ner(tokens):
    labels = []
    for token in tokens:
        if token in ORG_GAZETTEER:
            labels.append("B-ORG")
        elif token in GPE_GAZETTEER:
            labels.append("B-GPE")
        elif token in PRODUCT_GAZETTEER:
            labels.append("B-PRODUCT")
        else:
            labels.append("O")
    return labels
```

生产环境词典有数百万条目，来自维基百科和DBpedia，覆盖率极好，但歧义消解（例：公司Apple vs 水果Apple）很差。这就是统计模型胜出的原因。

### 步骤4：CRF步骤（简略示意，非完整实现）

50行完成的纯CRF源码没有概率论基础难以理解，推荐用`sklearn-crfsuite`：

```python
import sklearn_crfsuite

def to_features(tokens):
    out = []
    for i, tok in enumerate(tokens):
        prev = tokens[i - 1] if i > 0 else ""
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
        out.append({
            "word.lower()": tok.lower(),
            "word.isupper()": tok.isupper(),
            "word.istitle()": tok.istitle(),
            "word.isdigit()": tok.isdigit(),
            "word.suffix3": tok[-3:].lower(),
            "word.shape": word_shape(tok),
            "prev.word.lower()": prev.lower(),
            "next.word.lower()": nxt.lower(),
            "BOS": i == 0,
            "EOS": i == len(tokens) - 1,
        })
    return out


crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True)
X_train = [to_features(s) for s in sentences_tokenized]
crf.fit(X_train, bio_labels_train)
```

`c1`和`c2`是L1和L2正则化参数。设置`all_possible_transitions=True`让模型学会非法序列（如`O`后跟`I-ORG`）不可能，这就是CRF如何在不写约束的情况下保证BIO一致性的方式。

### 步骤5：BiLSTM-CRF带来的提升

特征改为自动学习。输入是词向量（GloVe或fastText）。LSTM左右双向读取，连接隐藏表征后送入CRF输出层。CRF确保标签序列一致，LSTM替代了手工特征。

```python
import torch
import torch.nn as nn


class BiLSTM_CRF_Head(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_labels):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, n_labels)

    def forward(self, token_ids):
        e = self.embed(token_ids)
        h, _ = self.lstm(e)
        emissions = self.fc(h)
        return emissions
```

CRF层可以用`torchcrf.CRF`（pip安装pytorch-crf）。和手工特征CRF相比，提升可量化，但除非你有数万标注句子，否则提升有限。

## 使用方式

spaCy开箱即用的生产级NER：

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple sued Google over its iPhone search deal in the US.")
for ent in doc.ents:
    print(f"{ent.text:20s} {ent.label_}")
```

```text
Apple                ORG
Google               ORG
iPhone               ORG
US                   GPE
```

注意iPhone被标记为ORG而非PRODUCT——spaCy小模型产品实体覆盖不好，使用大模型（`en_core_web_lg`）效果更佳，Transformer模型（`en_core_web_trf`）表现更优。

Hugging Face的BERT NER例子：

```python
from transformers import pipeline

ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
print(ner("Apple sued Google over its iPhone in the US."))
```

```text
[{'entity_group': 'ORG', 'word': 'Apple', ...},
 {'entity_group': 'ORG', 'word': 'Google', ...},
 {'entity_group': 'MISC', 'word': 'iPhone', ...},
 {'entity_group': 'LOC', 'word': 'US', ...}]
```

`aggregation_strategy="simple"`会合并相邻B-X、I-X的token成一个实体。没有这个参数你会得到token级标签，得自己做合并。

### 基于大语言模型（LLM）的NER（2026年的选项）

零样本和少样本的LLM NER在很多领域已能和微调模型媲美，且在标注数据稀少时表现大幅领先。

- **零样本提示。** 给LLM列出实体类型和示例schema，要求输出JSON。开箱即用，新领域准确度中等。  
- **ZeroTuneBio式提示。** 任务分解为候选抽取 → 意义解释 → 判断 → 复核。多阶段提示（非一次性）显著提升生物医学NER准确率，同样适用于法律、金融和科学领域。  
- **带RAG的动态提示。** 每次推理从小规模标注集检索最相似的例子，动态构建少样本提示。在2026年基准测试中，这能使GPT-4生物医学NER F1提高11-12%。  
- **按实体类型分解。** 长文档里一次提取所有实体类型会随着长度增长丢失召回。对每种实体类型单独提取，推理成本较高，准确率显著提升。临床记录和法律合同通常用此模式。

2026年生产推荐：先用LLM零样本基线，再考虑收集训练数据。F1通常足够好，无需微调。

### 经典NER仍然占优的场景

即使有LLM，经典NER胜出当：

- 延迟预算低于50毫秒。  
- 有数千标注样本需达到98%以上F1。  
- 领域有稳定本体，预训练CRF或BiLSTM转移表现好。  
- 监管要求使用本地部署非生成模型。  

### 局限和不足

- **领域迁移。** 在法律合同上用CoNLL训练的NER表现不及词典。需针对领域微调。  
- **嵌套实体。** “Bank of America Tower”同时是ORG和FACILITY，标准BIO无法表示重叠实体。需嵌套NER（多轮或基于跨度的模型）。  
- **长实体。** “United States Federal Deposit Insurance Corporation.” 词级模型有时会拆分。用`aggregation_strategy`或后处理。  
- **稀疏类型。** 医疗NER标签如DRUG_BRAND、ADVERSE_EVENT、DOSE，通用模型不识别。Scispacy和BioBERT是起点。

## 部署

保存为 `outputs/skill-ner-picker.md`：

```markdown
---
name: ner-picker
description: 为给定的抽取任务选择合适的 NER 方法。
version: 1.0.0
phase: 5
lesson: 06
tags: [nlp, ner, extraction]
---

给定任务描述（领域、标签集、语言、延迟、数据量），输出：

1. 方法。基于规则 + 地名词典（gazetteer）、CRF、BiLSTM-CRF，或 Transformer（Transformer 架构）微调。
2. 起始模型。命名（spaCy 模型 ID、Hugging Face 检查点 ID，或“自定义，重新训练”）。
3. 标注策略。BIO、BILOU，或基于跨度。用一句话说明理由。
4. 评估。使用 `seqeval`。始终报告实体级别的 F1 分数（不是 token 级别）。

除非用户已有预训练的领域模型，否则拒绝推荐在少于 500 个标注样本上微调 Transformer。标记嵌套实体需要基于跨度或多遍模型。如果用户提到“生产规模”且标签未改动自 CoNLL-2003，则要求进行地名词典审核。
```

## 练习

1. **简单。** 实现 `bio_to_spans`（与 `spans_to_bio` 相反），并在 10 句子上验证往返一致性。
2. **中等。** 在 CoNLL-2003 英文 NER 数据集上训练上述 sklearn-crfsuite CRF。使用 `seqeval` 报告实体级 F1。典型结果：约 84 F1。
3. **困难。** 在领域特定的 NER 数据集（医疗、法律或金融）上对 `distilbert-base-cased` 进行微调。与 spaCy 小模型比较。记录数据泄露检查并撰写令你惊讶的内容。

## 关键词

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| NER | 抽取名字 | 标注 token 跨度的类型（PERSON、ORG、GPE、DATE 等）。 |
| BIO | 标注方案 | `B-X` 开始，`I-X` 继续，`O` 表示外部。 |
| BILOU | 更好的 BIO | 增加 `L-X`（最后），`U-X`（单元）以明确边界。 |
| CRF | 结构化分类器 | 建模标签间的转换，不仅是发射。强制有效序列。 |
| 嵌套 NER | 重叠实体 | 一个跨度是不同于其子跨度的实体。BIO 无法表达。 |
| 实体级 F1 | 正确的 NER 指标 | 预测跨度必须与真实跨度完全匹配。Token 级 F1 夸大准确率。 |

## 深入阅读

- [Lample et al. (2016). Neural Architectures for Named Entity Recognition](https://arxiv.org/abs/1603.01360) — BiLSTM-CRF 论文。权威。
- [Devlin et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) — 引入了成为标准的 token 分类模式。
- [spaCy linguistic features — named entities](https://spacy.io/usage/linguistic-features#named-entities) — `Doc.ents` 和 `Span` 上每个属性的实用参考。
- [seqeval](https://github.com/chakki-works/seqeval) — 正确的评测库。务必使用。
