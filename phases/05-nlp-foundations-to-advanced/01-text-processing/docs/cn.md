# 文本处理 — 分词（Tokenization）、词干提取（Stemming）、词形还原（Lemmatization）

> 语言是连续的。模型是离散的。预处理是桥梁。

**类型：** 构建  
**语言：** Python  
**先决条件：** 第二阶段 · 14（朴素贝叶斯）  
**时间：** ~45分钟

## 问题

模型无法理解“The cats were running.” 它只能读取整数。

每个自然语言处理（NLP）系统一开始都会面对三个相同的问题：一个词从哪里开始？词的词根是什么？当对“run”、“running”、“ran”视为同一词（当有帮助时）或不同词（当没有帮助时），我们如何处理？

分词做不好模型就学垃圾。如果你的分词器将 `don't` 视为一个token，但 `do n't` 视为两个，训练分布就分裂了。如果你的词干提取器将 `organization` 和 `organ` 合并为同一词干，主题模型就坏了。如果你的词形还原器需要词性（POS）上下文但你没传递，动词会被当作名词处理。

本课从零构建这三步预处理，然后展示NLTK和spaCy是如何做相同工作的，方便你理解权衡。

## 概念

三个操作。各有职责和失败模式。

**分词（Tokenization）** 将字符串拆成token。“Token”特意保持模糊，因为合适的粒度取决于任务。经典NLP用词级（word-level），Transformer 用子词（subword），无空格语言用字符级（character）。

**词干提取（Stemming）** 用规则砍后缀。快速、激进、简单。`running -> run`。`organization -> organ`。后者是失败模式。

**词形还原（Lemmatization）** 用语法知识化简词到词典形。慢、准确，需要查表或形态分析器。`ran -> run`（需知道“ran”是“run”的过去式）。`better -> good`（需知道比较级形式）。

经验法则：当速度重要且能容忍噪声时用词干提取（搜索索引、粗分类）。当语义重要时用词形还原（问答、语义搜索、用户可见场景）。

## 构建它

### 步骤1：基于正则的词分词器

最简单实用的分词器对非字母数字字符分割，标点单独作为token。非完美，也非最终版，但一行代码可运行。

```python
import re

def tokenize(text):
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[^\sA-Za-z0-9]", text)
```

三个模式，优先级依次：带可选内部撇号的单词（`don't`, `it's`）；纯数字；单个非空白非字母数字字符作为独立token（标点）。

```python
>>> tokenize("The cats weren't running at 3pm.")
['The', 'cats', "weren't", 'running', 'at', '3', 'pm', '.']
```

注意失败模式：`3pm`被拆为`['3', 'pm']`，因为在字母串和数字串之间交替。对于多数任务已足够。URL、邮箱、标签都破坏。生产环境需在通用正则前加入更复杂模式。

### 步骤2：Porter词干提取器（仅第1a步）

完整Porter算法有五阶段规则。仅1a步覆盖最频繁英文后缀并演示方法。

```python
def stem_step_1a(word):
    if word.endswith("sses"):
        return word[:-2]
    if word.endswith("ies"):
        return word[:-2]
    if word.endswith("ss"):
        return word
    if word.endswith("s") and len(word) > 1:
        return word[:-1]
    return word
```

```python
>>> [stem_step_1a(w) for w in ["caresses", "ponies", "caress", "cats"]]
['caress', 'poni', 'caress', 'cat']
```

按顺序读规则。`ies -> i`规则造就`ponies -> poni`，非`pony`。真正Porter有1b步可修正。规则互相竞争，先匹配规则胜，顺序比单条规则重要。

### 步骤3：基于查表的词形还原器

真正的词形还原需要形态学。教学版用一个小词形表和兜底策略。

```python
LEMMA_TABLE = {
    ("running", "VERB"): "run",
    ("ran", "VERB"): "run",
    ("runs", "VERB"): "run",
    ("better", "ADJ"): "good",
    ("best", "ADJ"): "good",
    ("cats", "NOUN"): "cat",
    ("cat", "NOUN"): "cat",
    ("were", "VERB"): "be",
    ("was", "VERB"): "be",
    ("is", "VERB"): "be",
}

def lemmatize(word, pos):
    key = (word.lower(), pos)
    if key in LEMMA_TABLE:
        return LEMMA_TABLE[key]
    if pos == "VERB" and word.endswith("ing"):
        return word[:-3]
    if pos == "NOUN" and word.endswith("s"):
        return word[:-1]
    return word.lower()
```

```python
>>> lemmatize("running", "VERB")
'run'
>>> lemmatize("cats", "NOUN")
'cat'
>>> lemmatize("better", "ADJ")
'good'
>>> lemmatize("watched", "VERB")
'watched'
```

最后一例是关键教学点。`watched`没在表内，兜底仅处理 `ing`。真正的词形还原覆盖 `ed`、不规则动词、比较形容词、带音变复数（`children -> child`）。生产系统用 WordNet、spaCy形态分析器或完整形态分析器。

### 步骤4：将它们组合成管道

```python
def preprocess(text, pos_tagger=None):
    tokens = tokenize(text)
    stems = [stem_step_1a(t.lower()) for t in tokens]
    tags = pos_tagger(tokens) if pos_tagger else [(t, "NOUN") for t in tokens]
    lemmas = [lemmatize(word, pos) for word, pos in tags]
    return {"tokens": tokens, "stems": stems, "lemmas": lemmas}
```

缺少的是词性标注器。第五阶段 · 07（词性标注）将构建一个。目前全部默认为 `NOUN` 并说明限制。

## 使用它

NLTK和spaCy提供生产版本，几行代码即可。

### NLTK

```python
import nltk
nltk.download("punkt_tab")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger_eng")

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag

text = "The cats were running."
tokens = word_tokenize(text)
stems = [PorterStemmer().stem(t) for t in tokens]
lemmatizer = WordNetLemmatizer()
tagged = pos_tag(tokens)


def nltk_pos_to_wordnet(tag):
    if tag.startswith("V"):
        return "v"
    if tag.startswith("J"):
        return "a"
    if tag.startswith("R"):
        return "r"
    return "n"


lemmas = [lemmatizer.lemmatize(t, nltk_pos_to_wordnet(tag)) for t, tag in tagged]
```

`word_tokenize`处理缩写、Unicode、正则不及的边缘。`PorterStemmer`执行完整五步。`WordNetLemmatizer`需要将NLTK的Penn Treebank词性标注转换为WordNet缩写，上面的转换代码是多数教程省略的重点。

### spaCy

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The cats were running.")

for token in doc:
    print(token.text, token.lemma_, token.pos_)
```

```text
The      the     DET
cats     cat     NOUN
were     be      AUX
running  run     VERB
.        .       PUNCT
```

spaCy将整个处理管道封装在 `nlp(text)` 背后。分词、词性标注和词形还原都执行。相比NLTK规模更大时速度快，且准确率高。缺点是难以替换单个组件。

### 何时选用哪种

| 场景 | 选择 |
|-----------|------|
| 教学，研究，可替换组件 | NLTK |
| 生产，多语言，重速度 | spaCy |
| Transformer流水线（你反正要用模型自带分词器） | 使用 `tokenizers` / `transformers`，跳过传统预处理 |

### 两个没人提醒你的失败模式

大多数教程教授算法即止。真实预处理管道会被两件事坑，而它们几乎未被涉及。

**可复现性漂移。** NLTK和spaCy版本间分词和词形还原行为变。spaCy 2.x把`['do', "n't"]`分为二，3.x可能合成`["don't"]`。模型训练用的是一个分布，推理用的是另一种。准确率静悄悄变差原因不明。请在`requirements.txt`里锁定库版本。写预处理回归测试，对20条样例定死期望分词，升级时跑。

**训练 / 推理不匹配。** 训练阶段用激进预处理（小写、停用词过滤、词干提取），推理阶段用原始用户输入，性能坠落。这是生产NLP中最常见失败。训练用哪套预处理，推理就跑哪套。把预处理功能封装入模型包，不要靠服务团队重写Notebook。

## 部署它

一个重用的提示（prompt），帮助工程师选预处理策略，无需读三本教科书。

保存为 `outputs/prompt-preprocessing-advisor.md`:

```markdown
---
name: preprocessing-advisor
description: 为NLP任务推荐分词、词干提取和词形还原方案。
phase: 5
lesson: 01
---

你为经典NLP预处理提供建议。给定任务描述，输出：

1. 分词方案选择（正则、NLTK word_tokenize、spaCy或Transformer分词器），并说明原因。
2. 是否进行词干提取、词形还原，两者都做或都不做，说明理由。
3. 具体库调用，函数名。如涉及NLTK词性标签转换，引用转换代码。
4. 用户应测试的一个失败模式。

拒绝对用户可见文本推荐词干提取。拒绝无词性标签推荐词形还原。若非英语输入，标记需用不同管道。
```

## 练习

1. **简单。** 扩展`tokenize`函数，使其将URL保持为单一token。测试：`tokenize("Visit https://example.com today.")`应产生一个URL token。
2. **中等。** 实现Porter算法第1b步。若词含元音且以 `ed` 或 `ing` 结尾，则移除。处理双辅音规则（`hopping -> hop`，非`hopp`）。
3. **困难。** 构建一个词形还原器，使用WordNet做查表，WordNet无条目时退而用你的Porter词干提取器。对带标注语料测准确率，对比仅用WordNet和仅用Porter。

## 关键词

| 术语 | 一般说法 | 实际含义 |
|------|----------|----------|
| Token | 一个词 | 模型消耗的单元。可以是词、子词、字符或字节。 |
| Stem | 词根 | 基于规则的后缀去除结果，不一定是实际单词。 |
| Lemma | 词典形 | 你查词典时的词形，正确计算需上下文语法。 |
| POS tag | 词性标签 | 类别，如NOUN、VERB、ADJ。准确词形还原需要。 |
| Morphology | 形态学规则 | 词形根据时态、数、格变化的规则。词形还原依赖它。 |

## 延伸阅读

- [Porter, M. F. (1980). An algorithm for suffix stripping](https://tartarus.org/martin/PorterStemmer/def.txt) — 原始论文，五页，至今最清晰解释。
- [spaCy 101 — linguistic features](https://spacy.io/usage/linguistic-features) — 一个真实管道如何设计。
- [NLTK book, chapter 3](https://www.nltk.org/book/ch03.html) — 你未曾考虑的分词边缘情况。
