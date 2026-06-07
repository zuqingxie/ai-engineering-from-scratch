# 情感分析

> 经典的自然语言处理（NLP）任务。你需要了解的大多数经典文本分类知识都在这里体现。

**类型：** 构建  
**编程语言：** Python  
**先修知识：** 第5阶段 · 02（词袋模型（BoW）+ TF-IDF），第2阶段 · 14（朴素贝叶斯）  
**时间：** 约75分钟

## 问题

“食物不怎么样。” 是积极的还是消极的？

情感分析听起来很简单。评论者说他们喜欢或不喜欢某物，给句子打标签。它成为经典NLP任务的原因是每个看似简单的案例背后都隐藏着难题。否定词会改变意思。讽刺会颠倒意思。尽管出现两个负面词，“not bad at all” 实际上是积极的。表情符号携带的信息比周围文字还多。领域词汇也很重要（音乐评论中的 `tight` 和时尚评论中的 `tight` 意思不同）。

情感分析是经典NLP的实践实验室。理解了每个朴素基线为何会有特定失败模式，就理解了为何每种更复杂的模型被发明出来。本课将从零构建朴素贝叶斯基线，添加逻辑回归，并讲解使得生产环境中的情感分析成为合规级难题的陷阱。

## 概念

经典情感分析是两步法：

1. **表示。** 将文本转换为特征向量。词袋（BoW）、TF-IDF 或 n-gram。
2. **分类。** 在带标签的样本上拟合线性模型（朴素贝叶斯、逻辑回归、支持向量机（SVM））。

朴素贝叶斯是最简单有效的模型。假设给定标签后所有特征相互独立。根据计数估计 `P(word | positive)` 和 `P(word | negative)`。推断时将概率相乘。虽然“朴素”的独立性假设明显错误，但结果却异常强大。原因在于，稀疏的文本特征和适中的数据量下，分类器关心的是每个词更偏向哪一边，而不是其程度。

逻辑回归修正了独立性假设。它为每个特征学习一个权重，包括负权重。短语 `not good` 作为bigram特征会被赋予负权重，而朴素贝叶斯无法对未标注过的bigram赋值。

## 构建

### 第一步：一个真实的小型数据集

```python
POSITIVE = [
    "absolutely loved this movie",
    "beautiful cinematography and a great story",
    "one of the best films of the year",
    "brilliant acting from the lead",
    "heartwarming and funny",
]

NEGATIVE = [
    "boring and far too long",
    "not worth your time",
    "the plot made no sense",
    "terrible acting, awful script",
    "i want my two hours back",
]
```

刻意小样本。实际应用需要成千上万的样本（IMDb、SST-2、Yelp情感极性）。数学原理完全相同。

### 第二步：从零实现多项式朴素贝叶斯

```python
import math
from collections import Counter


def train_nb(docs_by_class, vocab, alpha=1.0):
    class_priors = {}
    class_word_probs = {}
    total_docs = sum(len(d) for d in docs_by_class.values())

    for cls, docs in docs_by_class.items():
        class_priors[cls] = len(docs) / total_docs
        counts = Counter()
        for doc in docs:
            for token in doc:
                counts[token] += 1
        total = sum(counts.values()) + alpha * len(vocab)
        class_word_probs[cls] = {
            w: (counts[w] + alpha) / total for w in vocab
        }
    return class_priors, class_word_probs


def predict_nb(doc, class_priors, class_word_probs):
    scores = {}
    for cls in class_priors:
        s = math.log(class_priors[cls])
        for token in doc:
            if token in class_word_probs[cls]:
                s += math.log(class_word_probs[cls][token])
        scores[cls] = s
    return max(scores, key=scores.get)
```

加法平滑（alpha=1.0）是拉普拉斯平滑。若没有它，某类别中未见过的词概率为零，取对数时会爆炸。实践中常用 `alpha=0.01`，教学默认用 `alpha=1.0`。

### 第三步：从零实现逻辑回归

```python
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def train_lr(X, y, epochs=500, lr=0.05, l2=0.01):
    n_features = X.shape[1]
    w = np.zeros(n_features)
    b = 0.0
    for _ in range(epochs):
        logits = X @ w + b
        preds = sigmoid(logits)
        err = preds - y
        grad_w = X.T @ err / len(y) + l2 * w
        grad_b = err.mean()
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b


def predict_lr(X, w, b):
    return (sigmoid(X @ w + b) >= 0.5).astype(int)
```

L2 正则化很重要。文本特征稀疏；无L2会让模型记忆训练样本。默认从 `0.01` 开始调优。

### 第四步：处理否定（失败模式）

考虑“not good”和“not bad”。词袋分类器看到 `{not, good}` 和 `{not, bad}`，会根据训练中出现次数多少学习权重。bigram分类器看到 `not_good` 和 `not_bad`，学习它们为不同特征，通常效果够用。

更粗糙但有效的方式（当没有bigram时）：**否定词范围标记**。对否定词后的词直到下一个标点符号前，加前缀 `NOT_`。

```python
NEGATION_WORDS = {"not", "no", "never", "nor", "none", "nothing", "neither"}
NEGATION_TERMINATORS = {".", "!", "?", ",", ";"}


def apply_negation(tokens):
    out = []
    negate = False
    for token in tokens:
        if token in NEGATION_TERMINATORS:
            negate = False
            out.append(token)
            continue
        if token in NEGATION_WORDS:
            negate = True
            out.append(token)
            continue
        out.append(f"NOT_{token}" if negate else token)
    return out
```

```python
>>> apply_negation(["not", "good", "at", "all", ".", "but", "funny"])
['not', 'NOT_good', 'NOT_at', 'NOT_all', '.', 'but', 'funny']
```

现在 `good` 和 `NOT_good` 是不同特征。分类器可以赋予相反权重。3行预处理，情感基准测试上准确率有明显提升。

### 第五步：重要的评估指标

仅准确率在类别不平衡时具有误导性。真实情感语料库通常70%-80%为正或负类；单一直观多数类别分类器就能达到80%准确率，毫无价值。必须报告以下所有指标：

- **分类别精确率（precision）和召回率（recall）。** 每个类别一对，计算宏平均得到一个兼顾类别平衡的数字。
- **宏平均F1（Macro-F1，主要指标用于不平衡数据）。** 是各类别F1的算术平均，权重均等。不平衡时用它替代准确率。
- **加权F1（Weighted-F1，备用指标）。** 与宏平均类似，但根据类别频率加权。业务中类别不平衡有意义时同时报告。
- **混淆矩阵（Confusion matrix）。** 原始计数。报任何标量指标前务必查看，可揭示模型混淆的类别对。
- **分类别错误样本。** 每个类别抽取5个错误预测样本。阅读它们。没有什么比看实际错误更重要。

极度不平衡数据（>95-5比例）应报告 **AUROC** 和 **AUPRC** 替代准确率。AUPRC 对少数类更敏感，通常是你关注的（垃圾邮件、欺诈、罕见情感）。

**常见坑。** 在不平衡数据上报告微平均F1（micro-F1）而非宏平均F1，会得到被多数类主导的假高分。宏平均F1迫使你关注少数类表现。

```python
def evaluate(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "precision": precision, "recall": recall, "f1": f1}
```

## 使用

scikit-learn六行代码即可正确完成：

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words=None)),
    ("clf", LogisticRegression(C=1.0, max_iter=1000)),
])
pipe.fit(X_train, y_train)
print(pipe.score(X_test, y_test))
```

三点注意。`stop_words=None` 保留否定词。`ngram_range=(1, 2)` 加入bigram使 `not_good` 成为特征。`sublinear_tf=True` 抑制重复词项频率。这三个标志使得SST-2基线准确率从75%提升到85%。

### 什么时候使用 Transformer（Transformer 架构）

- 讽刺（sarcasm）检测。经典模型在这里普遍失败。
- 长文本评论，情感中途变化。
- 面向方面的情感分析（aspect-based sentiment）。例如“相机很好，但是电池很糟。” 需要把情感归因到具体方面。只能用Transformer或结构化输出模型。
- 非英语、低资源语言。多语言BERT提供零样本基线。

如需以上任何功能，请跳至第7阶段（Transformer 深度学习）。否则，朴素贝叶斯或逻辑回归 + TF-IDF + bigram + 否定处理，是你2026年生产线基线。

### 可复现性陷阱（再次强调）

重新训练情感模型是常规，重新评估很难。论文中报告的准确率基于特定数据集划分、特定预处理和特定分词器。如果不使用完全相同的流水线与你的新模型对比基线，会得到误导结果。始终在你的流水线上重新生成基线结果，而非直接用论文数字。

## 部署

保存为 `outputs/prompt-sentiment-baseline.md`：

```markdown
---
name: sentiment-baseline
description: 为新数据集设计一个情感分析基线。
phase: 5
lesson: 05
---

给定数据集描述（领域、语言、规模、标签粒度、延迟预算），输出内容：

1. 特征抽取方案。指定分词器、n-gram范围、停用词策略（通常保留）、否定处理（范围前缀或bigram）。
2. 分类器。基线用朴素贝叶斯，生产环境用逻辑回归，有讽刺/方面/跨语种需求才用Transformer。
3. 评估计划。报告精确率、召回率、F1、混淆矩阵和分类别错误样本（非仅标量）。
4. 部署后监控一个失败模式。领域漂移和讽刺是前两位。

拒绝推荐为情感任务删除停用词。拒绝在类别不平衡（如90%为正）时仅报告准确率。提示富含子词的语言应使用FastText或Transformer词嵌入，而非词级TF-IDF。
```

## 练习

1. **简单。** 在scikit-learn流水线中加入 `apply_negation` 作为预处理步骤，并测量小型情感数据集上的F1提升。
2. **中等。** 实现类别加权逻辑回归（传参数 `class_weight="balanced"` 给scikit-learn，或自己推导梯度）。测量合成90:10类别不平衡上的影响。
3. **困难。** 构建讽刺检测器：在情感模型预测残差上训练第二分类器。记录实验方案。当准确率低于随机概率（2类讽刺约为50%）时提醒用户多数首次尝试都会跌落至此水平。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| Polarity（极性） | 积极或消极 | 二元标签；有时扩展为中性或细粒度（5星等级）。 |
| Aspect-based sentiment（基于方面的情感） | 每个方面的极性 | 将情感归因于文本中提到的具体实体或属性。 |
| Negation scoping（否定范围） | 反转附近的词元 | 在“not”后的词元前加 `NOT_`，直到标点符号。 |
| Laplace smoothing（拉普拉斯平滑） | 给计数加1 | 防止朴素贝叶斯中零概率特征。 |
| L2 regularization（L2正则化） | 缩小权重 | 在损失函数中加上 `lambda * sum(w^2)`。对于稀疏文本特征至关重要。 |

## 延伸阅读

- [Pang 和 Lee (2008). Opinion Mining and Sentiment Analysis（观点挖掘与情感分析）](https://www.cs.cornell.edu/home/llee/opinion-mining-sentiment-analysis-survey.html) — 基础性综述。篇幅较长，但前四节涵盖了所有经典内容。
- [Wang 和 Manning (2012). Baselines and Bigrams: Simple, Good Sentiment and Topic Classification（基线与二元组：简单且有效的情感和主题分类）](https://aclanthology.org/P12-2018/) — 展示了二元组 + 朴素贝叶斯在短文本上的强大表现。
- [scikit-learn 文本特征提取文档](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) — `CountVectorizer`、`TfidfVectorizer` 及所有参数调整的参考。
