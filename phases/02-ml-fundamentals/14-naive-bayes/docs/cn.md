# 朴素贝叶斯（Naive Bayes）

> “朴素”的假设是错误的，但它依然有效。这就是它的魅力所在。

**类型：**构建  
**语言：**Python  
**先决条件：**第二阶段，第01-07课（分类，贝叶斯定理）  
**时间：**约75分钟

## 学习目标

- 从头实现带拉普拉斯平滑（Laplace smoothing）的多项式朴素贝叶斯（Multinomial Naive Bayes）用于文本分类  
- 解释为什么朴素独立性假设在数学上是错误的，但在实际中能产生正确的类别排名  
- 比较多项式、伯努利（Bernoulli）和高斯（Gaussian）朴素贝叶斯变体，并为给定特征类型选择合适的模型  
- 在高维稀疏数据上评估朴素贝叶斯与逻辑回归（logistic regression）的性能，并解释偏差-方差权衡  

## 问题描述

你需要对文本进行分类。比如将电子邮件分为垃圾邮件或非垃圾邮件。将客户评论分为正面或负面。将支持工单分类。你有成千上万个特征（每个单词一个特征）和有限的训练数据。

大多数分类器都会在这里失灵。逻辑回归需要足够多的样本来可靠估计数千个权重。决策树一次只基于一个词切分，容易过拟合。高维中（如10,000维）的KNN毫无意义，因为每个点与其他点的距离几乎相等。

朴素贝叶斯能够处理这种情况。它做了一个数学上错误的假设（在给定类别的条件下，每个特征相互独立），但在文本分类上尤其是训练数据较少时，依然优于“更聪明”的模型。训练只需对数据进行一次遍历。它能扩展到数百万特征。它能输出概率估计（虽然由于独立性假设，这些估计往往校准得不好）。

理解为什么错误的假设能产生良好的预测，能够教会你机器学习的一个基本理念：最好的模型不一定是最正确的，而是对你的数据具有最佳偏差-方差权衡的模型。

## 概念

### 贝叶斯定理（快速回顾）

贝叶斯定理转换条件概率：

```text
P(class | features) = P(features | class) * P(class) / P(features)
```

我们想求的是 `P(class | features)` —— 给定文档中单词，文档属于某类别的概率。它可从以下计算得出：  
- `P(features | class)` —— 在该类别的文档中看到这些词的可能性  
- `P(class)` —— 类别的先验概率（比如垃圾邮件总体有多常见）  
- `P(features)` —— 证据，相同于所有类别，因此比较时可忽略  

概率最大的类别获胜。

### 朴素独立性假设

准确计算 `P(features | class)` 需要估计所有特征的联合概率。对于10,000个单词的词汇表，你需要估计2^10,000种可能组合的分布，完全不可能。

朴素假设是：对于给定类别，每个特征条件独立。

```text
P(w1, w2, ..., wn | class) = P(w1 | class) * P(w2 | class) * ... * P(wn | class)
```

你不需要估计一个巨大复杂的联合分布，而是估计n个简单的单特征分布，每个只需计数。

这个假设显然是错的。“machine”和“learning”这两个词在任何文档中都不是独立的。但分类器不需要准确的概率估计，只需要正确的排序——哪个类概率最高。独立假设会带来系统性误差，但这些误差对所有类别影响相似，因此排名仍然正确。

### 为什么它依然有效

三个原因： 

1. **排名优于校准。** 分类只需要排名最高的类别是正确的。即使P(垃圾邮件)=0.99999而真实概率是0.7，分类器仍会正确选中垃圾邮件。我们不需要准确的概率，只要正确的赢家。

2. **高偏差，低方差。** 独立假设是强先验。它强约束模型，防止过拟合。在有限训练数据下，略微错误但稳定的模型，胜过理论上正确但极不稳定的模型。这就是偏差-方差权衡的体现。

3. **特征冗余抵消。** 相关特征提供冗余证据。分类器会重复计数这些证据，但正确类别也被重复计数。如果“machine”和“learning”总是一起出现，它们都为“科技”类别提供证据。朴素贝叶斯两者都计数，但都计给正确类别。

第四个实际因素：朴素贝叶斯极快。训练只需通过数据计数一次，预测是矩阵乘法。可在几秒内训练百万文档。这速度让你能更快迭代，尝试更多特征集合，做更多实验。

### 数学步骤详解

举例说明。假设有两个类别：垃圾邮件和非垃圾邮件。词汇表包含三个词：“free”，“money”，“meeting”。

训练数据：  
- 垃圾邮件中，“free”出现80次，“money”60次，“meeting”10次，词总数150  
- 非垃圾邮件中，“free”5次，“money”10次，“meeting”100次，词总数115  
- 邮件中垃圾邮件占40%，非垃圾邮件占60%

带拉普拉斯平滑（alpha=1）：

```text
P(free | spam)    = (80 + 1) / (150 + 3) = 81/153 = 0.529
P(money | spam)   = (60 + 1) / (150 + 3) = 61/153 = 0.399
P(meeting | spam) = (10 + 1) / (150 + 3) = 11/153 = 0.072

P(free | not-spam)    = (5 + 1) / (115 + 3) = 6/118 = 0.051
P(money | not-spam)   = (10 + 1) / (115 + 3) = 11/118 = 0.093
P(meeting | not-spam) = (100 + 1) / (115 + 3) = 101/118 = 0.856
```

有封新邮件，包含：“free”出现2次，“money”1次，“meeting”0次。

```text
log P(spam | email) = log(0.4) + 2*log(0.529) + 1*log(0.399) + 0*log(0.072)
                    = -0.916 + 2*(-0.637) + (-0.919) + 0
                    = -3.109

log P(not-spam | email) = log(0.6) + 2*log(0.051) + 1*log(0.093) + 0*log(0.856)
                        = -0.511 + 2*(-2.976) + (-2.375) + 0
                        = -8.838
```

垃圾邮件概率大幅领先。单词“free”出现两次是检测垃圾邮件的强烈证据。注意“meeting”未出现对两边的对数和贡献为0（0乘以对数概率），因为多项式朴素贝叶斯中，缺失词无影响。反之，伯努利朴素贝叶斯明确建模了词的缺失。

### 三种变体

朴素贝叶斯有三种变体。分别以不同方式建模 `P(feature | class)`。

#### 多项式朴素贝叶斯（Multinomial Naive Bayes）

将每个特征建模为计数。适合文本数据，特征为词频或TF-IDF。

```text
P(word_i | class) = (word_i在该类别中的计数 + alpha) / (类别中单词总数 + alpha * 词汇量)
```

`alpha` 是拉普拉斯平滑参数（下文解释）。此变体是文本分类的主力。

#### 高斯朴素贝叶斯（Gaussian Naive Bayes）

将每个特征建模为高斯（正态）分布。适合连续特征。

```text
P(x_i | class) = (1 / sqrt(2 * pi * var)) * exp(-(x_i - mean)^2 / (2 * var))
```

每个类别对每个特征都有自己的均值和方差。若特征在每个类别内真实服从钟形分布，效果很好。

#### 伯努利朴素贝叶斯（Bernoulli Naive Bayes）

将每个特征视为二元（出现或未出现）。适合短文本或二元特征向量。

```text
P(word_i | class) = (类别中包含 word_i 的文档数 + alpha) / (类别中文档总数 + 2 * alpha)
```

与多项式不同，伯努利明确惩罚了词的缺失。如果“free”通常出现在垃圾邮件但未出现在这封邮件中，伯努利会将缺失视为垃圾邮件的反证据。

### 何时选择哪种变体

| 变体       | 特征类型     | 适用场景           | 示例                     |
|------------|-------------|--------------------|--------------------------|
| 多项式     | 计数或频率   | 文本分类，词袋模型 | 电子邮件垃圾分类，主题分类 |
| 高斯       | 连续值       | 表格数据，符合正态分布 | 鸢尾花分类，传感器数据      |
| 伯努利     | 二元（0/1）  | 短文本，二元特征矢量 | 短信垃圾，出现/缺失特征    |

### 拉普拉斯平滑（Laplace Smoothing）

测试集中出现但训练集中某类别未出现的词怎么办？

无平滑时：`P(word | class) = 0 / N = 0`。乘积中一旦出现0，即结果为0。单个从未见过的词会让整个类别概率归零，无法合理判断。

拉普拉斯平滑为每个特征计数加一个小值 `alpha`（通常为1）：

```text
P(word_i | class) = (count(word_i, class) + alpha) / (类别中单词总数 + alpha * 词汇量)
```

有了alpha=1，即使测试后文档出现“discombobulate”也不会让垃圾邮件概率变为零。平滑也能从贝叶斯角度解释：相当于在词分布上引入了均匀dirichlet先验。

alpha值越大，平滑越强（分布越均匀），alpha越小，模型越相信数据。alpha是超参数需调优。

alpha效果举例：

| Alpha | 作用               | 适用场景                      |
|-------|--------------------|------------------------------|
| 0.001 | 几乎无平滑，依赖数据 | 巨大训练集，无未见特征预期        |
| 0.1   | 轻度平滑           | 大型训练集                    |
| 1.0   | 标准拉普拉斯平滑    | 默认起点                      |
| 10.0  | 强平滑，分布更均匀  | 极小训练集，有大量未见词          |

### 对数空间计算

连续乘积多个小于1的概率会导致浮点数下溢。数值变成0，尽管数学值应是接近0但正值。

解决办法：用对数空间计算，概率乘积变为对数相加：

```text
log P(class | x1, x2, ..., xn) = log P(class) + sum_i log P(xi | class)
```

预测等价于点积运算：

```text
log_scores = X @ log_feature_probs.T + log_class_priors
prediction = argmax(log_scores)
```

矩阵乘法。这也是朴素贝叶斯预测极快的原因——它与单层线性模型的操作相同。

### 朴素贝叶斯 vs 逻辑回归

两者均是文本线性分类器。区别在于建模方式。

| 方面         | 朴素贝叶斯                        | 逻辑回归                      |
|--------------|---------------------------------|------------------------------|
| 类型         | 生成式（建模 P(X|Y)）           | 判别式（建模 P(Y|X)）         |
| 训练         | 计数频率                       | 优化损失函数                  |
| 小数据表现   | 更好（强先验有利）              | 较差（样本不足难估计权重）    |
| 大数据表现   | 较差（错误假设影响性能）        | 更好（边界更灵活）            |
| 特征处理     | 假设独立                       | 处理相关性                    |
| 速度         | 单遍历，极快                   | 迭代优化                      |
| 概率校准     | 概率差，校准较差               | 概率更好                      |

经验法则：先用朴素贝叶斯，若数据足够大且朴素贝叶斯性能停滞，再尝试逻辑回归。

### 分类流程图

```mermaid
flowchart LR
    A[原始文本] --> B[分词]
    B --> C[构建词汇表]
    C --> D[统计词频]
    D --> E[应用平滑]
    E --> F[计算对数概率]
    F --> G[预测：选取概率最高的类别]

    style A fill:#f9f,stroke:#333
    style G fill:#9f9,stroke:#333
```

在实践中，我们在对数空间工作以避免浮点数下溢。我们不是相乘许多小概率，而是相加它们的对数：

```text
log P(class | features) = log P(class) + sum_i log P(feature_i | class)
```

## 构建它

`code/naive_bayes.py` 中的代码从零实现了 MultinomialNB 和 GaussianNB。

### MultinomialNB

从零实现：

1. **fit(X, y)**：对于每个类别，统计每个特征的频率。添加拉普拉斯平滑（Laplace smoothing）。计算对数概率。存储类别先验（类别频率的对数）。

2. **predict_log_proba(X)**：对于每个样本，计算 log P(class) + 所有类别的 log P(feature_i | class) 之和。这是一次矩阵乘法：X @ log_probs.T + log_priors。

3. **predict(X)**：返回最大对数概率的类别。

```python
class MultinomialNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        classes = np.unique(y)
        n_classes = len(classes)
        n_features = X.shape[1]

        self.classes_ = classes
        self.class_log_prior_ = np.zeros(n_classes)
        self.feature_log_prob_ = np.zeros((n_classes, n_features))

        for i, c in enumerate(classes):
            X_c = X[y == c]
            self.class_log_prior_[i] = np.log(X_c.shape[0] / X.shape[0])
            counts = X_c.sum(axis=0) + self.alpha
            self.feature_log_prob_[i] = np.log(counts / counts.sum())

        return self
```

关键见解：拟合后，预测只是矩阵乘法加上一个偏置。这就是朴素贝叶斯（Naive Bayes）为什么如此快速。

### GaussianNB

对于连续特征，我们为每个类别的每个特征估计均值和方差：

```python
class GaussianNB:
    def __init__(self):
        pass

    def fit(self, X, y):
        classes = np.unique(y)
        self.classes_ = classes
        self.means_ = np.zeros((len(classes), X.shape[1]))
        self.vars_ = np.zeros((len(classes), X.shape[1]))
        self.priors_ = np.zeros(len(classes))

        for i, c in enumerate(classes):
            X_c = X[y == c]
            self.means_[i] = X_c.mean(axis=0)
            self.vars_[i] = X_c.var(axis=0) + 1e-9
            self.priors_[i] = X_c.shape[0] / X.shape[0]

        return self
```

预测时，使用每个特征的高斯概率密度函数（PDF），多个特征的概率在对数空间相加。

### 演示：文本分类

代码生成模拟两类（科技文章 vs 体育文章）的合成词袋（bag-of-words）数据。每个类别有不同的词频分布。MultinomialNB 使用词频进行分类。

合成数据构造方式是：我们创建了 200 个“词”（特征列）。词 0-39 在科技文章中频率高，在体育文章中频率低。词 80-119 在体育文章中频率高，在科技文章中频率低。词 40-79 在两者中频率中等。这模拟了一种真实情况，有些词是强分类指标，有些则是噪声。

### 演示：连续特征

代码生成类似鸢尾花（Iris）数据（三类，四特征，高斯簇）。GaussianNB 使用每类均值和方差进行分类。每个类别有不同的中心（均值向量）和不同的分布范围（方差），模拟现实中不同类别间测量的系统差异。

代码还演示了：
- **平滑比较（Smoothing comparison）：** 用不同 alpha 值训练 MultinomialNB，展示平滑强度对准确率的影响。
- **训练规模实验（Training size experiment）：** NB 精度随着训练样本从 20 增加到 1600 的变化。NB 即使样本很少也能达到不错准确率 —— 这是它的主要优势。
- **混淆矩阵（Confusion matrix）：** 显示各类别的精确度、召回率和 F1 分数，指出 NB 出错的地方。

### 预测速度

朴素贝叶斯预测就是一次矩阵乘法。对于 n 个样本，d 个特征和 k 个类别：
- MultinomialNB：一次矩阵乘法 (n x d) @ (d x k) = O(n * d * k)
- GaussianNB：n * k 次高斯 PDF 评估，每次覆盖 d 个特征 = O(n * d * k)

两者在所有维度上均为线性复杂度。相比 KNN（需要对所有训练点计算距离）或带 RBF 核的 SVM（需要对所有支持向量计算核），NB 在预测时快几个数量级。

## 使用它

用 sklearn，这两个变体都可以一行代码完成：

```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB

gnb = GaussianNB()
gnb.fit(X_train, y_train)
print(f"GaussianNB accuracy: {gnb.score(X_test, y_test):.3f}")

mnb = MultinomialNB(alpha=1.0)
mnb.fit(X_train_counts, y_train)
print(f"MultinomialNB accuracy: {mnb.score(X_test_counts, y_test):.3f}")
```

用 sklearn 做文本分类：

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

text_clf = Pipeline([
    ("vectorizer", CountVectorizer()),
    ("classifier", MultinomialNB(alpha=1.0)),
])

text_clf.fit(train_texts, train_labels)
accuracy = text_clf.score(test_texts, test_labels)
```

`naive_bayes.py` 中的代码将从零实现与 sklearn 在相同数据上的结果做比较，以验证正确性。

### TF-IDF 与朴素贝叶斯

原始词频计数对每个词的每次出现赋予相同权重，但常见词如 “the” 和 “is” 在所有类别中都频繁出现，根本不含信息。TF-IDF（词频-逆文档频率）会降低常见词的权重，提高稀有且有区分力的词权重。

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

text_clf = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", MultinomialNB(alpha=0.1)),
])
```

TF-IDF 值非负，因此适用于 MultinomialNB。TF-IDF+MultinomialNB 组合是文本分类中最强基线之一，常常超过训练样本数少于 10,000 的数据集上更复杂的模型。

### BernoulliNB 用于短文本

对于短文本（推文、短信、聊天消息），BernoulliNB 有时优于 MultinomialNB。短文本中的词频较低，所以 MultinomialNB 依赖的频率信息较为嘈杂。BernoulliNB 只关心词的出现与否，这在短文本中更可靠。

```python
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer

text_clf = Pipeline([
    ("vectorizer", CountVectorizer(binary=True)),
    ("classifier", BernoulliNB(alpha=1.0)),
])
```

CountVectorizer 中的 `binary=True` 标志将所有计数转换为 0/1。没有该标志，BernoulliNB 仍可工作，但输入的数据是计数，不是设计初衷。

### 校准 NB 概率

NB 概率的校准不佳。当 NB 给出 P(spam) = 0.95 时，真实概率可能是 0.7。如果你需要可靠的概率估计（例如设置阈值或融合其他模型），请使用 sklearn 的 CalibratedClassifierCV：

```python
from sklearn.calibration import CalibratedClassifierCV

calibrated_nb = CalibratedClassifierCV(MultinomialNB(), cv=5, method="sigmoid")
calibrated_nb.fit(X_train, y_train)
proba = calibrated_nb.predict_proba(X_test)
```

它使用交叉验证在 NB 的原始分数之上拟合一层逻辑回归。得到的概率更接近真实类别频率。

### 常见注意事项

1. **负特征值。** MultinomialNB 要求非负特征。如果有负值（比如某些 TF-IDF 设置或标准化特征），应改用 GaussianNB，或将特征平移为正数。

2. **零方差特征。** GaussianNB 需要除以方差。如果某类别的某特征方差为零（所有值相同），概率计算会出错。代码对所有方差加了一个小平滑项（1e-9）防止此问题。

3. **类别不平衡。** 如果 99% 邮件非垃圾邮件，先验 P(非垃圾) = 0.99 强得压倒似然证据。可以手动设置类别先验，或用 sklearn 的 class_prior 参数。

4. **特征缩放。** MultinomialNB 不需要缩放（它以计数为工作基础）。GaussianNB 也不需缩放（它估计每特征的统计量）。这优于对特征尺度敏感的逻辑回归和支持向量机（SVM）。

## 交付它

本课产出：
- `outputs/skill-naive-bayes-chooser.md` —— 一个用于选择合适 NB 变体的决策技能
- `code/naive_bayes.py` —— 从零实现的 MultinomialNB 和 GaussianNB，带 sklearn 对比

### 朴素贝叶斯失败时

当独立性假设导致错误排序（不仅仅是概率错误）时，NB 失败。出现于：

1. **强特征交互。** 类别依赖于两个特征的组合而不是任一单独特征（XOR 式模式），NB 完全捕捉不到。每个特征单独无证据，NB 无法非线性组合它们。

2. **高度相关但证据矛盾的特征。** 如果特征 A 表示“垃圾邮件”，特征 B 表示“非垃圾”，但 A 和 B 实际上完全相关（总是一致），NB 会看到不存在的矛盾证据。

3. **极大训练数据。** 拥有足够数据时，判别模型（如逻辑回归）学习到真实决策边界，超过 NB。独立假设在小样本时帮了大忙，数据多时反而拖后腿。

在文本分类中，这些失败情况很少见。文本特征多且单个弱，独立假设的错误常相互抵消。对有少量强相关特征的表格数据，推荐先使用逻辑回归或基于树的模型。

## 练习题

1. **平滑实验。** 在文本数据上训练 MultinomialNB，alpha 取 0.01、0.1、1.0、10.0 和 100.0。绘制准确率与 alpha 的关系。性能在哪个 alpha 取值时最高？为什么过大 alpha 会降低表现？

2. **特征独立性检测。** 用真实文本数据选取两个明显相关词（如 "machine" 和 "learning"）。计算 P(word1 | class) * P(word2 | class) 并与 P(word1 AND word2 | class) 比较。独立性假设错得有多严重？是否影响分类准确率？

3. **Bernoulli 实现。** 在代码中添加 BernoulliNB 类。将词袋转为二元（出现/不存在），并与 MultinomialNB 在文本数据上比较准确率。Bernoulli 何时表现更好？

4. **NB 与逻辑回归比较。** 在文本数据上训练两者，从 100 个训练样本增加到 10,000。绘制准确率与训练集大小的关系图。逻辑回归在哪个点超过 NB？

5. **垃圾邮件过滤器。** 完整构建垃圾邮件分类器：对原始邮件文本分词，构建词汇表，创建词袋特征，训练 MultinomialNB，用精确率和召回率评估（不仅仅是准确率 —— 为什么？）。

## 关键词

| 术语 | 常见说法 | 实际含义 |
|------|---------|----------|
| Naive Bayes | “简单概率分类器” | 一种在假设特征在类别条件下相互独立时应用贝叶斯定理的分类器 |
| Conditional independence（条件独立） | “特征互不影响” | P(A, B \| C) = P(A \| C) * P(B \| C) —— 知道 B 后，在已知 C 的条件下，A 没有新增信息 |
| Laplace smoothing（拉普拉斯平滑） | “加一平滑” | 对每个特征加一个小计数，防止零概率在预测中主导 |
| Prior（先验） | “看数据前的相信” | P(class) —— 未观察特征前的类别概率 |
| Likelihood（似然） | “数据拟合情况” | P(features \| class) —— 若类别确定，观察到这些特征的概率 |
| Posterior（后验） | “看数据后的相信” | P(class \| features) —— 观察特征后更新的类别概率 |
| Generative model（生成模型） | “模拟数据生成方式” | 学习 P(X \| Y) 和 P(Y)，再用贝叶斯定理求 P(Y \| X) 的模型 |
| Discriminative model（判别模型） | “直接建决策边界” | 直接学习 P(Y \| X)，不模拟 X 的生成 |
| Log probability（对数概率） | “避免下溢” | 用对数概率代替概率防止多个小数相乘导致浮点下溢 |

## 深入阅读

- [scikit-learn Naive Bayes 文档](https://scikit-learn.org/stable/modules/naive_bayes.html) -- 三种变体及其数学细节
- [McCallum 和 Nigam，事件模型在朴素贝叶斯文本分类中的比较（1998）](https://www.cs.cmu.edu/~knigam/papers/multinomial-aaaiws98.pdf) -- Multinomial（多项式）与 Bernoulli（伯努利）模型的经典对比
- [Rennie 等人，解决朴素贝叶斯文本分类器的弱假设问题（2003）](https://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf) -- 针对文本的朴素贝叶斯改进方法
- [Ng 和 Jordan，判别式分类器与生成式分类器的比较（2001）](https://ai.stanford.edu/~ang/papers/nips01-discriminativegenerative.pdf) -- 证明朴素贝叶斯在数据量较少时收敛速度快于逻辑回归（LR）
