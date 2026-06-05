# 概率与分布

> 概率是人工智能表达不确定性的语言。

**类型：** 学习  
**语言：** Python  
**先决条件：** 第一阶段，第01-04课  
**时间：** 约75分钟

## 学习目标

- 从零实现 Bernoulli（二项分布）、categorical（多项分布）、Poisson（泊松分布）、uniform（均匀分布）和 normal（正态分布）的 PMF（概率质量函数）和 PDF（概率密度函数）
- 计算期望、方差，并用中心极限定理解释为何高斯分布占主导
- 构建 softmax 和 log-softmax 函数，使用数值稳定技巧（减去最大 logit）
- 从 logits 计算交叉熵损失，并将其与负对数似然联系起来

## 问题背景

一个分类器输出 `[0.03, 0.91, 0.06]`。语言模型从 50,000 个候选词中选择下一个单词。扩散模型通过从学习到的分布采样生成图像。这些都是概率在实际中的表现。

模型做出的每一个预测都是一个概率分布。每个损失函数衡量预测分布与真实分布的距离。每一步训练调整参数，使一个分布更像另一个分布。没有概率，你无法阅读任何机器学习论文，调试任何模型，或理解为何训练损失会出现 NaN。

## 概念讲解

### 事件、样本空间与概率

样本空间 S 是所有可能结果的集合。事件是样本空间的子集。概率将事件映射到 0 到 1 之间的数字。

```text
抛硬币:
  S = {正面, 反面}
  P(正面) = 0.5,  P(反面) = 0.5

单次掷骰子:
  S = {1, 2, 3, 4, 5, 6}
  P(偶数) = P({2, 4, 6}) = 3/6 = 0.5
```

概率的三个公理定义了全部内容：
1. 对任何事件 A，有 P(A) >= 0  
2. P(S) = 1（必定发生某种情况）  
3. 当事件 A 和 B 不能同时发生时，P(A 或 B) = P(A) + P(B)  

其它内容（贝叶斯定理、期望、分布）都由这三条规则推导而来。

### 条件概率与独立性

P(A|B) 是在事件 B 发生的条件下，事件 A 的概率。

```text
P(A|B) = P(A 且 B) / P(B)

举例：扑克牌  
  P(国王牌 | 人脸牌) = P(国王牌 且 人脸牌) / P(人脸牌)
                    = (4/52) / (12/52)
                    = 4/12 = 1/3
```

当知道一个事件时，对另一个事件没有任何信息，即称两事件独立：

```text
独立事件:   P(A|B) = P(A)
等价于:    P(A 且 B) = P(A) * P(B)
```

抛硬币是独立事件。抽取扑克牌不放回则不是。

### 概率质量函数与概率密度函数

离散随机变量有概率质量函数（PMF），每个事件有明确概率，可以直接读取。

```text
PMF: P(X = k)

公平骰子:
  P(X = 1) = 1/6
  P(X = 2) = 1/6
  ...
  P(X = 6) = 1/6

所有概率加和 = 1
```

连续随机变量有概率密度函数（PDF）。一点处的密度不是概率，概率要从区间积分得到。

```text
PDF: f(x)

P(a <= X <= b) = 从 a 到 b 对 f(x) 的积分

f(x) 可大于 1（是密度，非概率）
f(x) 在全域积分（-∞ 到 +∞）等于 1
```

此区别在机器学习中很重要。分类输出是 PMF（离散选择），VAE 的潜变量空间使用 PDF（连续）。

### 常见分布

**Bernoulli（二项分布）:** 一次试验，两个结果。用于二分类。

```text
P(X = 1) = p
P(X = 0) = 1 - p
均值 = p， 方差 = p(1-p)
```

**Categorical（多项分布）:** 一次试验，k 个结果。用于多分类（softmax 输出）。

```text
P(X = i) = p_i，且 ∑p_i = 1
示例: P(猫) = 0.7，P(狗) = 0.2，P(鸟) = 0.1
```

**Uniform（均匀分布）:** 所有结果概率相等。用于随机初始化。

```text
离散: P(X = k) = 1/n，k ∈ {1, ..., n}
连续: f(x) = 1/(b-a)，x ∈ [a, b]
```

**Normal（高斯分布）:** 钟型曲线。参数为均值（mu）和方差（sigma^2）。

```text
f(x) = (1 / sqrt(2*pi*sigma^2)) * exp(-(x - mu)^2 / (2*sigma^2))

标准正态: mu = 0, sigma = 1
  68% 数据在1个标准差内
  95% 在2个标准差内
  99.7% 在3个标准差内
```

**Poisson（泊松分布）:** 固定区间内稀有事件计数。用于事件速率建模。

```text
P(X = k) = (lambda^k * e^(-lambda)) / k!
均值 = lambda，方差 = lambda
```

### 期望值与方差

期望值是加权平均结果。

```text
离散:   E[X] = ∑ x_i * P(X = x_i)
连续:   E[X] = ∫ x * f(x) dx
```

方差衡量结果围绕均值的离散程度。

```text
Var(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2
标准差 = sqrt(Var(X))
```

机器学习中，期望值表现为损失函数（对数据分布的平均损失），方差反映模型稳定性。梯度高方差意味着训练噪声大。

### 联合分布与边缘分布

联合分布 P(X, Y) 描述两个随机变量的联合概率。

联合 PMF 示例（X = 天气，Y = 是否带伞）:

|     | Y=0（不带伞）  | Y=1（带伞）  | 边缘分布 P(X)      |
|-----|----------------|--------------|--------------------|
| X=0（晴天） | 0.40           | 0.10         | P(X=0) = 0.50      |
| X=1（雨天） | 0.05           | 0.45         | P(X=1) = 0.50      |
| **边缘分布 P(Y)** | P(Y=0) = 0.45   | P(Y=1) = 0.55  | 1.00               |

边缘分布通过对另一变量求和得到：

```text
P(X = x) = ∑_y P(X = x, Y = y)
```

上表中的行和列合计即为边缘分布。

### 正态分布为何普遍出现

中心极限定理：许多独立随机变量之和（或均值）趋向于正态分布，与初始分布无关。

```text
抛1个骰子：均匀分布（平坦）
2个骰子均值：三角分布（尖峰）
30个骰子均值：近似完美钟形曲线

适用于任意起始分布。
```

因此：
- 测量误差近似正态（多源小误差叠加）
- 神经网络权重初始化使用正态分布
- SGD 中梯度噪声近似正态（多个梯度样本相加）
- 正态分布是固定均值和方差下最大熵分布

### 对数概率

原始概率会导致数值问题。多个小概率相乘很快下溢为 0。

```text
P(句子) = P(词1) * P(词2) * ... * P(词_n)
        = 0.01 * 0.003 * 0.02 * ...
        -> 0.0 （约30个乘积后发生下溢）
```

对数概率解决此问题。乘法变加法。

```text
log P(句子) = log P(词1) + log P(词2) + ... + log P(词_n)
            = -4.6 + -5.8 + -3.9 + ...
            -> 有限数值（无下溢）
```

规则：
- log(a * b) = log(a) + log(b)
- 对数概率总 ≤ 0（因为0 < P ≤ 1）
- 越负越不可能
- 交叉熵损失是正确类别概率的负对数概率

### Softmax 作为概率分布

神经网络输出原始得分（logits）。softmax 将它们转为合法概率分布。

```text
softmax(z_i) = exp(z_i) / ∑_j exp(z_j)

性质:
  - 输出范围在 (0, 1)
  - 输出加和为1
  - 保留输入的相对顺序
  - exp() 放大 logits 之间差异
```

softmax 技巧：在 exponent 前减去最大 logit 防止溢出。

```text
z = [100, 101, 102]
exp(102) = 溢出

z_shifted = z - max(z) = [-2, -1, 0]
exp(0) = 1  （安全）

结果相同，无溢出。
```

log-softmax 结合 softmax 和对数计算以获得数值稳定性。PyTorch 在计算交叉熵时内部使用。

### 采样

采样是从分布中随机抽取值。在机器学习中：
- Dropout 随机采样哪些神经元置零
- 数据增强采样随机变换
- 语言模型从预测分布采样下一个词
- 扩散模型采样噪声并逐步去噪

从任意分布采样需技巧，比如逆变换采样、拒绝采样，或重参数化技巧（VAE 中使用）。

## 实现步骤

### 步骤1：概率基础

```python
import math
import random

def factorial(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

def combinations(n, k):
    return factorial(n) // (factorial(k) * factorial(n - k))

def conditional_probability(p_a_and_b, p_b):
    return p_a_and_b / p_b

p_king_given_face = conditional_probability(4/52, 12/52)
print(f"P(King | Face card) = {p_king_given_face:.4f}")
```

### 步骤2：手写 PMF 和 PDF

```python
def bernoulli_pmf(k, p):
    return p if k == 1 else (1 - p)

def categorical_pmf(k, probs):
    return probs[k]

def poisson_pmf(k, lam):
    return (lam ** k) * math.exp(-lam) / factorial(k)

def uniform_pdf(x, a, b):
    if a <= x <= b:
        return 1.0 / (b - a)
    return 0.0

def normal_pdf(x, mu, sigma):
    coeff = 1.0 / (sigma * math.sqrt(2 * math.pi))
    exponent = -0.5 * ((x - mu) / sigma) ** 2
    return coeff * math.exp(exponent)
```

### 步骤3：期望与方差

```python
def expected_value(values, probabilities):
    return sum(v * p for v, p in zip(values, probabilities))

def variance(values, probabilities):
    mu = expected_value(values, probabilities)
    return sum(p * (v - mu) ** 2 for v, p in zip(values, probabilities))

die_values = [1, 2, 3, 4, 5, 6]
die_probs = [1/6] * 6
mu = expected_value(die_values, die_probs)
var = variance(die_values, die_probs)
print(f"Die: E[X] = {mu:.4f}, Var(X) = {var:.4f}, SD = {var**0.5:.4f}")
```

### 步骤4：从分布中采样

```python
def sample_bernoulli(p, n=1):
    return [1 if random.random() < p else 0 for _ in range(n)]

def sample_categorical(probs, n=1):
    cumulative = []
    total = 0
    for p in probs:
        total += p
        cumulative.append(total)
    samples = []
    for _ in range(n):
        r = random.random()
        for i, c in enumerate(cumulative):
            if r <= c:
                samples.append(i)
                break
    return samples

def sample_normal_box_muller(mu, sigma, n=1):
    samples = []
    for _ in range(n):
        u1 = random.random()
        u2 = random.random()
        z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        samples.append(mu + sigma * z)
    return samples
```

### 步骤5：Softmax 与对数概率

```python
def softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    exps = [math.exp(z) for z in shifted]
    total = sum(exps)
    return [e / total for e in exps]

def log_softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = max_logit + math.log(sum(math.exp(z) for z in shifted))
    return [z - log_sum_exp for z in logits]

def cross_entropy_loss(logits, target_index):
    log_probs = log_softmax(logits)
    return -log_probs[target_index]
```

### 步骤6：中心极限定理演示

```python
def demonstrate_clt(dist_fn, n_samples, n_averages):
    averages = []
    for _ in range(n_averages):
        samples = [dist_fn() for _ in range(n_samples)]
        averages.append(sum(samples) / len(samples))
    return averages
```

### 步骤7：可视化

```python
import matplotlib.pyplot as plt

xs = [mu + sigma * (i - 500) / 100 for i in range(1001)]
ys = [normal_pdf(x, mu, sigma) for x, mu, sigma in ...]
plt.plot(xs, ys)
```

完整实现及所有可视化见 `code/probability.py`。

## 使用方法

利用 NumPy（NumPy 库）和 SciPy（SciPy 库），上面所有内容都可以用一行代码实现：

```python
import numpy as np
from scipy import stats

normal = stats.norm(loc=0, scale=1)
samples = normal.rvs(size=10000)
print(f"Mean: {np.mean(samples):.4f}, Std: {np.std(samples):.4f}")
print(f"P(X < 1.96) = {normal.cdf(1.96):.4f}")

logits = np.array([2.0, 1.0, 0.1])
from scipy.special import softmax, log_softmax
probs = softmax(logits)
log_probs = log_softmax(logits)
print(f"Softmax: {probs}")
print(f"Log-softmax: {log_probs}")
```

你已经从头构建了这些。现在你知道库函数在做什么。

## 练习

1. 实现指数分布的逆变换采样。通过采样 10,000 个值并将直方图与真实 PDF（概率密度函数）比较来验证。

2. 构建两个加载骰子的联合分布表。计算边缘分布并检查骰子是否独立。

3. 计算一个 5 类分类器的交叉熵损失，其输出 logits 为 `[2.0, 0.5, -1.0, 3.0, 0.1]`，正确类别为索引 3。然后用 PyTorch 的 `nn.CrossEntropyLoss` 验证你的答案。

4. 编写一个函数，输入一组对数概率，返回最可能的序列、总对数概率和对应的原始概率。用一个句子（50 个词，每个词概率为 0.01）进行测试。

## 关键词

| 术语 | 常说法 | 实际含义 |
|------|--------|----------|
| Sample space（样本空间） | “所有可能” | 实验所有可能结果的集合 S |
| PMF（概率质量函数） | “概率函数” | 给出每个离散结果精确概率，所有概率和为 1 的函数 |
| PDF（概率密度函数） | “概率曲线” | 连续变量的密度函数。对区间积分得到概率 |
| Conditional probability（条件概率） | “给定某事的概率” | P(A\|B) = P(A 和 B) / P(B)。贝叶斯思维和贝叶斯定理基础 |
| Independence（独立性） | “不相互影响” | P(A 和 B) = P(A) * P(B)。知道一个事件对另一个无信息 |
| Expected value（期望值） | “平均值” | 所有结果的概率加权和。损失函数是期望值 |
| Variance（方差） | “分布离散程度” | 距离均值的平方偏差的期望。高方差 = 噪声多，估计不稳定 |
| Normal distribution（正态分布） | “钟形曲线” | f(x) = (1/sqrt(2πσ²)) * exp(-(x-μ)²/(2σ²))。中心极限定理导致其普遍出现 |
| Central Limit Theorem（中心极限定理） | “平均趋于正态” | 多个独立样本均值无论来源如何都趋于正态分布 |
| Joint distribution（联合分布） | “两个变量一起” | P(X, Y) 描述 X 和 Y 每种组合结果的概率 |
| Marginal distribution（边缘分布） | “对另一个变量求和” | P(X) = sum_y P(X, Y)。从联合分布恢复单个变量分布 |
| Log probability（对数概率） | “概率的对数” | log P(x)。将乘积变成求和，防止长序列数值下溢 |
| Softmax（Softmax 函数） | “把分数转成概率” | softmax(z_i) = exp(z_i) / sum(exp(z_j))。将实数 logits 映射为有效概率分布 |
| Cross-entropy（交叉熵） | “损失函数” | -sum(p_true * log(p_predicted))。衡量两个分布差异，值越小越好 |
| Logits（Logits） | “模型原始输出” | Softmax 之前的未归一化分数。名自 logistic 函数 |
| Sampling（采样） | “抽取随机值” | 根据概率分布生成数据。模型生成输出的方式 |

## 拓展阅读

- [3Blue1Brown: 但什么是中心极限定理？](https://www.youtube.com/watch?v=zeJD6dqJ5lo) - 关于为什么平均值会呈现正态分布的直观证明  
- [Stanford CS229 概率复习](https://cs229.stanford.edu/section/cs229-prob.pdf) - 几乎涵盖这里所有内容的简明参考  
- [对数和指数和技巧（Log-Sum-Exp Trick）](https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/) - 讲述数值稳定性的重要性及实现方法
