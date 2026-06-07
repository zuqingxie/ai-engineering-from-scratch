# 从零实现自注意力（Self-Attention）

> 注意力（Attention）是一个查找表，每个词都问“谁对我重要？”——并学习答案。

**类型：** 构建  
**语言：** Python  
**先决条件：** 第三阶段（深度学习核心），第五阶段第10课（序列到序列）  
**时间：** 约90分钟

## 学习目标

- 使用仅 NumPy 从零实现缩放点积自注意力（scaled dot-product self-attention），包括 query/key/value 投影和 softmax 加权求和
- 构建一个多头注意力层（multi-head attention layer），实现头部分割、并行计算注意力并拼接结果
- 跟踪注意力矩阵如何捕捉词元间关系，解释为何除以 sqrt(d_k) 防止 softmax 饱和
- 应用因果掩码（causal masking）将双向注意力转换为自回归（decoder 风格）注意力

## 问题背景

RNN 一次处理一个词元。到处理第50个词元时，第1个词元的信息已经经过50次压缩步骤被挤压。长距离依赖被压缩到固定大小的隐藏状态——这是一个瓶颈，LSTM 的门控机制无法完全解决。

2014年 Bahdanau 注意力论文提出了解决方案：让解码器回看每个编码器位置，决定哪些位置对当前步骤重要。但它仍依赖于 RNN。2017 年《Attention Is All You Need》论文提出了更严肃的问题：如果注意力是*唯一*机制怎么办？没有递归，没有卷积，只有注意力。

自注意力允许序列中的每个位置在单一步骤内关注序列中的所有其他位置。这就是 Transformer（Transformer 架构）快速、可扩展且主导性的原因。

## 基本概念

### 数据库查找类比

将注意力看作是一个软查找：

```text
传统数据库：
  查询：“法国首都”  -->  精确匹配  -->  “巴黎”

注意力：
  查询：“法国首都”  -->  与所有键的相似度  -->  所有值的加权融合
```

每个词元生成三个向量：  
- **Query（Q）**：“我在寻找什么？”  
- **Key（K）**：“我包含什么？”  
- **Value（V）**：“如果被选中，我提供什么信息？”

Query 和所有 Key 的点积产生注意力分数。高分表示“此键匹配我的查询”。分数用来权衡 Value，输出是加权值的和。

### Q，K，V 的计算

每个词元的嵌入通过三个学习得到的权重矩阵投影：

```text
输入嵌入（序列有 n 个词元，每个为 d 维）：

  X = [x1, x2, x3, ..., xn]       形状: (n, d)

三个权重矩阵：

  Wq  形状: (d, dk)
  Wk  形状: (d, dk)
  Wv  形状: (d, dv)

投影：

  Q = X @ Wq    形状: (n, dk)      每个词元的查询向量
  K = X @ Wk    形状: (n, dk)      每个词元的键向量
  V = X @ Wv    形状: (n, dv)      每个词元的值向量
```

视觉化单个词元：

```text
             Wq
  x_i ------[*]------> q_i    “我在寻找什么？”
       |
       |     Wk
       +----[*]------> k_i    “我包含什么？”
       |
       |     Wv
       +----[*]------> v_i    “我能提供什么？”
```

### 注意力矩阵

得到所有词元的 Q, K, V 后，注意力分数形成矩阵：

```text
Scores = Q @ K^T    形状: (n, n)

              k1    k2    k3    k4    k5
        +-----+-----+-----+-----+-----+
   q1   | 2.1 | 0.3 | 0.1 | 0.8 | 0.2 |   <- q1 对每个 key 的关注程度
        +-----+-----+-----+-----+-----+
   q2   | 0.4 | 1.9 | 0.7 | 0.1 | 0.3 |
        +-----+-----+-----+-----+-----+
   q3   | 0.2 | 0.6 | 2.3 | 0.5 | 0.1 |
        +-----+-----+-----+-----+-----+
   q4   | 0.9 | 0.1 | 0.4 | 1.7 | 0.6 |
        +-----+-----+-----+-----+-----+
   q5   | 0.1 | 0.3 | 0.2 | 0.5 | 2.0 |
        +-----+-----+-----+-----+-----+

每一行：一个词元对整个序列的注意力分布
```

### 为什么要缩放？

点积随着维度 dk 增大而变大。如果 dk=64，点积可能达到几十，导致 softmax 函数落入梯度消失区域。解决方案是除以 sqrt(dk)。

```text
缩放后的分数 = (Q @ K^T) / sqrt(dk)
```

这样数值保持在 softmax 能产生有效梯度的区间。

### Softmax 变权重

Softmax 将原始分数转换为每行的概率分布：

```text
q1 的原始分数:   [2.1, 0.3, 0.1, 0.8, 0.2]
                            |
                         softmax
                            |
注意力权重:   [0.52, 0.09, 0.07, 0.14, 0.08]   （和约为1.0）
```

现在每个词元有一组权重，表示它关注每个其他词元的程度。

### 加权值求和

每个词元最终输出是对所有值向量的加权和：

```text
output_i = sum( attention_weight[i][j] * v_j  for all j )

对于词元1：
  output_1 = 0.52 * v1 + 0.09 * v2 + 0.07 * v3 + 0.14 * v4 + 0.08 * v5
```

### 完整流程图

```text
                    +-------+
  X（输入）  ----->|  @ Wq  |-----> Q
                    +-------+
                    +-------+
  X（输入）  ----->|  @ Wk  |-----> K
                    +-------+                     +----------+
                    +-------+                     |          |
  X（输入）  ----->|  @ Wv  |-----> V ---------->| 加权求和  |----> 输出
                    +-------+          ^          |          |
                                       |          +----------+
                              +--------+--------+
                              |      softmax    |
                              +--------+--------+
                                       ^
                              +--------+--------+
                              | Q @ K^T / sqrt  |
                              +-----------------+
```

一行公式：

```text
Attention(Q, K, V) = softmax( Q @ K^T / sqrt(dk) ) @ V
```

## 实现步骤

### 步骤1：从零实现 Softmax

Softmax 将原始对数概率转换为概率。减去最大值以保证数值稳定。

```python
import numpy as np

def softmax(x):
    shifted = x - np.max(x, axis=-1, keepdims=True)  # 减去最大值稳定数值
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

logits = np.array([2.0, 1.0, 0.1])
print(f"logits:  {logits}")
print(f"softmax: {softmax(logits)}")
print(f"sum:     {softmax(logits).sum():.4f}")
```

### 步骤2：缩放点积注意力

核心函数。接收 Q, K, V 矩阵，返回注意力输出和权重矩阵。

```python
def scaled_dot_product_attention(Q, K, V):
    dk = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(dk)  # 点积后缩放
    weights = softmax(scores)       # 转成权重
    output = weights @ V            # 加权求和
    return output, weights
```

### 步骤3：带学习权重的自注意力类

完整自注意力模块，Wq、Wk、Wv 权重矩阵用类 Xaviar 风格初始化。

```python
class SelfAttention:
    def __init__(self, d_model, dk, dv, seed=42):
        rng = np.random.default_rng(seed)
        scale = np.sqrt(2.0 / (d_model + dk))
        self.Wq = rng.normal(0, scale, (d_model, dk))
        self.Wk = rng.normal(0, scale, (d_model, dk))
        scale_v = np.sqrt(2.0 / (d_model + dv))
        self.Wv = rng.normal(0, scale_v, (d_model, dv))
        self.dk = dk

    def forward(self, X):
        Q = X @ self.Wq
        K = X @ self.Wk
        V = X @ self.Wv
        output, weights = scaled_dot_product_attention(Q, K, V)
        return output, weights
```

### 步骤4：对一句话运行并查看注意力权重

生成假嵌入，观察注意力权重。

```python
sentence = ["The", "cat", "sat", "on", "the", "mat"]
n_tokens = len(sentence)
d_model = 8
dk = 4
dv = 4

rng = np.random.default_rng(42)
X = rng.normal(0, 1, (n_tokens, d_model))

attn = SelfAttention(d_model, dk, dv, seed=42)
output, weights = attn.forward(X)

print("注意力权重（每行：该词元关注的位置）:\n")
print(f"{'':>6}", end="")
for token in sentence:
    print(f"{token:>6}", end="")
print()

for i, token in enumerate(sentence):
    print(f"{token:>6}", end="")
    for j in range(n_tokens):
        w = weights[i][j]
        print(f"{w:6.3f}", end="")
    print()
```

### 步骤5：用 ASCII 热力图可视化注意力

用字符映射注意力权重，快速看趋势。

```python
def ascii_heatmap(weights, tokens, chars=" ░▒▓█"):
    n = len(tokens)
    print(f"\n{'':>6}", end="")
    for t in tokens:
        print(f"{t:>6}", end="")
    print()

    for i in range(n):
        print(f"{tokens[i]:>6}", end="")
        for j in range(n):
            level = int(weights[i][j] * (len(chars) - 1) / weights.max())
            level = min(level, len(chars) - 1)
            print(f"{'  ' + chars[level] + '   '}", end="")
        print()

ascii_heatmap(weights, sentence)
```

## 使用 PyTorch 的多头注意力

PyTorch 的 `nn.MultiheadAttention` 实现了我们搭建的功能，还包括多头分割和输出投影：

```python
import torch
import torch.nn as nn

d_model = 8
n_heads = 2
seq_len = 6

mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)

X_torch = torch.randn(1, seq_len, d_model)

output, attn_weights = mha(X_torch, X_torch, X_torch)

print(f"输入形状:            {X_torch.shape}")
print(f"输出形状:            {output.shape}")
print(f"注意力权重形状:      {attn_weights.shape}")
print(f"\n注意力权重（多头平均）:")
print(attn_weights[0].detach().numpy().round(3))
```

关键区别：多头注意力并行运行多个注意力函数，每个函数有自己的 Q,K,V，大小为 dk = d_model / n_heads，然后拼接结果。这允许模型同时关注不同类型的关系。

## 产出文件

本课生成：  
- `outputs/prompt-attention-explainer.md` - 用数据库查找类比解释注意力的提示词

## 练习

1. 修改 `scaled_dot_product_attention` 接收可选掩码矩阵，掩码位置设置为负无穷，softmax 前屏蔽（这就是因果/解码掩码效果）  
2. 从零实现多头注意力：将 Q, K, V 分成 `n_heads` 块，对每块计算注意力，拼接后通过最终权重矩阵 Wo 投影  
3. 用同一个 SelfAttention 实例输入两句不同的同长度句子，比较它们的注意力模式。哪些变化？哪些保持不变？

## 关键术语

| 术语 | 常见解释 | 实际含义 |
|------|----------|---------|
| Query (Q) | “问题向量” | 输入的学习投影，表示该词元在找什么信息 |
| Key (K) | “标签向量” | 代表该词元所含信息的学习投影，用来与查询匹配 |
| Value (V) | “内容向量” | 带有实际信息的学习投影，根据注意力得分被聚合 |
| 缩放点积注意力（Scaled dot-product attention） | “注意力公式” | softmax(QK^T / sqrt(dk)) @ V，缩放防止高维度下 softmax 饱和 |
| 自注意力（Self-attention） | “词元查看自己和其他词元” | Q, K, V 均来自相同序列，使每个位置关注序列所有位置 |
| 注意力权重（Attention weights） | “关注程度” | 由缩放点积的 softmax 产生的概率分布，表示关注位置权重 |
| 多头注意力（Multi-head attention） | “并行注意力” | 并行计算多个注意力函数，然后拼接结果，获得更丰富表示 |

## 深入阅读

- [Attention Is All You Need（Vaswani 等，2017）](https://arxiv.org/abs/1706.03762) - 原始 Transformer（Transformer 架构）论文
- [The Illustrated Transformer（Jay Alammar）](https://jalammar.github.io/illustrated-transformer/) - 最佳全架构视觉解读
- [The Annotated Transformer（Harvard NLP）](https://nlp.seas.harvard.edu/annotated-transformer/) - 带解释的逐行 PyTorch 实现
