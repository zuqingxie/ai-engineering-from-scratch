# Attention机制 — 突破点

> 解码器停止盯着压缩的摘要，开始关注整个源序列。从此之后，一切都是Attention加上工程实现。

**类型：** 构建  
**语言：** Python  
**先决条件：** 第5阶段·09课（序列到序列模型 Sequence-to-Sequence Models）  
**时间：** ~45分钟

## 问题

第09课以一个有限的失败告终。一个训练于玩具复制任务的GRU编码器-解码器，在长度为5时准确率90%左右，而长度80时准确率几乎接近随机。原因是结构性的，不是训练bug：编码器得到的所有信息必须压缩进一个固定大小的隐藏状态，解码器看不到别的任何信息。

Bahdanau、Cho和Bengio在2014年发表了一个三行代码的修正。不是只给解码器最终的编码器状态，而是保留所有编码器状态。解码器每一步计算编码器状态的加权平均，权重表示“当前解码器有多大程度需要关注编码器的第i个位置？”这个加权平均即上下文，在每个解码步骤都会变化。

这就是全部思想。Transformer进一步拓展它。自注意力（Self-attention）将它应用于单一序列。多头注意力（Multi-head attention）将其并行运行。但2014版已经打破了瓶颈，一旦有了它，转向Transformer主要是工程上的，而不是概念上。

## 概念

![Bahdanau attention: decoder queries all encoder states](../assets/attention.svg)

解码器在每一步`t`：

1. 使用上一步解码器隐藏状态`s_{t-1}`作为**query（查询）**。  
2. 对每个编码器隐藏状态`h_1, ..., h_T`打分。每个编码器位置一个标量。  
3. 对所有分数做softmax，得到和为1的注意力权重`α_{t,1}, ..., α_{t,T}`。  
4. 计算上下文向量`c_t = Σ α_{t,i} * h_i`，即编码器状态的加权平均。  
5. 解码器结合`c_t`和前一个输出token，生成下一个token。

加权平均才是重点。当解码器需要将“Je”翻译成“I”时，就重点加权编码器中代表“Je”的状态，权重高，其他低。当它需要“not”时，则把“pas”对应状态权重拉高。上下文向量每步都在调整。

## 形状（让大家第一次都卡住的地方）

这里是每个attention实现第一次常出错的地方。请慢慢读。

| 项目 | 形状 | 说明 |
|------|-------|-------|
| 编码器隐藏状态 `H` | `(T_enc, d_h)` | 双向LSTM时，`d_h = 2 * d_hidden` |
| 解码器隐藏状态 `s_{t-1}` | `(d_s,)` | 单个向量 |
| 注意力打分 `e_{t,i}` | 标量 | 每个编码器位置一个 |
| 注意力权重 `α_{t,i}` | 标量 | softmax过后，所有i权重和为1 |
| 上下文向量 `c_t` | `(d_h,)` | 与编码器状态形状一致 |

**Bahdanau（加性）得分公式：** `e_{t,i} = v_α^T * tanh(W_a * s_{t-1} + U_a * h_i)`。

- `s_{t-1}`形状为`(d_s,)`，`h_i`形状为`(d_h,)`。  
- `W_a`形状为`(d_attn, d_s)`，`U_a`形状为`(d_attn, d_h)`。  
- tanh内部的和的形状为`(d_attn,)`。  
- `v_α`形状为`(d_attn,)`。与`v_α`的内积降维为标量。**这就是`v_α`的作用。** 它不是魔法，是将注意力维度的向量投影成标量得分。

**Luong（乘积）得分公式。** 三种变体：

- `dot`：`e_{t,i} = s_t^T * h_i`。要求`d_s == d_h`，硬性约束。如果编码器是双向的，通常跳过这个。  
- `general`：`e_{t,i} = s_t^T * W * h_i`，其中`W`形状为`(d_s, d_h)`。去除了维度相等的限制。  
- `concat`：实质上是Bahdanau的形式。很少用，因为前两种更便宜。

**一个关于Bahdanau和Luong的致命坑。** Bahdanau用的是`s_{t-1}`（生成当前词前的解码器状态），Luong则用`s_t`（生成当前词后的状态）。混淆它们会导致梯度微妙错误，极难调试。挑一个论文并坚持其规范即可。

## 实现

### 步骤1：加性（Bahdanau）注意力

```python
import numpy as np


def additive_attention(decoder_state, encoder_states, W_a, U_a, v_a):
    projected_dec = W_a @ decoder_state
    projected_enc = encoder_states @ U_a.T
    combined = np.tanh(projected_enc + projected_dec)
    scores = combined @ v_a
    weights = softmax(scores)
    context = weights @ encoder_states
    return context, weights


def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()
```

对照上表检查形状。`encoder_states`形状为`(T_enc, d_h)`。`projected_enc`形状为`(T_enc, d_attn)`。`projected_dec`形状为`(d_attn,)`，支持广播。`combined`形状为`(T_enc, d_attn)`。`scores`形状为`(T_enc,)`。`weights`形状为`(T_enc,)`。`context`形状为`(d_h,)`。这样就可以了。

### 步骤2：Luong的dot和general注意力

```python
def dot_attention(decoder_state, encoder_states):
    scores = encoder_states @ decoder_state
    weights = softmax(scores)
    return weights @ encoder_states, weights


def general_attention(decoder_state, encoder_states, W):
    projected = W.T @ decoder_state
    scores = encoder_states @ projected
    weights = softmax(scores)
    return weights @ encoder_states, weights
```

每个三行代码。Luong论文因此流行。一致准确率，代码更简洁。

### 步骤3：数值示例

给定三个编码器状态（大致对应“cat”，“sat”，“mat”），以及一个解码器状态与第一个状态最匹配，注意力集中在位置0。当解码器状态挪到与第三个状态匹配时，注意力切换到位置2。上下文向量相应跟踪。

```python
H = np.array([
    [1.0, 0.0, 0.2],
    [0.5, 0.5, 0.1],
    [0.1, 0.9, 0.3],
])

s_close_to_cat = np.array([0.9, 0.1, 0.2])
ctx, w = dot_attention(s_close_to_cat, H)
print("weights:", w.round(3))
```

```text
weights: [0.464 0.305 0.231]
```

第一个项占优。然后把解码器状态移近第三个编码器状态，观察权重变化。这就是注意力，就是显式对齐。

### 步骤4：为什么这是通往Transformer的桥梁

将上述语言换成Q/K/V：

- **Query** = 解码器状态`s_{t-1}`  
- **Key** = 编码器状态（用于打分）  
- **Value** = 编码器状态（用于加权求和）

在经典attention中，Key和Value是一样的。自注意（Self-attention）分开它们：你对序列自身做查询，用不同的学习投影生成K和V。多头注意力并行多个不同投影。Transformer多次堆叠这个阶段，并舍弃RNN。

数学一样，形状一样。Bahdanau到缩放点积attention（scaled dot-product attention）的教学跳跃主要是符号变化。

## 使用

PyTorch和TensorFlow直接提供了注意力模块。

```python
import torch
import torch.nn as nn

mha = nn.MultiheadAttention(embed_dim=128, num_heads=8, batch_first=True)
query = torch.randn(2, 5, 128)
key = torch.randn(2, 10, 128)
value = torch.randn(2, 10, 128)

output, weights = mha(query, key, value)
print(output.shape, weights.shape)
```

```text
torch.Size([2, 5, 128]) torch.Size([2, 5, 10])
```

这就是一个Transformer注意力层。查询批次长度为5，键/值批次长度为10，均为128维，共8个头。`output`是新的上下文增强型query。`weights`是大小为5x10的对齐矩阵，方便可视化。

### 经典attention为何依旧重要

- 教学。单头、单层、基于RNN的版本让所有概念更直观。  
- 对于Transformer难以适用的设备端序列任务。  
- 2014-2017年发表的任何论文。不了解Bahdanau的约定容易误读。  
- 机器翻译中的细粒度对齐分析。原始注意力权重是解释性工具，哪怕在Transformer模型中，阅读需了解其含义。

### 注意力权重作为解释的陷阱

注意力权重看起来很直观。它们是对所有位置求和为1的权重，可以绘制图表；高权重意味着“模型关注此处”。评审们很喜欢。

但它们并没有看起来这么可解释。Jain和Wallace（2019）展示过注意力分布可以被置换甚至替换为任意分布，而模型预测不变。切勿在未做消融或反事实验证前，把注意力权重当作模型推理的确凿证据。

## 部署

保存为`outputs/prompt-attention-shapes.md`：

```markdown
---
name: attention-shapes
description: 调试attention实现中的形状错误。
phase: 5
lesson: 10
---

给定一个有问题的attention实现，你需要定位形状不匹配的地方。输出：

1. 哪个矩阵形状错误，标明tensor名称。  
2. 它应该的形状，基于(d_s, d_h, d_attn, T_enc, T_dec, batch_size)推导。  
3. 一行修复方案。可能是转置、重塑或投影。  
4. 一个回归测试。通常是：断言`output.shape == (batch, T_dec, d_h)`和`weights.shape == (batch, T_dec, T_enc)`且`weights.sum(dim=-1)`接近1。

拒绝建议静默广播的修复方案。隐藏广播的错误会导致后续默默的准确率下降，是注意力中最难察觉的Bug。

关于Bahdanau的混淆，要求输入解码器状态为`s_{t-1}`（生成当前词之前的状态）。Luong为`s_t`（生成当前词之后）。对于点积，重点是检查query和key维度不匹配，是第一次实现的最常见错误。
```

## 练习

1. **简单。** 实现带mask的softmax，使编码器的padding token注意力权重为零。用一个不定长序列的batch测试。  
2. **中等。** 在Luong的`general`形式上添加多头注意力。将`d_h`拆分为`n_heads`组，分别计算注意力，然后拼接。验证单头时结果与之前实现一致。  
3. **困难。** 训练一个带Bahdanau注意力的GRU编码器-解码器，处理第09课的玩具复制任务。绘制准确率随序列长度的变化。与无注意力基线比较，你应看到长度越长，差距越大，验证注意力解决了瓶颈问题。

## 关键词

| 术语 | 常见说法 | 实际含义 |
|--------|-------------|-------------|
| Attention（注意力） | 看东西 | 对value序列的加权平均，权重由query-key相似度计算得出。 |
| Query, Key, Value | QKV | 三个投影：Q发问，K是匹配对象，V是返回内容。 |
| Additive attention | Bahdanau（加性） | 前馈神经网络计算得分：`v^T tanh(W q + U k)`。 |
| Multiplicative attention | Luong dot / general（乘积） | 得分为`q^T k`或`q^T W k`。更便宜，多数任务准确率相当。 |
| Alignment matrix | 美观的图 | 注意力权重组成的`(T_dec, T_enc)`矩阵。可读来查看模型关注了什么。 |

## 相关阅读

- [Bahdanau, Cho, Bengio (2014). Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) — 原始论文。  
- [Luong, Pham, Manning (2015). Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025) — 详细介绍三种得分变体及其比较。  
- [Jain and Wallace (2019). Attention is not Explanation](https://arxiv.org/abs/1902.10186) — 注意力解释性警示。  
- [Dive into Deep Learning — Bahdanau Attention](https://d2l.ai/chapter_attention-mechanisms-and-transformers/bahdanau-attention.html) — PyTorch可运行示例讲解。
