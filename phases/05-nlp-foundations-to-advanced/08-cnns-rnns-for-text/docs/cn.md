# 用于文本的 CNN 和 RNN

> 卷积学习 n-gram。循环记忆信息。两者都被注意力机制取代。但在受限硬件上，两者仍然重要。

**类型：** 构建  
**语言：** Python  
**先决条件：** 第3阶段 · 11（PyTorch入门），第5阶段 · 03（词嵌入），第4阶段 · 02（从零实现卷积）  
**时长：** 约75分钟

## 问题

TF-IDF 和 Word2Vec 生成的是忽略词序的扁平向量。基于它们的分类器无法区分 `dog bites man` 和 `man bites dog`。词序有时携带关键信号。

在 Transformer 出现之前，有两大类架构填补了这一空白。

**文本卷积网络（TextCNN）。** 在词嵌入序列上应用一维卷积。宽度为3的滤波器是可学习的三元组检测器：跨越三个词并输出一个分数。堆叠不同的宽度（2，3，4，5）以检测多尺度模式。使用最大池化到固定大小表示。结构平坦、并行且快速。

**递归网络（RNN、LSTM、GRU）。** 逐个处理标记，保持一个携带前向信息的隐藏状态。顺序处理、带有记忆、支持灵活的输入长度。从2014年到2017年主导序列建模，之后注意力机制出现。

本课将构建两者，然后介绍促使注意力被提出的失败原因。

## 概念

**TextCNN**（Kim，2014）。标记先嵌入。宽度为 `k` 的一维卷积滑动窗口覆盖连续的 `k` 元组嵌入，生成一个特征图。对特征图进行全局最大池化，选取最强激活值。从多个滤波器宽度的最大池化输出中连接。送入分类器头部。

它的原理。滤波器是可学习的 n-gram。最大池化使得位置不变，所以“not good”无论在评论开头还是中间都会激活相同特征。三种宽度，每种100个滤波器，等于300个可学习的 n-gram 检测器。训练是并行的，无顺序依赖。

**RNN。** 每个时间步 `t`，隐藏状态由 `h_t = f(W * x_t + U * h_{t-1} + b)` 计算。`W`，`U`，`b` 在时间上共享。时间 `T` 时的隐藏状态是整个序列前缀的摘要。用于分类时，对 `h_1 ... h_T` 进行池化（最大、平均或最后一个）。

普通 RNN 存在梯度消失问题。**LSTM** 添加门控机制，决定遗忘、存储和输出，稳定了长序列的梯度。**GRU** 简化为两个门，参数更少，表现相似。

**双向 RNN** 同时运行一个正向和一个反向 RNN，连接隐藏状态。每个标记的表示均见到左右上下文。对序列标注任务至关重要。

## 构建

### 第1步：PyTorch 实现 TextCNN

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_classes, filter_widths=(2, 3, 4), n_filters=64, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, n_filters, kernel_size=k)
            for k in filter_widths
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(n_filters * len(filter_widths), n_classes)

    def forward(self, token_ids):
        x = self.embed(token_ids).transpose(1, 2)
        pooled = []
        for conv in self.convs:
            c = F.relu(conv(x))
            p = F.max_pool1d(c, c.size(2)).squeeze(2)
            pooled.append(p)
        h = torch.cat(pooled, dim=1)
        return self.fc(self.dropout(h))
```

`transpose(1, 2)` 将 `[batch, seq_len, embed_dim]` 转换为 `[batch, embed_dim, seq_len]`，因为 `nn.Conv1d` 将中间维度视为通道。最大池化输出固定大小，不受输入长度影响。

### 第2步：LSTM 分类器

```python
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_classes, bidirectional=True, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=bidirectional)
        factor = 2 if bidirectional else 1
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * factor, n_classes)

    def forward(self, token_ids):
        x = self.embed(token_ids)
        out, _ = self.lstm(x)
        pooled = out.max(dim=1).values
        return self.fc(self.dropout(pooled))
```

对序列进行最大池化，而非取最后状态池化。分类时，最大池化通常优于最后隐藏状态，因为长序列末端的信息会主导最后状态。

### 第3步：梯度消失演示（直觉理解）

没有门控的普通 RNN 无法学习长距离依赖。考虑任务：预测标记 `A` 是否出现在序列的任意位置。若 `A` 在位置1，序列长度100，损失反向传播的梯度需经过99次递归权重乘法。若权重小于 1，则梯度消失；大于 1，则梯度爆炸。

```python
def vanishing_gradient_sim(seq_len, recurrent_weight=0.9):
    import math
    return math.pow(recurrent_weight, seq_len)


# 权重=0.9，100步时：
#   0.9 ^ 100 ≈ 2.7e-5
# 第100步到第1步的梯度几乎为零。
```

LSTM 通过 **细胞状态** 实现只有加法交互（遗忘门乘法缩放，但梯度沿“高速公路”流动），从而稳定了100步甚至更长序列的训练。GRU 以更少参数实现类似功能。

### 第4步：为何仍不足够

即使有了 LSTM，依然存在三大问题。

1. **顺序瓶颈。** 训练长度为1000的序列需要1000步前后向传播，无法跨时间并行。
2. **编码器-解码器中固定大小上下文向量。** 解码器仅见编码器最终隐藏状态，长输入会丢失细节。第09课将专门讲解。
3. **远距离依赖的准确率天花板。** LSTM 优于普通 RNN，但仍难以跨200步准确传递特定信息。

注意力机制解决了上述三大问题。Transformer 完全放弃循环结构。第10课是转折点。

## 使用方法

PyTorch 的 `nn.LSTM`、`nn.GRU` 和 `nn.Conv1d` 已经可用于生产。训练代码标准。

Hugging Face 提供预训练的嵌入，可直接作为输入层：

```python
from transformers import AutoModel

encoder = AutoModel.from_pretrained("bert-base-uncased")
for param in encoder.parameters():
    param.requires_grad = False


class BertCNN(nn.Module):
    def __init__(self, n_classes, filter_widths=(2, 3, 4), n_filters=64):
        super().__init__()
        self.encoder = encoder
        self.convs = nn.ModuleList([nn.Conv1d(768, n_filters, kernel_size=k) for k in filter_widths])
        self.fc = nn.Linear(n_filters * len(filter_widths), n_classes)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        x = out.transpose(1, 2)
        pooled = [F.max_pool1d(F.relu(conv(x)), kernel_size=conv(x).size(2)).squeeze(2) for conv in self.convs]
        return self.fc(torch.cat(pooled, dim=1))
```

适用条件检查清单：

- **边缘/设备端推理。** 配合 GloVe 词嵌入的 TextCNN 体积比 Transformer 小 10 到 100 倍。若目标设备是手机，这是合适选择。
- **流式/在线分类。** RNN 每次处理一个标记；Transformer 需要完整序列。实时输入文本，LSTM 仍占优势。
- **微型模型基线。** 针对新任务快速迭代。CPU 上 5 分钟即可训练一个 TextCNN。
- **有限标注数据的序列标注。** BiLSTM-CRF（第06课）依然是标注 1k-10k 句子时的生产级命名实体识别架构。

其他场景均优先使用 Transformer。

## 部署

保存为 `outputs/prompt-text-encoder-picker.md`：

```markdown
---
name: text-encoder-picker
description: 根据约束条件选择文本编码器架构。
phase: 5
lesson: 08
---

给定约束（任务、数据量、延迟预算、部署目标、计算预算），输出：

1. 编码器架构：TextCNN、BiLSTM、BiLSTM-CRF、Transformer 微调，或“使用预训练 Transformer 作为冻结编码器 + 小型头部”。
2. 嵌入输入：随机初始化，GloVe / fastText 冻结，或上下文化 Transformer 嵌入。
3. 五行训练配方：优化器，学习率，批大小，训练轮数，正则化。
4. 一个监控信号。对于 RNN/CNN 模型：无注意力机制表明可能错过长距离依赖；检查不同长度的准确率。对于 Transformer：学习率过高导致微调崩溃；检查训练损失。

当数据少于约500条标注示例时，拒绝推荐 Transformer 微调，除非表现出 TextCNN / BiLSTM 基线已达到瓶颈。边缘部署需优先考虑架构选择。
```

## 练习

1. **简单。** 在一个3分类玩具数据集（自创数据）上训练 TextCNN。验证滤波器宽度组合（2，3，4）在平均 F1 上优于单一宽度（3）。
2. **中等。** 实现 LSTM 分类器的最大池化、平均池化和最后状态池化。在小数据集上比较并记录哪种池化效果最好，假设原因。
3. **困难。** 构建 BiLSTM-CRF 命名实体识别器（结合第06课和本课内容）。在 CoNLL-2003 上训练。与第06课的 CRF-alone 基线及 BERT 微调比较。报告训练时间、内存和 F1。

## 关键词

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| TextCNN | 文本卷积神经网络 | 词嵌入上的多层一维卷积及全局最大池化。Kim (2014)。 |
| RNN | 递归神经网络 | 每个时间步更新隐藏状态：`h_t = f(W x_t + U h_{t-1})`。 |
| LSTM | 门控递归神经网络 | 添加输入/遗忘/输出门和细胞状态。支持长序列稳定训练。 |
| GRU | 简化的 LSTM | 两个门替代三个门。精度相似，参数更少。 |
| 双向 | 双向处理 | 正反向 RNN 拼接。每个标记看到双侧上下文。 |
| 梯度消失 | 训练信号衰竭 | 纯 RNN 中权重 < 1 导致早期时间步梯度基本为零。 |

## 延伸阅读

- [Kim, Y. (2014). Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882) — TextCNN 论文，8页，可读性强。
- [Hochreiter, S. and Schmidhuber, J. (1997). Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf) — LSTM 论文，意外的清晰。
- [Olah, C. (2015). Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) — 让 LSTM 易懂的图示解读。
