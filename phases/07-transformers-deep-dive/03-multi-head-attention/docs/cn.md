# 多头注意力（Multi-Head Attention）

> 一个注意力头一次学习一种关系。八个头学习八种。头是免费的。多用几个。

**类型：** 构建  
**语言：** Python  
**先决条件：** 第7阶段·02（从零开始的自注意力）  
**时长：** 约75分钟

## 问题

单个自注意力头计算一个注意力矩阵。该矩阵捕捉一种关系——通常是最小化训练信号上的损失的那种关系。如果你的数据包含主谓一致、共指、长距离语篇、句法块等多种关系纠结混合，单个头会把它们混合到一个softmax分布中，丢失了一半信号。

2017年Vaswani论文的解决方案是：并行运行多个注意力函数，每个有自己独立的Q、K、V投影，然后将输出连接起来。每个头在维度为`d_model / n_heads`的较小子空间中操作。总参数量保持不变，但表达能力提升。

多头注意力是2026年所有Transformer默认搭载的机制。唯一的争议是*用多少头*以及键值是否共享投影（Grouped-Query注意力（Grouped-Query Attention）、多查询注意力（Multi-Query Attention）、多头潜在注意力（Multi-head Latent Attention））。

## 概念

![多头注意力分拆、注意、连接](../assets/multi-head-attention.svg)

**分拆（Split）。** 取形状为 `(N, d_model)` 的输入 `X`。投影到Q、K、V，形状均为 `(N, d_model)`。重塑为 `(N, n_heads, d_head)`，其中 `d_head = d_model / n_heads`。转置为 `(n_heads, N, d_head)`。

**并行注意力（Attend in parallel）。** 在每个头内运行缩放点积注意力。每个头产生形状为 `(N, d_head)` 的输出。各头在不同的子空间中独立操作，注意力计算过程中不交互。

**连接投影（Concatenate and project）。** 将各头堆叠回 `(N, d_model)`，并乘以一个学习到的输出矩阵 `W_o`，形状为 `(d_model, d_model)`。`W_o` 是各头混合的地方。

**为什么有效。** 每个头可以专门学习，而不会为表现空间竞争。2019-2024年的探查研究显示各头具有不同角色：位置头、注意上一个token的头、复制头、命名实体头、归纳头（支撑上下文学习）。

**2026年变体谱系：**

| 变体 | Q 头数 | K/V 头数 | 使用者         |
|-------|---------|----------|----------------|
| 多头（MHA） | N       | N        | GPT-2, BERT, T5 |
| 多查询（MQA） | N       | 1        | PaLM, Falcon    |
| 分组查询（GQA） | N       | G（例如N/8） | Llama 2 70B, Llama 3+, Qwen 2+, Mistral |
| 多头潜在（MLA） | N       | 压缩到低秩      | DeepSeek-V2, V3 |

GQA是现代默认，因为它通过将KV缓存减少`N/G`倍，几乎保持了完整质量。MLA则将K/V压缩到潜在空间，计算时再投影回去——消耗一些浮点运算，但节约更多内存。

## 构建它

### 步骤 1：从已有的单头注意力拆分头数

以第02课的`SelfAttention`为基础，用分拆/连接包裹。见`code/main.py`的numpy实现，逻辑如下：

```python
def split_heads(X, n_heads):
    n, d = X.shape
    d_head = d // n_heads
    return X.reshape(n, n_heads, d_head).transpose(1, 0, 2)  # (heads, n, d_head)

def combine_heads(H):
    h, n, d_head = H.shape
    return H.transpose(1, 0, 2).reshape(n, h * d_head)
```

一次重塑和一次转置。无循环。这正是PyTorch中`nn.MultiheadAttention`的实现方式。

### 步骤 2：对每个头运行缩放点积注意力

每个头获得自己的Q、K、V切片。注意力变成一个批量矩阵乘法：

```python
def mha_forward(X, W_q, W_k, W_v, W_o, n_heads):
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    Qh = split_heads(Q, n_heads)         # (heads, n, d_head)
    Kh = split_heads(K, n_heads)
    Vh = split_heads(V, n_heads)
    scores = Qh @ Kh.transpose(0, 2, 1) / np.sqrt(Qh.shape[-1])
    weights = softmax(scores, axis=-1)
    out = weights @ Vh                    # (heads, n, d_head)
    concat = combine_heads(out)
    return concat @ W_o, weights
```

在真实硬件上`Qh @ Kh.transpose(...)`是一次`bmm`。GPU看到的单个批处理矩阵乘法形状为 `(heads, N, d_head) × (heads, d_head, N) -> (heads, N, N)`。增加头是免费的。

### 步骤 3：分组查询注意力变体

只有键值投影变动。Q有`n_heads`组；K和V有更少的`n_kv_heads < n_heads`组，结果复制匹配Q头数：

```python
def gqa_project(X, W, n_kv_heads, n_heads):
    kv = split_heads(X @ W, n_kv_heads)       # (kv_heads, n, d_head)
    repeat = n_heads // n_kv_heads
    return np.repeat(kv, repeat, axis=0)      # (n_heads, n, d_head)
```

推理时，缓存中只存`n_kv_heads`份键值，减少内存。Llama 3 70B采用64查询头和8缓存头，缓存大小缩减8倍。

### 步骤 4：探查每个头学到的内容

用4个头的MHA运行一个短句子。打印每头的 `(N, N)` 注意力矩阵。你会看到不同头捕捉到不同结构，即使是随机初始化——这部分是信号，部分源于子空间的旋转对称性。

## 使用它

PyTorch中的一句话版本：

```python
import torch.nn as nn

mha = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
```

PyTorch 2.5+的GQA：

```python
from torch.nn.functional import scaled_dot_product_attention

# scaled_dot_product_attention 会自动在CUDA上调用Flash Attention。
# 对于GQA，传入Q形状为 (B, n_heads, N, d_head)，K、V形状为
# (B, n_kv_heads, N, d_head)。PyTorch会自动处理重复。
out = scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
```

**多少头合适？** 2026年生产模型的经验法则：

| 模型大小 | d_model | n_heads | d_head |
|----------|---------|---------|--------|
| 小型（约1.25亿） | 768     | 12      | 64     |
| 基础（约3.5亿） | 1024    | 16      | 64     |
| 大型（约10亿） | 2048    | 16      | 128    |
| 前沿（约700亿） | 8192    | 64      | 128    |

`d_head`几乎总是64或128。它是一个头“能看到”多少信息的单位。低于32时，头开始和缩放因子`sqrt(d_head)`冲突；超过256时则失去“多小专家”的好处。

## 部署它

见 `outputs/skill-mha-configurator.md`。该技能根据参数预算、序列长度和部署目标，推荐头数、kv头数和投影策略。

## 练习

1. **简单。** 取`code/main.py`中的MHA，将`n_heads`从1改成16，保持`d_model=64`。在一个合成复制任务上一层微型模型训练，绘制loss。更多头有助于性能提升还是达到瓶颈或退步？
2. **中等。** 实现MQA（所有查询头共享一个KV头）。测量参数量相比全MHA下降多少。计算N=2048时KV缓存体积缩减多少。
3. **困难。** 实现一个小型多头潜在注意力版本：将K、V压缩为秩为`r`的潜在表示，缓存该潜在，注意力计算时解压。哪个`r`下缓存内存下降到不足全MHA的1/8，同时验证集困惑度误差不超过1 bit？

## 关键词

| 术语       | 人们说的说法         | 实际含义                             |
|------------|----------------------|------------------------------------|
| 头（Head） | “单个注意力电路”     | 一个Q/K/V投影，维度为`d_head = d_model / n_heads`，对应一个注意力矩阵。 |
| d_head     | “头维度”             | 单头隐藏宽度；生产中几乎总为64或128。  |
| 分拆/组合  | “重塑技巧”           | `(N, d_model) ↔ (n_heads, N, d_head)` 的reshape+transpose，环绕注意力计算。 |
| W_o        | “输出投影”           | 连接头后应用的 `(d_model, d_model)` 矩阵；头混合之处。     |
| MQA        | “一个KV头”           | 多查询注意力：单一共享K/V投影。最小KV缓存，略有质量损失。   |
| GQA        | “Llama 2以来默认”    | 分组查询注意力：`n_kv_heads < n_heads`，K/V投影重复匹配查询头数。 |
| MLA        | “DeepSeek的技巧”     | 多头潜在注意力：K/V压缩成低秩潜在，计算时解压。          |
| 归纳头（Induction head） | “上下文学习背后的电路” | 一对检测之前出现并复制其后token的头。               |

## 相关阅读

- [Vaswani等（2017）。Attention Is All You Need §3.2.2](https://arxiv.org/abs/1706.03762) — 原始多头规范。  
- [Shazeer（2019）。快速Transformer解码：一个写头就是全部](https://arxiv.org/abs/1911.02150) — MQA论文。  
- [Ainslie等（2023）。GQA：从多头检查点训练广义多查询Transformer模型](https://arxiv.org/abs/2305.13245) — 训练后如何将MHA转为GQA。  
- [DeepSeek-AI（2024）。DeepSeek-V2技术报告](https://arxiv.org/abs/2405.04434) — MLA及其为何在缓存内存上优于MHA/GQA。  
- [Olsson等（2022）。上下文学习和归纳头](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) — 头实际做什么的机械视角。
