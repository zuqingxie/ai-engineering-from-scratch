# 位置编码 — 正弦、RoPE、ALiBi

> 注意力机制是对排列不变的。"The cat sat on the mat" 和 "mat the on sat cat the" 在没有位置信号时产生相同输出。解决这个问题的三种算法——每种对“位置”的定义各不相同。

**类型：** 构建  
**语言：** Python  
**先决条件：** 第7阶·02（自注意力 Self-Attention），第7阶·03（多头注意力 Multi-Head Attention）  
**时间：** 约45分钟

## 问题

缩放点积注意力（scaled dot-product attention）对顺序是无感知的。注意力矩阵 `softmax(Q K^T / √d) V` 是从成对相似度计算的。将输入 `X` 的行打乱，输出行会以相同方式打乱。注意力机制内部不关心位置。

这对词袋模型（bag-of-words）不算是错误。但对语言、代码、音频、视频——任何顺序携带意义的东西——是致命的。

解决方案是以某种方式将位置注入到嵌入（embedding）中。三代答案：

1. **绝对正弦编码**（Vaswani 2017）。将位置的 `sin`/`cos` 加到嵌入中。简单无需学习，超出训练长度时效果差。
2. **RoPE — 旋转位置嵌入 Rotary Position Embeddings**（Su 2021）。按位置比例旋转 Q 和 K 向量。在点积中直接编码*相对*位置。2026年主流方案。
3. **ALiBi — 带线性偏置的注意力 Attention with Linear Biases**（Press 2022）。完全跳过嵌入；对注意力分数按距离加上每个头的线性惩罚。极佳的长度外推能力。

截至2026年，几乎所有前沿开源模型都使用 RoPE：Llama 2/3/4，Qwen 2/3，Mistral，Mixtral，DeepSeek-V3，Kimi。少数长上下文模型使用 ALiBi 或其现代变种。绝对正弦属于历史方案。

## 概念

![正弦绝对位置编码 vs RoPE旋转 vs ALiBi距离偏置](../assets/positional-encoding.svg)

### 绝对正弦编码

预先计算一个固定矩阵 `PE`，形状为 `(max_len, d_model)`：

```text
PE[pos, 2i]   = sin(pos / 10000^(2i / d_model))
PE[pos, 2i+1] = cos(pos / 10000^(2i / d_model))
```

然后在注意力之前用 `X' = X + PE[:N]`。每个维度对应一个不同频率的正弦波。模型通过相位模式学会读取位置。当超出 `max_len` 时失效：模型没见过位置2048，只见过0–2047，就不知道该怎么办。

### RoPE

旋转 Q 和 K 向量（不是嵌入）。针对一对坐标 `(2i, 2i+1)`：

```text
[q'_2i    ]   [ cos(pos·θ_i)  -sin(pos·θ_i) ] [q_2i   ]
[q'_2i+1  ] = [ sin(pos·θ_i)   cos(pos·θ_i) ] [q_2i+1 ]

θ_i = base^(-2i / d_head),  base 默认为 10000
```

对位置为 `pos_k` 的 keys 应用相同旋转。点积 `q'_m · k'_n` 只依赖 `(m - n)`。即：**注意力分数仅依赖相对距离**，虽然旋转依据是绝对位置。巧妙设计。

RoPE 扩展：`base` 可缩放（NTK-aware，YaRN，LongRoPE），实现长上下文无重训外推。Llama 3即用此法将上下文扩展从8K到128K。

### ALiBi

跳过嵌入技巧。直接对注意力分数加偏置：

```text
attn_score[i, j] = (q_i · k_j) / √d  -  m_h · |i - j|
```

其中 `m_h` 是头专属斜率（如 `1 / 2^(8·h/H)`）。距离近的标记获增强，距离远的受惩罚。无训练成本。论文表明外推能力优于正弦方案，并在原训练长度下表现匹配RoPE。

### 2026年该选什么

| 变体               | 外推能力        | 训练成本         | 使用者                              |
|--------------------|-----------------|------------------|-----------------------------------|
| 绝对正弦编码       | 差              | 免费             | 原始Transformer，早期BERT          |
| 学习型绝对编码     | 无              | 极小             | GPT-2，GPT-3                      |
| RoPE               | 好，支持缩放    | 免费             | Llama 2/3/4，Qwen 2/3，Mistral，DeepSeek-V3，Kimi |
| RoPE + YaRN        | 极好            | 微调阶段         | Qwen2-1M，Llama 3.1 128K          |
| ALiBi              | 极好            | 免费             | BLOOM，MPT，百川                   |

RoPE胜在无需改动架构即可嵌入注意力，编码相对位置，且`base`超参便于长上下文微调。

## 构建它

### 步骤1：正弦编码

见 `code/main.py`。四行代码实现：

```python
def sinusoidal(N, d):
    pe = [[0.0] * d for _ in range(N)]
    for pos in range(N):
        for i in range(d // 2):
            theta = pos / (10000 ** (2 * i / d))
            pe[pos][2 * i]     = math.sin(theta)
            pe[pos][2 * i + 1] = math.cos(theta)
    return pe
```

在第一层注意力之前将该编码加到嵌入矩阵上。

### 步骤2：RoPE 应用于 Q, K

RoPE 直接对 Q 和 K 就地旋转。每对维度：

```python
def apply_rope(x, pos, base=10000):
    d = len(x)
    out = list(x)
    for i in range(d // 2):
        theta = pos / (base ** (2 * i / d))
        c, s = math.cos(theta), math.sin(theta)
        a, b = x[2 * i], x[2 * i + 1]
        out[2 * i]     = a * c - b * s
        out[2 * i + 1] = a * s + b * c
    return out
```

关键：对位置为 `m` 的 Q 和位置为 `n` 的 K 应用同样函数。它们点积在每对坐标计算时都会带入 `cos((m-n)·θ_i)` 因子。注意力机制免费学出相对位置。

### 步骤3：ALiBi 斜率与偏置

```python
def alibi_bias(n_heads, seq_len):
    # slope_h = 2 ** (-8 * h / n_heads) for h = 1..n_heads
    slopes = [2 ** (-8 * (h + 1) / n_heads) for h in range(n_heads)]
    bias = []
    for m in slopes:
        row = [[-m * abs(i - j) for j in range(seq_len)] for i in range(seq_len)]
        bias.append(row)
    return bias  # 在 softmax 前加到注意力分数矩阵
```

将 `bias[h]` 加到第 `h` 个头的 `(seq_len, seq_len)` 注意力分数矩阵上，再进行 softmax。

### 步骤4：验证 RoPE 的相对距离属性

随机选两向量 `a, b`。分别在 `(pos_a, pos_b)` 和 `(pos_a + k, pos_b + k)` 位置旋转。两者点积应在浮点误差范围内相等。此属性彰显 RoPE 只依赖相对偏移，不受绝对位置影响。

## 使用它

PyTorch 2.5+ 自带 RoPE 工具于 `torch.nn.functional`。大多数生产应用用 `flash_attn` 或 `xformers`，RoPE 在注意力核内实现。

```python
from transformers import AutoModel
model = AutoModel.from_pretrained("meta-llama/Llama-3.2-3B")
# model.config.rope_scaling → {"type": "yarn", "factor": 32.0, "original_max_position_embeddings": 8192}
```

**2026年长上下文技巧：**

- **NTK-aware 插值。** 扩展从4K到16K+时，缩放 `base` 为 `base * (scale_factor)^(d/(d-2))`。
- **YaRN。** 更先进的插值，能保持长上下文注意力熵。Llama 3.1 128K 采用。
- **LongRoPE。** 微软2024年方法，用进化搜索选维度别缩放因子。Phi-3-Long采用。
- **位置插值加微调。** 按扩展因子缩小位置，微调1–5B tokens。效果惊人。

## 部署它

见 `outputs/skill-positional-encoding-picker.md`。该技能根据目标上下文长度、外推需求和训练预算为新模型挑选编码策略。

## 练习

1. **简单。** 绘制 `max_len=512, d=128` 的正弦 `PE` 矩阵热力图。确认维度索引越大条纹越宽。
2. **中等。** 实现 NTK-aware RoPE 缩放。训练小型语言模型，训练序列长度256，测试时用和不用缩放对1024长度计算困惑度。
3. **困难。** 在同一注意力模块实现 ALiBi 与 RoPE。训练4层Transformer，复制任务，序列长度512。测试时外推至2048并比较性能退化。

## 关键词

| 术语               | 流行说法           | 实际含义                       |
|--------------------|--------------------|------------------------------|
| Positional encoding（位置编码） | “告诉注意力顺序” | 任何加入嵌入或注意力的位置信号。 |
| Sinusoidal（正弦编码）           | “最初的那个”       | 按几何频率的 `sin/cos` 加到嵌入；不支持外推。 |
| RoPE                          | “旋转嵌入”         | 依据位置角度旋转 Q、K；点积编码相对距离。 |
| ALiBi                         | “线性偏置技巧”     | 在注意力分数加 `-m·\|i-j\|`；无需嵌入，优越外推。 |
| base                         | “RoPE旋钮”         | RoPE中的频率缩放因子；推理时增大以扩展上下文。 |
| NTK-aware                    | “RoPE缩放技巧”     | 缩放 `base`，避免高频维度在扩展时被压缩。 |
| YaRN                         | “高级方法”         | 逐维插值+外推，保持注意力熵。   |
| Extrapolation（外推）          | “超出训练长度有效” | 位置编码能否在训练最大长度外提供正确输出。 |

## 深入阅读

- [Vaswani等（2017）。Attention Is All You Need §3.5](https://arxiv.org/abs/1706.03762) — 原始正弦编码。
- [Su等（2021）。RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) — RoPE论文。
- [Press, Smith, Lewis（2021）。Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation](https://arxiv.org/abs/2108.12409) — ALiBi论文。
- [Peng等（2023）。YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071) — 目前最先进的RoPE缩放。
- [Chen等（2023）。Extending Context Window of Large Language Models via Positional Interpolation](https://arxiv.org/abs/2306.15595) — Meta的Llama 2长上下文研究。
- [Ding等（2024）。LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens](https://arxiv.org/abs/2402.13753) — 微软方法，Phi-3-Long采用，参见“使用它”部分。
- [HuggingFace Transformers — `modeling_rope_utils.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_rope_utils.py) — 各类RoPE缩放方案（默认、线性、动态、YaRN、LongRoPE、Llama-3）生产级实现。
