# Full Transformer —— 编码器 + 解码器

> 注意力是明星。一切其他——残差、归一化、前馈、交叉注意力——是让你能堆叠它的脚手架。

**类型：** 构建  
**语言：** Python  
**先决条件：** 第7阶段 · 02（自注意力）、第7阶段 · 03（多头注意力）、第7阶段 · 04（位置编码）  
**时间：** ~75分钟

## 问题

单个注意力层是特征提取器，不是模型。每层一次矩阵乘法的容量不足以处理语言。你需要深度——没有正确的“管道”，深度就会断裂。

2017年Vaswani论文封装了六个设计决定，将一个注意力层变成可堆叠的模块。此后所有Transformer——仅编码器（BERT）、仅解码器（GPT）、编码器-解码器（T5）——都继承了相同的骨架。到了2026年，这些模块已被改进（RMSNorm、SwiGLU、预归一化、RoPE），但骨架依旧相同。

本课讲解骨架。后续课程专门讲解：06针对编码器，07针对解码器，08针对编码器-解码器。

## 概念

![编码器和解码器模块内部连接](../assets/full-transformer.svg)

### 六个组成部分

1. **嵌入 + 位置信号。** 将tokens转换成向量。位置通过RoPE（现代）或正弦函数（经典）注入。
2. **自注意力。** 每个位置关注所有其他位置。解码器中有遮罩。
3. **前馈网络（FFN）。** 位置点两层MLP：`W_2 · activation(W_1 · x)`。默认扩展比为4倍。
4. **残差连接。** `x + sublayer(x)`。没有它，梯度在大约6层后会消失。
5. **层归一化。** `LayerNorm`或`RMSNorm`（现代）。稳定残差流。
6. **交叉注意力（仅解码器）。** 查询来自解码器，键值来自编码器输出。

### 编码器模块（BERT、T5编码器使用）

```text
x → LN → MHA(自注意力) → + → LN → FFN → + → 输出
                     ^              ^
                     |              |
                     └── 残差 ──────┘
```

编码器是双向的。没有遮罩。所有位置都能看到所有其他位置。

### 解码器模块（GPT、T5解码器使用）

```text
x → LN → MHA(遮罩自注意力) → + → LN → MHA(交叉注意力，连接编码器) → + → LN → FFN → + → 输出
```

解码器每个模块有三个子层。中间的——交叉注意力——是信息从编码器流向解码器的唯一通道。在纯解码器架构（GPT）中，交叉注意力省略，只有遮罩自注意力和FFN。

### 预归一化 vs 后归一化

原始论文为：`x + sublayer(LN(x))` vs `LN(x + sublayer(x))`。后归一化在2019年前后失宠——没有仔细预热就难以训练深层网络。预归一化（在子层前做`LN`）是2026年的默认做法：Llama、Qwen、GPT-3+、Mistral都采用。

### 2026年现代化模块

Vaswani 2017版本使用LayerNorm + ReLU。现代架构更换了这两者。实际生产模块如下：

| 组件       | 2017版本           | 2026版本           |
|------------|--------------------|--------------------|
| 归一化     | LayerNorm          | RMSNorm            |
| FFN激活    | ReLU               | SwiGLU             |
| FFN扩展倍数 | 4×                 | 2.6×（SwiGLU用三个矩阵，总参数相当） |
| 位置编码   | 正弦绝对位置编码   | RoPE               |
| 注意力     | 全多头注意力（MHA）| GQA（或MLA）       |
| 偏置项     | 有                 | 无                 |

RMSNorm去掉了LayerNorm的均值中心化（少了一个减法操作），节省计算且经验上同样稳定。SwiGLU（`Swish(W1 x) ⊙ W3 x`）在Llama、PaLM和Qwen论文中一致优于ReLU/GELU的FFN，降低困惑度约0.5点。

### 参数量

对于一个`d_model = d`且FFN扩展比为`r`的模块：

- 多头注意力：`4 · d²`（Q、K、V、O投影）
- FFN（SwiGLU）：`3 · d · (r · d)` ≈ `3rd²`
- 归一化层：忽略不计

当`d=4096, r=2.6, 层数=32`（大致为Llama 3 8B）时，总计：  
`32 · (4·4096² + 3·2.6·4096²) ≈ 32 · (16 + 32) M = ~1.5B 参数/层 × 32 ≈ 7B`（加上嵌入和输出头）。与已公布数字一致。

## 构建它

### 第一步：构建模块组件

使用第03课微型`Matrix`类（为独立性复制到本文件）：

- `layer_norm(x, eps=1e-5)` —— 去均值，除以标准差。
- `rms_norm(x, eps=1e-6)` —— 除以RMS，不去均值。
- `gelu(x)`和`silu(x) * W3 x`（SwiGLU）。
- `ffn_swiglu(x, W1, W2, W3)`。
- `encoder_block(x, params)`和`decoder_block(x, enc_out, params)`。

完整连接见`code/main.py`。

### 第二步：组合2层编码器和2层解码器

堆叠它们。将编码器输出传入每个解码器的交叉注意力。输出投影前加一层LN。

```python
def encode(tokens, params):
    x = embed(tokens, params.emb) + sinusoidal(len(tokens), params.d)
    for block in params.encoder_blocks:
        x = encoder_block(x, block)
    return x

def decode(target_tokens, encoder_out, params):
    x = embed(target_tokens, params.emb) + sinusoidal(len(target_tokens), params.d)
    for block in params.decoder_blocks:
        x = decoder_block(x, encoder_out, block)
    return x
```

### 第三步：运行一个玩具示例的前向传播

输入6个tokens的源序列，和5个tokens的目标序列。验证输出shape是`(5, vocab)`。不训练——本课关注架构，不是loss。

### 第四步：替换为RMSNorm + SwiGLU

用RMSNorm和SwiGLU替换LayerNorm和ReLU-FFN。确认shape仍匹配。这是2026年现代化的用法，只改一个函数。

## 使用它

PyTorch/TF参考实现有：`nn.TransformerEncoderLayer`, `nn.TransformerDecoderLayer`。但2026年大部分生产代码自己写模块，因为：

- Flash Attention在注意力内部调用，不通过`nn.MultiheadAttention`。
- GQA / MLA非标准库自带。
- RoPE、RMSNorm、SwiGLU不是PyTorch默认。

HF `transformers`有干净的参考模块代码，值得阅读：`modeling_llama.py`是2026年标准的仅解码器模块，约500行，值得通读。

**编码器、解码器、编码器-解码器——什么时候选：**

| 需求                     | 选择     | 示例                         |
|--------------------------|----------|------------------------------|
| 分类、嵌入、文本问答      | 仅编码器  | BERT、DeBERTa、ModernBERT    |
| 文本生成、聊天、编码、推理 | 仅解码器  | GPT、Llama、Claude、Qwen     |
| 结构化输入→结构化输出（翻译、摘要） | 编码器-解码器 | T5、BART、Whisper             |

仅解码器模型胜出因为扩展性最好，既能理解也能生成。编码器-解码器仍最适合输入有清晰“源序列”身份的任务（翻译、语音识别、结构化任务）。

## 部署它

见`outputs/skill-transformer-block-reviewer.md`。该技能对比了新Transformer模块实现与2026默认规范，标记缺失部分（预归一化、RoPE、RMSNorm、GQA、FFN扩展倍数）。

## 练习

1. **简单。** 在`d_model=512, n_heads=8, ffn_expansion=4, swiglu=True`下计算`encoder_block`参数量。通过实现模块并用`sum(p.numel() for p in block.parameters())`验证。
2. **中等。** 从后归一化切换到预归一化。初始化两者，在随机输入上堆叠12层后测量激活范数。后归一化激活应爆炸；预归一化应保持有界。
3. **困难。** 实现4层编码器-解码器完成玩具复制任务（复制反转的`x`）。训练100步，报告loss。换成RMSNorm + SwiGLU + RoPE——loss减少了吗？

## 关键词

| 术语         | 常说是什么                      | 真实含义                                                  |
|--------------|-------------------------------|-----------------------------------------------------------|
| 模块（Block） | “一个Transformer层”             | 归一化+注意力+归一化+FFN的堆叠，带残差连接。               |
| 残差（Residual） | “跳跃连接”                     | 输出`x + f(x)`；使深层堆叠梯度能传递。                     |
| 预归一化（Pre-norm） | “归一化在子层前，而非后”        | 现代方案：`x + sublayer(LN(x))`。无预热即可训练更深网络。     |
| RMSNorm       | “无均值的LayerNorm”              | 除以RMS，少做一次减法，经验稳定性相当。                      |
| SwiGLU       | “大家换用的FFN激活”              | `Swish(W1 x) ⊙ W3 x → W2`。比ReLU/GELU在人语言模型上表现更优。|
| 交叉注意力（Cross-attention） | “解码器看编码器的方式”              | 多头注意力，Q来自解码器，K/V来自编码器输出。                 |
| FFN扩展倍数（FFN expansion） | “中间MLP隐藏层宽度”               | 隐层维度与`d_model`的比例，通常4（LayerNorm）或2.6（SwiGLU）。|
| 无偏置（Bias-free） | “省略+b项”                      | 现代模型中线性层省略偏置，略微降低困惑度且模型更小。           |

## 延伸阅读

- [Vaswani 等 (2017). Attention Is All You Need](https://arxiv.org/abs/1706.03762) —— 原始模块规范。
- [Xiong 等 (2020). On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745) —— 为什么预归一化深度优于后归一化。
- [Zhang, Sennrich (2019). Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) —— RMSNorm。
- [Shazeer (2020). GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) —— SwiGLU论文。
- [HuggingFace `modeling_llama.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) —— 2026年标准的仅解码器模块。
