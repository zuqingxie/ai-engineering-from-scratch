# 从零构建 Transformer（Transformer 架构）——结业项目

> 十三节课。一个模型。没有捷径。

**类型：** 构建  
**语言：** Python  
**先决条件：** 第7阶段 · 01至13节。不可跳过。  
**时间：** ~120分钟

## 问题描述

你阅读了所有论文，已经实现了 attention（注意力机制）、multi-head splits（多头拆分）、positional encodings（位置编码）、encoder 和 decoder 块、BERT 和 GPT 损失、MoE（专家混合）、KV cache（键值缓存）。现在让它们协同工作完成一个真实任务。

结业项目：端到端训练一个小型 decoder-only Transformer（仅解码器 Transformer 架构），在字符级语言建模任务上。它读取莎士比亚文本，生成新的莎士比亚式文本。体积足够小，在笔记本电脑上可在10分钟内训练完成。性能足够正确，更大的数据集和更长训练时间能得到实际的语言模型（LM）。

这是本课程的“nanoGPT”。不是原创——Karpathy 2023年的 nanoGPT 教程是参考实现，每个学生至少写过一遍。我们采用其结构，结合课程内容重新调整。

## 概念解析

![Transformer-from-scratch block diagram](../assets/capstone.svg)

架构，注释如下：

```text
input tokens (B, N)
   │
   ▼
token embedding + positional embedding  ◀── 第04课（RoPE 选项）
   │
   ▼
┌──── block × L ────────────────────┐
│  RMSNorm                          │  ◀── 第05课
│  MultiHeadAttention (causal)      │  ◀── 第03 + 07课（因果掩码）
│  residual                         │
│  RMSNorm                          │
│  SwiGLU FFN                       │  ◀── 第05课
│  residual                         │
└────────────────────────────────── ┘
   │
   ▼
final RMSNorm
   │
   ▼
lm_head (与 token embedding 共享权重)
   │
   ▼
logits (B, N, V)
   │
   ▼
shift-by-one cross-entropy          ◀── 第07课
```

### 我们交付的内容

- `GPTConfig` — 一个地方配置所有超参数。  
- `MultiHeadAttention` — 因果、多批处理，支持可选的Flash方式路径（PyTorch 的 `scaled_dot_product_attention`）。  
- `SwiGLUFFN` — 现代 FFN（前馈网络）。  
- `Block` — 预归一化，残差包装的注意力 + FFN。  
- `GPT` — 嵌入层、堆叠的块、LM 头、`generate()`方法。  
- 训练循环包括 AdamW 优化器、余弦学习率（LR）调度和梯度裁剪。  
- 基于莎士比亚文本的字符级分词器。  

### 我们未交付的内容

- RoPE（旋转位置编码）——在第04课中概念实现，这里为了简单使用学习到的位置嵌入。练习会让你用上 RoPE。  
- 生成时的 KV cache——每步生成都重新计算整个前缀的注意力。速度较慢但更简单。练习中让你添加 KV cache。  
- Flash Attention —— PyTorch 2.0+ 会自动调度；我们使用 `F.scaled_dot_product_attention`。  
- MoE —— 每个块单个 FFN。你在第11课见过 MoE。  

### 目标指标

在 Mac M2 笔记本上，4层、4头、d_model=128 的 GPT 模型，在 `tinyshakespeare.txt` 训练2,000步：

- 训练损失从约4.2（随机）收敛到约1.5，耗时约6分钟。  
- 采样输出呈现莎士比亚风格：古语、换行、出现“ROMEO:”这类专有名词。  
- 验证损失（最终10%文本的保留集）紧跟训练损失；此规模和预算下没有过拟合。  

## 构建步骤

本课使用 PyTorch。安装 `torch`（CPU版本可）。见 `code/main.py`。脚本支持：

- 如果缺少，则下载 `tinyshakespeare.txt`（或读取本地副本）。  
- 字节级字符分词器。  
- 90/10的训练/验证拆分。  
- 在支持硬件上启用 bf16 自动类型转换的训练循环。  
- 训练完成后采样。  

### 第一步：数据处理

```python
text = open("tinyshakespeare.txt").read()
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda xs: "".join(itos[x] for x in xs)
```

共65个唯一字符。词汇量非常小。适合4字节 vocab_size。无 BPE（字节对编码），无复杂分词器。

### 第二步：模型搭建

见 `code/main.py`。Block 模块来自第05课 —— 预归一化（pre-norm）、RMSNorm、SwiGLU、因果 MHA。4层4头128维参数约80万。

### 第三步：训练循环

获取随机长度为256的 token 窗口批次。前向。shift-by-one 交叉熵。反向。AdamW 步进。日志。重复。

```python
for step in range(max_steps):
    x, y = get_batch("train")
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad()
```

### 第四步：采样生成

给定提示，重复前向，基于 top-p logits 采样，拼接继续。生成500个 token 后停止。

### 第五步：查看输出

训练2000步后示例：

```text
ROMEO:
Away and mild will not thy friend, that thou shalt wit:
The chief that well shame and hath been his friends,
...
```

不是莎士比亚原文，但形态相似。约80万参数、6分钟训练在笔记本上取得的明显成果。

## 使用指南

该结业项目是参考架构。3个扩展，将其用到实际层面：

1. **更换分词器。** 使用 BPE（例如 `tiktoken.get_encoding("cl100k_base")`）。词汇量从65跃升至约5万。模型容量需相应扩展。  
2. **更大语料训练。** 使用 `OpenWebText` 或 `fineweb-edu`（HuggingFace数据集）。在单个 A100 上训练125M参数模型，处理100亿token约需24小时。  
3. **加入 RoPE + KV 缓存 + Flash Attention。** 以下练习指导逐步实现。  

最终形成125M参数 GPT，生成流畅英文。不是前沿模型，但同一路径——更大规模——2026年 Karpathy、EleutherAI 和 Allen Institute 用它训练研究模型。

## 交付内容查看

参见 `outputs/skill-transformer-review.md`。该技能复盘检验了13节课全部知识点下 Transformer 从零实现的正确性。

## 练习题

1. **简单。** 运行 `code/main.py`。验证你训练模型最后步的验证损失低于2.0。将 `max_steps` 从2000改为5000，验证损失是否持续下降？  
2. **中等。** 用 RoPE 替换学习的位置嵌入。对 `MultiHeadAttention` 内 Q 和 K 应用旋转。训练并确认验证损失至少达到同样低。  
3. **中等。** 在采样循环中实现 KV 缓存。对比有无缓存采样500个 token，墙钟时间应提速5至20倍。  
4. **困难。** 加一个额外头，预测下下一个token（MTP——DeepSeek-V3 的多token预测）。联合训练。是否有助于性能？  
5. **困难。** 用4专家 MoE 替换块中的单一 FFN。含 Router + top-2 路由。检验匹配活跃参数时验证损失变化。  

## 关键词汇

| 术语                | 常见说法               | 实际含义                                         |
|---------------------|------------------------|------------------------------------------------|
| nanoGPT             | “Karpathy的教程仓库”   | 极简仅解码器Transformer训练代码，约300行；权威参考实现。|
| tinyshakespeare     | “标准玩具语料库”       | ~1.1 MB文本；自2015年起所有字符级LM教程都会用。    |
| Tied embeddings     | “共享输入/输出矩阵”    | LM 头权重=token embedding矩阵转置；节省参数，提升质量。 |
| bf16 autocast       | “训练精度技巧”         | 计算前后向用bf16，优化器状态用fp32；2021年以来标准。|
| Gradient clipping   | “防止梯度爆炸”         | 全局梯度范数限制为1.0；防止训练不稳定。             |
| Cosine LR schedule  | “2020+默认方案”        | 学习率先线性预热后余弦衰减至峰值10%。               |
| MFU                 | “模型FLOP使用率”       | 实际FLOPs/理论峰值；2026年40%密集，30%MoE属于优秀。|
| Val loss            | “保留集损失”           | 模型没见过的数据的交叉熵；检测过拟合。               |

## 深入阅读

- [The Annotated Transformer (Harvard NLP)](https://nlp.seas.harvard.edu/annotated-transformer/) — 经典注释实现。
