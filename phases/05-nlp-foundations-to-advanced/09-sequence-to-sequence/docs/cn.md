# 序列到序列模型（Sequence-to-Sequence Models）

> 两个 RNN 假装自己是翻译器。他们遇到的瓶颈正是注意力（attention）存在的原因。

**类型：** 构建  
**语言：** Python  
**先决条件：** 第5阶段 · 08课（用于文本的CNN和RNN），第3阶段 · 11课（PyTorch 入门）  
**时间：** 约75分钟

## 问题描述

分类任务将可变长度序列映射到单一标签。翻译任务则将一个可变长度序列映射到另一个可变长度序列。输入和输出可能属于不同词汇表，甚至不同语言，且长度不保证相等。

序列到序列架构（seq2seq，Sutskever, Vinyals, Le，2014）通过一个刻意简单的方案破解了这个问题。用两个RNN。第一个读取源句子并生成一个固定大小的上下文向量（context vector）。第二个以该向量为输入，逐个生成目标句子标记。和第08课写的代码相同，只是组装方式不同。

研究这个模型有两个原因。首先，上下文向量瓶颈是NLP最有教学价值的失败点。它激发了所有注意力机制和Transformer（Transformer 架构）的发展。其次，这个训练方案（teacher forcing（教师强制），scheduled sampling（计划采样），推断时的beam search（束搜索））依然适用于包括大语言模型（LLMs）在内的所有现代生成系统。

## 概念说明

**编码器（Encoder）**。一个 RNN，读取源句子。其最终隐藏状态即为**上下文向量**——整个输入的固定大小摘要。理论上不丢失任何关键信息。

**解码器（Decoder）**。另一个以上下文向量初始化的 RNN。每个时间步输入上一步生成的标记并产生目标词汇表的概率分布。通过采样或argmax选择下一个标记。将选择的标记作为下一步输入，重复直到生成`<EOS>`标记或者达到最大长度。

**训练：** 对解码器每一步应用交叉熵损失，整个序列的损失求和。通过时间反向传播（BPTT）训练两个网络。

**教师强制（Teacher forcing）**。训练时，在步骤`t`，解码器的输入是位置`t-1`的*真实*标记，而非模型之前的预测。这稳定了训练；否则早期错误会级联，模型无法收敛。推断时必须使用模型自己的预测，因此存在训练/推断分布差异，这种差异称为**曝光偏差（exposure bias）**。

**瓶颈（The bottleneck）**。编码器对源句子的所有理解必须压缩到一个上下文向量中。长句子细节丢失。稀有词汇变得模糊。句序重排（chat noir 与 black cat）需要记忆而非计算。

注意力机制（第10课）通过让解码器访问*每一个*编码器隐藏状态而非仅最后一个，解决了瓶颈问题。这就是注意力设计的核心动机。

## 构建步骤

### 步骤1：编码器

```python
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(src_vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)

    def forward(self, src):
        e = self.embed(src)
        outputs, hidden = self.gru(e)
        return outputs, hidden
```

`outputs` 形状为 `[batch, seq_len, hidden_dim]`——每个输入位置对应一个隐状态。`hidden` 形状为 `[1, batch, hidden_dim]`——最后一个时间步的隐状态。第08课说过“对outputs做池化用于分类”。这里我们使用最后隐状态作为上下文向量，忽略每步输出。

### 步骤2：解码器

```python
class Decoder(nn.Module):
    def __init__(self, tgt_vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(tgt_vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, tgt_vocab_size)

    def forward(self, token, hidden):
        e = self.embed(token)
        out, hidden = self.gru(e, hidden)
        logits = self.fc(out)
        return logits, hidden
```

解码器每次调用处理单步。输入：单标记批次和当前隐状态。输出：下一标记的词汇表概率分布logits及更新后的隐状态。

### 步骤3：训练循环，带教师强制

```python
def train_batch(encoder, decoder, src, tgt, bos_id, optimizer, teacher_forcing_ratio=0.9):
    optimizer.zero_grad()
    _, hidden = encoder(src)
    batch_size, tgt_len = tgt.shape
    input_token = torch.full((batch_size, 1), bos_id, dtype=torch.long)
    loss = 0.0
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    for t in range(tgt_len):
        logits, hidden = decoder(input_token, hidden)
        step_loss = loss_fn(logits.squeeze(1), tgt[:, t])
        loss += step_loss
        use_teacher = torch.rand(1).item() < teacher_forcing_ratio
        if use_teacher:
            input_token = tgt[:, t].unsqueeze(1)
        else:
            input_token = logits.argmax(dim=-1)

    loss.backward()
    optimizer.step()
    return loss.item() / tgt_len
```

要点：`ignore_index=0`跳过对填充标记的损失计算。`teacher_forcing_ratio`是每步使用真实标记而非模型预测的概率。训练时从1.0（完全教师强制）逐渐缓降到约0.5，以减少曝光偏差。

### 步骤4：推断循环（贪心解码）

```python
@torch.no_grad()
def greedy_decode(encoder, decoder, src, bos_id, eos_id, max_len=50):
    _, hidden = encoder(src)
    batch_size = src.shape[0]
    input_token = torch.full((batch_size, 1), bos_id, dtype=torch.long)
    output_ids = []
    for _ in range(max_len):
        logits, hidden = decoder(input_token, hidden)
        next_token = logits.argmax(dim=-1)
        output_ids.append(next_token)
        input_token = next_token
        if (next_token == eos_id).all():
            break
    return torch.cat(output_ids, dim=1)
```

贪心解码每步选概率最高标记，可能会从正确路径偏离：一旦选了标记就不能改回。**束搜索（beam search）**维护每步top-k部分序列，最终选择得分最高的完整序列。束宽一般为3-5。

### 步骤5：瓶颈演示

在一个玩具复制任务上训练：源序列 `[a, b, c, d, e]`，目标序列同源。逐步增加序列长度，观察准确率。

```text
seq_len=5   复制准确率: 98%
seq_len=10  复制准确率: 91%
seq_len=20  复制准确率: 62%
seq_len=40  复制准确率: 23%
```

单个GRU隐状态无法无损地记忆40个标记输入。信息在编码器每个步骤都有，但解码器只看最后状态。注意力机制直接解决这一问题。

## 使用方法

PyTorch 有`nn.Transformer`和基于`nn.LSTM`的seq2seq模板。Hugging Face的`transformers`库提供全套编码器-解码器模型（BART、T5、mBART、NLLB），训练数据规模达数十亿标记。

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tok = AutoTokenizer.from_pretrained("facebook/bart-base")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-base")

src = tok("Translate this to French: Hello, how are you?", return_tensors="pt")
out = model.generate(**src, max_new_tokens=50, num_beams=4)
print(tok.decode(out[0], skip_special_tokens=True))
```

现代编码器-解码器弃用RNN，采用Transformer（Transformer 架构）。总体结构（编码器、解码器、逐步生成）同2014年seq2seq论文相同，内部机制不同。

### 何时仍用RNN seq2seq

几乎不用于新项目。特殊例外：

- 流式翻译，输入逐步消费且内存有限。
- 设备端文本生成，Transformer开销过高时。
- 教学——理解编码器-解码器瓶颈是理解Transformer成功的最快路径。

### 曝光偏差及缓解方法

- **计划采样（Scheduled sampling）**。训练中逐渐降低教师强制比例，使模型学会从自身错误恢复。
- **最小风险训练（Minimum risk training）**。用句子级BLEU代替标记级交叉熵，更符合实际需求。
- **强化学习微调**。用指标（reward）奖励序列生成器。现代LLM RLHF方法采用。

以上方法均适用于Transformer生成模型。

## 部署方案

保存为 `outputs/prompt-seq2seq-design.md`：

```markdown
---
name: seq2seq-design
description: 为给定任务设计序列到序列流水线。
phase: 5
lesson: 09
---

给定任务（翻译，摘要，复述，问题重写），输出：

1. 架构。默认使用预训练Transformer编码器-解码器（BART，T5，mBART，NLLB）。RNN seq2seq仅限特殊约束。
2. 预训练检查点。指定名称（`facebook/bart-base`，`google/flan-t5-base`，`facebook/nllb-200-distilled-600M`）。匹配任务和语言覆盖。
3. 解码策略。确定性输出用贪心，质量优先用束搜索（宽度4-5），多样化用带温度采样。一句说明理由。
4. 部署前需验证的失败模式。曝光偏差导致长输出内容漂移；抽取90百分位长度20个生成结果逐一检查。

拒绝推荐训练百万级并行文本以下的seq2seq模型。对用户前端内容使用贪心解码的流水线均视为脆弱（重复与循环）。
```

## 练习

1. **简单。** 实现玩具复制任务。训练GRU seq2seq，输入输出相同。评测5、10、20长度的准确率，复现瓶颈现象。
2. **中等。** 添加束搜索解码（宽度3）。在小型平行语料库上评测BLEU，对比贪心。记录束搜索获胜位置（通常是最后几个标记）和无差异位置。
3. **困难。** 对`facebook/bart-base`在1万对复述数据集上微调。比较微调模型和原始模型在测试集束宽为4输出上的表现。报告BLEU并挑选10个质性例子。

## 关键词

| 术语 | 普通说法 | 实际意义 |
|------|----------|----------|
| Encoder（编码器） | 输入RNN | 读取输入，输出每步隐状态和最终上下文向量。 |
| Decoder（解码器） | 输出RNN | 由上下文向量初始化，逐步生成目标标记。 |
| Context vector（上下文向量） | 摘要 | 编码器最终隐状态，固定大小，是注意力机制解决的瓶颈。 |
| Teacher forcing（教师强制） | 使用真实标记 | 训练时输入前一步真实标记，稳定训练。 |
| Exposure bias（曝光偏差） | 训练/推断偏差 | 训练用真实标记，推断不用，导致模型不习惯自我纠错。 |
| Beam search（束搜索） | 更优解码 | 每步保留top-k部分序列，非贪心选择。 |

## 拓展阅读

- [Sutskever, Vinyals, Le (2014). Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215) — 基础seq2seq论文，四页。
- [Cho et al. (2014). Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078) — 首次引入GRU及编码器-解码器框架。
- [Bahdanau, Cho, Bengio (2014). Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) — 注意力机制论文，建议课后立即阅读。
- [PyTorch NLP from Scratch tutorial](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html) — 可构建的seq2seq与注意力代码教程。
