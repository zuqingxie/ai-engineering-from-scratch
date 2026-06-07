# T5, BART — 编码器-解码器模型（Encoder-Decoder Models）

> 编码器负责理解。解码器负责生成。将它们组合起来，你就得到一个专为输入→输出任务设计的模型：翻译、摘要、重写、转录。

**类型：** 学习（Learn）  
**语言：** Python  
**先修课程：** 第7阶段 · 05（完整Transformer），第7阶段 · 06（BERT），第7阶段 · 07（GPT）  
**耗时：** 约45分钟

## 问题背景

仅解码器的GPT和仅编码器的BERT分别为不同目标简化了2017年的架构。但许多任务本质上是输入-输出形式：

- 翻译：英语 → 法语。  
- 摘要：5000个标记的文章 → 200个标记的摘要。  
- 语音识别：音频标记 → 文本标记。  
- 结构化抽取：普通文本 → JSON。

对于这些任务，编码器-解码器的格式是最合适的。编码器生成源文本的密集表示。解码器在每一步生成输出，同时跨注意力（cross-attention）关注该表示。训练时输出侧的目标是向右移一位的预测。损失与GPT相同，只是条件依赖于编码器输出。

现代实践主要由两篇论文定义：

1. **T5** (Raffel et al. 2019)。“文本到文本迁移Transformer（Text-to-Text Transfer Transformer）”。所有NLP任务都重新表述为文本输入，文本输出。单一架构，单一词汇表，单一损失。预训练任务是掩码跨度预测（输入中破坏连续的跨度，在输出中恢复它们）。
2. **BART** (Lewis et al. 2019)。“双向自回归Transformer（Bidirectional and Auto-Regressive Transformer）”。去噪自动编码器：以多种方式破坏输入（打乱、掩码、删除、旋转），要求解码器重构原文。

到2026年，编码器-解码器格式仍然适用于结构化输入非常关键的场景：

- Whisper（语音 → 文本）。  
- 谷歌的翻译流水线。  
- 一些代码补全/修复模型，其上下文和编辑结构区分明显。  
- Flan-T5及变体用于结构化推理任务。

尽管仅解码器模型赢得了更大关注，编码器-解码器架构从未消失。

## 概念介绍

![编码器-解码器与跨注意力（cross-attention）](../assets/encoder-decoder.svg)

### 正向传播流程

```text
source tokens ─▶ encoder ─▶ (N_src, d_model)  ──┐
                                                 │
target tokens ─▶ decoder block                   │
                 ├─▶ masked self-attention       │
                 ├─▶ cross-attention ◀───────────┘
                 └─▶ FFN
                ↓
              下一个标记的logits
```

关键点是，编码器针对每个输入运行一次。解码器自回归运行，但每一步都跨注意力关注相同的编码器输出。对编码器输出做缓存能免费加速长输入的处理。

### T5预训练 — 跨度（span）破坏

随机选取输入的若干跨度（平均长度3个标记，占总标记15%），用唯一的哨兵令牌替换这些跨度：`<extra_id_0>`, `<extra_id_1>` 等。解码器输出只生成被破坏的跨度及其哨兵前缀：

```text
source: The quick <extra_id_0> fox jumps <extra_id_1> dog
target: <extra_id_0> brown <extra_id_1> over the lazy
```

比预测整个序列信号更廉价。T5论文的消融实验显示结果可和MLM（BERT）及prefix-LM（UniLM）竞争。

### BART预训练 — 多种噪声去噪

BART尝试了五种加噪函数：

1. 标记掩码（token masking）。  
2. 标记删除（token deletion）。  
3. 文本填充（text infilling，掩码一段连续标记，解码器输出正确长度的填充）。  
4. 句子重排（sentence permutation）。  
5. 文档旋转（document rotation）。

文本填充加句子置换组合能带来最佳下游表现。解码器始终重构整段原文。相比T5，BART的预训练计算成本更高，因为其输出不是仅破坏的跨度，而是完整序列。

### 推理

与GPT一样的自回归生成。可用贪心、束搜索（beam search）、top-p采样。翻译和摘要常用宽度4-5的束搜索，因为输出分布更窄。

### 2026年如何选择架构变体

| 任务           | 编码器-解码器？ | 原因                                                   |
|--------------|--------------|------------------------------------------------------|
| 翻译           | 通常选            | 源序列清晰；输出分布固定；束搜索效果好                              |
| 语音转文本       | 是（如Whisper）      | 输入模态与输出不同；编码器负责塑造音频特征                                  |
| 聊天 / 推理       | 不，选解码器         | 无持续的“输入”；对话本身即序列                                     |
| 代码补全         | 通常不            | 解码器单独处理长上下文的胜出；代码模型如Qwen 2.5 Coder均为解码器架构           |
| 摘要           | 两者皆可           | BART、PEGASUS 超过早期纯解码器基线；现代纯解码器LLM能匹配                      |
| 结构化抽取       | 两者皆可           | T5因“文本→文本”范式整合多种输出格式较为清晰                               |

自2022年以来的趋势：纯解码器模型占据原本由编码器-解码器模型主导的任务，原因是（a）基于指令微调的解码器模型通过提示能泛化所有任务，（b）单架构比双架构更容易扩展，（c）RLHF训练假设用解码器。编码器-解码器架构在输入模态不同（语音、图像）或者束搜索质量关键任务中保持优势。

## 构建它

参见 `code/main.py`。我们为玩具语料实现了T5风格的跨度破坏，这是本课最有价值的部分，因为这几乎是所有编码器-解码器预训练配方的核心。

### 第一步：跨度破坏

```python
def corrupt_spans(tokens, mask_rate=0.15, mean_span=3.0, rng=None):
    """选取总计约mask_rate比例的标记跨度。返回（破坏后的输入, 目标序列）。"""
    n = len(tokens)
    n_mask = max(1, int(n * mask_rate))
    n_spans = max(1, int(round(n_mask / mean_span)))
    ...
```

目标格式遵循T5惯例：`<sent0> span0 <sent1> span1 ...`。破坏后的输入在非破坏标记之间插入哨兵标记。

### 第二步：验证往返正确

给定破坏输入和目标，重建原始句子。如果破坏过程可逆，则正向过程明确。这是健全性检查——真正训练不会这么做，但该测试低成本且能捕获跨度索引的越界错误。

### 第三步：BART噪声函数

五个函数：`token_mask`、`token_delete`、`text_infill`、`sentence_permute`、`document_rotate`。组合两种，展示效果。

## 使用它

HuggingFace参考：

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer
tok = T5Tokenizer.from_pretrained("google/flan-t5-base")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")

inputs = tok("translate English to French: Attention is all you need.", return_tensors="pt")
out = model.generate(**inputs, max_new_tokens=32)
print(tok.decode(out[0], skip_special_tokens=True))
```

T5的妙招是任务名称放入输入文本。相同模型接管几十个任务，因为所有任务都是文本输入，文本输出。到了2026年，基于指令微调的纯解码器模型已推广此模式，但T5是首创。

## 交付它

参见 `outputs/skill-seq2seq-picker.md`。此技能模块根据输入-输出结构、延迟和质量目标，为新任务选取编码器-解码器或解码器单独架构。

## 练习

1. **简单。** 运行 `code/main.py`，对一条30标记的句子应用跨度破坏，验证拼接非哨兵源标记和解码目标跨度是否可还原原始句子。  
2. **中等。** 实现BART的 `text_infill` 噪声：用单个 `<mask>` 替换随机跨度，解码器必须推断正确的跨度长度和内容。展示示例。  
3. **困难。** 在小型英→猪拉丁语语料（200对）上微调 `flan-t5-small`，在持出50对上测BLEU。与对相同数据、耗费相同算力的 `Llama-3.2-1B` 微调结果对比。

## 关键词

| 术语           | 常用表述              | 实际含义                                                  |
|--------------|------------------|---------------------------------------------------------|
| 编码器-解码器（Encoder-decoder） | “Seq2seq transformer” | 两个堆栈：输入的双向编码器，输出的带cross-attention的自回归解码器。                |
| 跨注意力（Cross-attention）   | “源序列和目标序列对话”   | 解码器的Query×编码器的Key/Value。编码器信息唯一进入解码器的通道。                    |
| 跨度破坏（Span corruption）  | “T5的预训练妙招”        | 用哨兵标记替换随机跨度，解码器输出被替换的所有跨度。                               |
| 去噪目标（Denoising objective）| “BART的核心任务”         | 对输入添加噪声，训练解码器重建干净序列。                                       |
| 哨兵令牌（Sentinel token）    | “`<extra_id_N>`位置占位符”| 特殊标记，用于标注源和目标中被破坏的跨度位置。                                    |
| Flan          | “指令微调的T5”         | 在1800多个任务上微调T5，使其在指令跟随上编码器-解码器架构具竞争力。                  |
| 束搜索（Beam search）      | “解码策略”             | 每步保留top-k部分序列；翻译/摘要的标准解码方法。                                  |
| 教师强制（Teacher forcing）  | “训练时的输入”           | 训练中，解码器输入真实的前一个输出标记，而非采样标记。                             |

## 延伸阅读

- [Raffel et al. (2019). Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683) — T5。  
- [Lewis et al. (2019). BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/abs/1910.13461) — BART。  
- [Chung et al. (2022). Scaling Instruction-Finetuned Language Models](https://arxiv.org/abs/2210.11416) — Flan-T5。  
- [Radford et al. (2022). Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) — Whisper，2026年典型编码器-解码器模型。  
- [HuggingFace `modeling_t5.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py) — 参考实现。
