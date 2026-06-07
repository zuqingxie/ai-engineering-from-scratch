# BERT — 掩码语言模型（Masked Language Modeling）

> GPT 预测下一个词。BERT 预测缺失的词。一句话的差别——以及半个十年所有嵌入式模型的变革。

**类型：** 构建  
**语言：** Python  
**先决条件：** 阶段 7 · 05（完整 Transformer（Transformer 架构）），阶段 5 · 02（文本表示）  
**时间：** 约 45 分钟

## 问题

在 2018 年，每个 NLP 任务——情感分析、命名实体识别（NER）、问答（QA）、蕴涵识别（entailment）——都要在自己的标注数据上从零训练模型。那时没有可以微调的预训练“理解英语”的检查点。ELMo（2018）显示可以用双向 LSTM 预训练上下文嵌入；这带来了一定帮助，但泛化能力有限。

BERT（Devlin 等，2018）提出：如果我们采一个 Transformer 编码器，把它放在互联网上的所有句子上训练，让它用两边的上下文预测缺失词呢？然后在你的下游任务上仅微调一个头。参数效率让人震惊。

结果：18 个月内，BERT 及其变体（RoBERTa、ALBERT、ELECTRA）主导了所有 NLP 排行榜。到 2020 年，所有搜索引擎、内容审核管线和语义搜索系统都有了 BERT 内核。

到了 2026 年，纯编码器模型仍是分类、检索和结构化抽取的正确工具——每个 Token 的运行速度是解码器的 5–10 倍，它们的嵌入成为现代检索栈的骨干。ModernBERT（2024 年 12 月）采用 Flash Attention + RoPE + GeGLU 将架构推至 8K 上下文长度。

## 概念

![掩码语言模型：选择词元，掩码它们，预测原始词元](../assets/bert-mlm.svg)

### 训练信号

取一句话：`the quick brown fox jumps over the lazy dog`。

随机遮蔽 15% 的词元：

```text
输入： the [MASK] brown fox jumps [MASK] the lazy dog
目标： the  quick brown fox jumps  over  the lazy dog
```

训练模型预测被掩码位置的原始词元。因为编码器是双向的，预测位置 1 的 `[MASK]` 可以利用 2+ 位置的 `brown fox jumps`。这正是 GPT 无法做到的。

### BERT 掩码规则

在选中预测的 15% 词元中：

- 80% 替换为 `[MASK]`。
- 10% 替换为随机词元。
- 10% 保持不变。

为何不总是用 `[MASK]`？因为 `[MASK]` 在推理阶段从不出现。如果把 100% 掩码位置都训练成期待 `[MASK]`，会造成预训练和微调时的分布转移。10% 随机 + 10% 不变保持模型“诚实”。

### 下一句预测（NSP）——及其被废弃的原因

原始 BERT 同时训练 NSP：给定两句 A 和 B，预测 B 是否紧随 A。RoBERTa（2019）废除了 NSP，证明 NSP 反而有害。现代编码器跳过 NSP。

### 2026 年的变化：ModernBERT

2024 年 ModernBERT 论文用 2026 年的基元重建了模块：

| 组件       | 原始 BERT（2018）             | ModernBERT（2024）            |
|------------|-------------------------------|------------------------------|
| 位置编码   | 学习的绝对位置                 | RoPE                         |
| 激活函数   | GELU                          | GeGLU                        |
| 归一化     | LayerNorm                    | 预归一化 RMSNorm             |
| 注意力机制 | 全连接密集                    | 交替本地（128）+全局         |
| 上下文长度 | 512                           | 8192                         |
| 分词器     | WordPiece                    | BPE                          |

且不同于 2018 年架构，ModernBERT 天生支持 Flash-Attention。在 8K 序列长度推理时比 DeBERTa-v3 快 2–3 倍，同时 GLUE 任务得分更高。

### 2026 年仍选择编码器的应用场景

| 任务               | 编码器胜过解码器的原因                          |
|--------------------|-----------------------------------------------|
| 检索 / 语义搜索嵌入 | 双向上下文 = 每个词元更高质量的嵌入              |
| 分类（情感、意图、有害内容检测） | 单次前向，无生成开销                         |
| 命名实体识别 / 词元标注 | 按位置输出，本身支持双向                      |
| 零样本蕴涵识别（NLI） | 编码器顶部的分类头                            |
| RAG 重排器          | 交叉编码器打分，速度比大型语言模型重排器快 10 倍 |

## 构建步骤

### 第1步：掩码逻辑

见 `code/main.py`。函数 `create_mlm_batch` 接收词元 ID 列表、词表大小和掩码概率。返回输入 ID（带掩码）和标签（仅掩码位置值，其他为 -100，符合 PyTorch 忽略索引规则）。

```python
def create_mlm_batch(tokens, vocab_size, mask_prob=0.15, rng=None):
    input_ids = list(tokens)
    labels = [-100] * len(tokens)
    for i, t in enumerate(tokens):
        if rng.random() < mask_prob:
            labels[i] = t
            r = rng.random()
            if r < 0.8:
                input_ids[i] = MASK_ID
            elif r < 0.9:
                input_ids[i] = rng.randrange(vocab_size)
            # else: 保持原始词元
    return input_ids, labels
```

### 第2步：在小语料上跑 MLM 预测

训练一个 2 层编码器 + MLM 头，词表大小 20，语料 200 句。无梯度，仅做前向传递的合理性检查。完整训练需 PyTorch。

### 第3步：对比掩码类型

展示三种掩码规则如何让模型在无 `[MASK]` 的情况下仍可用。在无掩码和掩码句子上预测，二者均能输出合理词元分布，因为训练时模型见过两种模式。

### 第4步：微调头部

用玩具情感数据集替换 MLM 头为分类头。仅训练头部，编码器冻结。这是所有 BERT 用例的套路。

## 使用方法

```python
from transformers import AutoModel, AutoTokenizer

tok = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
model = AutoModel.from_pretrained("answerdotai/ModernBERT-base")

text = "Attention is all you need."
inputs = tok(text, return_tensors="pt")
out = model(**inputs).last_hidden_state   # (1, N, 768)
```

**嵌入模型是针对 BERT 的微调。** `sentence-transformers` 模型如 `all-MiniLM-L6-v2` 是用对比损失训练的 BERT。编码器相同，只是损失函数变了。

**交叉编码器重排器也是微调的 BERT。** 在 `[CLS] query [SEP] doc [SEP]` 上做对分类。查询和文档之间的双向注意力使交叉编码器在质量上优于双编码器。

**2026 年何时不选 BERT。** 任何生成任务都不适合。编码器不能自回归生成词元。另外：参数少于 10 亿时，小型解码器因灵活性和相当质量（如 Phi-3-Mini、Qwen2-1.5B）更优。

## 部署

见 `outputs/skill-bert-finetuner.md`。该技能定义 BERT 微调范畴（骨干选型、头部定义、数据、评估、早停）以适配新分类或抽取任务。

## 练习

1. **简单。** 运行 `code/main.py` 并打印 10,000 个词元的掩码分布。确认约 15% 被选中，其中约 80% 替换为 `[MASK]`。
2. **中等。** 实现整词掩码：若一个词被拆分成子词，要么全部掩码，要么一个都不掩码。测量是否提升了 500 句语料的 MLM 准确率。
3. **困难。** 用公开数据集的 10,000 句训练一个小型（2 层，隐藏维度 64）BERT。微调 `[CLS]` 词元以完成 SST-2 情感分类。和同等参数的纯解码器基线比，哪个更好？

## 关键词

| 术语           | 常用说法                   | 实际含义                                              |
|----------------|----------------------------|-------------------------------------------------------|
| MLM            | “掩码语言建模”             | 训练信号：随机替换 15% 词元为 `[MASK]`，预测原始词元。  |
| 双向（Bidirectional） | “两边看”               | 编码器注意力无因果掩码——各位置可见所有其他位置。        |
| `[CLS]`        | “池化词元”                 | 序列前置特殊词元，其最终嵌入用作句子级别表示。           |
| `[SEP]`        | “分段符”                   | 分隔配对的序列（如查询/文档，句子 A/B）。                |
| NSP            | “下一句预测”               | BERT 的第二预训练任务；RoBERTa 证实无用，2019 年后废弃。 |
| 微调（Fine-tuning） | “适配任务”             | 编码器大部分冻结；仅训练任务头部。                       |
| 交叉编码器（Cross-encoder） | “重排器”          | 同时输入查询和文档，输出相关性分数的 BERT。              |
| ModernBERT     | “2024 年刷新版”            | 用 RoPE、RMSNorm、GeGLU、交替本地/全局注意力和 8K 上下文重构的编码器。 |

## 参考文献

- [Devlin 等（2018）. BERT：用于语言理解的深度双向 Transformer 预训练](https://arxiv.org/abs/1810.04805) — 原始论文。
- [Liu 等（2019）. RoBERTa：一种鲁棒优化的 BERT 预训练方法](https://arxiv.org/abs/1907.11692) — 如何正确训练 BERT；废除 NSP。
- [Clark 等（2020）. ELECTRA：将文本编码器作为判别器而非生成器预训练](https://arxiv.org/abs/2003.10555) — 替换词检测在相同期算力下胜过 MLM。
- [Warner 等（2024）. 更智能、更好、更快、更长：现代双向编码器](https://arxiv.org/abs/2412.13663) — ModernBERT 论文。
- [HuggingFace `modeling_bert.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py) — 权威编码器参考源码。
