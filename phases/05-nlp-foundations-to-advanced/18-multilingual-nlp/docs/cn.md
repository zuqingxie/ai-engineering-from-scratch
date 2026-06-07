# 多语言自然语言处理（Multilingual NLP）

> 一个模型，支持100多种语言，大多数语言无训练数据。跨语言迁移是2020年代的实用奇迹。

**类型：** 学习  
**语言：** Python  
**先修知识：** 第5阶段 · 04（GloVe, FastText, Subword）、第5阶段 · 11（机器翻译）  
**时间：** 约45分钟

## 问题描述

英语有数十亿标注样本。乌尔都语有数千个。迈蒂利语几乎没有。任何面向全球用户的实用NLP系统都必须在无任务特定训练数据的语言长尾上工作。

多语言模型通过同时在多种语言上训练一个模型来解决这个问题。共享表示让模型可以将高资源语言学到的技能迁移到低资源语言。对模型进行英语情感分析的微调，开箱即用地对乌尔都语产生惊人的情感预测效果。这就是零样本跨语言迁移（zero-shot cross-lingual transfer），它重塑了NLP面向世界的交付方式。

本课将阐明权衡、典型模型，以及常让初学多语言工作的团队犯错的一项决策：选择迁移的源语言。

## 概念

![通过共享多语言嵌入空间实现跨语言迁移](../assets/multilingual.svg)

**共享词汇表。** 多语言模型使用在所有目标语言文本上训练的SentencePiece或WordPiece分词器。词汇表是共享的：相同的子词单元表示相关语言中的相同词素。英语和意大利语中的 `anti-` 由同一个token表示。

**共享表示。** Transformer（Transformer 架构）在多语言的掩码语言模型预训练中学会不同语言中语义相似的句子会产生相似的隐藏状态。mBERT、XLM-R和NLLB都是如此。英文的“cat”嵌入聚集在法语“chat”和西班牙语“gato”附近，整个句子的嵌入也是如此。

**零样本迁移。** 在一种语言（通常是英语）上带标签数据微调模型，推理时运行在模型支持的任何其他语言，无需目标语言标签。对类型相近的语言效果很好，远缘语言效果较弱。

**少样本微调。** 在目标语言上增加100-500个标注样本。分类任务准确率跃升至英语基线的95-98%。这是多语言NLP中最具性价比的杠杆。

## 模型列表

| 模型    | 年份 | 覆盖语言数 | 备注                                            |
|---------|------|------------|-------------------------------------------------|
| mBERT   | 2018 | 104语言    | 在维基百科训练。第一个实用多语言LM。低资源表现弱。 |
| XLM-R   | 2019 | 100语言    | 在CommonCrawl训练（远比维基百科大）。设定跨语言基线。Base 2.7亿，Large 5.5亿参数。 |
| XLM-V   | 2023 | 100语言    | XLM-R 1百万token词汇表版本（vs 25万）。低资源表现更好。 |
| mT5     | 2020 | 101语言    | T5架构的多语言生成模型。                         |
| NLLB-200| 2022 | 200语言    | Meta的翻译模型，涵盖55种低资源语言。              |
| BLOOM   | 2022 | 46语言 + 13编程语言 | 开源176B多语言大模型。                          |
| Aya-23  | 2024 | 23语言     | Cohere的多语言大模型。对阿拉伯语、印地语、斯瓦希里语表现强。 |

根据用例选择。分类任务通常选XLM-R-base（270M）作为理智默认。生成任务视译文或开放生成，选择mT5或NLLB。大语言模型任务配合Aya-23或Claude，使用明确的多语言提示。

## 源语言决策（2026年研究）

大多数团队默认用英语微调为源语言。2026年最新研究显示这常常错误。

语言相似度比语料量更能预测迁移质量。对斯拉夫语目标，德语或俄语常优于英语。对印度语系目标，印地语常优于英语。**qWALS**相似度指标（2026年，基于世界语音结构图）量化了此点。**LANGRANK**（Lin等，ACL 2019）是另一较早方法，通过语言相似性、语料大小和系统发育关系加权排序候选源语言。

实用准则：若目标语言有一个类型学上接近且资源丰富的亲属语言，优先在那个语言上微调，再与英语微调比较。

## 实操

### 第1步：零样本跨语言分类

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tok = AutoTokenizer.from_pretrained("joeddav/xlm-roberta-large-xnli")
model = AutoModelForSequenceClassification.from_pretrained("joeddav/xlm-roberta-large-xnli")


def classify(text, candidate_labels, hypothesis_template="This text is about {}."):
    scores = {}
    for label in candidate_labels:
        hypothesis = hypothesis_template.format(label)
        inputs = tok(text, hypothesis, return_tensors="pt", truncation=True)
        with torch.no_grad():
            logits = model(**inputs).logits[0]
        entail_score = torch.softmax(logits, dim=-1)[2].item()
        scores[label] = entail_score
    return dict(sorted(scores.items(), key=lambda x: -x[1]))


print(classify("I love this product!", ["positive", "negative", "neutral"]))
print(classify("मुझे यह उत्पाद पसंद है!", ["positive", "negative", "neutral"]))
print(classify("J'adore ce produit !", ["positive", "negative", "neutral"]))
```

一个模型，三种语言，相同API。XLM-R用NLI数据训练，通过蕴涵技巧良好迁移至分类任务。

### 第2步：多语言嵌入空间

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

pairs = [
    ("The cat is sleeping.", "Le chat dort."),
    ("The cat is sleeping.", "El gato está durmiendo."),
    ("The cat is sleeping.", "Die Katze schläft."),
    ("The cat is sleeping.", "The dog is barking."),
]

for eng, other in pairs:
    emb_eng = model.encode([eng], normalize_embeddings=True)[0]
    emb_other = model.encode([other], normalize_embeddings=True)[0]
    sim = float(np.dot(emb_eng, emb_other))
    print(f"  {eng!r} <-> {other!r}: cos={sim:.3f}")
```

翻译句子在嵌入空间中靠得很近。不同的英文句子则位于较远处。这使得跨语言检索、聚类和相似度计算成为可能。

### 第3步：少样本微调策略

```python
from transformers import TrainingArguments, Trainer
from datasets import Dataset


def few_shot_finetune(base_model, base_tokenizer, examples):
    ds = Dataset.from_list(examples)

    def tokenize_fn(ex):
        out = base_tokenizer(ex["text"], truncation=True, max_length=128)
        out["labels"] = ex["label"]
        return out

    ds = ds.map(tokenize_fn)
    args = TrainingArguments(
        output_dir="out",
        per_device_train_batch_size=8,
        num_train_epochs=5,
        learning_rate=2e-5,
        save_strategy="no",
    )
    trainer = Trainer(model=base_model, args=args, train_dataset=ds)
    trainer.train()
    return base_model
```

对于100-500个目标语言样本，`num_train_epochs=5` 和 `learning_rate=2e-5` 是安全默认配置。更高学习率会导致多语言对齐崩溃，得到只懂英语的模型。

## 有效评估

- **按语言划分的验证集准确率。** 不做汇总。汇总隐藏了长尾效果。
- **与单语言基线对比。** 有足够数据的语言，单语言模型训练有时优于多语言。测试验证。
- **实体级测试。** 目标语言中命名实体。多语言模型对非拉丁脚本的分词常较弱。
- **跨语言一致性。** 语义相同的两种语言应产出相同预测。测量差距。

## 实用推荐（2026年版本）

| 任务                     | 推荐方案                                    |
|--------------------------|---------------------------------------------|
| 分类任务，100种语言        | XLM-R-base (~2.7亿参数) 微调                 |
| 零样本文本分类             | `joeddav/xlm-roberta-large-xnli`            |
| 多语言句子嵌入             | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| 翻译，200种语言            | `facebook/nllb-200-distilled-600M`（见第11课）  |
| 生成多语言任务              | Claude、GPT-4、Aya-23、mT5-XXL               |
| 低资源语言NLP             | XLM-V或相关高资源语言的领域微调               |

如果性能重要，务必预留目标语言微调预算。零样本是起点，不是终点。

### 分词税（低资源语言出错原因）

多语言模型共用一个基于英语、法语、西班牙语、中文、德语主导语料训练的分词器。对非主流语言，会产生以下三重无声损耗：

- **分裂税（Fertility tax）。** 低资源语言文本分词后每词token数远超英语。印地语句子可能是等效英语句子的3-5倍。这3-5倍消耗上下文窗口、训练效率和延迟。
- **变体恢复税（Variant recovery tax）。** 每个拼写错误、变音符号变体、Unicode标准化差异或大小写变化都会成为嵌入空间中的冷启动独立序列。模型学不到原生者认为理所当然的正字法对应关系。
- **容量溢出税（Capacity spillover tax）。** 税1和税2消耗了上下文位置、层深和嵌入维度。实际用于推理的容量系统性低于高资源语言使用该模型时的容量。

实际现象：模型在印地语上训练正常，损失曲线合理，评估困惑度尚可，但生产结果细微错误。形态学在句中崩溃，罕见词形无法恢复。**坏的分词器无法通过数据规模解决。**

缓解措施：选覆盖目标语言良好的分词器（XLM-V的100万个token词汇表是直接解决方案）；训练前在目标语保留集上验证分词率；对极长尾脚本使用字节级回退（SentencePiece的`byte_fallback=True`，或GPT-2风格的字节级BPE），确保没有OOV。

## 交付

保存为 `outputs/skill-multilingual-picker.md`：

```markdown
---
name: multilingual-picker
description: 为多语言NLP任务选择源语言、目标模型和评估方案。
version: 1.0.0
phase: 5
lesson: 18
tags: [nlp, multilingual, cross-lingual]
---

给定要求（目标语言、任务类型、每种语言可用标注数据），输出：

1. 微调源语言。默认是英语；如果目标语言有类型学接近且资源丰富亲属，参考LANGRANK或qWALS优先尝试。
2. 基础模型。分类用XLM-R，生成用mT5，翻译用NLLB，生成LLM用Aya-23。
3. 少样本预算。优先使用目标语言100-500样本；若标注不可行则用零样本。
4. 评估方案。按语言准确率（不汇总）、跨语言一致性、非拉丁脚本实体F1。

拒绝交付未做按语言评估的多语言模型——汇总指标掩盖长尾失败。脚本如阿姆哈拉语、提格利尼亚语和许多非洲语言的低分词覆盖，应使用带字节回退（SentencePiece带`byte_fallback=True`或GPT-2类字节级分词器）的模型。
```

## 练习

1. **简单。** 在英语、法语、印地语和阿拉伯语中，每种语言对10句话运行 zero-shot classification pipeline（零样本分类流水线）。报告每种语言的准确率。你应该会看到法语表现强劲，印地语表现尚可，阿拉伯语表现波动较大。
2. **中等。** 使用 `paraphrase-multilingual-MiniLM-L12-v2` 构建一个跨语言检索器，针对一个混合语言的小型语料库。用英语查询，可以检索任何语言的文档。测量 recall@5。
3. **困难。** 对比以英语为源和以印地语为源的微调，在印地语分类任务上。两种方案下均使用500个目标语言的样本进行少样本微调。报告哪种源语言产生更好的印地语准确率及差距。这是 LANGRANK 论文的缩影。

## 关键词

| 术语             | 大众说法                 | 实际含义                                       |
|------------------|-------------------------|------------------------------------------------|
| Multilingual model（多语言模型） | 一个模型，多种语言             | 跨语言共享词汇表和参数。                         |
| Cross-lingual transfer（跨语言迁移） | 在一种语言上训练，在另一种语言上运行 | 在无目标语言标签情况下，基于源语言微调并在目标语言评估。             |
| Zero-shot（零样本）       | 无目标语言标签               | 无需在目标语言上微调即完成迁移。                     |
| Few-shot（少样本）        | 少量目标语言标签             | 使用100-500个目标语言样本进行微调。                  |
| mBERT               | 首个多语言语言模型           | 在维基百科预训练的104种语言BERT。                   |
| XLM-R               | 标准跨语言基线               | 在CommonCrawl预训练的100种语言RoBERTa。              |
| NLLB                | Meta的200语言机器翻译        | No Language Left Behind（无语言被遗忘）。包含55种低资源语言。 |

## 延伸阅读

- [Conneau et al. (2019). Unsupervised Cross-lingual Representation Learning at Scale](https://arxiv.org/abs/1911.02116) — XLM-R论文。
- [Pires, Schlinger, Garrette (2019). How Multilingual is Multilingual BERT?](https://arxiv.org/abs/1906.01502) — 开启跨语言迁移研究方向的分析论文。
- [Costa-jussà et al. (2022). No Language Left Behind](https://arxiv.org/abs/2207.04672) — NLLB-200论文。
- [Üstün et al. (2024). Aya Model: An Instruction Finetuned Open-Access Multilingual Language Model](https://arxiv.org/abs/2402.07827) — Aya，Cohere的多语言大语言模型。
- [Language Similarity Predicts Cross-Lingual Transfer Learning Performance (2026)](https://www.mdpi.com/2504-4990/8/3/65) — qWALS / LANGRANK源语言论文。
