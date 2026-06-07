# 自然语言推断 — 文本蕴含（Natural Language Inference, NLI）

> “t 蕴含 h”意味着一个读者在读到 t 后会认为 h 为真。NLI 是预测蕴含 / 矛盾 / 中立的任务。表面看乏味，生产环境中却极其重要。

**类型：** 学习  
**语言：** Python  
**先决条件：** 第5阶段 · 05（情感分析）, 第5阶段 · 13（问答系统）  
**时长：** 约60分钟

## 问题描述

你做了一个摘要器。它生成了一个摘要。你怎么知道摘要里没有产生幻觉（hallucination）？

你做了一个聊天机器人。它回答了“是的”。你怎么知道这个答案是被检索到的段落支持的？

你要对1万篇新闻文章按主题分类。你没有训练标签。能复用已有模型吗？

这三个问题都归约为自然语言推断任务。NLI 问：给定前提 `t` 和假设 `h`，`h` 是被 `t` 蕴含、被矛盾，还是中立（无关）？

- **幻觉检测：** `t` = 源文档，`h` = 摘要声明。不蕴含＝幻觉。  
- **基于证据的问答（Grounded QA）：** `t` = 检索到的段落，`h` = 生成的答案。不蕴含＝捏造。  
- **零样本分类：** `t` = 文档，`h` = 标签的文本化（“这是关于体育的”）。蕴含＝预测标签。

一个任务，三种生产场景。这就是为什么每个 RAG 评估框架底层都搭载了 NLI 模型。

## 概念

![NLI: three-way classification, premise vs hypothesis](../assets/nli.svg)

**三种标签**

- **蕴含（Entailment）**。`t` → `h`。例如“猫在垫子上”蕴含“有一只猫”。  
- **矛盾（Contradiction）**。`t` → ¬`h`。例如“猫在垫子上”与“没有猫”矛盾。  
- **中立（Neutral）**。没有推断关系。例如“猫在垫子上”和“猫饿了”是中立的。

**非严格逻辑蕴含。** NLI 是自然语言推断——一个典型人类读者会推断出的结论，而非严格逻辑。比如“NLI 中‘约翰遛狗’蕴含‘约翰有一只狗’”，但严格的一阶逻辑需要公理化“拥有关系”才能成立。

**数据集。**

- **SNLI**（2015年）。57万人人标注对，以图像标题为前提。领域狭窄。  
- **MultiNLI**（2017年）。43.3万对，涵盖10个体裁。2026年的标准训练语料。  
- **ANLI**（2019年）。对抗性 NLI。人类特意编写例子来挑战现有模型。更难。  
- **DocNLI、ConTRoL**（2020–21年）。文档级前提。测试多跳和长距离推断。

**架构。** Transformer 编码器（BERT、RoBERTa、DeBERTa）读取 `[CLS] premise [SEP] hypothesis [SEP]`，`[CLS]` 向量经过3分类 softmax。训练于 MNLI，评估在保留集，分布内对达到90%+准确率。

**零样本 NLI。** 给定文档和候选标签，将每个标签转成假设（“本文是关于体育的”），计算蕴含概率，选最大概率标签。这就是 Hugging Face `zero-shot-classification` 管道的机制。

## 实战

### 步骤1：运行预训练 NLI 模型

```python
from transformers import pipeline

nli = pipeline("text-classification",
               model="facebook/bart-large-mnli",
               top_k=None)  # 返回所有标签；取代过时的 return_all_scores=True

premise = "The cat is sleeping on the couch."
hypothesis = "There is a cat in the room."

result = nli({"text": premise, "text_pair": hypothesis})[0]
print(result)
# [{'label': 'entailment', 'score': 0.97},
#  {'label': 'neutral', 'score': 0.02},
#  {'label': 'contradiction', 'score': 0.01}]
```

在生产环境中，`facebook/bart-large-mnli` 和 `microsoft/deberta-v3-large-mnli` 是公开的默认选择。DeBERTa-v3 排行榜领先。

### 步骤2：零样本分类

```python
zs = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

text = "The stock market rallied after the central bank cut interest rates."
labels = ["finance", "sports", "politics", "technology"]

result = zs(text, candidate_labels=labels)
print(result)
# {'labels': ['finance', 'politics', 'technology', 'sports'],
#  'scores': [0.92, 0.05, 0.02, 0.01]}
```

默认模板为“This example is about {label}.”，可用 `hypothesis_template` 自定义。无训练数据要求，无需微调，开箱即用。

### 步骤3：RAG 的忠实度检查

```python
def is_faithful(answer, context, threshold=0.5):
    result = nli({"text": context, "text_pair": answer})[0]
    entail = next(s for s in result if s["label"] == "entailment")
    return entail["score"] > threshold
```

这是 RAGAS 忠实度检查的核心。将生成的答案拆解成原子声明。针对每条声明与检索到的上下文进行检查。报告蕴含比例。

### 步骤4：手写 NLI 分类器（概念）

详见 `code/main.py`，使用标准库完成的玩具示例：通过词汇重叠和否定检测做前提和假设比较。虽不及 Transformer 模型，但展现了任务形态：两个文本输入，三分类输出，损失是对 `{entail, contradict, neutral}` 的交叉熵。

## 注意陷阱

- **仅靠假设的捷径。** 模型仅用假设就能在 SNLI 达到约60%的正确率，因为“not”、“nobody”、“never”等词与矛盾相关。检测标签泄露的强基线。  
- **词汇重叠启发。** 子序列启发式（“每个子序列都被蕴含”）能通过 SNLI，却会在 HANS/ANLI 失败。使用对抗性基准。  
- **长文本性能下降。** 单句 NLI 模型在文档级前提上 F1 下降20+。长上下文用 DocNLI 训练模型。  
- **零样本模板敏感度。** “This example is about {label}” vs “{label}” vs “The topic is {label}” 会造成准确率10+点差异。调优模板。  
- **领域不匹配。** MNLI 训练于通用英文。法律、医疗、科研文本需领域专用 NLI 模型（如 SciNLI、MedNLI）。

## 使用推荐

2026年主流架构：

| 用例 | 模型 |
|---------|-------|
| 通用 NLI | `microsoft/deberta-v3-large-mnli` |
| 快速 / 边缘设备 | `cross-encoder/nli-deberta-v3-base` |
| 零样本分类（轻量） | `facebook/bart-large-mnli` |
| 文档级 NLI | `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` |
| 多语言 | `MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli` |
| RAG 中幻觉检测 | RAGAS / DeepEval 内置 NLI 层 |

2026 元模式：NLI 是文本理解的“万能胶”。只要你需要“A 是否支持 B？”或“A 是否与 B 矛盾？”——先用 NLI 再考虑调用大模型。

## 发布建议

保存为 `outputs/skill-nli-picker.md`：

```markdown
---
name: nli-picker
description: 选择 NLI 模型、标签模板和评估方案，用于分类 / 忠实度检测 / 零样本任务。
version: 1.0.0
phase: 5
lesson: 21
tags: [nlp, nli, zero-shot]
---

对于一个用例（忠实度检测、零样本分类、文档级推断），输出：

1. 模型。指定 NLI checkpoint。结合领域、长度和语言给出理由。  
2. 模板（零样本时）。标签的文本化模式。举例。  
3. 阈值。蕴含概率的决策门槛。基于校准给出理由。  
4. 评估。保留标注集准确率，假设仅基线，对抗样本子集表现。

拒绝在无100例标注检验的情况下交付零样本分类功能。拒绝用句子级 NLI 模型处理文档级前提。警告任何声称 NLI 可完全解决幻觉的说法——它只能减少，无法根除。
```

## 练习

1. **简单。** 用 `facebook/bart-large-mnli` 在20个手工制作的（前提、假设、标签）三元组上测试，覆盖全部三类。测准确率。加入对抗性的“子序列启发”陷阱（“我没吃蛋糕”vs“我吃了蛋糕”）并观察是否被破坏。  
2. **中等。** 在100个 AG News 新闻标题上比较零样本模板“This text is about {label}”与“The topic is {label}”和“{label}”，报告准确率差异。  
3. **困难。** 构建 RAG 忠实度检查器：拆分为原子声明，对每条声明用 NLI 检测。用50个 RAG 生成答案和其黄金上下文进行评测。比较误报率和漏报率与人工标注。

## 关键词

| 术语 | 常见说法 | 实际含义 |
|------|---------|----------|
| NLI | 自然语言推断 | 前提-假设关系的三分类。 |
| RTE | 识别文本蕴含（Recognizing Textual Entailment） | NLI 的旧称，任务相同。 |
| Entailment | “t 蕴含 h” | 读者基于 t 会认为 h 为真。 |
| Contradiction | “t 排除 h” | 读者基于 t 会认为 h 为假。 |
| Neutral | “未定” | t 对 h 无推断关系。 |
| Zero-shot classification | 用 NLI 做分类 | 把标签文本化为假设，选最大蕴含。 |
| Faithfulness | 答案是否有依据？ | 对（检索上下文，生成答案）做 NLI。 |

## 延伸阅读

- [Bowman 等 (2015). 用于学习自然语言推断的大型注释语料库](https://arxiv.org/abs/1508.05326) — SNLI。  
- [Williams, Nangia, Bowman (2017). 通过推断进行句子理解的广覆盖挑战语料库](https://arxiv.org/abs/1704.05426) — MultiNLI。  
- [Nie 等 (2019). 对抗性自然语言推断](https://arxiv.org/abs/1910.14599) — ANLI 基准。  
- [Yin, Hay, Roth (2019). 零样本文本分类基准](https://arxiv.org/abs/1909.00161) — NLI 作为分类器。  
- [He 等 (2021). DeBERTa：具有解码增强和解耦注意力的 BERT](https://arxiv.org/abs/2006.03654) — 2026 年 NLI 主力。
