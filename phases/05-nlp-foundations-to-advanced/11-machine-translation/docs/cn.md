# 机器翻译

> 翻译是促成 NLP 研究三十年发展的任务，现在仍在推动它的发展。

**类型：** 构建  
**语言：** Python  
**先决条件：** 第5阶段 · 10（Attention Mechanism 注意力机制），第5阶段 · 04（GloVe, FastText, Subword 子词）  
**时间：** ~75分钟

## 问题

模型读取一种语言的句子，并生成另一种语言的句子。长度可变。词序可变。某些源语言词会对应多个目标词，反之亦然。习语无法一一对应。英文 "I miss you" 在法语中是 "tu me manques" —— 字面意思是“你对我来说缺少”。没有一种词级对齐能存活下来。

机器翻译是迫使 NLP 发明编码器-解码器、注意力机制、Transformer（Transformer 架构），最终形成整个大型语言模型（LLM）范式的任务。 每一步前进都是因为翻译质量可度量，且人机之间的差距顽固存在。

本课跳过历史讲解，教授 2026 年的工作流程：预训练多语言编码器-解码器（NLLB-200 或 mBART），子词分词，束搜索（beam search），BLEU 和 chrF 评价，以及仍会出现在生产环境中的少量失败模式。

## 概念

![MT pipeline: tokenize → encode → decode with attention → detokenize](../assets/mt-pipeline.svg)

现代机器翻译是基于 Transformer 编码器-解码器架构的平行文本训练模型。编码器按源语言的分词方式读取输入。解码器通过交叉注意力（第10课）使用编码器的输出，每次生成一个子词。解码使用束搜索，避免贪心解码陷阱。输出解码后进行去分词、去大小写处理，并与参考译文评分。

三个操作层面决定真实世界机器翻译的质量。

- **分词器。** 基于 SentencePiece BPE，训练于多语言混合语料。跨语言共享的词汇表使 NLLB 支持零样本翻译对。
- **模型大小。** NLLB-200 蒸馏版 600M 可运行于笔记本；NLLB-200 3.3B 是公开的生产默认版；54.5B 是研究天花板。
- **解码方式。** 一般内容选择束宽 4-5。使用长度惩罚避免输出过短。需要术语一致性时使用约束解码。

## 构建它

### 第1步：调用预训练机器翻译模型

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_id = "facebook/nllb-200-distilled-600M"
tok = AutoTokenizer.from_pretrained(model_id, src_lang="eng_Latn")
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

src = "The cats are running."
inputs = tok(src, return_tensors="pt")

out = model.generate(
    **inputs,
    forced_bos_token_id=tok.convert_tokens_to_ids("fra_Latn"),
    num_beams=5,
    length_penalty=1.0,
    max_new_tokens=64,
)
print(tok.batch_decode(out, skip_special_tokens=True)[0])
```

```text
Les chats courent.
```

这里有三点重要：`src_lang` 告诉分词器应该使用哪种脚本和分词方式；`forced_bos_token_id` 告诉解码器生成哪个语言输出。两者都是 NLLB 特有的技巧；mBART 和 M2M-100 使用自己的规范，不能互换。

### 第2步：BLEU 和 chrF

BLEU 测量输出与参考译文间的 n-gram 重叠，统计 1-4 阶 n-gram 的几何平均精度，带简短输出惩罚。分数范围在 [0, 100]。是常用指标，但难以直观理解：30 分表示“可用”；40 分表示“良好”；50 分表示“卓越”；小于 1 分的差异属于噪声。

chrF 是字符级 F-score，更适合形态丰富语言，BLEU 在这类语言中匹配数量不足。常与 BLEU 一起报告。

```python
import sacrebleu

hypotheses = ["Les chats courent."]
references = [["Les chats courent."]]

bleu = sacrebleu.corpus_bleu(hypotheses, references)
chrf = sacrebleu.corpus_chrf(hypotheses, references)
print(f"BLEU: {bleu.score:.1f}  chrF: {chrf.score:.1f}")
```

始终使用 `sacrebleu`。它对分词做了规范化，从而保证跨论文的分数可比。自己实现 BLEU 很容易导致误导的基准。

### 三层评估体系（2026 年）

现代机器翻译评价使用三大互补指标族。生产环境至少搭配两种。

- **启发式指标**（BLEU，chrF）：快速、基于参考、可解释，对同义改写不敏感。用于历史比较和回归检测。
- **学习型指标**（COMET，BLEURT，BERTScore）：基于人类判断训练的神经模型，比较译文与源文和参考的语义相似度。COMET 自 2023 年以来在 MT 研究关联最高，2026 年生产默认用。  
- **LLM 评判**（无参考）：提示大型语言模型对译文在流利度、充分度、语气、文化适切性进行打分。GPT-4 作为评分者，在评分标准设计合理时与人工一致率约 80%。适用于无参考文本的开放内容。

2026 年实用堆栈：使用 `sacrebleu` 计算 BLEU 和 chrF，`unbabel-comet` 计算 COMET，最后用提示的大型语言模型给出最终人工视角的评分信号。部署前请基于 50-100 个人工标注样本校准各指标。

无参考指标（COMET-QE，BLEURT-QE，以及 LLM-as-judge）能够评估无参考的翻译，适合长尾语言对无参考译文存在的情况。

### 第3步：生产环境中会出错的地方

上述工作流程可令机器翻译 80% 流利输出，20% 静默失败。已知的失败模式：

- **幻觉**。模型生成了源文中不存在的内容。常见于不熟悉领域词汇。症状是输出流畅，但陈述了源文未述及的事实。缓解方法：对领域术语使用约束解码、规管内容人工审核、监控输出长度远超输入的情况。
- **脱靶生成**。模型翻译成错误的语言。NLLB 在罕见语言对中尤易出现。缓解方法：确认 `forced_bos_token_id`，并始终对输出进行语言 ID 校验。
- **术语漂移**。如“Sign up”在文档1中翻译为 "s'inscrire"，在文档2中变成 "créer un compte"。在 UI 文本和面向用户的字符串中，一致性比单纯的质量更重要。缓解方法：术语表约束解码或后期编辑字典。
- **敬语不匹配**。法语中的“tu”对“vous”，日语的敬语等级。模型选用训练中出现较多的形式。面向客户的内容通常错误。缓解方法：对支持的模型加敬语标记的提示前缀，或微调单一敬语语料的小模型。
- **短句长度爆炸**。非常短的输入句子常产生过长的翻译，因为长度惩罚在源句长度约 5 以下时骤然减弱。缓解方法：根据源文长度硬限制最大输出长度。

### 第4步：针对领域微调

预训练模型是通用型。法律、医疗或游戏对话翻译可通过领域平行语料微调得到明显提升。步骤不复杂：

```python
from transformers import Trainer, TrainingArguments
from datasets import Dataset

pairs = [
    {"src": "The defendant pleaded guilty.", "tgt": "L'accusé a plaidé coupable."},
]

ds = Dataset.from_list(pairs)


def preprocess(ex):
    return tok(
        ex["src"],
        text_target=ex["tgt"],
        truncation=True,
        max_length=128,
        padding="max_length",
    )


ds = ds.map(preprocess, remove_columns=["src", "tgt"])

args = TrainingArguments(output_dir="out", per_device_train_batch_size=4, num_train_epochs=3, learning_rate=3e-5)
Trainer(model=model, args=args, train_dataset=ds).train()
```

几千条高质量平行样本，胜过几十万条网络抓取的噪声样本。训练数据质量是生产环境中最大可控因素。

## 使用它

2026 年机器翻译生产环境推荐堆栈：

| 用例 | 推荐起点 |
|---------|---------------------------|
| 任意语种相互翻译，200 语种 | `facebook/nllb-200-distilled-600M`（笔记本）或 `nllb-200-3.3B`（生产） |
| 以英语为中心，高质量，50 语种 | `facebook/mbart-large-50-many-to-many-mmt` |
| 快速批量，廉价推理，英法/德/西 | Helsinki-NLP / Marian 相关模型 |
| 低延迟浏览器端部署 | ONNX 量化 Marian（~50 MB） |
| 追求最高质量，预算充足 | GPT-4 / Claude / Gemini 配合翻译提示 |

2026 年，部分语言对上大模型已超过专用机器翻译模型，尤其在成语内容和长上下文翻译方面。权衡是每个 token 成本及延迟。上下文长度、风格一致性或提示式的领域适配比吞吐率重要时，宜选大模型。

## 发布它

保存为 `outputs/skill-mt-evaluator.md`：

```markdown
---
name: mt-evaluator
description: Evaluate a machine translation output for shipping.
version: 1.0.0
phase: 5
lesson: 11
tags: [nlp, translation, evaluation]
---

Given a source text and a candidate translation, output:

1. Automatic score estimate. BLEU and chrF ranges you would expect. State whether a reference is available.
2. Five-point human-verifiable check list: (a) content preservation (no hallucinations), (b) correct language, (c) register / formality match, (d) terminology consistency with glossary if provided, (e) no truncation or length explosion.
3. One domain-specific issue to probe. E.g., for legal: named entities and statute citations. For medical: drug names and dosages. For UI: placeholder variables `{name}`.
4. Confidence flag. "Ship" / "Ship with review" / "Do not ship". Tie to the severity of issues found in step 2.

Refuse to ship a translation without a language-ID check on output. Refuse to evaluate without a reference unless the user explicitly opts in to reference-free scoring (COMET-QE, BLEURT-QE). Flag any content over 1000 tokens as likely needing chunked translation.
```

## 练习

1. **简单。** 使用 `nllb-200-distilled-600M` 将一个 5 句的英文段落翻译成法语，再翻译回英文。测量往返翻译与原文的接近度，你应能观察到语义保留但词汇选择漂移。
2. **中等。** 实现对翻译输出的语言 ID 检查，使用 `fasttext lid.176` 或 `langdetect`。集成到 MT 调用中，确保返回前捕捉错误语言生成。
3. **困难。** 对 `nllb-200-distilled-600M` 进行微调，使用你选择的 5,000 条领域语料。微调前后在保留集上测量 BLEU，报告哪些句子类型改善了，哪些退步了。

## 关键词

| 术语 | 所谓 | 实际含义 |
|------|-----------------|-----------------------|
| BLEU | 翻译评分 | 带简短惩罚的 n-gram 精度，[0, 100]。 |
| chrF | 字符的 F 分数 | 字符级 F-score。形态丰富语言更敏感。 |
| NMT | 神经机器翻译 | 基于平行文本训练的 Transformer 编码器-解码器架构，2017 年后主流。 |
| NLLB | No Language Left Behind（不让任何语言落下） | Meta 出品的 200 语言机器翻译模型系列。 |
| 约束解码 | 受控输出 | 强制输出中出现或不出现某些 token 或 n-gram。 |
| 幻觉 | 虚构内容 | 模型输出中不为源文支持的内容。 |

## 进一步阅读

- [Costa-jussà 等人（2022）。No Language Left Behind: Scaling Human-Centered Machine Translation（没有语言被落下：面向人类的机器翻译扩展）](https://arxiv.org/abs/2207.04672) — NLLB 论文。
- [Post（2018）。A Call for Clarity in Reporting BLEU Scores（呼吁清晰报告 BLEU 分数）](https://aclanthology.org/W18-6319/) — 说明为什么 `sacrebleu` 是报告 BLEU 唯一正确的方法。
- [Popović（2015）。chrF: character n-gram F-score for automatic MT evaluation（chrF: 用于自动机器翻译评估的字符 n-gram F 分数）](https://aclanthology.org/W15-3049/) — chrF 论文。
- [Hugging Face MT 指南](https://huggingface.co/docs/transformers/tasks/translation) — 实用微调教程。
