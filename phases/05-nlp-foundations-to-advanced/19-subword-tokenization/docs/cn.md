# 子词分词 — BPE、WordPiece、Unigram、SentencePiece

> 词级分词器无法处理未见过的单词。字符分词器导致序列长度爆炸。子词分词器折中两者。每个现代大语言模型（LLM）都内置了一个。

**类型：** 学习  
**语言：** Python  
**先决条件：** 第5阶段 · 01（文本处理）、第5阶段 · 04（GloVe / FastText / 子词）  
**时间：** ~60分钟

## 问题

你的词汇表有5万个单词。用户输入 "untokenizable"。你的分词器返回 `[UNK]`。模型因此对该词没有任何信号。更糟的是：你语料库中第90百分位的文档有40个罕见词，也就是说每篇文档丢失40比特信息。

子词分词解决了这个问题。常见单词保持单个token。罕见单词分解成有意义的片段：`untokenizable` → `un`，`token`，`izable`。训练数据覆盖所有内容，因为任何字符串最终都是字节序列。

2026年每个前沿LLM都使用三种算法之一（BPE、Unigram、WordPiece），包裹在三种库之一中（tiktoken、SentencePiece、HF Tokenizers）。你无法发布语言模型而不选定其中之一。

## 概念

![BPE vs Unigram vs WordPiece，逐字符展示](../assets/subword-tokenization.svg)

**BPE（Byte-Pair Encoding 字节对编码）。** 从字符级词汇表开始。统计每对相邻字符。合并出现频率最高的对形成新token。反复进行直到达到目标词汇表大小。主流算法：GPT-2/3/4、Llama、Gemma、Qwen2、Mistral。

**字节级BPE。** 同样算法但作用于原始字节（256个基础token）而非Unicode字符。保证不出现`[UNK]`token——任何字节序列都能编码。GPT-2使用50,257个token（256字节基础 + 50,000合并 + 1特殊token）。

**Unigram。** 从超大词汇表开始。为每个token分配单元概率（unigram probability）。反复剪枝——去除后对语料的对数似然影响最小的token。推理中采用概率采样分词（有助于通过子词正则化实现数据增强）。用于T5、mBART、ALBERT、XLNet、Gemma。

**WordPiece。** 合并对以最大化训练语料的似然，而非单纯频率。用于BERT、DistilBERT、ELECTRA。

**SentencePiece vs tiktoken。** SentencePiece是一个*训练*词汇表（BPE或Unigram）的库，直接在原始Unicode文本上训练，将空格编码为 `▁`。tiktoken是OpenAI的快速*编码器*，针对预建词汇表；不进行训练。

经验法则：

- **训练新词汇表：** SentencePiece（多语言，无需预分词）或HF Tokenizers。  
- **对GPT词汇快速推理：** tiktoken（cl100k_base，o200k_base）。  
- **两者兼顾：** HF Tokenizers — 单一库，训练+服务。

## 动手实现

### 第一步：从零实现BPE

参见 `code/main.py`。循环过程：

```python
def train_bpe(corpus, num_merges):
    vocab = {tuple(word) + ("</w>",): count for word, count in corpus.items()}
    merges = []
    for _ in range(num_merges):
        pairs = Counter()
        for symbols, freq in vocab.items():
            for a, b in zip(symbols, symbols[1:]):
                pairs[(a, b)] += freq
        if not pairs:
            break
        best = pairs.most_common(1)[0][0]
        merges.append(best)
        vocab = apply_merge(vocab, best)
    return merges
```

算法蕴含三点：`</w>`标记单词结尾，确保 `"low"`（后缀）和 `"lower"`（前缀）分开；频率权重确保高频对优先合并；合并列表有序，推理时按训练顺序应用。

### 第二步：用学习合并进行编码

```python
def encode_bpe(word, merges):
    symbols = list(word) + ["</w>"]
    for a, b in merges:
        i = 0
        while i < len(symbols) - 1:
            if symbols[i] == a and symbols[i + 1] == b:
                symbols = symbols[:i] + [a + b] + symbols[i + 2:]
            else:
                i += 1
    return symbols
```

朴素版本复杂度为 O(n·|merges|)。生产实现（tiktoken、HF Tokenizers）通过合并等级查找和优先队列实现近线性时间。

### 第三步：实际使用SentencePiece

```python
import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input="corpus.txt",
    model_prefix="my_tokenizer",
    vocab_size=8000,
    model_type="bpe",          # 或 "unigram"
    character_coverage=0.9995, # 对中日韩字符可设低一点（如英文0.9995，日本语0.995）
    normalization_rule_name="nmt_nfkc",
)

sp = spm.SentencePieceProcessor(model_file="my_tokenizer.model")
print(sp.encode("untokenizable", out_type=str))
# ['▁un', 'token', 'izable']
```

注意：无需预分词，空格编码为 `▁`，`character_coverage` 控制稀有字符保留和映射到 `<unk>` 的权衡。

### 第四步：tiktoken支持OpenAI兼容词汇

```python
import tiktoken
enc = tiktoken.get_encoding("o200k_base")
print(enc.encode("untokenizable"))        # [127340, 101028]
print(len(enc.encode("Hello, world!")))   # 4
```

仅编码。快速（Rust后端）。与GPT-4/5分词完全匹配，用于字节计数、成本估算、上下文窗口预算。

## 2026年仍需警惕的问题

- **分词器漂移。** 训练时用词汇A，部署时用词汇B。Token ID不匹配，模型输出垃圾。CI中校验`tokenizer.json`哈希值。  
- **空白歧义。** BPE中 `"hello"` 和 `" hello"`分词不同。须显式指定 `add_special_tokens` 和 `add_prefix_space`。  
- **多语言训练不足。** 以英语为主的语料产生的词汇表，会把非拉丁文字拆成5-10倍更多token。相同输入日语/阿拉伯语在GPT-3.5中代价5-10倍更高。`o200k_base`部分缓解。  
- **表情符号拆分。** 一个表情符号可能占5个token。预算上下文时注意表情处理。

## 如何使用

2026年的方案：

| 场景                      | 选择             |
|---------------------------|------------------|
| 从零训练单语言模型        | HF Tokenizers（BPE） |
| 训练多语言模型            | SentencePiece（Unigram，`character_coverage=0.9995`） |
| 提供OpenAI兼容API服务     | tiktoken（GPT-4+用`o200k_base`） |
| 领域专用词汇（代码，数学，蛋白质） | 在领域语料上训练自定义BPE，和基础词汇表合并 |
| 边缘推理，小模型          | Unigram（小词汇表效果更佳） |

词汇表大小是扩展决策，不是固定。粗略估计：<1B参数使用32k，1-10B参数用50-100k，多语言和前沿模型用200k+。

## 交付成果

保存为 `outputs/skill-bpe-vs-wordpiece.md`：

```markdown
---
name: tokenizer-picker
description: 针对给定语料和部署目标，选择分词算法、词汇大小和库。
version: 1.0.0
phase: 5
lesson: 19
tags: [nlp, tokenization]
---

根据语料（大小、语言、领域）与部署目标（从零训练/微调/API兼容推理），输出：

1. 算法。BPE、Unigram 或 WordPiece。简短原因说明。
2. 库。SentencePiece、HF Tokenizers 或 tiktoken。原因说明。
3. 词汇大小。四舍五入到最接近的1000。理由与模型规模和语言覆盖相关。
4. 覆盖设置。`character_coverage`、`byte_fallback`、特殊token列表。
5. 验证方案。保留集的平均词tokens数、OOV率、压缩率、往返解码一致性。

拒绝训练 character_coverage < 0.995 的分词器于含有罕见字符的语料。拒绝发布没有CI冻结的 `tokenizer.json` 哈希检测的词汇表。标记词汇少于16k的单语分词器可能规格不足。
```

## 练习

1. **简单。** 在 `code/main.py` 示例的微型语料上训练500次合并的BPE。编码三个保留词。多少词准确产生1个token，多少产生超过1个token？  
2. **中等。** 在100句英文维基百科句子上，比较 `cl100k_base`、`o200k_base` 和你训练的32k词汇SentencePiece BPE的token计数。报告各自压缩率。  
3. **困难。** 利用相同语料，分别训练BPE、Unigram和WordPiece。在微小情感分类器上使用每种分词结果测量下游准确率。分词算法选择是否影响F1分数超过1分？

## 关键词

| 术语            | 俗称                     | 实际含义                                                         |
|-----------------|--------------------------|------------------------------------------------------------------|
| BPE             | Byte-Pair Encoding       | 贪心合并最频繁的字符对，直到达到目标词汇表大小。                  |
| 字节级BPE       | 永不出现未知token        | 在256字节基础上做BPE；GPT-2 / Llama采用。                         |
| Unigram         | 概率分词器               | 从候选集剪枝，依据对数似然；T5、Gemma采用。                      |
| SentencePiece   | 空格编码分词器           | 直接训练BPE/Unigram的库，空格编码为 `▁`。                        |
| tiktoken        | 快速编码器               | OpenAI基于Rust的BPE编码器，用于预训练词汇表，不训练。             |
| 合并列表        | 合并魔法数字             | `(a, b) → ab`合并操作的有序列表，推理时按顺序应用。               |
| 字符覆盖度      | 多稀有才算稀有           | 训练语料的字符被分词器覆盖的比例；典型为 ~0.9995。               |

## 进一步阅读

- [Sennrich, Haddow, Birch (2015). Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) — BPE 论文。
- [Kudo (2018). Subword Regularization with Unigram Language Model](https://arxiv.org/abs/1804.10959) — Unigram 论文。
- [Kudo, Richardson (2018). SentencePiece: A simple and language independent subword tokenizer](https://arxiv.org/abs/1808.06226) — 该库介绍。
- [Hugging Face — Summary of the tokenizers](https://huggingface.co/docs/transformers/tokenizer_summary) — 简洁参考。  
- [OpenAI tiktoken repo](https://github.com/openai/tiktoken) — 使用手册与编码列表。
