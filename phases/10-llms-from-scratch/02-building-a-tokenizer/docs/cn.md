# 从零开始构建分词器（Tokenizer）

> 课程01给了你一个玩具。本课给你一把武器。

**类型：** 构建  
**语言：** Python  
**先决条件：** 第10阶段，课程01（分词器：BPE、WordPiece、SentencePiece）  
**时间：** ~90分钟

## 学习目标

- 构建一个生产级别的BPE分词器，支持Unicode，空白符规范化以及特殊标记  
- 实现字节级回退，确保分词器能对任何输入（包括表情符号（emoji）、中日韩（CJK）和代码）进行编码，且没有未知标记  
- 添加预分词（pre-tokenization）的正则表达式模式，先于BPE合并在单词边界处拆分文本  
- 在语料库上训练自定义分词器，并在多语言文本上评估其压缩率，比较tiktoken的效果  

## 存在的问题

课程01中的BPE分词器只能处理英文文本。现在试着输入日语、表情符号或带有混合制表符和空格的Python代码。

结果崩溃了。

这不是因为BPE算法错误，而是因为实现不完整。生产级分词器需要处理任意编码的原始字节，对Unicode进行规范化后再拆分，管理永远不会合并的特殊标记，连接预分词和子词拆分，而且速度要足够快，不成为处理15万亿tokens训练管线的瓶颈。

GPT-2的词表有50,257个token，Llama 3有128,256个，GPT-4约10万个。这些不是玩具数字。词表背后的合并表训练于百GB级文本，周边机制——规范化、预分词、特殊token注入、聊天模板格式化——决定了分词器是只能处理“hello world”还是能应付整个互联网。

你将构建这些机制。

## 概念

### 完整流水线

生产级分词器不是单一算法，而是包含五个阶段的流水线，每个阶段解决不同的问题。

```mermaid
graph LR
    A[原始文本] --> B[规范化]
    B --> C[预分词]
    C --> D[BPE合并]
    D --> E[特殊标记]
    E --> F[Token ID映射]

    style A fill:#1a1a2e,stroke:#e94560,color:#fff
    style B fill:#1a1a2e,stroke:#e94560,color:#fff
    style C fill:#1a1a2e,stroke:#e94560,color:#fff
    style D fill:#1a1a2e,stroke:#e94560,color:#fff
    style E fill:#1a1a2e,stroke:#e94560,color:#fff
    style F fill:#1a1a2e,stroke:#e94560,color:#fff
```

每个阶段的具体职责：

| 阶段 | 功能 | 重要性 |
|-------|-------|--------|
| 规范化（Normalize） | NFKC Unicode，大小写可选，小写后可选去重音符号 | “fi”连字（U+FB01）变成“fi”（两个字符）。不做规范化，同一词会生成不同token。 |
| 预分词（Pre-Tokenize） | 在BPE之前将文本分割成块 | 防止BPE跨单词边界合并，“the cat”不应生成“e c”这类token。 |
| BPE合并（BPE Merge） | 对字节序列应用学习到的合并规则 | 核心压缩算法，将原始字节转换成子词token。 |
| 特殊标记（Special Tokens） | 注入[BOS]、[EOS]、[PAD]、聊天模板标记 | 这些token有固定ID，绝不参与BPE合并，模型需要它们来构建结构。 |
| ID映射（ID Mapping） | 将token字符串转为整数ID | 模型操作的是整数ID而非字符串。 |

### 字节级BPE（Byte-Level BPE）

课程01的分词器使用UTF-8字节，这样做是正确的。但有个重要问题没处理好：如果字节不是有效UTF-8怎么办？

字节级BPE通过将所有可能的字节值（0-255）视为有效token来解决这个问题。你的基础词表正好256个条目。任何文件——文本、二进制、损坏——均可被分词而不会产生未知token。

GPT-2加了个技巧：将每个字节映射成一个人类可读的Unicode字符，例如字节0x20（空格）映射为字符“G”。这纯粹是为了美观，算法并不在意这个映射。

真正强大的地方在于字节级BPE能处理地球上任何语言。中文字符每个3个UTF-8字节，日语可能3-4个，阿拉伯文、天城文（Devanagari）、表情符号——它们都是字节序列。BPE算法在这些字节序列中找模式的方式，和分析英文ASCII字节完全一样。

### 预分词（Pre-Tokenization）

在BPE算法处理文本之前，你需要先将文本拆分成块，防止算法合并跨单词的token。

GPT-2用的正则表达式模式如下：

```text
'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
```

此模式针对英文中的缩写（“don't”变成“don” + “‘t’”）、带可选前置空格的单词、数字、标点符号和空白符进行分割。前置空格会附着于单词前，如“the cat”变成[" the", " cat"]，而非["the", " ", "cat"]。

Llama采用SentencePiece，完全跳过正则表达式，直接把原始字节流视为一个长序列，让BPE自动找边界。这样更简单，但允许BPE创造跨词token。

这两种策略各有利弊。GPT-2的正则防止分词器把两个“the”合并成一个token，而SentencePiece允许这种合并，有时实现更高效的压缩但可读性较差。

### 特殊标记（Special Tokens）

每个生产级分词器都会保留部分token ID，用于结构性标记：

| Token | 作用 | 使用模型 |
|-------|--------|---------|
| `[BOS]` / `<s>` | 序列开始 | Llama 3，GPT |
| `[EOS]` / `</s>` | 序列结束 | 全部模型 |
| `[PAD]` | 批处理对齐用填充 | BERT，T5 |
| `[UNK]` | 未知token（字节级BPE消除了未知token） | BERT，WordPiece |
| `<\|im_start\|>` | 聊天消息边界开始 | ChatGPT，Qwen |
| `<\|im_end\|>` | 聊天消息边界结束 | ChatGPT，Qwen |
| `<\|user\|>` | 用户对话标记 | Llama 3 |
| `<\|assistant\|>` | 助手对话标记 | Llama 3 |

特殊token绝不被BPE拆分。它们在合并算法运行前被精确匹配，替换成固定ID，周围的文本则正常分词。

### 聊天模板（Chat Templates）

大部分人困惑点和很多实现出错的地方。

用于聊天模型时，API接受消息列表：

```text
[
  {"role": "system", "content": "You are helpful."},
  {"role": "user", "content": "Hello"},
  {"role": "assistant", "content": "Hi there!"}
]
```

模型实际上看不到JSON，而是看到扁平的token序列。聊天模板用特殊token把消息列表转换成扁平序列。不同模型格式不同：

```text
Llama 3:
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are helpful.<|eot_id|><|start_header_id|>user<|end_header_id|>

Hello<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Hi there!<|eot_id|>

ChatGPT:
<|im_start|>system
You are helpful.<|im_end|>
<|im_start|>user
Hello<|im_end|>
<|im_start|>assistant
Hi there!<|im_end|>
```

模板搞错了模型会输出垃圾。训练只用过某一严苛格式，缺少换行、标记错位、额外空格都会使输入超出训练分布。

### 速度

Python在生产分词上太慢。

tiktoken（OpenAI）用Rust写，并提供Python绑定。HuggingFace tokenizers同样是Rust。SentencePiece是C++实现。比纯Python快10~100倍。

举例：Llama 3预训练需切分15万亿tokens，按100万tokens/秒（Python快码）需174天，按1亿tokens/秒（Rust方案）则1.7天。

这里用Python实现，目的是理解算法。生产环境用编译语言实现，然后用Python封装调用。

## 构建它

### 步骤1：字节级编码

基础。将任何字符串转为字节序列，为显示将字节映射成可打印字符，并实现反向过程。

```python
def bytes_to_tokens(text):
    return list(text.encode("utf-8"))

def tokens_to_text(token_bytes):
    return bytes(token_bytes).decode("utf-8", errors="replace")
```

多语言文本测试字节数：

```python
texts = [
    ("English", "hello"),
    ("Chinese", "你好"),
    ("Emoji", "🔥"),
    ("Mixed", "hello你好🔥"),
]

for label, text in texts:
    b = bytes_to_tokens(text)
    print(f"{label}: {len(text)} chars -> {len(b)} bytes -> {b}")
```

“hello”有5个字节。“你好”6个（每字3字节）。火焰表情是4字节。字节级分词器不关心是什么语言，字节就是字节。

### 步骤2：用正则预分词

用GPT-2模式用正则分割文本块。每块后续独立做BPE。

```python
import re

try:
    import regex
    GPT2_PATTERN = regex.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )
except ImportError:
    GPT2_PATTERN = re.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?[a-zA-Z]+| ?[0-9]+| ?[^\s\w]+|\s+(?!\S)|\s+"""
    )

def pre_tokenize(text):
    return [match.group() for match in GPT2_PATTERN.finditer(text)]
```

`regex`模块支持Unicode属性转义（`\p{L}`表示字母，`\p{N}`表示数字），内置`re`模块不支持，故用ASCII类做后备。生产多语言分词器建议用`regex`安装。

试试：

```python
print(pre_tokenize("Hello, world! Don't stop."))
# [' Hello', ',', ' world', '!', " Don", "'t", ' stop', '.']
```

前置空格附着于单词，缩写在撇号处分开，标点作为独立块。BPE决不会跨这些块合并。

### 步骤3：对字节序列做BPE

课程01的核心算法，只不过现在对预分词块独立操作。

```python
from collections import Counter

def get_byte_pairs(chunks):
    pairs = Counter()
    for chunk in chunks:
        byte_seq = list(chunk.encode("utf-8"))
        for i in range(len(byte_seq) - 1):
            pairs[(byte_seq[i], byte_seq[i + 1])] += 1
    return pairs

def apply_merge(byte_seq, pair, new_id):
    merged = []
    i = 0
    while i < len(byte_seq):
        if i < len(byte_seq) - 1 and byte_seq[i] == pair[0] and byte_seq[i + 1] == pair[1]:
            merged.append(new_id)
            i += 2
        else:
            merged.append(byte_seq[i])
            i += 1
    return merged
```

### 步骤4：特殊标记处理

特殊token需要精确匹配和固定ID，跳过BPE过程。

```python
class SpecialTokenHandler:
    def __init__(self):
        self.special_tokens = {}
        self.pattern = None

    def add_token(self, token_str, token_id):
        self.special_tokens[token_str] = token_id
        escaped = [re.escape(t) for t in sorted(self.special_tokens.keys(), key=len, reverse=True)]
        self.pattern = re.compile("|".join(escaped))

    def split_with_specials(self, text):
        if not self.pattern:
            return [(text, False)]
        parts = []
        last_end = 0
        for match in self.pattern.finditer(text):
            if match.start() > last_end:
                parts.append((text[last_end:match.start()], False))
            parts.append((match.group(), True))
            last_end = match.end()
        if last_end < len(text):
            parts.append((text[last_end:], False))
        return parts
```

### 第五步：完整的 Tokenizer 类

将所有步骤串联起来：归一化，基于特殊 token 分割，预分词，BPE 合并，映射到 ID。

```python
import unicodedata

class ProductionTokenizer:
    def __init__(self):
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.special_handler = SpecialTokenHandler()
        self.next_id = 256

    def normalize(self, text):
        return unicodedata.normalize("NFKC", text)

    def train(self, text, num_merges):
        text = self.normalize(text)
        chunks = pre_tokenize(text)
        chunk_bytes = [list(chunk.encode("utf-8")) for chunk in chunks]

        for i in range(num_merges):
            pairs = Counter()
            for seq in chunk_bytes:
                for j in range(len(seq) - 1):
                    pairs[(seq[j], seq[j + 1])] += 1
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            new_id = self.next_id
            self.next_id += 1
            self.merges[best] = new_id
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]
            chunk_bytes = [apply_merge(seq, best, new_id) for seq in chunk_bytes]

    def add_special_token(self, token_str):
        token_id = self.next_id
        self.next_id += 1
        self.special_handler.add_token(token_str, token_id)
        self.vocab[token_id] = token_str.encode("utf-8")
        return token_id

    def encode(self, text):
        text = self.normalize(text)
        parts = self.special_handler.split_with_specials(text)
        all_ids = []
        for part_text, is_special in parts:
            if is_special:
                all_ids.append(self.special_handler.special_tokens[part_text])
            else:
                for chunk in pre_tokenize(part_text):
                    byte_seq = list(chunk.encode("utf-8"))
                    for pair, new_id in self.merges.items():
                        byte_seq = apply_merge(byte_seq, pair, new_id)
                    all_ids.extend(byte_seq)
        return all_ids

    def decode(self, ids):
        byte_parts = []
        for token_id in ids:
            if token_id in self.vocab:
                byte_parts.append(self.vocab[token_id])
        return b"".join(byte_parts).decode("utf-8", errors="replace")

    def vocab_size(self):
        return len(self.vocab)
```

### 第六步：多语言测试

真正的测试。用英语、中文、表情符号和代码测试。

```python
corpus = (
    "The quick brown fox jumps over the lazy dog. "
    "The quick brown fox runs through the forest. "
    "Machine learning models process natural language. "
    "Deep learning transforms how we build software. "
    "def train(model, data): return model.fit(data) "
    "def predict(model, x): return model(x) "
)

tok = ProductionTokenizer()
tok.train(corpus, num_merges=50)

bos = tok.add_special_token("<|begin|>")
eos = tok.add_special_token("<|end|>")

test_texts = [
    "The quick brown fox.",
    "你好世界",
    "Hello 🌍 World",
    "def foo(x): return x + 1",
    f"<|begin|>Hello<|end|>",
]

for text in test_texts:
    ids = tok.encode(text)
    decoded = tok.decode(ids)
    print(f"输入:   {text}")
    print(f"Token数: {len(ids)} 个 id")
    print(f"解码:   {decoded}")
    print()
```

中文字符每个产生3个字节。表情符号产生4个字节。没有一个导致 tokenizer 崩溃。没有一个产生未知 token。这就是基于字节级别 BPE 的强大之处。

## 使用它

### 比较真实的 Tokenizer

加载来自 Llama 3、GPT-4 和 Mistral 的实际 tokenizer。看看它们如何处理同一段多语言文本。

```python
import tiktoken

gpt4_enc = tiktoken.get_encoding("cl100k_base")

test_paragraph = "Machine learning is powerful. 机器学习很强大。 L'apprentissage automatique est puissant. 🤖💪"

tokens = gpt4_enc.encode(test_paragraph)
pieces = [gpt4_enc.decode([t]) for t in tokens]
print(f"GPT-4 ({len(tokens)} tokens): {pieces}")
```

```python
from transformers import AutoTokenizer

llama_tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
mistral_tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

for name, tok in [("Llama 3", llama_tok), ("Mistral", mistral_tok)]:
    tokens = tok.encode(test_paragraph)
    pieces = tok.convert_ids_to_tokens(tokens)
    print(f"{name} ({len(tokens)} tokens): {pieces[:20]}...")
```

你会看到相同文本的 token 数不同。具有128K词汇量的 Llama 3 在合并常见模式方面更激进。具有100K词汇的 GPT-4 处于中间。32K词汇的 Mistral 生成更多的 tokens，但嵌入层更小。

取舍总是一样的：更大的词汇量意味着更短的序列，但参数更多。

## 发布它

本课程提供了一个用于构建和调试生产级 tokenizer 的提示。见 `outputs/prompt-tokenizer-builder.md`。

## 练习

1. **简单：** 添加一个 `get_token_bytes(id)` 方法，显示任意 token ID 的原始字节。用它来检查你最常见的合并 token 实际代表什么。
2. **中等：** 实现 Llama 风格的预分词，将文本按空白字符和数字分割，但保留前导空格。在相同语料上与 GPT-2 的正则表达式方法比较其词汇表。
3. **困难：** 添加一个聊天模板方法，接收一个包含 `{"role": ..., "content": ...}` 消息的列表，生成符合 Llama 3 聊天格式的正确 token 序列。并与 HuggingFace 实现进行测试对比。

## 关键词

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|---------|
| Byte-level BPE | “基于字节的 tokenizer” | 以256个字节值为基础词汇的 BPE —— 处理任何输入，且不会产生未知 token |
| Pre-tokenization（预分词） | “BPE 前的拆分” | 基于正则或规则的拆分，防止 BPE 跨单词边界合并 |
| NFKC normalization（NFKC 归一化） | “Unicode 清理” | 先做规范分解再做兼容性组合 —— “fi”连字变“fi”，全角“A”变“A” |
| Chat template（聊天模板） | “消息变 token 的方式” | 将角色/内容消息列表转换为平坦 token 序列的精确格式 —— 与模型训练格式对应 |
| Special tokens（特殊 token） | “控制 token” | 预留的 token ID，绕过 BPE —— 如 [BOS]、[EOS]、[PAD]、聊天标记 —— 在合并前精确匹配 |
| Fertility（Token 数量率） | “每个单词的 token 数” | 输出 token 数与输入单词数的比率 —— GPT-4 对英语约1.3，韩语为2-3，更高表示上下文浪费 |
| tiktoken | “OpenAI tokenizer” | Rust 实现的 BPE，带 Python 绑定 —— 比纯 Python 快 10-100 倍 |
| Merge table（合并表） | “词汇表” | 训练期间学习到的字节对合并有序列表 —— 它就是 tokenizer 的知识库 |

## 深入阅读

- [OpenAI tiktoken 源码](https://github.com/openai/tiktoken) —— GPT-3.5/4 使用的 Rust BPE 实现
- [HuggingFace tokenizers](https://github.com/huggingface/tokenizers) —— 支持 BPE、WordPiece、Unigram 的 Rust tokenizer 库
- [Llama 3 论文（Meta, 2024）](https://arxiv.org/abs/2407.21783) —— 128K词汇和 tokenizer 训练细节
- [SentencePiece (Kudo & Richardson, 2018)](https://arxiv.org/abs/1808.06226) —— 语言无关的分词方法
- [GPT-2 tokenizer 源码](https://github.com/openai/gpt-2/blob/master/src/encoder.py) —— 原始字节到 Unicode 映射方法
