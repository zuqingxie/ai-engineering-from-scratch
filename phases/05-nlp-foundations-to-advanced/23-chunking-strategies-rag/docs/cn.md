# RAG 的分块策略（Chunking Strategies）

> 分块配置对检索质量的影响与嵌入模型（embedding model，嵌入模型）的选择同等重要（Vectara NAACL 2025）。分块做错了，没有任何重排（reranking）能救你。

**类型：** 练习构建  
**语言：** Python  
**先决条件：** 阶段 5 · 14（信息检索 Information Retrieval）、阶段 5 · 22（嵌入模型 Embedding Models）  
**时间：** ~60分钟

## 问题描述

你将一份 50 页的合同放入 RAG 系统。用户问：“终止条款（termination clause）是什么？”检索器却返回了封面页。为什么？因为模型是在 512 token 分块上训练的，而终止条款正好跨越了两页断页，并且没有本地关键词与查询相关联。

解决方法不是“买更好的嵌入模型”，而是分块。多大合适？重叠多少？在哪儿分割？带不带上下文？

2026 年 2 月的基准测试显示了令人惊讶的结果：

- Vectara 2026 研究：递归 512-token 分块超过语义分块，准确率69% 对 54%。
- SPLADE + Mistral-8B 在 Natural Questions 上：重叠没有带来可衡量的提升。
- 上下文断崖（context cliff）：响应质量在约 2500 token 上下文处急剧下降。

“显而易见”的答案（语义分块，20% 重叠，1000 token）往往是错的。此课程帮你建立六种策略的直觉，并告诉你何时选用哪种。

## 概念介绍

![六种分块策略在同一文本上的可视化](../assets/chunking.svg)

**固定分块（Fixed chunking）**。每隔 N 个字符或 token 切分。最简单的基线方法。会断句中。压缩效果好，但连贯性差。

**递归分块（Recursive）**。LangChain 的 `RecursiveCharacterTextSplitter`。先尝试用 `\n\n` 分割，再是 `\n`，然后是 `.`，最后是空格。回退机制完善。2026 年默认方案。

**语义分块（Semantic）**。对每句话做嵌入，计算相邻句子间的余弦相似度。相似度低于阈值时切分。保持主题连贯。速度慢，有时产生只有 40 token 的碎片，影响检索。

**句子分块（Sentence）**。按句子分界切分。每块包括一句或 N 句。与语义分块在 ~5k token 范围内效果相当，成本更低。

**父文档分块（Parent-document）**。检索时存储小的子块和较大的父块。通过子块检索，返回父块。退化表现优雅：子块不好，父块仍合理。

**后期分块（Late chunking, 2024）**。先在 token 级别嵌入整个文档，再池化成分块向量。保留跨块上下文。适合长上下文嵌入器（如 BGE-M3、Jina v3）。计算成本更高。

**上下文检索（Contextual retrieval，Anthropic 2024）**。在每块前加上 LLM 生成的位置摘要（“该块是终止条款第 3.2 节……”）。Anthropic 自测提升 35-50% 检索效果。索引成本高。

### 战胜所有默认设置的规则

匹配分块大小与查询类型：

| 查询类型 | 分块大小 |
|----------|----------|
| 事实型（“CEO 的名字是什么？”） | 256-512 token |
| 分析型 / 多跳 | 512-1024 token |
| 整节理解 | 1024-2048 token |

NVIDIA 2026 基准测试。分块要足够大以包含答案及局部上下文，且要足够小使检索器返回的前 K 个命中聚焦于答案而非上下文噪声。

## 构建示例

### 第一步：固定和递归分块

```python
def chunk_fixed(text, size=512, overlap=0):
    step = size - overlap
    return [text[i:i + size] for i in range(0, len(text), step)]


def chunk_recursive(text, size=512, seps=("\n\n", "\n", ". ", " ")):
    if len(text) <= size:
        return [text]
    for sep in seps:
        if sep not in text:
            continue
        parts = text.split(sep)
        chunks = []
        buf = ""
        for p in parts:
            if len(p) > size:
                if buf:
                    chunks.append(buf)
                    buf = ""
                chunks.extend(chunk_recursive(p, size=size, seps=seps[1:] or (" ",)))
                continue
            candidate = buf + sep + p if buf else p
            if len(candidate) <= size:
                buf = candidate
            else:
                if buf:
                    chunks.append(buf)
                buf = p
        if buf:
            chunks.append(buf)
        return [c for c in chunks if c.strip()]
    return chunk_fixed(text, size)
```

### 第二步：语义分块

```python
def chunk_semantic(text, encoder, threshold=0.6, min_chars=200, max_chars=2048):
    sentences = split_sentences(text)
    if not sentences:
        return []
    embs = encoder.encode(sentences, normalize_embeddings=True)
    chunks = [[sentences[0]]]
    for i in range(1, len(sentences)):
        sim = float(embs[i] @ embs[i - 1])
        current_len = sum(len(s) for s in chunks[-1])
        if sim < threshold and current_len >= min_chars:
            chunks.append([sentences[i]])
        else:
            chunks[-1].append(sentences[i])

    result = []
    for group in chunks:
        text_group = " ".join(group)
        if len(text_group) > max_chars:
            result.extend(chunk_recursive(text_group, size=max_chars))
        else:
            result.append(text_group)
    return result
```

根据你的领域调整 `threshold`。阈值过高 → 产生碎片。阈值过低 → 出现超大分块。

### 第三步：父文档分块

```python
def chunk_parent_child(text, parent_size=2048, child_size=256):
    parents = chunk_recursive(text, size=parent_size)
    mapping = []
    for p_idx, parent in enumerate(parents):
        children = chunk_recursive(parent, size=child_size)
        for child in children:
            mapping.append({"child": child, "parent_idx": p_idx, "parent": parent})
    return mapping


def retrieve_parent(child_query, mapping, encoder, top_k=3):
    child_embs = encoder.encode([m["child"] for m in mapping], normalize_embeddings=True)
    q_emb = encoder.encode([child_query], normalize_embeddings=True)[0]
    scores = child_embs @ q_emb
    top = np.argsort(-scores)[:top_k]
    seen, parents = set(), []
    for i in top:
        if mapping[i]["parent_idx"] not in seen:
            parents.append(mapping[i]["parent"])
            seen.add(mapping[i]["parent_idx"])
    return parents
```

关键见解：要去重父文档。多个子块映射到同一父文档；返回所有会浪费上下文。

### 第四步：上下文检索（Anthropic 模式）

```python
def contextualize_chunks(document, chunks, llm):
    context_prompts = [
        f"""<document>{document}</document>
Here is the chunk to situate: <chunk>{c}</chunk>
Write 50-100 words placing this chunk in the document's context."""
        for c in chunks
    ]
    contexts = llm.batch(context_prompts)
    return [f"{ctx}\n\n{c}" for ctx, c in zip(contexts, chunks)]
```

对上下文化后的分块建索引。查询时检索收益于额外的上下文信号。

### 第五步：评估

```python
def recall_at_k(queries, corpus_chunks, encoder, k=5):
    chunk_embs = encoder.encode(corpus_chunks, normalize_embeddings=True)
    hits = 0
    for q_text, gold_idxs in queries:
        q_emb = encoder.encode([q_text], normalize_embeddings=True)[0]
        top = np.argsort(-(chunk_embs @ q_emb))[:k]
        if any(i in gold_idxs for i in top):
            hits += 1
    return hits / len(queries)
```

一定要做基准测试。你语料库的“最佳”策略可能与任何博客文章都不相符。

## 陷阱

- **只在事实型查询上评估分块。**多跳查询揭示完全不同的赢家。使用按查询类型分层的评估集。
- **语义分块无最小长度限制。**会产生只有 40 token 的碎片，影响检索。务必设置 `min_tokens`。
- **盲目使用重叠（overlap）。**2026 年研究发现重叠常无效且索引成本翻倍。一定要测量，不要假设。
- **无最小/最大长度约束。**5 token 或 5000 token 的分块都会破坏检索体验。要限制。
- **跨文档分块。**绝不要让分块跨越两个文档。始终先单文档分块，再合并。

## 使用建议

2026 年方案：

| 情形 | 策略 |
|------|------|
| 初次构建，语料未知 | 递归分块，512 token，无重叠 |
| 事实型问答 | 递归分块，256-512 token |
| 分析型 / 多跳问答 | 递归分块，512-1024 token + 父文档分块 |
| 强烈交叉引用（合同、论文） | 后期分块或上下文检索 |
| 会话 / 对话语料 | 逐轮分块 + 说话人元数据 |
| 短文本（推文、评论） | 一文档即一分块 |

从递归 512 token 开始。用 50 条查询评估集测量 recall@5，然后调优。

## 产出示例

保存为 `outputs/skill-chunker.md`：

```markdown
---
name: chunker
description: Pick a chunking strategy, size, and overlap for a given corpus and query distribution.
version: 1.0.0
phase: 5
lesson: 23
tags: [nlp, rag, chunking]
---

Given a corpus (document types, avg length, domain) and query distribution (factoid / analytical / multi-hop), output:

1. Strategy. Recursive / sentence / semantic / parent-document / late / contextual. Reason.
2. Chunk size. Token count. Reason tied to query type.
3. Overlap. Default 0; justify if >0.
4. Min/max enforcement. `min_tokens`, `max_tokens` guards.
5. Evaluation plan. Recall@5 on 50-query stratified eval set (factoid, analytical, multi-hop).

Refuse any chunking strategy without min/max chunk size enforcement. Refuse overlap above 20% without an ablation showing it helps. Flag semantic chunking recommendations without a min-token floor.
```

## 练习

1. **简单。**用 fixed(512, 0)、recursive(512, 0)、recursive(512, 100) 三种方法分块同一 20 页文档。比较分块数和边界质量。
2. **中等。**构建一个包含 5 篇文档、30 条查询的评估集。测量 recursive、semantic 和 parent-document 的 recall@5，哪个最好？是否符合博客结论？
3. **困难。**实现上下文检索。测量相较基线递归的 MRR 提升。报告索引成本（LLM 调用次数）与准确率增益。

## 关键词

| 词汇 | 普通说法 | 实际含义 |
|------|----------|----------|
| Chunk（分块） | 文档片段 | 被嵌入、索引和检索的子文档单位。 |
| Overlap（重叠） | 安全边界 | 相邻分块共享的 N 个 token；2026 年基准多无效用。 |
| Semantic chunking（语义分块） | 智能分块 | 在相邻句子嵌入相似度下降处切分。 |
| Parent-document（父文档） | 双层检索 | 先检索小子块，再返回大父块。 |
| Late chunking（后期分块） | 嵌入后分块 | 先在 token 级嵌入全文，再池化到分块向量。 |
| Contextual retrieval（上下文检索） | Anthropic 技巧 | LLM 生成的上下文摘要加在每块前索引前缀。 |
| Context cliff（上下文断崖） | 2500-token 门槛 | RAG 约 2500 token 上下文处质量骤降（2026 年 1 月观察）。 |

## 延伸阅读

- [Yepes et al. / LangChain — 递归字符分割文档](https://python.langchain.com/docs/how_to/recursive_text_splitter/) — 生产环境默认。
- [Vectara (2024，NAACL 2025) 分块配置分析](https://arxiv.org/abs/2410.13070) — 分块和嵌入模型一样重要。
- [Jina AI — 长上下文嵌入模型中的后期分块（2024）](https://jina.ai/news/late-chunking-in-long-context-embedding-models/) — 后期分块论文。
- [Anthropic — 上下文检索](https://www.anthropic.com/news/contextual-retrieval) — LLM 生成上下文前缀提升 35-50%。
- [NVIDIA 2026 分块大小基准 — Premai 总结](https://blog.premai.io/rag-chunking-strategies-the-2026-benchmark-guide/) — 按查询类型定分块大小。
