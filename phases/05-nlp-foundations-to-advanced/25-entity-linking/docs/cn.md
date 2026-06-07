# 实体链接与消歧

> 命名实体识别（NER）找到了“Paris”。实体链接决定：是法国巴黎？巴黎希尔顿？美国德州巴黎？还是特洛伊王子Paris？如果不链接，知识图谱就会保持模糊不清。

**类型：** 构建  
**语言：** Python  
**先决条件：** 第5阶段 · 06（NER），第5阶段 · 24（共指消解）  
**时间：** ~60分钟

## 问题描述

一句话是：“Jordan beat the press.” 你的NER标注“Jordan”为PERSON，很好。但是*哪个* Jordan？

- Michael Jordan（篮球运动员）？  
- Michael B. Jordan（演员）？  
- Michael I. Jordan（伯克利机器学习教授 —— 在机器学习论文中这确实会混淆）？  
- Jordan（国家约旦）？  
- Jordan（希伯来语名字）？

实体链接（Entity linking，EL）将每个提及解析到知识库中的唯一实体：Wikidata，Wikipedia，DBpedia，或者你自己的领域知识库。包含两个子任务：

1. **候选生成。** 给定“Jordan”，哪些知识库条目是合理的候选？  
2. **消歧。** 给定上下文，哪个候选才是正确的？

两个步骤都可以通过学习实现，也都有基准。整体管道已稳定运行十年——变化的是消歧器的质量。

## 概念

![实体链接管道：提及 → 候选 → 消歧实体](../assets/entity-linking.svg)

**候选生成。** 给定提及表面形式（“Jordan”），在别名索引中查找候选。维基百科的别名字典覆盖大部分命名实体：“JFK” → 约翰·F·肯尼迪，杰奎琳·肯尼迪，JFK机场，电影《JFK》。典型索引每个提及返回10-30个候选。

**消歧：三种方法。**

1. **先验+上下文（Milne & Witten, 2008）。** `P(entity | mention) × context-similarity(entity, text)`。效果好、快速，无需训练。  
2. **基于嵌入（ESS / REL / Blink）。** 对提及+上下文编码。对每个候选实体描述编码。选择最大余弦相似度。2020-2024年的默认选择。  
3. **生成式（GENRE, 2021；基于大语言模型，2023+）。** 逐字解码实体的规范名称。约束在有效实体名称的Trie中，确保输出为有效知识库ID。

**端到端与流水线。** 现代模型（ELQ, BLINK, ExtEnD, GENRE）一遍完成NER、候选生成与消歧。流水线系统在生产环境仍占主导，因为可替换组件。

### 两个评价指标

- **提及召回（候选生成）。** 正确知识库条目出现在候选列表中的比例。整个管道的下限。  
- **消歧准确率/F1。** 在正确候选给定的情况下，top-1正确的比例。

两者均需报告。消歧准确率99%且候选召回80%，总体管道效果即80%。

## 构建它

### 第1步：从维基百科重定向构建别名索引

```python
alias_to_entities = {
    "jordan": ["Q41421 (Michael Jordan)", "Q810 (Jordan, country)", "Q254110 (Michael B. Jordan)"],
    "paris":  ["Q90 (Paris, France)", "Q663094 (Paris, Texas)", "Q55411 (Paris Hilton)"],
    "apple":  ["Q312 (Apple Inc.)", "Q89 (apple, fruit)"],
}
```

维基百科别名数据：约1800万（别名，实体）对。从Wikidata dump下载。存为倒排索引。

### 第2步：基于上下文的消歧

```python
def disambiguate(mention, context, alias_index, entity_desc):
    candidates = alias_index.get(mention.lower(), [])
    if not candidates:
        return None, 0.0
    context_words = set(tokenize(context))
    best, best_score = None, -1
    for entity_id in candidates:
        desc_words = set(tokenize(entity_desc[entity_id]))
        union = len(context_words | desc_words)
        score = len(context_words & desc_words) / union if union else 0.0
        if score > best_score:
            best, best_score = entity_id, score
    return best, best_score
```

Jaccard重叠是玩具方法。用基于嵌入的余弦相似度替换（参见`code/main.py`第2步，Transformer版本）。

### 第3步：基于嵌入（BLINK风格）

```python
from sentence_transformers import SentenceTransformer
encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed_mention(text, mention_span):
    start, end = mention_span
    marked = f"{text[:start]} [MENTION] {text[start:end]} [/MENTION] {text[end:]}"
    return encoder.encode([marked], normalize_embeddings=True)[0]

def embed_entity(entity_id, description):
    return encoder.encode([f"{entity_id}: {description}"], normalize_embeddings=True)[0]
```

索引时，每个知识库实体编码一次。查询时，提及+上下文编码一次，与候选池点积，选最大。

### 第4步：生成式实体链接（概念）

GENRE逐字符解码实体的维基百科标题。约束解码（参考第20课）确保只输出有效标题。紧密结合知识库支持的Trie。现代版本是REL-GEN和基于大语言模型的结构化输出EL。

```python
prompt = f"""Text: {text}
Mention: {mention}
List the best Wikipedia title for this mention.
Respond with JSON: {{"title": "..."}}"""
```

结合白名单（轮廓中提到的`choice`），这是2026年最简单可部署的EL管道。

### 第5步：在AIDA-CoNLL上评估

AIDA-CoNLL是标准的EL基准：1,393篇Reuters文章，34k提及，维基百科实体。报告知识库内准确率(`P@1`)和知识库外（NIL）的检测率。

## 陷阱

- **NIL处理。** 部分提及不在知识库中（新兴实体、鲜为人知的人物）。系统必须预测NIL，而不是胡乱猜错实体。单独评测。  
- **提及边界错误。** 上游NER漏标部分跨度（“Bank of America”只标为“Bank”）。EL召回下降。  
- **流行度偏差。** 训练系统过度预测高频实体。“Michael I. Jordan”的一篇ML论文中提及，经常链接成篮球的Jordan。  
- **跨语言EL。** 中文文本提及映射到英文维基百科实体。需要多语言编码器或翻译步骤。  
- **知识库过时。** 新公司、事件、人物未包含在去年维基百科dump中。生产管道需刷新循环。

## 使用它

2026年技术栈：

| 情况 | 选用 |
|-----------|------|
| 通用英语 + 维基百科 | BLINK 或 REL |
| 跨语言，知识库为维基百科 | mGENRE |
| 适合LLM，少量提及/天 | 用候选列表+约束JSON提示Claude/GPT-4 |
| 特定领域知识库（医疗、法律） | 定制BERT结合知识库检索+领域AIDA风格数据微调 |
| 极低延迟 | 仅先验精确匹配（Milne-Witten基线） |
| 研究SOTA | GENRE / ExtEnD / 生成式LLM-EL |

2026年生产流程模式：NER → 共指消解 → 每个提及EL → 合并簇内实体为单一规范实体。输出：文档中每个实体一个知识库ID，而非每个提及一个。

## 发布它

保存为 `outputs/skill-entity-linker.md`：

```markdown
---
name: entity-linker
description: 设计一个实体链接管道 — 知识库，候选生成器，消歧器，评估。
version: 1.0.0
phase: 5
lesson: 25
tags: [nlp, entity-linking, knowledge-graph]
---

给定用例（领域知识库、语言、规模、延迟预算），输出：

1. 知识库。Wikidata / Wikipedia / 自定义知识库。版本日期。刷新频率。  
2. 候选生成器。别名索引、嵌入或混合型。目标提及召回@K。  
3. 消歧器。先验+上下文、基于嵌入、生成式或LLM提示式。  
4. NIL策略。基于最高分阈值、分类器或显式NIL候选。  
5. 评估。提及召回@30，top-1准确率，持出集上的NIL检测F1。

拒绝任何无提及召回基线的EL管道（没有候选生成确认正确实体，无法评估消歧器）。拒绝任何未对输出进行有效KB ID约束的LLM提示EL。检测到流行度偏差影响小众实体（如重名）且无领域微调的系统应标记。
```

## 练习

1. **简单。** 在`code/main.py`实现先验+上下文消歧，针对10个模糊提及（Paris、Jordan、Apple）手工标注正确实体。测量准确率。  
2. **中等。** 用句子变换器编码50个模糊提及。编码每个候选实体描述。比较基于嵌入的消歧与Jaccard上下文重叠。  
3. **困难。** 构建一个包含1000实体的领域知识库（例如公司员工和产品）。端到端实现NER + EL。对100条持出句子测量精度和召回率。

## 关键词

| 术语 | 常用说法 | 实际含义 |
|------|---------|-----------|
| 实体链接（EL） | 链接到维基百科 | 将提及对应到唯一知识库实体。 |
| 候选生成 | 可能是谁？ | 返回一份合理候选知识库实体。 |
| 消歧 | 选正确的那个 | 利用上下文给候选评分，选最高。 |
| 别名索引 | 查找表 | 从表面形式映射到候选实体。 |
| NIL | 不在知识库中 | 明确预测无匹配知识库条目。 |
| 知识库（KB） | 知识库 | Wikidata、Wikipedia、DBpedia或你的领域知识库。 |
| AIDA-CoNLL | 基准 | 1,393篇Reuters文章带有金标准实体链接。 |

## 深入阅读

- [Milne, Witten (2008). Learning to Link with Wikipedia](https://www.cs.waikato.ac.nz/~ihw/papers/08-DM-IHW-LearningToLinkWithWikipedia.pdf) — 基础的先验+上下文方法。  
- [Wu et al. (2020). Zero-shot Entity Linking with Dense Entity Retrieval (BLINK)](https://arxiv.org/abs/1911.03814) — 基于嵌入的主力。  
- [De Cao et al. (2021). Autoregressive Entity Retrieval (GENRE)](https://arxiv.org/abs/2010.00904) — 生成式EL与约束解码。  
- [Hoffart et al. (2011). Robust Disambiguation of Named Entities in Text (AIDA)](https://www.aclweb.org/anthology/D11-1072.pdf) — 基准论文。  
- [REL: An Entity Linker Standing on the Shoulders of Giants (2020)](https://arxiv.org/abs/2006.01969) — 开源生产堆栈。
