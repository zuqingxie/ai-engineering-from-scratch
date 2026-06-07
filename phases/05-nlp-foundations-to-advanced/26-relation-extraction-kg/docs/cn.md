# 关系抽取与知识图谱构建

> 命名实体识别（NER）发现实体。实体链接锚定它们。关系抽取（RE）找出它们之间的边。知识图谱是节点、边及其来源的总和。

**类型：** 构建  
**语言：** Python  
**先决条件：** 第5阶段 · 06（NER），第5阶段 · 25（实体链接）  
**时间：** ~60分钟

## 问题

分析师看到：“Tim Cook 于2011年成为Apple的CEO。” 四条事实：

- `(Tim Cook, role, CEO)`
- `(Tim Cook, employer, Apple)`
- `(Tim Cook, start_date, 2011)`
- `(Apple, type, Organization)`

关系抽取（Relation Extraction，RE）将自由文本转化为结构化三元组 `(subject, relation, object)`。在整个语料库中聚合便形成知识图谱。聚合并查询后，你将获得用于RAG、分析或合规审计的推理基础。

2026年的问题：大型语言模型（LLMs）过于热情地提取关系，它们会幻觉出文本没有支持的三元组。没有来源信息，你无法辨别真实三元组和看似合理的虚构。2026年的答案是AEVS风格的锚定-验证流水线。

## 概念

![文本 → 三元组 → 知识图谱](../assets/relation-extraction.svg)

**三元组形式。** `(subject_entity, relation_type, object_entity)`。关系来自封闭本体（Wikidata属性、FIBO、UMLS）或开放集（OpenIE风格，任意关系）。

**三种抽取方法。**

1. **基于规则/模式。** Hearst模式：“X such as Y” → `(Y, isA, X)`。加上人工写的正则表达式。脆弱、精准、可解释。  
2. **监督分类器。** 给定句子中的两个实体提及，从固定集合中预测关系。训练数据来自TACRED、ACE、KBP。2015-2022年的标准方法。  
3. **生成型LLM。** 通过提示让模型输出三元组。开箱即用。需要来源信息，否则幻觉出貌似合理的无用数据。

**AEVS（Anchor-Extraction-Verification-Supplement，2026）。** 当前缓解幻觉的框架：

- **锚定（Anchor）。** 标记每个实体跨度和关系短语的精确位置。  
- **抽取（Extract）。** 生成链接于锚点的三元组。  
- **验证（Verify）。** 将每个三元组元素与源文本匹配；拒绝任何无支持的内容。  
- **补充（Supplement）。** 覆盖检查确保没有锚定的跨度被遗漏。

幻觉大幅下降。需要更多计算资源，但可审计。

**开放与封闭的权衡。**

- **封闭本体。** 固定属性列表（如Wikidata的11000+属性）。可预测、可查询。难以发明新关系。  
- **开放IE。** 任意动词短语皆可为关系。召回高，准确率低。查询混乱。

生产知识图谱通常混合：开放IE用于发现，然后将关系规范化为封闭本体属性后合并到主图中。

## 构建流程

### 步骤1：基于模式的抽取

```python
PATTERNS = [
    (r"(?P<s>[A-Z]\w+) (?:is|was) (?:a|an|the) (?P<o>[A-Z]?\w+)", "isA"),
    (r"(?P<s>[A-Z]\w+) (?:is|was) born in (?P<o>\w+)", "bornIn"),
    (r"(?P<s>[A-Z]\w+) works? (?:at|for) (?P<o>[A-Z]\w+)", "worksAt"),
    (r"(?P<s>[A-Z]\w+) founded (?P<o>[A-Z]\w+)", "founded"),
]
```

完整的玩具抽取器见 `code/main.py`。Hearst模式仍在领域特定流水线中使用，因为易于调试。

### 步骤2：监督关系分类

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tok = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
model = AutoModelForSequenceClassification.from_pretrained("Babelscape/rebel-large")

text = "Tim Cook was born in Alabama. He later became CEO of Apple."
encoded = tok(text, return_tensors="pt", truncation=True)
output = model.generate(**encoded, max_length=200)
triples = tok.batch_decode(output, skip_special_tokens=False)
```

REBEL是一种序列到序列的关系抽取工具：输入文本，输出三元组，已映射到Wikidata属性ID。基于远程监督数据微调。标准的开放权重基线。

### 步骤3：基于LLM提示的抽取及锚定

```python
prompt = f"""Extract (subject, relation, object) triples from the text.
For each triple, include the exact character span in the source text.

Text: {text}

Output JSON:
[{{"subject": {{"text": "...", "span": [start, end]}},
   "relation": "...",
   "object": {{"text": "...", "span": [start, end]}}}}, ...]

Only include triples fully supported by the text. No inference beyond what is stated.
"""
```

逐一核对返回的跨度与源文本是否一致。若 `text[start:end] != triple_entity` 则拒绝该三元组。这是AEVS“验证”步骤的简化形式。

### 步骤4：规范化到封闭本体

```python
RELATION_MAP = {
    "is the CEO of": "P169",       # "chief executive officer"
    "was born in":   "P19",         # "place of birth"
    "founded":        "P112",       # "founded by"（主客体倒置）
    "works at":       "P108",       # "employer"
}


def canonicalize(relation):
    rel_low = relation.lower().strip()
    if rel_low in RELATION_MAP:
        return RELATION_MAP[rel_low]
    return None   # 舍弃未映射的开放关系或走人工审核流程
```

规范化通常占工程量的60-80%。请预留预算。

### 步骤5：构建小型图谱并查询

```python
triples = extract(text)
graph = {}
for s, r, o in triples:
    graph.setdefault(s, []).append((r, o))


def neighbors(node, relation=None):
    return [(r, o) for r, o in graph.get(node, []) if relation is None or r == relation]


print(neighbors("Tim Cook", relation="P108"))    # -> [(P108, Apple)]
```

这是一切基于KG的RAG系统的核心。大规模可使用RDF三元组库（Blazegraph、Virtuoso）、属性图（Neo4j）或向量增强图库扩展。

## 陷阱

- **RE之前需做共指消解。** “He founded Apple”——RE需要知道“he”指谁。先运行共指消解（第24课）。  
- **实体规范化。** “Apple Inc”和“Apple”需映射至同一节点。实体链接先行（第25课）。  
- **幻觉三元组。** LLM生成文本不支持的三元组。强制做跨度验证。  
- **关系规范化漂移。** 开放IE里的关系不一致（如“was born in”、“came from”、“is a native of”）。需折叠至规范id，否则图谱不可查询。  
- **时态错误。** “Tim Cook is CEO of Apple”——现在为真，2005年为假。许多关系有时间限制。使用限定符（Wikidata中的`P580`起始时间，`P582`结束时间）。  
- **领域不匹配。** REBEL在维基百科上训练。法律、医疗、科学文本常需领域微调模型。

## 使用指南

2026年堆栈：

| 场景             | 选择                                    |
|------------------|---------------------------------------|
| 通用领域高速生产 | REBEL或LlamaPred + Wikidata规范化       |
| 领域特定（生物医药、法律） | SciREX风格领域微调 + 自定义本体         |
| LLM提示，审计输出 | AEVS流水线：锚定 → 抽取 → 验证 → 补充     |
| 高频新闻信息抽取   | 基于模式 + 监督方法混合                   |
| 从零构建知识图谱   | 开放IE + 人工规范化流程                     |
| 时间维度知识图谱   | 抽取带有限定符（开始/结束时间、具体时点）     |

集成模式：NER → 共指消解 → 实体链接 → 关系抽取 → 本体映射 → 图谱载入。每一步都是潜在质量关口。

## 发布示例

保存为 `outputs/skill-re-designer.md`：

```markdown
---
name: re-designer
description: 设计一个具有来源追踪与规范化的关系抽取流水线。
version: 1.0.0
phase: 5
lesson: 26
tags: [nlp, relation-extraction, knowledge-graph]
---

给定语料库（领域、语言、规模）和下游用途（KG-RAG、分析、合规），输出：

1. 抽取器。基于模式/监督/LLM/AEVS混合。基于精准率与召回目标权衡设计。
2. 本体。封闭属性列表（Wikidata/领域）或开放IE加规范化环节。
3. 来源。每个三元组携带源文本字符跨度 + 文档ID。审计必需。
4. 合并策略。规范实体ID + 关系ID + 时间限定符；去重规则。
5. 评估。在200个人工标注的三元组上测精准率/召回率，检验LLM抽取样本的幻觉率。

拒绝任何无跨度验证（无来源追踪）的LLM关系抽取流水线。拒绝未经规范化直接流入生产知识图谱的开放IE输出。标记无时间限定符的时态关系（雇主、配偶、职位）流水线。
```

## 练习

1. **简单。** 使用 `code/main.py` 中的模式抽取器对5句新闻文本抽取关系。手动检验精准率。  
2. **中等。** 对相同句子使用REBEL（或小型LLM）抽取关系。比较三元组。哪个抽取器精准率更高？召回率更高？  
3. **困难。** 构建AEVS流水线：用LLM抽取 + 验证跨度与源文本匹配。在50句维基百科风格句子上测量验证前后的幻觉率。

## 关键术语

| 术语               | 常见说法               | 实际含义                                           |
|--------------------|------------------------|---------------------------------------------------|
| 三元组（Triple）     | 主-关系-宾             | `(s, r, o)`元组，是知识图谱的原子单元。           |
| 开放IE（Open IE）    | 抽取任何关系           | 开放词汇的关系短语；召回高、精准率低。              |
| 封闭本体（Closed ontology） | 固定模式               | 有界的关系类型集合（如Wikidata、UMLS、FIBO）。     |
| 规范化（Canonicalization） | 统一标准化             | 将表面名称/关系映射到规范ID。                       |
| AEVS               | 有据可查的抽取         | 锚定-抽取-验证-补充流水线（2026年设计）。          |
| 来源（Provenance）   | 真相来源链接           | 每三元组携带文档ID及字符跨度，保证可审计。           |
| 远程监督（Distant supervision） | 便宜的标注             | 用现有知识图谱对齐文本以生成训练数据。               |

## 延伸阅读

- [Mintz et al. (2009). Distant supervision for relation extraction without labeled data](https://www.aclweb.org/anthology/P09-1113.pdf) — 远程监督关系抽取论文。  
- [Huguet Cabot, Navigli (2021). REBEL: Relation Extraction By End-to-end Language generation](https://aclanthology.org/2021.findings-emnlp.204.pdf) — 序列到序列关系抽取主力。  
- [Wadden et al. (2019). Entity, Relation, and Event Extraction with Contextualized Span Representations (DyGIE++)](https://arxiv.org/abs/1909.03546) — 联合信息抽取。  
- [AEVS — Anchor-Extraction-Verification-Supplement framework](https://www.mdpi.com/2073-431X/15/3/178) — 2026年幻觉缓解设计。  
- [Wikidata SPARQL 教程](https://www.wikidata.org/wiki/Wikidata:SPARQL_tutorial) — 规范图谱查询。
