# 指代消解（Coreference Resolution）

> “她打电话给他。他没有接。医生正在吃午饭。” 三个提及涉及两个人，但没有名字。指代消解就是搞清楚谁是谁。

**类型：** 学习  
**语言：** Python  
**先决条件：** 阶段5 · 06（命名实体识别 NER）、阶段5 · 07（词性标注与句法分析 POS & Parsing）  
**时长：** 约60分钟

## 问题

从一篇约300字的文章中抽取所有涉及 Apple Inc. 的提及。当文章直接说“Apple”时很简单。当文章说“the company（这家公司）”、“they（他们）”、“Cupertino's technology giant（库比蒂诺的科技巨头）”或“Jobs's firm（乔布斯的公司）”时就很难了。如果不将这些提及解析为同一个实体，你的NER流水线将漏掉60-80%的提及。

指代消解将所有指向同一现实世界实体的表达连接成一个簇。它是表层NLP（NER、句法分析）与下游语义（信息抽取 IE、问答 QA、摘要、知识图谱 KG）之间的桥梁。

2026年为什么它很重要：

- 摘要： “The CEO announced...” 与 “Tim Cook announced...” —— 摘要应该点名这位CEO。
- 问答：“她打电话给谁？”需要解析“她”指代谁。
- 信息抽取：知识图谱中，“PER1 创立 Apple” 与 “Jobs 创立 Apple” 分开处理是错误的。
- 多文档信息抽取：将关于同一事件的多篇文章中的提及合并即为跨文档指代消解。

## 概念

![指代聚类：提及 → 实体](../assets/coref.svg)

**任务。** 输入：一个文档。输出：对提及（短语）进行聚类，每个聚类指代一个实体。

**提及类型。**

- **命名实体（Named entity）。** “Tim Cook”
- **指定名词短语（Nominal）。** “the CEO（CEO）”，“the company（公司）”
- **代词（Pronominal）。** “he（他）”，“she（她）”，“they（他们）”，“it（它）”
- **同位语（Appositive）。** “Tim Cook, Apple's CEO,”

**架构。**

1. **基于规则（Rule-based，Hobbs，1978）。** 使用句法树和语法规则进行代词消解。良好的基线模型，代词消解特别难超越它。
2. **提及对分类器（Mention-pair classifier）。** 对每对提及（m_i, m_j）预测是否指代同一个实体。通过传递闭包实现聚类。2016年前的标准方法。
3. **提及排名（Mention-ranking）。** 对每个提及，给候选先行词排序（包括“无先行词”），选排名第一的。
4. **基于跨度的端到端模型（Span-based end-to-end，Lee 等，2017）。** Transformer 编码器。枚举所有候选跨度（长度有上限），预测提及分数，预测每个跨度的先行词概率。贪婪聚类。现代默认方法。
5. **生成式模型（Generative，2024+）。** 提示大语言模型（LLM）：“列出文本中所有代词及其先行词。”对简单情况表现良好，但对长文本和罕见指代表现较差。

**评估指标。** 标准有五种（MUC、B³、CEAF、BLANC、LEA），单一指标难以完全衡量聚类质量。通常报告前三个指标的平均作为 CoNLL F1。2026年 CoNLL-2012 最高水平约为83 F1。

**已知难点。**

- 明确定义的描述指向数页前引入的实体。
- 桥接指代（Bridging anaphora）：“the wheels（轮子）”→ 先前提到的车。
- 零指代（Zero anaphora），如中文和日文中存在。
- 逆指代（Cataphora，代词在指称词之前）：“When **she** walked in, Mary smiled.”（当她走进来时，玛丽笑了。）

## 构建方法

### 第一步：预训练神经指代消解（AllenNLP / spaCy-experimental）

```python
import spacy
nlp = spacy.load("en_coreference_web_trf")   # 实验性模型
doc = nlp("Apple announced new products. The company said they would ship soon.")
for cluster in doc._.coref_clusters:
    print(cluster, "->", [m.text for m in cluster])
```

在较长文档中会类似输出：

- 簇1：[Apple, The company, they]
- 簇2：[new products]

### 第二步：基于规则的代词解析器（教学用途）

请参见 `code/main.py` 中的标准库纯Python实现：

1. 提取提及：命名实体（首字母大写的短语）、代词（字典查找）、定指描述（"the X"）。
2. 对每个代词，查看之前的 K 个提及，按以下标准评分：
   - 性别/数量一致性（启发式）
   - 新近性（距离越近越优）
   - 句法角色（主语优先）
3. 连接得分最高的先行词。

性能不及神经模型，但展示了搜索空间和端到端模型要做的决策。

### 第三步：使用LLM进行指代消解

```python
prompt = f"""Text: {text}

List every pronoun and noun phrase that refers to a person or company.
Cluster them by what they refer to. Output JSON:
[{{"entity": "Apple", "mentions": ["Apple", "the company", "it"]}}, ...]
"""
```

注意两种失败模式。第一，LLM过度合并（如“him”和“her”实际上指两不同人）。第二，LLM在长文档中会默默遗落提及。务必用跨度偏移校验结果。

### 第四步：评估

标准 Conll-2012 脚本计算 MUC、B³、CEAF-φ4 并报告平均值。内部评估建议从标注集上的跨度级别精确率和召回率开始，再加入提及链接的 F1。

## 注意事项

- **单例爆炸（Singleton explosion）。** 一些系统把每个提及当作独立簇。B³指标较宽容，MUC严厉惩罚。务必检查所有三种指标。
- **长文本中的代词。** 数据超过2000个token，性能大约下降15 F1。需合理分块处理。
- **性别假设。** 硬编码的性别规则在处理非二元性别指代、组织、动物时失效。更推荐学习模型或中性评分。
- **长文本中LLM漂移。** 一次API调用无法可靠跨50+段落聚类。推荐滑动窗口处理后合并。

## 使用建议

2026年堆栈：

| 情况 | 选择 |
|-----------|------|
| 英文单文档 | `en_coreference_web_trf`（spaCy 实验模型）或 AllenNLP 神经指代模型 |
| 多语种 | SpanBERT / XLM-R，训练于 OntoNotes 或多语言 CoNLL |
| 跨文档事件指代 | 专用端到端模型（2025–26年SOTA） |
| 快速LLM基线 | GPT-4o / Claude，用结构化输出的指代消解提示 |
| 生产对话系统 | 规则基fallback+神经主模型+关键槽人工复核 |

2026年典型集成模式：先运行NER，再执行指代消解，将指代簇合并进NER实体。后续任务看到的是每簇对应一个实体，而非每个提及一个实体。

## 部署方案

保存为 `outputs/skill-coref-picker.md`：

```markdown
---
name: coref-picker
description: 选择指代消解方法、评估计划和集成策略。
version: 1.0.0
phase: 5
lesson: 24
tags: [nlp, coref, information-extraction]
---

给定用例（单文档/多文档、领域、语言），输出：

1. 方法。基于规则 / 神经跨度 / LLM 提示 / 混合。简述理由。
2. 模型。神经模型给出明确检查点。
3. 集成。操作顺序：分词 → NER → coref → 下游任务。
4. 评估。CoNLL F1（MUC + B³ + CEAF-φ4平均）于保留集 + 20文档的人工簇复核。

不接受单纯LLM处理超过2000token文档且无滑动窗口合并方案。不接受未报告提及级别精确率召回率的指代流水线。针对在人口统计多样文本中部署的性别启发式系统提出警告。
```

## 练习

1. **简单。** 在5段手工编写文本上运行`code/main.py`中的基于规则解析器。测量提及连接准确率与标注集对比。
2. **中等。** 使用预训练的神经指代模型处理新闻文章。将自动聚类与人工标注对比。失败出现在什么地方？
3. **困难。** 构建一个增强的指代消解NER流水线：先NER再通过核心簇合并。对100篇文章测量实体覆盖率相较仅NER的提升。

## 关键词

| 术语 | 通用说法 | 实际含义 |
|------|----------|----------|
| Mention（提及） | 一个指称 | 文本中指代实体的短语（姓名、代词、名词短语）。 |
| Antecedent（先行词） | “它”指什么 | 后一个提及所指向的早先提及。 |
| Cluster（簇） | 实体的所有提及 | 所有指向同一现实实体的提及集合。 |
| Anaphora（回指） | 向前指 | 后续提及指向之前的实体（“he” → “John”）。 |
| Cataphora（逆指代） | 向后指 | 先出现的提及指向后出现的实体（“当他到达时，John…”）。 |
| Bridging（桥接） | 隐式指代 | “I bought a car. The wheels were bad.”（“轮子”隐指该车。） |
| CoNLL F1 | 排行榜指标 | MUC、B³、CEAF-φ4三个F1的平均值。 |

## 延伸阅读

- [Jurafsky & Martin, SLP3 第26章——指代消解与实体链接](https://web.stanford.edu/~jurafsky/slp3/26.pdf) — 标准教材章节。
- [Lee et al. (2017). End-to-end Neural Coreference Resolution](https://arxiv.org/abs/1707.07045) — 基于跨度的端到端方法。
- [Joshi et al. (2020). SpanBERT](https://arxiv.org/abs/1907.10529) — 提升指代消解性能的预训练方法。
- [Pradhan et al. (2012). CoNLL-2012 共享任务](https://aclanthology.org/W12-4501/) — 基准数据集。
- [Hobbs (1978). Resolving Pronoun References](https://www.sciencedirect.com/science/article/pii/0024384178900064) — 经典规则基代词消解参考文献。
