# Alignment Research Ecosystem — MATS, Redwood, Apollo, METR

> 五个组织定义了2026年非实验室对齐研究层。MATS（ML Alignment & Theory Scholars，机器学习对齐与理论学者）：自2021年末以来有527+研究人员，发表180+篇论文，引用10K+次，h-index为47；2024年夏季学员团体作为501(c)(3)法人注册，拥有约90名学者和40名导师；80%的2025年前校友从事安全/保障工作，200多人在Anthropic、DeepMind、OpenAI、英国AISI、RAND、Redwood、METR、Apollo等机构。Redwood Research：由Buck Shlegeris创立的应用对齐实验室；提出了AI Control（第10课）；与英国AISI合作制定控制安全案例。Apollo Research：为前沿实验室提供部署前的阴谋评估；发表了In-Context Scheming（第8课）和Towards Safety Cases for AI Scheming。METR（Model Evaluation and Threat Research，模型评估与威胁研究）：基于任务的能力评估，自主任务时间视野研究；《Common Elements of Frontier AI Safety Policies》比较了各实验室框架。Eleos AI Research：模型福利部署前评估（第19课）；进行了Claude Opus 4福利评估。

**类型：** 学习  
**语言：** 无  
**先决条件：** 阶段18 · 01-27（先前的阶段18课程）  
**时长：** ~45分钟

## 学习目标

- 识别非实验室对齐研究生态系统中的五个组织及其核心成果。
- 描述MATS的规模（学者、论文、h-index）及其作为人才输送管道的角色。
- 描述Redwood的AI Control议程及其与英国AISI的合作。
- 描述METR的基于任务的评估方法。

## 问题

前沿实验室（第18课）内部进行安全评估并发布部分结果。实验室外的生态系统是评估验证的场所，是首次发现新失败模式以及人才培养的地点。理解该生态系统有助于判断哪些研究成果被谁信任。

## 概念

### MATS（ML Alignment & Theory Scholars）

始于2021年末。研究指导项目；学者们与高级研究员合作10-12周，专注于特定对齐问题。

规模（2026年）：  
- 自成立以来有527+研究人员。  
- 发表180+篇论文。  
- 10000+次引用。  
- h-index为47。  
- 2024年夏季：90名学者 + 40名导师；作为501(c)(3)法人注册。

职业发展成果：约80%的2025年前校友从事安全/保障工作。200多人在Anthropic、DeepMind、OpenAI、英国AISI、RAND、Redwood、METR、Apollo工作。

### Redwood Research

应用对齐实验室。由Buck Shlegeris创立。引入AI Control议程（第10课）。与英国AISI合作制定控制安全案例。为DeepMind和Anthropic提供评估设计咨询。

代表论文：Greenblatt, Shlegeris等人，《AI Control》（arXiv:2312.06942，ICML 2024）；Alignment Faking（Greenblatt, Denison, Wright等人，arXiv:2412.14093，与Anthropic联合）。

风格：具体威胁模型，最坏情况的对抗者，可进行压力测试的具体协议。

### Apollo Research

为前沿实验室做部署前的阴谋评估。发表了In-Context Scheming（第8课，arXiv:2412.04984）。是2025年OpenAI反阴谋训练合作伙伴。发布《Towards Safety Cases for AI Scheming》（2024年）。

风格：智能体环境下的评估，关注可能出现欺骗行为；采用三柱分解法（失调、目标导向性、情境意识）。

### METR（Model Evaluation and Threat Research）

基于任务的能力评估。自主任务完成时间视角研究。《Common Elements of Frontier AI Safety Policies》（metr.org/common-elements，2025年）比较了实验室框架。

是AI Scheming安全案例草图的Apollo联合作者。

风格：长时间跨度的任务评估，经验能力测量，框架综合。

### Eleos AI Research

模型福利部署前评估。进行过《Claude Opus 4》福利评估，在系统卡第5.3节中有记录。为第19课福利相关声明提供外部方法论验证。

### 流程

MATS培养研究人员。毕业生进入Anthropic、DeepMind、OpenAI（实验室安全团队）或Redwood、Apollo、METR、Eleos（外部评估）。外部评估者与实验室及英国AISI / CAISI合作。出版物反哺MATS，供下一届学员使用。

### 该层的重要性

单一来源的评估是不可靠的：实验室自己评估自己的模型存在结构性利益冲突。外部评估者能发现并验证实验室可能隐瞒的失败模式。2024年《Sleeper Agents》论文（第7课）由Anthropic + Redwood共同完成；Alignment Faking由Anthropic + Redwood合作；In-Context Scheming出自Apollo；Anti-Scheming出自Apollo + OpenAI。多组织结构是质量控制保证。

### 在阶段18中的位置

第7-11课引用了Redwood和Apollo的工作；第18课提及METR的框架比较；第19课提及Eleos。第28课是该生态系统的明确组织地图，是阶段其余部分的基础。

## 使用方法

无代码要求。可阅读METR的《Common Elements of Frontier AI Safety Policies》，作为外部综合如何为实验室内部政策工作增值的示例。

## 交付成果

本课产生 `outputs/skill-ecosystem-map.md`。给定对齐主张或评估，识别所属组织、发表场所、方法论风格，并与已知合作组织交叉核对。

## 练习

1. 从第7-15课中选择一篇论文，识别涉及的组织。将作者与MATS校友及当前生态系统成员对照核查。
2. 阅读METR的《Common Elements of Frontier AI Safety Policies》。识别他们强调的跨实验室三大共识和两个最大分歧点。
3. MATS的职业成果显示约80%从事安全/保障工作。论证这种选择性压力是适应性的（培养领域）还是偏见性的（排除异端观点）。
4. Redwood和Apollo都做控制/阴谋工作，但风格不同。选择一个失败模式，描述两者如何调查它。
5. Eleos AI是唯一专注模型福利的组织。设计一个假设的第二个组织，聚焦不同的福利相关问题（认知自由、机器人具身等），并阐述其方法论。

## 关键词

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| MATS | “导师项目” | ML Alignment & Theory Scholars；自2021年起527+研究人员 |
| Redwood Research | “控制实验室” | 应用对齐；AI Control作者；英国AISI合作伙伴 |
| Apollo Research | “阴谋评估” | 面向前沿实验室的部署前阴谋评估 |
| METR | “任务-时间视野评估” | 基于任务的能力评估；框架综合 |
| Eleos AI | “福利实验室” | 模型福利部署前评估 |
| 人才管道 | “MATS -> 实验室” | MATS毕业生流入Anthropic、DM、OpenAI、Redwood、Apollo、METR |
| 外部评估 | “非实验室检查” | 非模型原生产者进行的评估；增加可信度 |

## 进一步阅读

- [MATS (ML Alignment & Theory Scholars)](https://www.matsprogram.org/) — 导师项目  
- [Redwood Research](https://www.redwoodresearch.org/) — AI Control 论文  
- [Apollo Research](https://www.apolloresearch.ai/) — 阴谋评估  
- [METR — Common Elements of Frontier AI Safety Policies](https://metr.org/blog/2025-03-26-common-elements-of-frontier-ai-safety-policies/) — 框架比较  
- [Eleos AI Research](https://www.eleosai.org/research) — 模型福利方法论
