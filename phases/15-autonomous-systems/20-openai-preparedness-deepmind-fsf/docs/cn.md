# OpenAI 准备框架和 DeepMind 前沿安全框架

> OpenAI 准备框架 v2（2025 年 4 月）引入了研究类别（Research Categories）——长程自主（Long-range Autonomy）、故意低估（Sandbagging）、自主复刻与适应（Autonomous Replication and Adaptation）、破坏安全措施（Undermining Safeguards）——与追踪类别（Tracked Categories）区别开。追踪类别会触发能力报告（Capabilities Reports）及安全措施报告（Safeguards Reports），由安全咨询组（Safety Advisory Group）审核。DeepMind 的 FSF v3（2025 年 9 月，2026 年 4 月 17 日新增追踪能力等级）将自主权整合到机器学习研发（ML R&D）与网络安全（Cyber）领域（ML R&D 自主等级 1 表示以相比人类+AI 工具更有竞争力的成本完全自动化 AI 研发流程）。FSF v3 明确通过自动化监测针对工具性推理（instrumental reasoning）滥用应对欺骗性对齐（deceptive alignment）。事实上，准备框架 v2 中的研究类别（包括长程自主）并不自动触发缓解措施，政策语言用的是“潜在（potential）”。DeepMind 自身表示，如果工具性推理强化，自动监控“长远来看将不足够”。

**类型：** 学习  
**语言：** Python（标准库，三框架决策表差异工具）  
**先决条件：** 第 15·19 课（Anthropic RSP）  
**时间：** 约 45 分钟

## 问题

第 19 课仔细研读了 Anthropic 的扩展政策。本课通过阅读 OpenAI 和 DeepMind 策略，补全了全貌。三份文档是同类产物，均在探讨一个问题——前沿实验室何时应暂停或限制模型——它们在一小部分类别上达成共识，但在具体细节上有差异。

共识：三者都将长程自主视为值得追踪的能力类别。三者皆认可欺骗性行为（对齐伪装、故意低估）是一种特定风险。均有内部审查机构。差异：OpenAI 将类别分为“追踪”（必须缓解）和“研究”（不自动触发）。DeepMind 将自主权分摊到两个领域，而非单独命名。实验室依其命名“追踪”与“研究”、或 “关键” 与“中等”、或“一级”与“二级”，对能力落在哪个类别的实际操作影响不同。

将它们合并阅读是有益的。同一能力在 Anthropic 是“必须缓解”，在 OpenAI 是“监控但未触发”，在 DeepMind 是“特定领域追踪”。政策立场至关重要。

## 概念

### OpenAI 准备框架 v2（2025 年 4 月）

结构：

- **追踪类别（Tracked Categories）**：触发能力报告（模型能力）和安全措施报告（缓解措施），部署前由安全咨询组审核。
- **研究类别（Research Categories）**：模型能力实验室正在关注，但尚未承诺具体缓解措施。包括长程自主、故意低估、自主复刻与适应、破坏安全措施。

v2 的关键框架：研究类别不自动触发缓解。政策用语是“潜在缓解”。这是关键的差别：类别位于追踪或研究之间是“受限”与“监控”的分水岭。

### DeepMind 前沿安全框架 v3（2025 年 9 月；2026 年 4 月 17 日新增追踪能力等级）

结构：

- **关键能力等级（Critical Capability Levels，CCLs）**：涵盖 5 个领域的能力阈值——网络安全（Cyber）、生物（Bio）、机器学习研发（ML R&D）、化学、生物、放射性和核（CBRN）、自主（自主被整合进 ML R&D 和 Cyber）。
- **追踪能力等级（Tracked Capability Levels）**：2026 年 4 月加入更细化等级。明确定义：ML R&D 自主等级 1 = 以竞争成本全面自动化 AI 研发流程。
- **欺骗性对齐监测**：明确承诺自动监控工具性推理滥用。

自主框架与 OpenAI 不同。DeepMind 不将自主权作为顶级领域，合并于自主造成伤害的领域（ML R&D 和 Cyber）。理据是，没有领域的自主只是能力没有风险；反论是跨领域自主是框架应命名的元风险。

### 三者共识

- 内部安全咨询组（Anthropic SAG、OpenAI SAG、DeepMind 内部委员会）。高能力模型部署前审查。
- 明确提及欺骗性对齐/对齐伪装风险类别。
- 定期发布官方文档（Anthropic：前沿安全路线图和风险报告；OpenAI：能力与安全报告；DeepMind：FSF 更新周期）。
- 认可仅监控的防御有上限。DeepMind 明言：“自动监测长远来看不足够。”

### 差异之处

- **Anthropic：** v3.0 取消暂停承诺；AI 研发-4 阈值是下一道门槛。
- **OpenAI：** 追踪与研究区分；研究类别（包含长程自主）不自动触发暂停。
- **DeepMind：** 自主权合并进其他领域；追踪能力等级 2026 年 4 月进一步细化。

### 故意低估（Sandbagging）：复杂化三者的特定能力

故意低估（模型在评估中策略性低表现）被纳入 OpenAI 研究类别。Anthropic 在 RSP v3.0 通过评估环境差异（Lesson 1）解决。DeepMind 在 FSF v3 中通过欺骗性对齐监测应对。

若模型低估评估，所有框架能力阈值都会被低估。框架仅当测量有效时才起作用。这就是为何除实验室自评外，还需要外部测量（Lesson 21，METR）和对抗性评估。

### 政策解读能力

- 定位：你关注的每项能力应在政策中可找到，找不到即政策不涵盖。
- 分类：是追踪（触发缓解）还是研究（被追踪但未触发）？OpenAI 直白命名，Anthropic 与 DeepMind 有自身对应。
- 频率：政策是按固定时间更新，还是仅在特定事件后更新？固定频率更强。
- 独立性：外部审查是必需还是可选？Anthropic 与 Apollo 和 US AI Safety Institute 合作；OpenAI 与 METR；DeepMind 主要内部 SAG。

## 使用说明

`code/main.py` 实现了一个小型决策表差异工具。给定某项能力（自主权、欺骗性对齐、研发自动化、网络提升等），输出三框架对该能力的分类和触发的缓解措施。只做解读辅助，不是政策工具。

## 输出结果

`outputs/skill-cross-policy-diff.md` 产出处于三框架参考下的特定能力跨政策比较。

## 练习

1. 运行 `code/main.py`。确认差异工具输出与源文档对至少两项能力的政策匹配。

2. 完整阅读 OpenAI 准备框架 v2。识别每个研究类别。对每个类别写一句话说明为何它处于研究类而非追踪类。

3. 完整阅读 DeepMind FSF v3 及 2026 年 4 月的追踪能力等级更新。确定 ML R&D 自主等级 1 的具体评估标准。你会如何进行外部测量？

4. 故意低估属于 OpenAI 研究类别。设计一个评估迫使策略性低估模型暴露其真实能力。参考 Lesson 1 中的评估环境操纵讨论。

5. 对比三份政策中某项你选的能力。指出你认为什么政策分类最严谨、哪项最宽松。用原文做出理由。

## 关键词

| 术语                   | 常见说法                 | 实际含义                                 |
|-----------------------|-------------------------|------------------------------------------|
| Preparedness Framework | “OpenAI 扩展政策”        | PF v2（2025年4月）；追踪与研究类别           |
| Tracked Category      | “强制性缓解”             | 触发能力及安全报告；SAG 审核                 |
| Research Category     | “仅监控”                 | 被追踪但无自动缓解；含长程自主                 |
| Frontier Safety Framework | “DeepMind 扩展政策”       | FSF v3（2025年9月）+追踪能力等级（2026年4月）    |
| CCL                   | “关键能力等级”           | DeepMind 各领域阈值（Cyber, Bio, ML R&D, CBRN） |
| ML R&D autonomy level 1 | “研发自动化”             | 以竞争成本全面自动化 AI 研发流程                |
| Sandbagging           | “策略性低表现”           | 模型在评估中表现不足；OpenAI 研究类别             |
| Instrumental reasoning | “工具性推理”             | 关于如何实现目标的推理；DeepMind 监测目标             |

## 延伸阅读

- [OpenAI — 更新我们的准备框架](https://openai.com/index/updating-our-preparedness-framework/) — v2 发布公告。  
- [OpenAI — Preparedness Framework v2 PDF](https://cdn.openai.com/pdf/18a02b5d-6b67-4cec-ab64-68cdfbddebcd/preparedness-framework-v2.pdf) — 完整文档。  
- [DeepMind — 加强我们的前沿安全框架](https://deepmind.google/blog/strengthening-our-frontier-safety-framework/) — FSF v3 发布公告。  
- [DeepMind — 更新前沿安全框架（2026 年 4 月）](https://deepmind.google/blog/updating-the-frontier-safety-framework/) — 追踪能力等级新增。  
- [Gemini 3 Pro FSF 报告](https://storage.googleapis.com/deepmind-media/gemini/gemini_3_pro_fsf_report.pdf) — FSF 格式风险报告示例。
