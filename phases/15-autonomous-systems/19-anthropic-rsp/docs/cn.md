# Anthropic 责任扩展策略 v3.0

> RSP v3.0 于 2026 年 2 月 24 日生效，取代了 2023 年的政策。采用两级缓解措施：Anthropic 单方面采取的措施与作为行业范围推荐（包括 RAND SL-4 安全标准）的分开陈述。将 Frontier Safety Roadmaps（前沿安全路线图）和 Risk Reports（风险报告）提升为常设文件，而非一次性成果。取消了 2023 年的暂停承诺。引入了 AI R&D-4 阈值：一旦跨越，Anthropic 必须发布一份积极理由，阐明不匹配风险和缓解措施。Claude Opus 4.6 未跨越该阈值。Anthropic 在 v3.0 公告中声明，“自信地排除这一点变得困难”。SaferAI 对 2023 年 RSP 评分为 2.2；对 v3.0 降级至 1.9，将 Anthropic 列为“薄弱”RSP 类别，与 OpenAI 和 DeepMind 同列。定性阈值取代了 2023 年的定量承诺；撤销暂停条款是最明显的倒退。

**类型：** 学习  
**语言：** Python（stdlib，RSP 阈值决策引擎）  
**先决条件：** 第 15 阶段 · 06（AAR）、第 15 阶段 · 07（RSI）  
**时长：** 约 45 分钟

## 问题

前沿实验室发布的扩展策略既是部分技术文档，又是治理文档，还承担着向监管机构传达信号的作用。RSP v3.0 是当前 Anthropic 的文档。仔细阅读它很重要，并非因为其合规性具备约束力（实际上没有），而是因为其框架塑造了实验室如何认知灾难性风险以及如何向公众传递权衡。

v3.0 与 v2.0 的差异是有用的分析单元。新增内容：Frontier Safety Roadmaps，Risk Reports，AI R&D-4 阈值。移除内容：2023 年的暂停承诺。重新框定内容：将缓解措施分为 Anthropic 单方面行动和行业推荐两级。外部审查机构 SaferAI 将评分从 2.2（v2）降至 1.9（v3.0）。这就是扩展策略看似更精炼却变得不那么严格的例子。

## 概念

### 两级缓解结构

- **Anthropic 单方面行动**：Anthropic 无论其他实验室如何都会采取的措施。包括超阈值时停止训练、特定安全措施、特定部署门控。
- **行业范围推荐**：Anthropic 认为行业整体应采纳的措施。包括 RAND SL-4 安全标准。这不是 Anthropic 自身的承诺，而是政策倡议。

两级结构在 v2 中不存在。这意味着读者需要关注每项承诺所属的栏位。位于“行业范围推荐”栏的安全措施不是 Anthropic 的承诺，而是其期望。

### AI R&D-4 阈值

这是 RSP v3.0 指定的重要下一个能力级别。具体指：一个模型能够以有竞争力的成本自动化完成 AI 研究中的大量工作。一旦 Anthropic 认为模型跨越此阈值，须发布一份积极理由，说明不匹配风险和缓解措施，然后才能继续扩展。

根据 v3.0 公告，Claude Opus 4.6 未跨越此阈值。文档中补充道：“自信地排除这一点变得困难。”此措辞重要，表明该阈值已足够接近，成为一个实际关注点，而非纯粹假设。

第 6 课（自动化对齐研究）和第 7 课（递归自我改进）与此阈值密切相关。自动化对齐研究人员达到研究质量水平，证明 AI R&D-4 阈值正在逼近。

### Frontier Safety Roadmaps 和 Risk Reports

v3.0 将两种文档类型提升为常设文件：

- **Frontier Safety Roadmap（前沿安全路线图）**：前瞻性文档，描述规划中的安全工作、能力预期和缓解研究。  
- **Risk Report（风险报告）**：事后文档，针对特定模型发布后，描述观察到的能力及剩余风险。

两者均公开，均按既定节奏更新。用处在于：读者可以跟踪 Anthropic 在路线图中计划的工作与风险报告中实际反馈的对比。

### 去除暂停条款

2023 年 RSP 包含明确的暂停承诺：如果模型跨越某些能力阈值，训练将暂停直到缓解措施就位。v3.0 将明确暂停替换为较温和表述（发布积极理由，若缓解充分则继续）。SaferAI 和其他分析人士直言这是新文档中最大的倒退。

政策变更理由为：2023 年的定量阈值由于基准调整，到了 2026 年能力基准已无法适用。反对意见是：暂停条款是扩展政策中的承诺装置，撤销它会削弱政策的可信度。

### SaferAI 降级

SaferAI 是独立组织，负责评估类似 RSP 的文件。其公开评分：2023 年 Anthropic RSP 得分 2.2（满分为 4.0，1.0 为最低标准），v3.0 得分 1.9。这将 Anthropic 从“中等”类别降至“薄弱”，与 OpenAI 和 DeepMind 一同列入。

SaferAI 降级理由：
- 定性阈值取代了定量阈值。
- 移除了暂停承诺。
- AI R&D-4 阈值缓解仅用“积极理由”描述，未列出具体措施。
- 审查机制依赖于 Anthropic 的安全顾问组，缺少独立监督。

### 本课非教条

本课非关于合规。RSP v3.0 非法规，Anthropic 无强制遵守义务。课程重点在于以应有的具体性和怀疑态度解读文档。扩展策略是前沿实验室向公众发出的关于灾难风险态度的主要信号。善于阅读它们是依赖前沿能力人员的重要实用技能。

## 练习示例

`code/main.py` 实现了一个小型决策引擎，模拟 RSP 阈值评估逻辑：给定候选模型及其能力测量，返回是否跨越 AI R&D-4 阈值、所需积极理由部分、是否允许部署。实现故意简化，目的是明确文档逻辑。

## 实战

`outputs/skill-scaling-policy-review.md` 针对 v3.0 参考框架，审查一份扩展策略（Anthropic、OpenAI、DeepMind 或内部）：两级结构、阈值、暂停承诺、独立审查。

## 练习题

1. 运行 `code/main.py`。输入三个不同能力水平的合成模型。确认阈值评估器行为符合预期，输出正确的积极理由模板。

2. 通读 RSP v3.0（32 页）。列出所有属于“行业范围推荐”层级的承诺。哪些在 v2 中属于“Anthropic 单方面”？

3. 研读 SaferAI 的 RSP 评分方法。用其评分规则对 v3.0 文档打分，复现 1.9 分。哪个评分项对降级影响最大？

4. 2023 年暂停承诺被移除。提出替代承诺，既保持政策可信度，又考虑到 2026 年基准调整问题。

5. 对比 RSP v3.0 与 OpenAI Preparedness Framework v2（第 20 课）。指出 v3.0 强的一个方面，以及 Preparedness Framework 强的一个方面。

## 关键词

| 术语 | 通俗说法 | 实际含义 |
|---|---|---|
| RSP | “Anthropic 的扩展政策” | Responsible Scaling Policy（责任扩展策略）；v3.0 自 2026 年 2 月 24 日生效 |
| AI R&D-4 | “研究自动化阈值” | 具备以竞争成本自动化大量 AI 研究的能力 |
| Affirmative case | “安全论证” | 发布的论证，确认风险已识别且缓解措施充分 |
| Frontier Safety Roadmap | “前瞻计划” | 关于计划安全工作和预期能力的常设文档 |
| Risk Report | “模型回顾” | 发布后观察能力及剩余风险的常设文档 |
| Two-tier mitigation | “单方面与行业” | Anthropic 承诺与行业推荐独立列示 |
| Pause commitment | “2023 条款” | 明确暂停训练的承诺，v3.0 已移除 |
| SaferAI rating | “独立 RSP 评级” | 第三方评分；v3.0 得 1.9（v2 为 2.2） |

## 延伸阅读

- [Anthropic — Responsible Scaling Policy v3.0](https://anthropic.com/responsible-scaling-policy/rsp-v3-0) — 完整 32 页政策文档。  
- [Anthropic — RSP v3.0 announcement](https://www.anthropic.com/news/responsible-scaling-policy-v3) — v2 变更总结。  
- [Anthropic — Frontier Safety Roadmap](https://www.anthropic.com/research/frontier-safety) — RSP v3.0 引用的常设文档。  
- [Anthropic — Risk Report: Claude Opus 4.6](https://www.anthropic.com/research/risk-report-claude-opus-4-6) — 当前前沿模型的回顾报告。  
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — 将 AI R&D-4 与测量的自治性联系起来。
