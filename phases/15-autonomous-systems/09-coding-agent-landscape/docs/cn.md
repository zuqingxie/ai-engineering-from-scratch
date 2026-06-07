# 自动编码代理全景（2026）

> SWE-bench Verified 在不到三年的时间里从4％提升到了80.9％。同一个 Claude Sonnet 4.5 在 SWE-agent v1 上得分为43.2％，而在 Cline autonomous 上得分为59.8％——模型周围的支架（scaffolding）现在和模型本身同样重要。OpenHands（前身为 OpenDevin）是最活跃的 MIT 许可平台，其 CodeAct 循环直接在沙箱中执行 Python 操作，而不是调用 JSON 工具。表面的数据掩盖了方法论问题：500个 SWE-bench Verified 任务中有161个只需1–2行改动，而 SWE-bench Pro（10+行改动的任务）在相同前沿模型上的得分为23–59%。

**类型：** 学习  
**语言：** Python（标准库，CodeAct 与 JSON 工具调用对比）  
**前置知识：** 第14阶段 · 07（工具使用），第15阶段 · 01（长远代理）  
**时间：** ~45分钟

## 问题

“哪个编码代理最好”是错误的问题。正确的问题是：在与我的工作匹配的任务分布上，采用我将在生产中运行的支架，最终端到端的可靠性是多少？

2022年至2026年间，领域认识到支架——检索层、规划器、沙箱、编辑-验证循环、反馈格式——是承重的。Claude Sonnet 4.5 在 SWE-agent v1 上得分43.2％；相同模型在 Cline 的 autonomous 支架中得分59.8％。同权重下差了16.6个绝对点。基础模型只是组件；循环才是产品。

伴随的问题是，基准测试的饱和掩盖了回退。SWE-bench Verified 已接近饱和，且任务中容易的尾部（161个任务需要≤2行）拉高了最高分。现实质量更适合用 SWE-bench Pro（10+ 行改动）这类分布衡量，当前领跑者得分为23–59%。

## 概念

### SWE-bench 简述

SWE-bench（Jimenez 等人）采用真实 GitHub 问题及其对应的真实补丁，要求代理生成补丁使测试套件通过。SWE-bench Verified（OpenAI，2024）是经过人工筛选的500任务子集，剔除了模糊和错误的任务。SWE-bench Pro 是更难的后继版本——任务需要10+行改动，当前前沿代理得分在23–59%。

### 2022 → 2026 曲线实际表现

- **2022年**：研究模型在原始 SWE-bench 上约4％。  
- **2024年**：GPT-4 + Devin 风格支架约14%；SWE-agent约12%。  
- **2025年**：Claude 3.5/3.7 Sonnet 纳入 Aider 和 SWE-agent，推高至40–55%。  
- **2026年**：Claude Sonnet 4.5及竞争者在 SWE-bench Verified 上达到70–80%+。Epoch AI 实时跟踪排行榜。

这斜率来自三个复合因素：更好基础模型、更好支架（CodeAct、反思、验证器循环）和更好基准（Verified 去噪）。

### CodeAct 与 JSON 工具调用对比

OpenHands（All-Hands-AI, arXiv:2407.16741，前身为 OpenDevin）做了一个特定架构赌注：模型不是输出 JSON 工具调用由宿主解码执行，而是输出 Python 代码，由类似 Jupyter 的内核在沙箱中执行。代理可以在一个动作内循环多个文件、链式使用工具并捕获自身异常。

权衡如下：

- **JSON 工具调用**：每个动作是一轮，易审计，组合性有限，默认安全因为每次调用都经过明确验证器。  
- **CodeAct**：一个动作可以是整段程序，具备组合性，需加固沙箱（OpenHands 使用 Docker 隔离），失败模式为沙箱运行时允许的一切。

两种架构均在生产中使用。CodeAct 在开源平台（OpenHands、smolagents）中主导。JSON 工具调用在托管服务（Anthropic 管理代理、OpenAI 助手）中仍占主导，因供应商控制执行环境。

### 2026 生态中的支架

| 支架 | 许可 | 执行模型 | 特色 |
|---|---|---|---|
| OpenHands (OpenDevin) | MIT | Docker 中的 CodeAct | 最活跃开源平台；事件流可重放 |
| SWE-agent | MIT | Agent-Computer Interface (ACI) | 首个端到端 SWE-bench 支架 |
| Aider | Apache-2 | 本地仓库通过差异编辑 | 极简支架，回归稳定性强 |
| Cline | Apache-2 | VS Code 代理+工具策略 | Sonnet 4.5 上最高分开源支架 |
| Devin (Cognition) | 专有 | 托管虚拟机+规划器 | 首个“AI 软件工程师”产品类别 |
| Claude Code | 专有 | 权限模式+例程 | 第10课详细覆盖代理循环 |

### 支架为何主导

编码过程是长远轨迹（第1课）。可靠性在步骤间复合。支架带来提升的三个环节：

1. **检索**：找到正确文件静悄悄地限制性能。SWE-agent 的 ACI、OpenHands 的文件索引和 Aider 的仓库映射都致力于此。  
2. **验证循环**：运行测试，读堆栈，重试补丁，对于 SWE-bench 有10+分提升。  
3. **失败隔离**：错误时回滚的沙箱能防止复合损坏。同模型有无验证循环就像是两款产品。

### 基准测试饱和及真实分布

OpenHands 作者和 Epoch AI 均指出 SWE-bench Verified 包含易任务尾巴：161/500 任务只需1–2行改动。高分部分由此驱动。SWE-bench Pro 限定10+行改动，即使是前沿模型得分也在23–59%。您的生产分布几乎肯定更接近 Pro 而非 Verified。

选代理意义：用类似 Pro 的子集运行自有缺陷积压。关键分数是代表出货任务的得分。

## 实践

`code/main.py` 比较了两个玩具代理支架在固定小任务分布上的表现：

1. 一个**JSON 工具调用**支架，每轮执行一个动作。  
2. 一个**CodeAct**支架，每个动作可输出一小段 Python 代码。

二者都用一个存根“模型”（确定性规则），对比纯粹支架差异。输出显示 CodeAct 支架以更少轮次解决更多任务，但每动作波及范围更大。

## 交付

`outputs/skill-scaffold-audit.md` 有助于在采用前审计拟议的编码代理支架：检索质量、验证器存在、沙箱隔离及基准与分布匹配。

## 练习

1. 运行 `code/main.py`。两个支架解决相同任务集分别用了多少轮？各自动作的波及范围如何？

2. 阅读 OpenHands 论文（arXiv:2407.16741）。论文论证 CodeAct 在复杂任务上优于 JSON 工具调用。指出论文承认的一个失败模式，并用一句话描述该模式在生产环境下何时成为主导。

3. 从缺陷积压中选一个跨两个文件需10+行改动的任务，估计前沿模型采用（a）JSON 工具调用、（b）CodeAct 的端到端成功概率，并解释差距。

4. SWE-bench Verified 中有161个单文件、1–2行任务。构造一个排除它们的得分指标。排行榜如何变化？

5. 阅读 “Introducing SWE-bench Verified”（OpenAI）。解释用于剔除模糊任务的具体方法，列举一个该筛选可能遗漏的任务类别。

## 关键术语

| 术语 | 人们说 | 实际含义 |
|---|---|---|
| SWE-bench | “编码基准” | 带有真实补丁和测试套件的真实 GitHub 问题 |
| SWE-bench Verified | “清理子集” | 500个人工筛选任务，包含易任务尾巴 |
| SWE-bench Pro | “更难子集” | 需10+行改动；前沿模型得分23–59% |
| CodeAct | “代码即动作” | 代理输出 Python，由 Jupyter 风格内核沙箱执行 |
| JSON tool call | “函数调用” | 每个动作是结构化 JSON，有验证后执行 |
| Scaffold（支架） | “代理框架” | 基础模型周围的检索+规划+执行+验证循环 |
| ACI (Agent-Computer Interface) | “SWE-agent 格式” | 针对大语言模型设计的命令集，非人类 shell |
| Verifier loop（验证循环） | “测试和重试” | 运行测试、读取输出、修正补丁；非模型层面最大可靠性提升 |

## 拓展阅读

- [Jimenez 等人——SWE-bench](https://www.swebench.com/) ——原始基准及方法论。  
- [OpenAI——Introducing SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/) ——如何构建筛选子集。  
- [Wang 等人——OpenHands：AI 软件开发者开源平台](https://arxiv.org/abs/2407.16741) ——CodeAct 架构及事件流设计。  
- [Epoch AI——SWE-bench 排行榜](https://epoch.ai/benchmarks) ——实时跟踪得分。  
- [Anthropic——测量代理自主性](https://www.anthropic.com/research/measuring-agent-autonomy) ——长程编码代理可靠性框架。
