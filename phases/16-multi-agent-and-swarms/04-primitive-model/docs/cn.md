# 多智能体原始模型（Multi-Agent Primitive Model）

> 每一个将在2026年发布的多智能体框架——AutoGen、LangGraph、CrewAI、OpenAI Agents SDK、微软Agent框架——都是四维设计空间中的一个点。四个原语，仅此而已：agent（智能体）、handoff（交接）、shared state（共享状态）、orchestrator（协调者）。本课从零构建它们，在这四个原语上运行一个玩具系统，然后将每个主流框架映射到相同坐标轴，这样你就能在一段话内读懂任何新版本。

**类型：** 学习  
**语言：** Python（标准库）  
**前置条件：** 第14阶段（Agent工程）、第16阶段·01（为何多智能体）  
**时长：** ~60分钟

## 问题

每半年都会有一个新的多智能体框架发布。2023年的AutoGen，2024年的CrewAI，2024年的LangGraph和OpenAI Swarm，2025年4月的Google ADK，2026年2月的微软Agent框架RC。每个新闻稿都宣称自己是“正确的抽象”。

如果你试图一个接一个地学习它们，你会筋疲力尽。API看起来不同。文档对于“agent”的定义各执一词。一个框架把共享内存称为“blackboard（黑板）”，另一个叫“message pool（消息池）”，第三个叫“StateGraph（状态图）”。你开始怀疑这个领域只是在不断翻篇。

其实不是。在营销背后，这四个原语是稳定的。学会一次，就能用一句话理解每个新框架。

## 概念

### 四个原语

1. **Agent（智能体）** — 一个系统提示和工具列表的组合。无状态；每次运行都从它的系统提示和当前消息历史开始。  
2. **Handoff（交接）** — 从一个agent到另一个的有结构的控制转移。机械上是一种调用工具，返回一个新的agent或一条条件触发的图边。  
3. **Shared state（共享状态）** — 多个agent均可读取（有时写入）的任何数据结构。消息池、黑板、键值存储、向量存储。  
4. **Orchestrator（协调者）** — 决定下一个发言者的人。选项：显式图（确定性）、LLM发言者选择器（软选择）、上一个发言者的handoff调用（OpenAI Swarm）、或基于队列的调度器（群集架构）。

这就是整个设计空间。每个框架为每个轴选择默认值，剩下的就是表层语法。

### 每个2026年框架的映射

| 框架                        | Agent（智能体）                         | Handoff（交接）             | Shared state（共享状态）  | Orchestrator（协调者）               |
|----------------------------|--------------------------------------|----------------------------|--------------------------|------------------------------------|
| OpenAI Swarm / Agents SDK   | `Agent(instructions, tools)`         | 工具返回Agent              | 调用者的问题              | LLM的下一次handoff调用              |
| AutoGen v0.4 / AG2          | `ConversableAgent`                   | GroupChat上的发言者选择器  | 消息池                    | 选择器函数（LLM或轮询）             |
| CrewAI                     | `Agent(role, goal, backstory)`       | `Process.Sequential / Hierarchical` | 任务输出串联              | 管理LLM或静态顺序                   |
| LangGraph                  | 节点函数                             | 图边+条件                   | `StateGraph` reducer     | 图，确定性                         |
| 微软Agent框架              | agent + 编排模式                     | 模式特定                    | 线程 / 上下文             | 模式特定                          |
| Google ADK                 | agent + A2A卡片                     | A2A任务                     | A2A工件                  | 主机决定                          |

表层差异看起来很大。实际上：同样的四个旋钮。

### 为什么这很重要

一旦理解了原语，框架比对就变成了一个简短的清单：

- 协调者是否信任LLM来做路由（Swarm），还是在代码中固定路由（LangGraph）？  
- 共享状态是完整历史（GroupChat）还是投影视图（StateGraph reducer）？  
- 智能体是否能修改彼此的提示（CrewAI管理者）还是只能进行handoff（Swarm）？

这三个问题决定了80%适合某个问题的框架。你不再去挑“最好的多智能体框架”，而是设计真正关心的轴向。

### 无状态洞察

除共享状态外，每个原语都是无状态的。Agent是(prompt, tools)的函数，Handoff是函数调用，Orchestrator是调度器。**系统中唯一有状态的部分是共享状态。** 有趣的bug都出现在这里：内存污染（第15课）、消息顺序、版本控制、写入冲突。

隐藏共享状态的框架（Swarm）把问题推给调用者。中心化管理共享状态的框架（LangGraph检查点、AutoGen池）使其可检查，但协调成本转嫁给共享状态实现。

### 单个原语的结构

#### Agent（智能体）

```python
Agent = (system_prompt, tools, model, optional_name)
```

无记忆，无状态。两个拥有相同系统提示和工具的agent是可互换的。所有看似智能体状态的内容其实都在共享状态或handoff协议中。

#### Handoff（交接）

```python
Handoff = (from_agent, to_agent, reason, payload)
```

主要有三种实现：

- **函数返回** — 工具返回下一个agent。这是OpenAI Swarm模式。智能体在工具定义中携带路由逻辑。  
- **图边** — LangGraph。图边是声明式的，LLM输出一个值，条件选中下一个节点。  
- **发言者选择** — AutoGen GroupChat。选择函数（有时自己是一条LLM调用）读取消息池，选择下一个发言者。

#### Shared state（共享状态）

```python
SharedState = { messages: [], artifacts: {}, context: {} }
```

至少有一组消息。通常还有更多：结构化工件（CrewAI任务输出）、类型化上下文（LangGraph reducers）、外部记忆（MCP，向量数据库）。

两种拓扑：**全池**（所有智能体都看到所有消息）和**投影池**（智能体看到基于角色的视图）。全池简单但扩展性差，投影池扩展性好但需预先设计schema。

#### Orchestrator（协调者）

```python
Orchestrator = ({state, last_speaker}) -> next_agent
```

四种类型：

- **静态** — 构建时固定图（LangGraph确定性、CrewAI顺序）。  
- **LLM选择** — LLM读取消息池选下一个发言者（AutoGen、CrewAI层次化）。  
- **handoff驱动** — 当前智能体通过调用handoff工具决定（Swarm）。  
- **队列驱动** — 工作者从共享队列拉任务；无显式下一个发言者（群集架构，Matrix）。

### 框架之间的差异

原语确定后，剩下设计决策是：

- **内存策略** — 瞬态 vs 持久检查点（LangGraph检查点器）。  
- **安全边界** — 谁能批准handoff（人机交互环）。  
- **成本核算** — 按智能体的token预算。  
- **可观察性** — 跟踪handoff，持久化状态以便回放。

都可以基于原语实现，没有新的原语。

## 实现它

`code/main.py`用约150行纯标准库Python实现了四个原语。无真正LLM——每个agent是脚本策略，重点在协调结构。

导出内容：

- `Agent` — 一个包含名称、系统提示、工具、策略函数的数据类。  
- `Handoff` — 返回新agent的函数。  
- `SharedState` — 线程安全的消息池。  
- `Orchestrator` — 三种变体：`StaticOrchestrator`、`HandoffOrchestrator`、`LLMSelectorOrchestrator`（模拟）。

演示运行相同的三智能体流程（研究 → 写作 → 审阅），用三种协调者类型，并打印最终消息池。你会发现输出差异仅在于*谁选择下一个发言者*；智能体和共享状态在运行间保持相同。

运行命令：

```bash
python3 code/main.py
```

预期输出：三种协调者模式各跑一遍。每次打印最终消息池。handoff驱动的执行如果研究者早早决定完成，触达的智能体会更少——这就是LLM路由的缩影。

## 使用它

`outputs/skill-primitive-mapper.md`是一个技能，读取任何多智能体代码库或框架文档，返回四个原语映射。在新框架版本上线时运行，可先用一句话理解，再深入看文档。

## 发布它

采用新框架前，写下它的原语映射。写不出来说明文档不全或框架发明了第五个原语（罕见——检查是否有未见过的共享状态类型）。

把映射写入架构文档。新成员加入时，把映射先发给他们，再发API文档。框架版本变更时，比对映射，而非发布日志。

## 练习

1. 用不同智能体策略运行`code/main.py`三次。观察协调者选择如何影响运行的智能体。  
2. 实现第四种协调者类型：一个队列驱动的，实现智能体轮询共享状态找工作。会出现什么死锁？如何检测？  
3. 拿LangGraph快速入门（https://docs.langchain.com/oss/python/langgraph/workflows-agents），用四个原语重写。哪些LangGraph抽象一一映射，哪些是便利包装？  
4. 阅读OpenAI Swarm手册（https://developers.openai.com/cookbook/examples/orchestrating_agents）。找出四个原语中Swarm哪个最优雅，哪个交给调用者。  
5. 找表格中完全隐藏共享状态的框架。解释当智能体需要跨handoff协调但无法重读历史时会发生什么。

## 关键词

| 术语              | 人们说                 | 实际含义                                 |
|-------------------|------------------------|------------------------------------------|
| Agent（智能体）   | “一个带工具的LLM”       | `(system_prompt, tools, model)`三元组。无状态。  |
| Handoff（交接）   | “控制转移”             | 命名下一个agent和可选负载的结构化调用。有三种实现：函数返回、图边、发言者选择。 |
| Shared state（共享状态） | “记忆”/“上下文”        | 多智能体系统中唯一有状态的部分。消息池或黑板。    |
| Orchestrator（协调者） | “协调者”               | 决定谁下一个运行的人。静态图、LLM选择、handoff驱动或队列驱动。   |
| Primitive（原语） | “抽象”                 | 每个框架参数化的四个轴之一。不是框架特性。         |
| Message pool（消息池） | “共享聊天记录”           | 完整历史共享状态。易推理，扩展性差。              |
| Projected state（投影状态） | “作用域视图”            | 针对角色的共享状态视图。可扩展，需设计schema。      |
| Speaker selection（发言者选择） | “谁下一个说话”          | 协调者模式，函数（通常是LLM）从一组中选下一个智能体。 |

## 推荐阅读

- [OpenAI手册：Orchestrating Agents——流程与交接](https://developers.openai.com/cookbook/examples/orchestrating_agents) — 最清晰的handoff驱动编排说明  
- [AutoGen稳定文档](https://microsoft.github.io/autogen/stable/) — GroupChat + 发言者选择是LLM选择编排的参考实现  
- [LangGraph工作流和智能体](https://docs.langchain.com/oss/python/langgraph/workflows-agents) — 图边编排和基于reducer的共享状态  
- [CrewAI介绍](https://docs.crewai.com/en/introduction) — 角色-目标-背景智能体，顺序/层次进程  
- [AG2（社区AutoGen延续）](https://github.com/ag2ai/ag2) — 微软将v0.4转为维护后活跃的AutoGen v0.2分支
