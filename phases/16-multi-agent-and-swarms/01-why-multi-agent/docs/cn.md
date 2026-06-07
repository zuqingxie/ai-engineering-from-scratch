# 为什么选择多智能体（Multi-Agent）？

> 一个智能体撞到了墙。聪明的做法不是打造更强大的单个智能体，而是使用更多的智能体。

**类型：** 学习  
**编程语言：** TypeScript  
**先决条件：** 阶段14（Agent 工程学）  
**时长：** 约60分钟

## 学习目标

- 识别单智能体的瓶颈（上下文溢出、混合专业知识、顺序瓶颈），并解释何时拆分成多个智能体是正确选择  
- 比较编排模式（流水线、并行扇出、监督者、层级结构），并为特定任务结构选择合适的模式  
- 设计具有明确角色边界、共享状态和通信契约的多智能体系统  
- 分析多智能体复杂性（延迟、成本、调试难度）与单智能体简洁性的权衡

## 问题描述

你在阶段14构建了一个单智能体。它能工作。能读取文件、运行命令、调用 API、并推理结果。然后你把它应用到一个真实代码库：200个文件，三种语言，依赖基础设施的测试，并且要求在写代码前先调研外部 API。

智能体崩溃了。这不是因为大语言模型（LLM）智能不够，而是任务超出了一个智能体循环能处理的范围。上下文窗口被文件内容填满。智能体忘记了40次工具调用之前阅读的内容。它试图同时做研究员、编码者和审查员，但三者都做得很差。

这就是单智能体的天花板。每当任务需要：

- **更多上下文，而一窗口无法容纳** —— 读取50个文件远超20万token  
- **不同阶段需不同专业知识** —— 研究所需的提示不同于代码生成  
- **任务可并行完成** —— 为什么要顺序读取三个文件，而不是同时读取？

时，你就会遇到单智能体的限制。

## 核心概念

### 单智能体的天花板

单智能体是一轮循环，一个上下文窗口，一个系统提示。形象地看：

```text
┌─────────────────────────────────────────┐
│            单智能体（SINGLE AGENT）      │
│                                         │
│  ┌───────────────────────────────────┐  │
│  │         上下文窗口（Context Window）│  │
│  │                                   │  │
│  │  研究笔记                         │  │
│  │  + 代码文件                       │  │
│  │  + 测试输出                      │  │
│  │  + 审查反馈                      │  │
│  │  + API 文档                      │  │
│  │  + ...                          │  │
│  │                                   │  │
│  │  ██████████████████████ 满了 ███  │  │
│  └───────────────────────────────────┘  │
│                                         │
│  一个系统提示试图涵盖                    │
│  研究 + 编码 + 审查 + 测试              │
│                                         │
│  结果：什么都做，但效果平庸              │
└─────────────────────────────────────────┘
```

有三个主要问题：

1. **上下文饱和** - 工具结果堆积。到第30个回合时，智能体已消耗15万个token的文件内容、命令输出和之前推理。回合5的关键信息丢失。  

2. **角色混淆** - 系统提示说“你同时是研究员、编码者、审查员和测试员”，结果是它只是半研究、半编码，永远完成不了审查。  

3. **顺序瓶颈** - 智能体顺序读取文件A、文件B、文件C。三次串行的 LLM 调用，三次串行的工具执行，没有并行。  

### 多智能体解决方案

拆分工作。给每个智能体一个任务，一个上下文窗口，一个针对任务调优的系统提示：

```text
┌──────────────────────────────────────────────────────────┐
│                    编排者（ORCHESTRATOR）                  │
│                                                          │
│  “构建一个用户管理的REST API”                            │
│                                                          │
│         ┌──────────┬──────────┬──────────┐               │
│         │          │          │          │               │
│         ▼          ▼          ▼          ▼               │
│   ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│   │ 研究员    │ │ 编码者    │ │ 审查员    │ │ 测试员    │  │
│   │          │ │          │ │          │ │          │  │
│   │ 读取文档  │ │ 基于研究 │ │ 检查代码 │ │ 运行测试  │  │
│   │，发现    │ │ 与规范写 │ │ 质量，   │ │ 并报告   │  │
│   │模式     │ │ 代码      │ │ 发现缺陷 │ │ 结果     │  │
│   │          │ │          │ │          │ │          │  │
│   └─────┬────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘  │
│         │           │            │             │         │
│         └───────────┴────────────┴─────────────┘         │
│                          │                               │
│                     合并结果                            │
└──────────────────────────────────────────────────────────┘
```

每个智能体都有：  
- 专注的系统提示（“你是代码审查员，你唯一的任务是发现缺陷。”）  
- 独立的上下文窗口（不会被其他智能体的工作污染）  
- 明确的输入/输出契约（接收研究笔记，输出代码）  

### 真实系统实例

**Claude Code 子智能体** —— 当 Claude Code 使用 `Task` 生成子智能体时，它创建了带有限定任务的子智能体，父智能体维持干净上下文，子智能体专注处理并返回摘要。  

**Devin** —— 运行计划者智能体、编码者智能体和浏览器智能体。计划者拆分工序，编码者写代码，浏览器调研文档。各自上下文独立。  

**多智能体编程团队（SWE-bench）** —— SWE-bench 的顶级系统使用研究员读取代码库，计划者设计修复，编码者实施。单智能体系统得分较低。  

**ChatGPT 深度调研** —— 并行生成多个搜索智能体，各自探索不同角度，最后合成结果。  

### 多智能体的谱系

多智能体不是非此即彼，而是一个连续谱系：

```text
简单 ─────────────────────────────────────────── 复杂

 单智能体    子智能体       流水线       团队        群集

 ┌───┐      ┌───┐        ┌───┐───┐    ┌───┐───┐    ┌─┐┌─┐┌─┐
 │ A │      │ A │        │ A │ B │    │ A │ B │    │ ││ ││ │
 └───┘      └─┬─┘        └───┘─┬─┘    └─┬─┘─┬─┘    └┬┘└┬┘└┬┘
              │              │       │   │       ┌┴──┴──┴┐
            ┌─┴─┐          ┌───┘───┐  │   │       │共享状态│
            │ a │          │ C │ D │ ┌─┴───┴─┐    └───────┘
            └───┘          └───┘───┘ │  消息总线│
                                      │          │
  1轮循环    父 + 子任务         阶段式       N份合作者，
  1上下文窗口                   执行          涌现行为
                                      明确角色
```

- **单智能体（Single agent）** —— 单轮循环，一个提示。适合简单任务。  
- **子智能体（Subagents）** —— 父智能体生成子任务的子智能体。父智能体维护计划，子智能体报告进展。这是 Claude Code 的做法。  
- **流水线（Pipeline）** —— 智能体顺序执行。A的输出是B的输入。适合“研究 - 编码 - 审查 - 测试”等分阶段工作流。  
- **团队（Team）** —— 智能体并行运行，共享消息总线。各自扮演角色，由编排者协调。适合同时需要不同专业技能的场景。  
- **群集（Swarm）** —— 大量相同或相似智能体共享状态。无固定编排者，智能体从队列取任务。适合高吞吐并行任务。

### 四种多智能体模式

#### 模式1：流水线（Pipeline）

```text
输入 ──▶ 智能体A ──▶ 智能体B ──▶ 智能体C ──▶ 输出
          （研究）   （编码）     （审查）
```

每个智能体加工数据并传递。逻辑简单，一环失败，后续阻塞。

#### 模式2：扇出/扇入（Fan-out / Fan-in）

```text
                ┌──▶ 智能体A ──┐
                │              │
输入 ──▶ 拆分 ├──▶ 智能体B ──├──▶ 合并 ──▶ 输出
                │              │
                └──▶ 智能体C ──┘
```

工作分配给并行智能体，结果合并。适合可拆解成独立子任务的场景。

#### 模式3：编排者-工作者（Orchestrator-Worker）

```text
                    ┌──────────┐
                    │ 编排者   │
                    └──┬───┬───┘
                  任务 │   │ 任务
                 ┌─────┘   └─────┐
                 ▼               ▼
           ┌──────────┐   ┌──────────┐
           │ 工作者A  │   │ 工作者B  │
           └──────────┘   └──────────┘
```

智能编排者决定任务分配、委派工作者，整合结果。编排者本身是可生成工作者的智能体。

#### 模式4：点对点群集（Peer Swarm）

```text
         ┌───┐ ◄──── 消息 ────▶ ┌───┐
         │ A │                  │ B │
         └─┬─┘                  └─┬─┘
           │                      │
      消息 │    ┌───────────┐     │ 消息
           └───▶│ 共享状态  │◄────┘
                │  / 队列   │
           ┌───▶│           │◄────┐
           │    └───────────┘     │
      消息 │                      │ 消息
         ┌─┴─┐                  ┌─┴─┐
         │ C │ ◄──── 消息 ────▶ │ D │
         └───┘                  └───┘
```

无中央编排者。智能体点对点通信。决策从交互中涌现。调试更难，但更适合大规模智能体。

### 何时不使用多智能体

多智能体增加复杂性。智能体间的每条消息都是潜在故障点。调试从“读取一段对话”变成“追踪五个智能体间消息”。

**保持单智能体时机：**  
- 任务能放进一个上下文窗口（少于约10万token工作数据）  
- 不需要不同系统提示覆盖不同阶段  
- 顺序执行速度够快  
- 任务简单，拆分反而带来额外开销

**复杂性代价：**  
- 每个智能体边界是一种有损压缩：智能体A的全部上下文被总结为发送给智能体B的消息  
- 协调逻辑（谁做什么、几点做、顺序如何）是新的错误源  
- 增加延迟：N个智能体意味着至少N轮串行 LLM 调用，且可能需要往返通信  
- 成本倍增：每个智能体独立消耗计算资源

经验法则：如果任务调用工具次数少于20次且能放进10万token，则保持单智能体。

## 实践构建

### 步骤1：超载的单智能体

这是一个尝试做所有事情的单智能体。它有一个巨大的系统提示，和一个上下文窗口，包含研究、代码和审查内容：

```typescript
type AgentResult = {
  content: string;
  tokensUsed: number;
  toolCalls: number;
};

async function singleAgentApproach(task: string): Promise<AgentResult> {
  const systemPrompt = `You are a full-stack developer. You must:
1. Research the requirements
2. Write the code
3. Review the code for bugs
4. Write tests
Do ALL of these in a single conversation.`;

  const contextWindow: string[] = [];
  let totalTokens = 0;
  let totalToolCalls = 0;

  const research = await fakeLLMCall(systemPrompt, `Research: ${task}`);
  contextWindow.push(research.output);
  totalTokens += research.tokens;
  totalToolCalls += research.calls;

  const code = await fakeLLMCall(
    systemPrompt,
    `Given this research:\n${contextWindow.join("\n")}\n\nNow write code for: ${task}`
  );
  contextWindow.push(code.output);
  totalTokens += code.tokens;
  totalToolCalls += code.calls;

  const review = await fakeLLMCall(
    systemPrompt,
    `Given all previous context:\n${contextWindow.join("\n")}\n\nReview the code.`
  );
  contextWindow.push(review.output);
  totalTokens += review.tokens;
  totalToolCalls += review.calls;

  return {
    content: contextWindow.join("\n---\n"),
    tokensUsed: totalTokens,
    toolCalls: totalToolCalls,
  };
}
```

这种方法存在的问题：
- 上下文窗口随着每个阶段增长。在审查步骤时，它包含了研究笔记、代码和先前的推理。
- 系统提示是通用的，无法针对每个阶段进行调整。
- 没有任何并行执行。

### 第 2 步：专业代理（Specialist Agents）

现在拆分它。每个代理只负责一项任务：

```typescript
type SpecialistAgent = {
  name: string;
  systemPrompt: string;
  run: (input: string) => Promise<AgentResult>;
};

function createSpecialist(name: string, systemPrompt: string): SpecialistAgent {
  return {
    name,
    systemPrompt,
    run: async (input: string) => {
      const result = await fakeLLMCall(systemPrompt, input);
      return {
        content: result.output,
        tokensUsed: result.tokens,
        toolCalls: result.calls,
      };
    },
  };
}

const researcher = createSpecialist(
  "researcher",
  "You are a technical researcher. Read documentation, find patterns, and summarize findings. Output only the facts needed for implementation."
);

const coder = createSpecialist(
  "coder",
  "You are a senior TypeScript developer. Given requirements and research notes, write clean, tested code. Nothing else."
);

const reviewer = createSpecialist(
  "reviewer",
  "You are a code reviewer. Find bugs, security issues, and logic errors. Be specific. Cite line numbers."
);
```

每个专业代理有专注的提示。每个代理获得一个干净的上下文窗口，只有它所需的输入。

### 第 3 步：通过消息协调

使用显式消息传递将专业代理连接起来：

```typescript
type AgentMessage = {
  from: string;
  to: string;
  content: string;
  timestamp: number;
};

async function multiAgentApproach(task: string): Promise<AgentResult> {
  const messages: AgentMessage[] = [];
  let totalTokens = 0;
  let totalToolCalls = 0;

  const researchResult = await researcher.run(task);
  messages.push({
    from: "researcher",
    to: "coder",
    content: researchResult.content,
    timestamp: Date.now(),
  });
  totalTokens += researchResult.tokensUsed;
  totalToolCalls += researchResult.toolCalls;

  const coderInput = messages
    .filter((m) => m.to === "coder")
    .map((m) => `[From ${m.from}]: ${m.content}`)
    .join("\n");

  const codeResult = await coder.run(coderInput);
  messages.push({
    from: "coder",
    to: "reviewer",
    content: codeResult.content,
    timestamp: Date.now(),
  });
  totalTokens += codeResult.tokensUsed;
  totalToolCalls += codeResult.toolCalls;

  const reviewerInput = messages
    .filter((m) => m.to === "reviewer")
    .map((m) => `[From ${m.from}]: ${m.content}`)
    .join("\n");

  const reviewResult = await reviewer.run(reviewerInput);
  messages.push({
    from: "reviewer",
    to: "orchestrator",
    content: reviewResult.content,
    timestamp: Date.now(),
  });
  totalTokens += reviewResult.tokensUsed;
  totalToolCalls += reviewResult.toolCalls;

  return {
    content: messages.map((m) => `[${m.from} -> ${m.to}]: ${m.content}`).join("\n\n"),
    tokensUsed: totalTokens,
    toolCalls: totalToolCalls,
  };
}
```

每个代理仅接收发给它的消息。没有上下文污染。研究者读取的 5 万个 token 文档永远不会进入审查者的上下文。

### 第 4 步：比较

```typescript
async function compare() {
  const task = "Build a rate limiter middleware for an Express.js API";

  console.log("=== Single Agent ===");
  const single = await singleAgentApproach(task);
  console.log(`Tokens: ${single.tokensUsed}`);
  console.log(`Tool calls: ${single.toolCalls}`);

  console.log("\n=== Multi-Agent ===");
  const multi = await multiAgentApproach(task);
  console.log(`Tokens: ${multi.tokensUsed}`);
  console.log(`Tool calls: ${multi.toolCalls}`);
}
```

多代理版本使用了更多的总 token（三个代理，三次独立的 LLM 调用），但每个代理的上下文保持干净。每个阶段的质量提高是因为系统提示更加专业化。

## 使用它

本节课生成了一个用于决定何时采取多代理的可重用提示。见 `outputs/prompt-multi-agent-decision.md`。

## 练习

1. 添加第四个专业代理：一个“测试者”代理，接收来自 coder 的代码和来自 reviewer 的审查反馈，然后编写测试
2. 修改流水线，使审查者可以将反馈发回 coder 以进行修订循环（最多两轮）
3. 将顺序流水线转换为扇出（fan-out）：并行运行 researcher 和“需求分析器”代理，然后合并它们的输出再传给 coder

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| Swarm | “一群 AI 代理的蜂巢智能” | 一组具有共享状态且无固定领导的对等代理。从局部交互中涌现行为。 |
| Orchestrator | “老板代理” | 一种工具包括生成和管理其他代理的代理。它负责计划和委派，但可能不执行实际工作。 |
| Coordinator | “交通警察” | 非代理组件（通常只是代码，不是 LLM），根据规则在代理间路由消息。 |
| Consensus | “代理达成一致” | 多个代理必须达成协议后方可继续的协议。用于需要解决冲突输出时。 |
| Emergent behavior | “代理自己想出来了” | 代理交互产生的系统级模式，没被明确编程。可能有用也可能有害。 |
| Fan-out / fan-in | “代理版的 Map-Reduce” | 将任务分配给多个并行代理（扇出），然后合并它们的结果（扇入）。 |
| Message passing | “代理互通有无” | 代理间的通信机制：由一个代理发送给另一个代理的结构化数据，取代共享上下文窗口。 |

## 进一步阅读

- [The Landscape of Emerging AI Agent Architectures](https://arxiv.org/abs/2409.02977) - 多代理模式调研
- [AutoGen: Enabling Next-Gen LLM Applications](https://arxiv.org/abs/2308.08155) - 微软多代理对话框架
- [Claude Code subagents documentation](https://docs.anthropic.com/en/docs/claude-code) - Claude Code 如何用 Task 委派
- [CrewAI documentation](https://docs.crewai.com/) - 基于角色的多代理框架
