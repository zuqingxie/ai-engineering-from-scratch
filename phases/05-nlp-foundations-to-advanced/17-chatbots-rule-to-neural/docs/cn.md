# 聊天机器人 — 规则驱动到神经网络再到大模型代理（LLM Agents）

> ELIZA 通过模式匹配回复。DialogFlow 映射意图。GPT 从权重中回答。Claude 运行工具并验证。每个时代都解决了前一个时代最严重的失败。

**类型：** 学习  
**语言：** Python  
**前置知识：** 第5阶段 · 13（问答），第5阶段 · 14（信息检索）  
**时间：** ~75分钟  

## 问题

用户说“我想改签航班。”系统必须弄清他们想要什么，缺少哪些信息，如何获取这些信息，以及如何完成操作。然后用户说“等等，如果我改成取消怎么办？”系统必须记住上下文，切换任务，并保存状态。

对机器学习（ML）系统来说，对话非常困难。输入是开放的。输出必须在多轮交流中保持连贯。系统可能需要对现实世界采取行动（改签航班，刷卡收费）。每一个错误步骤用户都能看到。

聊天机器人架构经历了四种范式迭代，每次引入新范式都是因为前一个范式失败得太明显。本课依次介绍这四种架构。2026年的生产环境是后两者的混合体。

## 概念

![聊天机器人演变：规则驱动 → 检索 → 神经网络 → 代理](../assets/chatbot.svg)

**规则驱动（ELIZA，AIML，DialogFlow）**。手工编写的模式匹配用户输入并生成回答。意图分类器路由到预定义流程。槽位填充状态机收集必需信息。在其设计的狭窄范围内效果极佳。超出该范围则立刻失败。仍在安全关键领域（银行认证、航空订票）中使用，因为不能有幻觉。

**检索驱动**。类似FAQ系统。对每对（话语，回应）进行编码。运行时，编码用户消息并检索最相近的储存回复。比如 Zendesk 经典的“相似文章”功能。比规则更能处理同义重述。无生成，故无幻觉。

**神经网络（seq2seq）**。基于话语日志训练的编码器-解码器。生成回复。流畅但往往泛泛而谈（“我不知道”）且事实易偏离主题。话题连贯性不稳定。谷歌、Facebook、微软在2016-2019年这段时间的聊天机器人表现差，原因就在这里。

**大模型代理（LLM agents）**。一个内置循环的语言模型，能规划、调用工具、验证结果。不是采用长提示的聊天机器人。代理循环：规划 → 调用工具 → 观察结果 → 决定下一步。先检索再生成（RAG）防止幻觉。工具调用实现实际操作。这就是2026年的架构。

这四种范式不是顺序替代关系。2026年生产环境的聊天机器人会根据需求路由使用四者：规则驱动用于认证和破坏性操作，检索处理FAQ，神经生成处理自然表述，LLM代理处理模糊和开放式查询。

## 构建指南

### 第1步：规则驱动的模式匹配

```python
import re


class RulePattern:
    def __init__(self, pattern, response_template):
        self.regex = re.compile(pattern, re.IGNORECASE)
        self.template = response_template


PATTERNS = [
    RulePattern(r"my name is (\w+)", "Nice to meet you, {0}."),
    RulePattern(r"i (need|want) (.+)", "Why do you {0} {1}?"),
    RulePattern(r"i feel (.+)", "Why do you feel {0}?"),
    RulePattern(r"(.*)", "Tell me more about that."),
]


def rule_based_respond(user_input):
    for pattern in PATTERNS:
        m = pattern.regex.match(user_input.strip())
        if m:
            return pattern.template.format(*m.groups())
    return "I don't understand."
```

ELIZA 的20行代码版本。反射技巧（“我感到难过” → “你为什么感到难过”）是Weizenbaum 1966年经典的心理治疗师演示，至今仍很有启发性。

### 第2步：基于检索的FAQ

此示例代码需要安装 `sentence-transformers`（会自动安装 torch）。本课提供的可运行`code/main.py`版本用标准库Jaccard相似度实现，无需外部依赖即可运行。

```python
from sentence_transformers import SentenceTransformer
import numpy as np


FAQ = [
    ("how do i reset my password", "Go to Settings > Security > Reset Password."),
    ("how do i cancel my order", "Go to Orders, find the order, click Cancel."),
    ("what is your return policy", "30-day returns on unused items, original packaging."),
]


encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
faq_questions = [q for q, _ in FAQ]
faq_embeddings = encoder.encode(faq_questions, normalize_embeddings=True)


def faq_respond(user_input, threshold=0.5):
    q_emb = encoder.encode([user_input], normalize_embeddings=True)[0]
    sims = faq_embeddings @ q_emb
    best = int(np.argmax(sims))
    if sims[best] < threshold:
        return None
    return FAQ[best][1]
```

基于阈值的拒绝机制是关键设计点。若最佳匹配不够接近，返回 `None`，让系统升级处理。

### 第3步：神经生成（基线）

使用小型的指令调优的编码器-解码器模型（FLAN-T5）或经过微调的对话模型。单独用在2026年已无生产可用性（自相矛盾、偏题、事实错误），但混合系统中用于自然表述仍有应用。DialoGPT这类只解码模型需要明确的对话轮分隔符和EOS处理才能输出连贯回答；而FLAN-T5的text2text流水线开箱即用，适合教学示例。

```python
from transformers import pipeline

chatbot = pipeline("text2text-generation", model="google/flan-t5-small")

response = chatbot("Respond politely to: Hi there!", max_new_tokens=40)
print(response[0]["generated_text"])
```

### 第4步：LLM代理循环

2026年生产架构示例：

```python
def agent_loop(user_message, tools, llm, max_steps=5):
    history = [{"role": "user", "content": user_message}]
    for _ in range(max_steps):
        response = llm(history, tools=tools)
        tool_call = response.get("tool_call")
        if tool_call:
            tool_name = tool_call.get("name")
            args = tool_call.get("arguments")
            if not isinstance(tool_name, str) or tool_name not in tools:
                history.append({"role": "assistant", "tool_call": tool_call})
                history.append({"role": "tool", "name": str(tool_name), "content": f"error: unknown tool {tool_name!r}"})
                continue
            if not isinstance(args, dict):
                history.append({"role": "assistant", "tool_call": tool_call})
                history.append({"role": "tool", "name": tool_name, "content": f"error: arguments must be a dict, got {type(args).__name__}"})
                continue
            fn = tools[tool_name]
            result = fn(**args)
            history.append({"role": "assistant", "tool_call": tool_call})
            history.append({"role": "tool", "name": tool_name, "content": result})
        else:
            return response["content"]
    return "I could not complete the task in the step budget."
```

需明确三件事。工具是LLM可调用的函数。循环终止条件是模型返回最终答案而非工具调用。步骤预算防止任务不明确时陷入无限循环。

真实生产环境中会加入：先检索再生成的本地知识（每次调用LLM前注入相关文档）、防护措施（无确认时拒绝破坏性操作）、可观察性（记录每一步）、评估（自动检测代理行为是否符合规范）。

### 第5步：混合路由

```python
def hybrid_chat(user_input):
    if is_destructive_action(user_input):
        return structured_flow(user_input)

    faq_answer = faq_respond(user_input, threshold=0.6)
    if faq_answer:
        return faq_answer

    return agent_loop(user_input, tools, llm)


def is_destructive_action(text):
    danger_words = ["delete", "cancel", "charge", "refund", "transfer"]
    return any(w in text.lower() for w in danger_words)
```

模式是：破坏性操作用确定性规则，FAQ用检索，其他用LLM代理。这就是2026年客户支持系统的应用。

## 使用场景

2026年架构栈：

| 用例                  | 架构                                  |
|---------------------|-------------------------------------|
| 订票、支付、认证          | 规则驱动状态机 + 槽位填充                     |
| 客服FAQ                 | 对策划答案检索                           |
| 开放式辅助聊天            | 带RAG和工具调用的LLM代理                    |
| 内部工具 / IDE助手        | 带工具调用的LLM代理（搜索、读取、写入）             |
| 陪伴 / 角色扮演聊天机器人     | 经过调优的大模型带角色系统提示，带知识检索               |

生产环境务必使用混合路由。无单一架构能很好应对所有请求。路由层通常是小型意图分类器。

## 仍然需要容忍的失败模式

- **自信的虚构（Confident fabrication）。** LLM代理声称完成了未做的操作。缓解措施：核实结果，记录工具调用，禁止LLM在无成功工具返回时声明已完成操作。  
- **提示词注入（Prompt injection）。** 用户插入文本覆盖系统提示。2025年OWASP LLM应用Top10之LLM01。两类攻击：直接注入（粘贴进聊天），间接注入（隐藏在文档、邮件或代理读取的工具输出中）。

  攻击成功率根据场景不同，一般工具和编码基准在~0.5%-8.5%之间。特定高风险场景（针对AI编码代理的自适应攻击，脆弱调度）达约84%。生产安全漏洞包括EchoLeak（CVE-2025-32711，CVSS 9.3）——微软365 Copilot中的零点击数据泄露漏洞，由攻击者邮件触发。

  缓解措施：将用户输入视为不可信；调用工具前进行净化；将工具输出隔离出主提示；采用计划-验证-执行（PVE）模式——代理先计划，验证每步行动是否符合计划后再执行（阻止工具结果注入未计划操作）；破坏性操作需用户确认；工具权限最小化。

  无论多少提示词工程都无法完全消除风险。必须借助外部运行时防护层（如LLM Guard，白名单验证，语义异常检测）。
- **范围蔓延（Scope creep）。** 代理因工具调用返回的相关但非核心信息而偏离主题。缓解：缩小工具接口范围；聚焦系统提示；增加脱轨率评估。
- **无限循环（Infinite loops）。** 代理反复调用同一工具。缓解：步骤预算，工具调用去重，LLM判定是否有进展。
- **上下文窗口耗尽（Context window exhaustion）。** 长对话将最早轮次挤出上下文。缓解：摘要旧轮次，按相似度检索相关旧轮次，或使用长上下文模型。

## 交付成果

保存为 `outputs/skill-chatbot-architect.md`：

```markdown
---
name: chatbot-architect
description: Design a chatbot stack for a given use case.
version: 1.0.0
phase: 5
lesson: 17
tags: [nlp, agents, chatbot]
---

Given a product context (user need, compliance constraints, available tools, data volume), output:

1. Architecture. Rule-based, retrieval, neural, LLM agent, or hybrid (specify which paths go where).
2. LLM choice if applicable. Name the model family (Claude, GPT-4, Llama-3.1, Mixtral). Match to tool-use quality and cost.
3. Grounding strategy. RAG sources, retrieval method (see lesson 14), tool contracts.
4. Evaluation plan. Task success rate, tool-call correctness, off-task rate, hallucination rate on held-out dialogs.

Refuse to recommend a pure-LLM agent for any destructive action (payments, account deletion, data modification) without a structured confirmation flow. Refuse to skip the prompt-injection audit if the agent has write access to anything.
```

## 练习

1. **简单。** 实现上述基于规则的响应，包含10个模式，针对咖啡店点单机器人。测试边界情况：重复订单、修改、取消、意图不明确。
2. **中等。** 构建混合 FAQ + 大语言模型（LLM）回退系统。为 SaaS 产品准备50条固定 FAQ 条目，LLM 通过文档网站检索实现回退。对100个真实支持问题测量拒绝率和准确率。
3. **困难。** 实现上述代理循环，集成三种工具（搜索、读取用户数据、发送邮件）。运行包含50个测试场景的评估，包括提示注入攻击。报告任务偏离率、任务失败率和任何注入成功情况。

## 关键词

| 术语 | 俗称 | 实际含义 |
|------|------|---------|
| Intent | 用户意图 | 分类标签（book_flight，reset_password），路由到处理器。 |
| Slot | 插槽 | 机器人需要的信息参数（日期、目的地）。插槽填充是依次询问的过程。 |
| RAG | 检索增强生成 | 先检索相关文档，再将 LLM 的回答基于文档。 |
| Tool call | 工具调用 | LLM 发出结构化调用，包含名称和参数，运行时执行并返回结果。 |
| Agent loop | 代理循环 | 控制器在 LLM 调用与工具调用间交替运行，直到任务完成。 |
| Prompt injection | 提示注入 | 用户发起的恶意输入，试图覆盖系统提示内容。 |

## 延伸阅读

- [Weizenbaum (1966). ELIZA — A Computer Program For the Study of Natural Language Communication](https://web.stanford.edu/class/cs124/p36-weizenabaum.pdf) — 最初的基于规则聊天机器人论文。
- [Thoppilan et al. (2022). LaMDA: Language Models for Dialog Applications](https://arxiv.org/abs/2201.08239) — 谷歌晚期神经聊天机器人论文，就在大型语言模型代理普及之前。
- [Yao et al. (2022). ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) — 首次命名“代理循环”模式的论文。
- [Anthropic's guide on building effective agents](https://www.anthropic.com/research/building-effective-agents) — 2024年的生产级指导，2026年依然适用。
- [Greshake et al. (2023). Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection](https://arxiv.org/abs/2302.12173) — 提示注入攻击论文。
- [OWASP Top 10 for LLM Applications 2025 — LLM01 Prompt Injection](https://genai.owasp.org/llmrisk/llm01-prompt-injection/) — 促使提示注入成为首要安全关切的排名。
- [AWS — Securing Amazon Bedrock Agents against Indirect Prompt Injections](https://aws.amazon.com/blogs/machine-learning/securing-amazon-bedrock-agents-a-guide-to-safeguarding-against-indirect-prompt-injections/) — 包括 Plan-Verify-Execute 和用户确认流程的实用编排层防御。
- [EchoLeak (CVE-2025-32711)](https://www.vectra.ai/topics/prompt-injection) — 来自间接提示注入的经典零点击数据泄露漏洞（CVE）案例。强调写访问代理需要运行时防护。
