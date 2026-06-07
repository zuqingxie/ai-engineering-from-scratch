# 群聊与发言者选择

> AutoGen GroupChat 和 AG2 GroupChat 在 N 个代理之间共享一个对话；通过选择函数（LLM、大轮询（round-robin）或自定义）决定下一个发言者。这是涌现型多代理对话的典型模式——代理并不知道它们在静态图中的角色，它们只是对共享池做出反应。AutoGen v0.2 的 GroupChat 语义在 AG2 分支中得以保留；AutoGen v0.4 则重写为事件驱动的 actor 模型。微软于 2026 年 2 月将 AutoGen 置于维护模式，并将其与 Semantic Kernel 合并为 Microsoft Agent Framework（2026 年 2 月 RC）。GroupChat 原语在 AG2 和 Microsoft Agent Framework 中均存活——学一次，随处可用。

**类型：** 学习 + 构建  
**语言：** Python（标准库）  
**先决条件：** 阶段 16 · 04（原语模型）  
**时间：** ~60 分钟

## 问题

静态图（LangGraph）在工作流程已知时非常好用。真正的对话不是静态的：有时编码者问审阅者，有时问研究者，有时问写作者。硬编码每个可能的交接会产生指数级边数。你需要*代理对共享池做出反应*，并用某个函数决定谁接下来发言。

这正是 AutoGen GroupChat 所做的。

## 概念

### 结构示意

```text
              ┌─── 共享池 ────┐
              │   m1  m2  m3  ...  │
              └─────────┬──────────┘
                        │（所有人都读取所有消息）
      ┌───────┬─────────┼─────────┬───────┐
      ▼       ▼         ▼         ▼       ▼
    代理 A  代理 B   代理 C   代理 D   选择器
                                           │
                                           ▼
                                  “下一个发言者 = C”
```

每个代理都能看到每条消息。每轮调用一次选择函数，决定谁接下来发言。

### 三种选择器类型

**大轮询（Round-robin）。** 固定周期。确定性。随代理数线性扩展但不考虑上下文——即使主题是法律审核，编码者也有其发言机会。

**LLM 选择。** 调用 LLM，读取最近的消息池，返回最佳的下一个发言者。上下文感知，但较慢：每轮都要调用 LLM。AutoGen 默认选择。

**自定义。** 你用任意逻辑写的 Python 函数。典型用法是 LLM 选择带后备规则（例如，“编码者后总是给核验者发言”）。

### ConversableAgent API

```python
agent = ConversableAgent(
    name="coder",
    system_message="You write Python.",
    llm_config={...},
)
chat = GroupChat(agents=[coder, reviewer, tester], messages=[])
manager = GroupChatManager(groupchat=chat, llm_config={...})
```

`GroupChatManager` 持有选择器。当代理完成一轮发言后，管理者调用选择器返回下一个代理。循环直到满足终止条件。

### 终止条件

三个常用模式：

- **最大轮数。** 对总轮数设硬上限。
- **“TERMINATE” 令牌。** 代理可发送哨兵消息；管理者检测到该消息则停止。
- **目标达成检测。** 轻量级校验器每轮运行，完成时停止对话。

### AutoGen → AG2 分支及微软 Agent Framework 合并

2025 年初，微软围绕事件驱动 actor 模型对 AutoGen（v0.4）进行重大重写。社区分叉出 AutoGen v0.2 的 GroupChat 语义形成 AG2，保留了早期采用者集成的 API。

2026 年 2 月，微软宣布对 AutoGen 进入维护模式，事件驱动 actor 模型合并进 **Microsoft Agent Framework**（2026 年 2 月 RC，现与 Semantic Kernel 合并）。GroupChat 概念在两个发展路径中均存续，具体实现略有差异。AG2 是兼容 v0.2 代码的首选上游。

### 何时适用 GroupChat

- **涌现型对话。** 不想预先写死所有可能的下一个发言者。
- **角色多变任务。** 编码者问研究者，研究者问档案员，档案员再问编码者。流程不是 DAG。
- **探索性问题解决。** 想“头脑风暴会议”，而非“流水线”。

### 何时不适用

- **严格确定性。** LLM 选择器可能不一致。同一提示不同次运行，可能有不同的下一个发言者。
- **谄媚级联。** 代理总是顺从说话最自信的那个。需明确对抗提示。
- **上下文膨胀。** 每个代理读所有消息；十轮后上下文巨大。使用投影（第15课）限定视野。
- **热门发言者。** 某代理因选择器偏好其专长而主导对话。可添加发言平衡机制作选择器特性。

### 群聊 vs 监督者（Supervisor）

相同原语，默认行为差异：

- 监督者：一个代理做规划，其他执行。选择器即“问规划者做什么”。
- 群聊：所有代理为平级；选择器基于共享池的函数。

两者均用第04课的四个原语。群聊默认使用 LLM 选择编排和全池共享状态。

## 构建它

`code/main.py` 从零实现了一个标准库版本的 GroupChat。三代理（coder、reviewer、manager）、大轮询和 LLM 选择变体，以及遇到 `TERMINATE` 令牌终止的逻辑。

演示打印对话记录和两种选择器决策轨迹。

运行：

```text
python3 code/main.py
```

## 使用它

`outputs/skill-groupchat-selector.md` 配置了一个 GroupChat 选择器针对给定任务——大轮询 vs LLM 选择 vs 自定义，以及所用的选择器输入（最近消息、代理专长、轮数统计等）。

## 交付它

清单：

- **最大轮数限制。** 必须有。典型任务 10-20 轮。
- **发言平衡度指标。** 追踪每代理发言轮数；失衡超阈值报警。
- **终止令牌。** `TERMINATE` 或专门的校验代理。
- **投影或作用域记忆。** ~10 条消息后，考虑给每代理有限作用域视图防止上下文爆炸。
- **选择器日志。** LLM 选择版本必须记录选择器输入和决策，否则无法调试。

## 练习

1. 运行 `code/main.py`。比较大轮询和 LLM 选中下的对话。每种中哪个代理主导？
2. 在选择器中增加“每代理最大发言数”规则。对话记录受到什么影响？
3. 实现目标达成终止：当审阅者返回“approved”时停止。通常在最大轮数前几轮触发？
4. 阅读 AutoGen GroupChat 正式文档（https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/design-patterns/group-chat.html）。识别 `GroupChatManager` 默认使用的选择器。
5. 阅读 AG2 仓库（https://github.com/ag2ai/ag2），比较 v0.2 的 GroupChat 与 v0.4 事件驱动版本。v0.4 最具体增加了什么性质（吞吐量、容错性、可组合性）？

## 关键词

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| GroupChat | “多个代理在一个聊天房间” | 共享消息池 + 选择函数。AutoGen / AG2 原语。 |
| Speaker selection | “谁接着说” | 选出下一个代理的函数。可能是大轮询、LLM 选中或自定义。 |
| GroupChatManager | “会议主持人” | AutoGen 组件，拥有选择器并控制轮次循环。 |
| ConversableAgent | “基础代理” | AutoGen 基类；能发送和接收消息的代理。 |
| Termination token | “停止词” | 哨兵字符串（一般为 `TERMINATE`），结束聊天。 |
| Hot speaker | “某代理主导对话” | 失败模式，选择器反复选同一个代理。 |
| Context bloat | “消息池无限增长” | 每代理阅读全部历史消息，上下文随轮次扩大。 |
| Projection | “作用域视图” | 角色特定的共享池子视图，防止上下文爆炸。 |

## 延伸阅读

- [AutoGen 群聊文档](https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/design-patterns/group-chat.html) — 参考实现  
- [AG2 仓库](https://github.com/ag2ai/ag2) — 社区版 AutoGen v0.2 续作  
- [微软 Agent Framework 文档](https://microsoft.github.io/agent-framework/) — 合并继任者，2026 年 2 月 RC  
- [AutoGen v0.4 发布说明](https://microsoft.github.io/autogen/stable/) — 事件驱动 actor 模型重写细节
