# FIPA-ACL 和言语行为的传承

> 在 MCP 之前，在 A2A 之前，有 FIPA-ACL。2000 年，IEEE 智能物理代理基金会（Foundation for Intelligent Physical Agents）批准了一种代理通信语言，包括二十个 performative（行为动词）、两种内容语言和一套交互协议——合同网（contract net）、订阅/通知（subscribe/notify）、请求-当（request-when）。因为本体论（ontology）开销对网络来说太重，该语言逐渐在业界衰退，但大语言模型（LLM）复兴多代理系统的过程中，悄悄地用没有形式语义的思路重新实现了这些理念：JSON 合约取代了 performative，自然语言替代了本体论。本课认真解读 FIPA-ACL，帮助你看清 2026 年的协议决策中哪些是重新发明，哪些是创新，以及当前浪潮将重新遇到 2000 年代已解决的问题。

**类型：**学习  
**语言：**Python（标准库）  
**先决条件：**阶段 16 · 01（为何多代理）  
**时间：**约 60 分钟

## 问题

2026 年的代理协议格局异常繁忙：MCP 用于工具，A2A 用于代理，ACP 用于企业审计，ANP 用于去中心化信任，NLIP 用于自然语言内容，还有 CA-MCP 和数十个研究提案。每个规范都自称是基础。

诚实地说，它们中的大多数都在重新发现一个二十多年前的决策树。Austin（1962）和 Searle（1969）的言语行为理论（speech-act theory）告诉我们“话语即行为”。KQML（1993）将其变成了一个线协议。FIPA-ACL（2000 年批准）成为参考标准：二十个 performative，内容语言 SL0/SL1，合同网和订阅-通知的交互协议。JADE 和 JACK 是 Java 的参考平台。由于本体论开销过大且网络技术占优，2010 年左右该努力逐渐衰退。

当你看到 MCP 的 `tools/call`、A2A 的任务生命周期或 CA-MCP 的共享上下文存储，你所见的是 FIPA 决策的一次更轻、更原生 JSON 风格的重述。了解传统会告诉你两件事：哪些所谓的新“创新”其实是旧知的再发明，哪些旧有失败模式会被新规范重新遇见。

## 概念

### 言语行为，一段话说明

Austin 注意到，某些句子不是在描述世界——它们改变世界。“我承诺。”“我请求。”“我声明。”他称这些为 performative utterances（施为话语）。Searle 将其形式化为五类：断言（assertive）、指令（directive）、承诺（commissive）、表达（expressive）、声明（declarative）。KQML（Finin 等，1993）使之对软件代理可操作：消息由 performative（动作）加内容（动作主题）组成。FIPA-ACL 纠正了 KQML 的不足，标准化了二十个 performative。

### 二十个 FIPA performative（部分列表）

| Performative（行为动词） | 意图 |
|---|---|
| `inform` | “我告诉你 P 为真” |
| `request` | “我请求你做 X” |
| `query-if` | “P 是否为真？” |
| `query-ref` | “X 的值是多少？” |
| `propose` | “我提议我们做 X” |
| `accept-proposal` | “我接受该提议” |
| `reject-proposal` | “我拒绝该提议” |
| `agree` | “我同意做 X” |
| `refuse` | “我拒绝做 X” |
| `confirm` | “我确认 P 为真” |
| `disconfirm` | “我否认 P” |
| `not-understood` | “你的消息无法解析” |
| `cfp` | “呼叫针对 X 的提案” |
| `subscribe` | “当 X 变化时通知我” |
| `cancel` | “取消正在进行的 X” |
| `failure` | “我尝试做 X 失败了” |

完整列表见 `fipa00037.pdf`（FIPA ACL 消息结构）。重点不是记忆它——而是每个 performative 都对应一个大语言模型协议最终重用的原语。

### 标准的 FIPA-ACL 消息

```text
(inform
  :sender       agent1@platform
  :receiver     agent2@platform
  :content      "((price IBM 83))"
  :language     SL0
  :ontology     finance
  :protocol     fipa-request
  :conversation-id   conv-42
  :reply-with   msg-17
)
```

七个字段构成协议信封；一个字段（`content`）承载有效载荷。其余字段就是你每次为 JSON 协议增加重试、线程和本体时都会重新发明的。

### 两个传统平台

**JADE**（Java Agent DEvelopment framework，1999–2020s）是最常用的 FIPA 合规运行时。代理继承基类、交换 ACL 消息、在容器内运行，并通过“行为”（behaviors）协调。其交互协议库包含合同网、订阅-通知、请求-当和提议-接受。

**JACK**（Agent Oriented Software，商业）强调基于 BDI（信念-欲望-意图）模型的推理，建立在 FIPA 消息上。更形式化，但采用较少。

两者随着 Web 堆栈占领多代理用例而衰落。MCP 和 A2A 是 2026 年的运行时“容器”。

### FIPA 衰落原因

- **本体过重。**FIPA 需要共享本体解析 `content`，达成一致本体是多年标准过程。网络直接用 HTTP + JSON。
- **没人用的形式语义。**SL（语义语言）提供严格的真值条件，多数生产系统用自由格式内容，忽视该形式化。
- **工具锁闭。**JADE 只支持 Java；JACK 为商业。多语言团队绕开二者。
- **互联网赢得了协议层。**REST、JSON-RPC、gRPC 替代了 ACL 的传输。

### LLM 复兴是 FIPA-lite

对比 FIPA 的 `request` 和 MCP `tools/call`：

```text
(request                                {
  :sender  agent1                         "jsonrpc": "2.0",
  :receiver tool-server                   "method":  "tools/call",
  :content "(lookup stock IBM)"           "params":  {"name":"lookup_stock",
  :ontology finance                                   "arguments":{"symbol":"IBM"}},
  :conversation-id c42                    "id": 42
)                                        }
```

相同信封，不同语法。都包含：发送者、接收者、意图、载荷、关联 ID。彼此非革命，而是同一设计的不同权衡。

刘等人 2025 年综述（“A Survey of Agent Interoperability Protocols: MCP, ACP, A2A, ANP”，arXiv:2505.02279）明确了渊源：MCP 对应工具使用言语行为，A2A 对应代理对等言语行为，ACP 对应审计轨迹言语行为，ANP 对应去中心化身份扩展。新协议是用 JSON 语法、语义松散的 ACL 后代。

### 权衡，直白说

**FIPA 给而现代规范弃的：**

- 形式语义——你能证明 `inform` 蕴含发送者信念内容为真。
- 标准 performative 目录——不用再争论“要不要 `cancel`？”。
- 数十年交互协议模式——合同网、订阅-通知、提议-接受，伴随已知正确性属性。

**现代规范给而 FIPA 没有的：**

- 与现代工具兼容的原生 JSON 载荷。
- LLM 可理解、无须硬编码本体的自然语言内容。
- 网络堆栈传输（HTTP，SSE，WebSocket）。
- 通过自描述文档实现能力发现（MCP `listTools`，A2A Agent Card）。

语义意图更宽松，方便实现。这就是核心权衡。

### 值得移植的交互协议

FIPA 定义了约 15 种交互协议。三种值得带入 LLM 多代理系统：

1. **合同网协议（Contract Net Protocol，CNP）。** 管理者发出 `cfp`（呼叫提案）；竞标者回应 `propose`；管理者接受或拒绝。经典任务市场模式（阶段 16 · 16 谈判）。
2. **订阅/通知（Subscribe/Notify）。** 订阅者发送 `subscribe`；发布者在主题变化时发 `inform`。2026 年的事件总线范式。
3. **请求-当（Request-When）。** “当条件 Y 成立时做 X。”带前置条件的延迟动作。2026 的类比是耐久工作流引擎的延期任务（阶段 16 · 22 生产扩展）。

都能清晰映射到现代消息队列、HTTP + 轮询或 SSE 流。

### 放弃本体论后会破坏什么

没有共享本体，代理只能从自然语言内容推断意义。2026 年记录的失败模式是 **语义漂移（semantic drift）**：两个代理用同一个词（如 `"customer"`）表示细微不同的概念，接收代理根据错误理解采取行动，且无模式验证器发现。FIPA 的本体要求会在解析时拒绝此类消息。

非全本体论的缓解措施：

- `content` 上加 JSON Schema——线上拒绝结构错误。
- 类型化工件（A2A）——拒绝错误模态。
- 信封中明确 performative——即使内容是自然语言，也让意图明确无歧义。

### 2026 规范，映射到言语行为传承

| 现代规范 | FIPA 类比 | 保留 | 弃用 |
|---|---|---|---|
| MCP `tools/call` | `request` | 明确意图，关联 ID | 形式语义，本体论 |
| MCP `resources/read` | `query-ref` | 明确意图，关联 ID | 形式语义 |
| A2A 任务生命周期 | 合同网 + 请求-当 | 异步生命周期，状态转换 | 形式完整性保证 |
| A2A 流事件 | 订阅/通知 | 异步推送 | 类型谓词订阅 |
| CA-MCP 共享上下文 | 黑板（Hayes-Roth 1985） | 多写者共享内存 | 逻辑一致性模型 |
| NLIP | 自然语言内容 | LLM 原生 | 模式 |

从表中自上而下看，模式是：保留结构性原语，放弃形式主义，让 LLM 掩盖歧义。

## 构建

`code/main.py` 实现了一个纯标准库的 FIPA-ACL 翻译器。它编解码标准 ACL 信封，演示每个 MCP / A2A 消息形态如何归约到相同的七个字段。演示内容：

- 将五个 MCP 风格和 A2A 风格的消息编码为 FIPA-ACL。
- 将 FIPA-ACL 解码回现代等价形式。
- 运行一个简单的合同网谈判，含一个管理者与三个竞标者，使用 `cfp`、`propose`、`accept-proposal`、`reject-proposal`。

运行：

```text
python3 code/main.py
```

输出为并排跟踪，展示每个现代消息在 2026 JSON 形式和 FIPA-ACL 形式的对比，然后是合同网竞标的往返过程。同样的协议原语在往返中保留，唯有语法不同。

## 使用

`outputs/skill-fipa-mapper.md` 是一个技能，它读取任何代理协议规范并生成 FIPA-ACL 的映射。采纳新协议前用它答疑：“这真的是新协议，还是带 JSON 语法的 `inform`？”

## 交付

不要复活 FIPA-ACL，复活它的清单：

- 每条消息的意图原语（performative）是什么？
- 请求-响应和取消是否有关联 ID？
- 是否有明确的内容语言（JSON-RPC、纯文本、结构化类型工件）？
- 交互协议是否为一等公民，还是从零重实现合同网？
- 两代理对内容意义不一致时（语义漂移）会怎样？

在新协议推向生产前，必须文档化这五个问题。

## 练习

1. 运行 `code/main.py`，观察往返编码。识别哪个 FIPA performative 对应 `tools/call`、`resources/read` 和 A2A 任务创建。
2. 在合同网演示中添加 `cancel` performative，允许管理者在竞标中途撤回任务。`cancel` 解决了哪些仅靠重试无法解决的失败场景？
3. 阅读 FIPA ACL 消息结构（http://www.fipa.org/specs/fipa00037/）第 4.1–4.3 节。选择本课未覆盖的一个 performative，描述其现代 JSON-RPC 对应。
4. 阅读 Liu 等人，arXiv:2505.02279。列表 MCP、A2A、ACP、ANP 各自保留和放弃的 FIPA performative 家族。
5. 为你自己的系统中 `request` performative 的 `content` 字段设计一个最小 JSON-Schema。该模式提供哪些纯自然语言不具备的优势？其代价是什么？

## 关键词

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| Speech act | “一个完成某事的发话” | Austin/Searle：将发话视为行动。ACL（代理通信语言）的理论基础。 |
| FIPA | “那个老旧的 XML 东西” | IEEE 智能物理代理基金会。2000 年标准化了 ACL。 |
| ACL | “代理通信语言” | FIPA 的信封格式：performative（施为动词）+ content（内容）+ metadata（元数据）。 |
| Performative | “动词” | 消息的意图类别：`inform`、`request`、`propose`、`cfp` 等。 |
| KQML | “FIPA 的前身” | 知识查询与操作语言（1993）。更简单，范围更窄。 |
| Ontology | “共享词汇” | 内容语言所谈论概念的形式定义。 |
| SL0 / SL1 | “FIPA 内容语言” | 语义语言级别 0 和 1 —— 形式内容语言家族。 |
| Contract Net（合同网） | “任务市场” | 管理者发出 cfp（征求提案）；竞标者提出方案；管理者接受。典型交互协议。 |
| Interaction protocol（交互协议） | “消息模式” | 一序列已知正确性的 performative（施为动词）：request-when、subscribe-notify 等。 |

## 扩展阅读

- [Liu et al. — 代理互操作协议综述：MCP、ACP、A2A、ANP](https://arxiv.org/html/2505.02279v1) — 2025 年权威综述，连接现代规范与 FIPA 传统
- [FIPA ACL 消息结构规范（fipa00037）](http://www.fipa.org/specs/fipa00037/) — 2000 年批准的信封格式
- [FIPA 交际行为库规范（fipa00037）](http://www.fipa.org/specs/fipa00037/) — 完整的 performative（施为动词）目录
- [MCP 规范 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25) — 现代的工具使用对应 `request`/`query-ref`
- [A2A 规范](https://a2a-protocol.org/latest/specification/) — 现代代理对等的合同网与订阅通告等价物
