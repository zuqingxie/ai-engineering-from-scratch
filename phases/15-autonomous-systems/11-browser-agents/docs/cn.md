# 浏览器代理和长时域网页任务

> ChatGPT 代理（2025 年 7 月）将 Operator 和深度研究合并为一个浏览器/终端代理，并将 BrowseComp 设定的新状态（SOTA）提高至 68.9%。OpenAI 于 2025 年 8 月 31 日关闭了 Operator —— 产品层面合并。Anthropic 收购 Vercept 后，将 Claude Sonnet 在 OSWorld 的表现从不足 15% 提升至 72.5%。WebArena-Verified（ServiceNow，ICLR 2026）解决了原始 WebArena 大约 11.3 个百分点的假阴性率，并发布了包含 258 个任务的 Hard 子集。数据真实，攻击面也是真实的：OpenAI 的应急负责人公开表示，间接提示注入（indirect prompt injection）进入浏览器代理“不是能完全修补的漏洞”。2025–2026 年有文档记录的攻击包括：Tainted Memories（Atlas CSRF）、HashJack（Cato Networks）以及 Perplexity Comet 的一键劫持。

**类型：** 学习  
**语言：** Python（标准库，间接提示注入攻击面模型）  
**先决条件：** 第 15 阶段·10（权限模式），第 15 阶段·01（长时域代理）  
**时间：** 约 45 分钟

## 问题

浏览器代理是一个长时域代理，它会读取不受信任的内容并采取重要操作。代理访问的每个页面都是用户未写入的输入。每个页面上的每个表单都是潜在的命令通道。2025–2026 年的攻击案例表明这不是假设：Tainted Memories 允许攻击者通过特制页面将恶意指令绑定到代理的内存；HashJack 将命令隐藏在代理访问的 URL 片段中；Perplexity Comet 的劫持只需一键即可发生。

防御形势不容乐观。OpenAI 的应急负责人坦言了这个事实：间接提示注入“不是可以完全修补的漏洞”。这是因为攻击存在于代理的读取—执行边界，该边界在架构上模糊不清 —— 理论上模型读到的每个 token 都可以被理解为指令。

本课命名了攻击面，介绍了基准测试场景（BrowseComp、OSWorld、WebArena-Verified），并模拟了一个最小的间接提示注入场景，方便你在第 14 和 18 课中推理实际防御。

## 概念

### 2026 年形势，以每个系统一句话概括

**ChatGPT 代理（OpenAI）。** 2025 年 7 月发布。统一了 Operator（浏览）和深度研究（多小时研究）。2025 年 8 月 31 日关闭独立的 Operator。BrowseComp SOTA 达 68.9%；在 OSWorld 和 WebArena-Verified 上表现稳健。

**Claude Sonnet + Vercept（Anthropic）。** Anthropic 收购 Vercept，专注于计算机使用能力。将 Claude Sonnet 在 OSWorld 的成绩从低于 15% 提升到 72.5%。Claude 计算机使用以工具 API 形式发布。

**Gemini 3 Pro 带浏览器使用（DeepMind）。** 浏览器使用功能中支持计算机使用操作；FSF v3（2026 年 4 月，第 20 课）专门追踪机器学习研发领域的自主能力。

**WebArena-Verified（ServiceNow，ICLR 2026）。** 解决了一个已知问题：原始 WebArena 的假阴性率约为 11.3%（被标记为失败但实际上已完成的任务）。Verified 版本采用人工策划的成功标准重新评分，并增加了包含 258 个任务的 Hard 子集（ICLR 2026 论文，openreview.net/forum?id=94tlGxmqkN）。

### BrowseComp、OSWorld 和 WebArena 比较

| 基准 | 测量内容 | 时间范围 |
|---|---|---|
| BrowseComp | 在开放网页中在时间压力下寻找具体事实 | 几分钟 |
| OSWorld | 代理操作完整桌面（鼠标、键盘、终端） | 数十分钟 |
| WebArena-Verified | 模拟网页站点中的事务性任务 | 几分钟 |
| Hard 子集 | WebArena-Verified 中涉及多页面状态转换的任务 | 数十分钟 |

不同的维度。BrowseComp 高分说明代理能找出事实；但不代表代理能预订航班。OSWorld 成绩则更贴近“在我的桌面环境中能否正常工作”。WebArena-Verified 倾向于“能否完成一个工作流”。实际产品决策需要匹配任务分布的基准。

### 命名攻击面

1. **间接提示注入（Indirect prompt injection）。** 不受信任的页面内容包含指令。代理读取它们并执行。公开例子：2024 Kai Greshake 等人，2025 年 Tainted Memories 论文，2026 年 HashJack（Cato Networks）。
2. **URL 片段 / 查询注入。** 抓取的 URL 的 `#fragment` 或查询字符串中包含命令。不会可视化渲染，但仍在代理上下文中。
3. **内存绑定攻击。** 页面指示代理写入持久内存（第 12 课涉及持久状态）。下一会话时，内存触发有效载荷，无需明显触发器。
4. **针对已认证会话的 CSRF 形攻击。** Tainted Memories 类攻击：代理已登录，攻击者页发出状态变更请求，代理携带用户 cookie 执行。
5. **一键劫持。** 一个视觉上无害的按钮承载载荷，代理执行。Comet 类攻击。
6. **代理主机面内容安全策略漏洞。** 渲染和工具层本身可能是攻击向量；浏览器嵌入浏览器代理堆栈的攻击面宽广。

### 为什么“无法完全修补”

攻击与代理能力同构。代理必须读取不受信任内容才能完成工作。代理读取的任何内容都可能包含指令。代理执行的任何指令都可能与用户实际请求不符。防御措施（信任边界、分类器、工具白名单、关键操作的人类干预 HITL）能够增加攻击成本，减少影响范围，但不能根本关闭这类攻击。

此理理由与 Lob 定理（第 8 课）相同：代理无法证明下一个 token 是安全的，只能建立检测非法 token 的系统。

### 实际部署的防御姿态

- **读/写边界。** 读取永远不产生后果。写入（提交表单、发布内容、调用有副作用工具）需要新的人类审批，前提是启动内容来自信任边界之外。
- **每任务工具白名单。** 代理能浏览，但除非明确允许，不得发起转账等操作。第 13 课讲预算。
- **会话隔离。** 浏览器代理会话仅带有作用域凭证运行。不含生产认证，无个人邮箱。每次 HTTP 请求均被记录审计。
- **内容净化器。** 抓取的 HTML 移除已知恶意模式后才并入模型上下文。（减少低级攻击，但难阻高级载荷。）
- **关键操作人类干预（HITL）。** 先提议后执行模式（第 15 课）。
- **内存金丝雀令牌。** 如果内存条目触发，用户可见（第 14 课）。

## 使用方法

`code/main.py` 模拟一个简易浏览器代理运行在三个合成页面上。一个页面正常，一个页面在可见文字中含有直接提示注入代码块，一个页面含有 URL 片段注入（不可见但在代理上下文内）。脚本展示了：（a）天真代理的行为，（b）读/写边界的拦截，（c）净化器的拦截，（d）两者均未拦截的攻击。

## 部署指南

`outputs/skill-browser-agent-trust-boundary.md` 规划了建议的浏览器代理部署：涉及的信任区域，授权写入范围，以及首次运行前应具备的防御措施。

## 练习

1. 运行 `code/main.py`。找出哪一种攻击被净化器拦截但读/写边界未拦截，哪一种攻击仅被读/写边界拦截。
2. 扩展净化器，检测一类 HashJack 风格的 URL 片段注入。在正常含片段的 URL 上测量误报率。
3. 选取一个你熟悉的真实浏览器代理工作流（例如“预订航班”）。列出所有读取和写入操作。标记哪些写操作需要 HITL，原因是什么。
4. 阅读 WebArena-Verified ICLR 2026 论文。确定原始 WebArena 哪类任务评分不可靠，并解释 Verified 子集如何解决该问题。
5. 设计一个浏览器代理场景下的内存金丝雀。你会存储什么，存储在哪里，什么情况报警？

## 关键词

| 术语 | 常见说法 | 实际意义 |
|---|---|---|
| 间接提示注入 | “恶意页面文本” | 代理读取的不可信页面内容隐藏指令，代理执行这些指令 |
| Tainted Memories | “内存攻击” | 代理将攻击者提供的指令写入持久内存，下一会话触发 |
| HashJack | “URL 片段攻击” | 有效载荷藏在 URL 片段/查询字符串中，处于代理上下文但未渲染 |
| 一键劫持 | “恶意按钮” | 可视化可操作元素携带后续有效载荷，代理执行 |
| BrowseComp | “网页搜索基准” | 在开放网页上找具体事实；分钟级时域 |
| OSWorld | “桌面基准” | 完整系统控制，多步骤 GUI 任务 |
| WebArena-Verified | “修正网页任务基准” | ServiceNow 重新标注的 WebArena，含 Hard 子集 |
| 读/写边界 | “副作用门槛” | 读取无后果；写操作若内容超出信任边界需新审批 |

## 延伸阅读

- [OpenAI — Introducing ChatGPT agent](https://openai.com/index/introducing-chatgpt-agent/) — Operator 和深度研究合并；BrowseComp SOTA。
- [OpenAI — Computer-Using Agent](https://openai.com/index/computer-using-agent/) — Operator 衍生及成为 ChatGPT 代理的架构。
- [Zhou 等 — WebArena](https://webarena.dev/) — 原始基准测试。
- [WebArena-Verified (OpenReview)](https://openreview.net/forum?id=94tlGxmqkN) — ICLR 2026 修正子集论文。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — 包含计算机使用代理的攻击面讨论。
