# Prompt Caching 和 Context Caching（上下文缓存）

> 你的 system prompt（系统提示）有 4,000 个 tokens。你的 RAG context（检索增强生成上下文）有 20,000 个 tokens。你每次请求都发送这两个部分。你也为这两部分付费——每次请求都付。Prompt caching（提示缓存）让服务商在他们端保持该前缀热态，重复使用时只收取正常费用的 10%。正确使用时，它可降低 50–90% 推理成本及 40–85% 首个 token 延迟。

**类型：** 构建  
**语言：** Python  
**先修知识：** Phase 11 · 01（Prompt Engineering（提示工程）），Phase 11 · 05（Context Engineering（上下文工程）），Phase 11 · 11（Caching and Cost（缓存与成本））  
**时间：** 约 60 分钟  

## 问题

一个编码代理在每次对话轮次都会向 Claude 发送相同的 15,000-token 系统提示。20 轮请求，输入 token 价格为 $3/M，单纯输入成本就是 $0.90 —— 还没算用户的实际消息。每天 10,000 次对话，则账单达到 $9,000/天，而文本内容从不变更。

你无法缩减提示内容而不牺牲质量。你也不能不发送它 —— 模型每轮都需要。这时唯一做法就是停止为服务商已经见过的前缀支付全价。

这就是 prompt caching。Anthropic 于 2024 年 8 月推出（2025 年带 1 小时可拓展 TTL 版本），OpenAI 同年晚些时候自动支持，Google 则在 Gemini 1.5 旁边推出显式 context caching。现在这三家均把它作为旗舰模型的一级功能。

## 概念

![Prompt caching: write once, read cheap](../assets/prompt-caching.svg)

**机制。** 请求的 prefix（前缀）如果与最近一次请求的一致，服务商会直接返回上次运行时 KV-cache（键值缓存），而非重新编码 tokens。首次付少量写入溢价，之后每次大幅读折扣。

**2026 年三种服务商实践。**

| 服务商 | API 形式 | 缓存命中折扣 | 写入溢价 | 默认 TTL | 最小缓存单位 |
|---------|-----------|--------------|-----------|-------------|---------------|
| Anthropic | 内容块上显式 `cache_control` 标记 | 输入费 9 折（90% off） | 加价 25% | 5 分钟（可扩展至 1 小时） | 1,024 tokens（Sonnet/Opus），2,048（Haiku） |
| OpenAI | 自动前缀检测 | 输入费 5 折（50% off） | 无 | 最多 1 小时（尽力保证） | 1,024 tokens |
| Google（Gemini） | 显式 `CachedContent` API | 存储费用；读取约正常输入费的 25% | 按 token·小时计存储费 | 用户设定（默认 1 小时） | 4,096 tokens（Flash），32,768（Pro） |

**恒等式。** 三家都只缓存前缀。如果请求之间任何 token 有差异，第一处不同后均为未命中。请将*稳定*部分置顶，*可变*部分置底。

### 缓存友好型布局

```text
[system prompt]          <-- 缓存该部分
[tool definitions]       <-- 缓存该部分
[few-shot examples]      <-- 缓存该部分
[retrieved documents]    <-- 复用时缓存，否则不缓存
[conversation history]   <-- 缓存至上一轮
[current user message]   <-- 不缓存（每次不同）
```

如果破坏这个顺序——例如，将用户消息放在 system prompt 上方、在 few-shot 中穿插动态检索——缓存永远命中不了。

### 收支平衡计算

Anthropic 的 25% 写入溢价意味着缓存块须至少被读取两次才能实现净节省。1 次写入 + 1 次读取，每次请求平均成本为 0.675 倍（节省 32%）；1 次写入 + 10 次读取，平均成本为 0.205 倍（节省 80%）。经验法则：缓存你期望在 TTL 范围内被复用至少 3 次的内容。

## 构建流程

### 第 1 步：Anthropic 显式标记的提示缓存

```python
import anthropic

client = anthropic.Anthropic()

SYSTEM = [
    {
        "type": "text",
        "text": "You are a senior Python reviewer. Follow the rubric exactly.\n\n" + RUBRIC_15K_TOKENS,
        "cache_control": {"type": "ephemeral"},
    }
]

def review(code: str):
    return client.messages.create(
        model="claude-opus-4-7",
        max_tokens=1024,
        system=SYSTEM,
        messages=[{"role": "user", "content": code}],
    )
```

`cache_control` 标记告诉 Anthropic 缓存该块 5 分钟。该窗口内复用命中，过期后重写。

**响应中的用量字段：**

```python
response = review(code_a)
response.usage
# InputTokensUsage(
#     input_tokens=120,
#     cache_creation_input_tokens=15023,   # 按 1.25 倍计费
#     cache_read_input_tokens=0,
#     output_tokens=340,
# )

response_b = review(code_b)
response_b.usage
# cache_creation_input_tokens=0
# cache_read_input_tokens=15023           # 按 0.1 倍计费
```

在 CI 中检查两个字段——如果 `cache_read_input_tokens` 始终为零，说明你的缓存键（cache keys）在漂移。

### 第 2 步：一小时可扩展 TTL

对于长时批处理，默认 5 分钟 TTL 可能在作业间过期。设置 `ttl`：

```python
{"type": "text", "text": RUBRIC, "cache_control": {"type": "ephemeral", "ttl": "1h"}}
```

1 小时 TTL 导致写入溢价翻倍（由 25% 涨至 50%），但只要批次复用超过 5 次，很快收回成本。

### 第 3 步：OpenAI 自动缓存

OpenAI 不需任何配置。任何超过 1,024 tokens 的前缀如果匹配最近请求，自动享受 50% 折扣。

```python
from openai import OpenAI
client = OpenAI()

resp = client.chat.completions.create(
    model="gpt-5",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},   # 稳定且较长
        {"role": "user", "content": user_msg},
    ],
)
resp.usage.prompt_tokens_details.cached_tokens  # 折扣部分
```

同样适用缓存友好布局规则。有两点会使 OpenAI 缓存失效（Anthropic 不会）：修改 `user` 字段（作为缓存键），以及工具顺序变动。

### 第 4 步：Gemini 显式上下文缓存

Gemini 把缓存视为一级对象，你新建并命名它：

```python
from google import genai
from google.genai import types

client = genai.Client()

cache = client.caches.create(
    model="gemini-3-pro",
    config=types.CreateCachedContentConfig(
        display_name="rubric-v3",
        system_instruction=RUBRIC,
        contents=[FEW_SHOT_EXAMPLES],
        ttl="3600s",
    ),
)

resp = client.models.generate_content(
    model="gemini-3-pro",
    contents=["Review this code:\n" + code],
    config=types.GenerateContentConfig(cached_content=cache.name),
)
```

Gemini 按 token·小时计存储费，读取成本约为正常输入的 25%。适合多天内多会话重复使用同一巨量提示。

### 第 5 步：生产中计量命中率

查看 `code/main.py`，其中模拟了支持三家服务商的账户系统，跟踪写入/读取/未命中次数，计算每 1,000 请求的混合成本。根据目标命中率控制发布 —— 大多数 Anthropic 生产配置预热后读取比例应超 80%。

## 2026 年仍然存在的坑

- **顶部带动态时间戳。** `"Current time: 2026-04-22 15:30:02"` 出现在 system prompt 顶部。每次请求均未命中。应把时间戳放在缓存断点以下。
- **工具顺序变动。** 工具顺序必须稳定，字典顺序在部署间若变动，将导致缓存全部失效。
- **文本近似重复。** “You are helpful.” 和 “You are a helpful assistant.” 只差一个字节即完全未命中。
- **区块太小。** Anthropic 要求至少 1,024 tokens（Haiku 为 2,048）。小于此的区块默默不缓存。
- **盲目看成本面板。** 将“输入 token”细分为缓存与非缓存，否则流量波动会误导为缓存节省。

## 使用指南

2026 年缓存策略：

| 场景 | 方案选择 |
|-----------|------|
| 稳定的 10k+ system prompt，多轮对话 | Anthropic `cache_control`，5 分钟 TTL |
| 批处理作业，30 分钟以上复用前缀 | Anthropic，使用 `ttl: "1h"` |
| 无自定义基础设施的 GPT-5 无服务器端点 | OpenAI 自动（保持前缀稳定且较长） |
| 多天复用巨量代码/文档语料 | Gemini 显式 `CachedContent` |
| 跨服务商回退方案 | 按同一缓存前缀布局保持一致，多服务商均可命中 |

结合 Phase 11 · 11 的 semantic caching（语义缓存）用于用户消息层：prompt caching 处理 *token 完全一致* 的复用，semantic caching 处理 *语义一致* 的复用。

## 发版指南

保存到 `outputs/skill-prompt-caching-planner.md`：

```markdown
---
name: prompt-caching-planner
description: 设计缓存友好的提示布局，并选择合适的服务商缓存模式。
version: 1.0.0
phase: 11
lesson: 15
tags: [llm-engineering, caching, cost]
---

给定一个提示（system + tools + few-shot + retrieval + history + user）和使用画像（每小时请求数、TTL 需求、服务商），输出：

1. 布局。重排段落，标记唯一缓存断点；说明哪段稳定，哪段易变。
2. 服务商模式。Anthropic `cache_control`、OpenAI 自动、或 Gemini `CachedContent`。根据 TTL 和复用模式给出选择理由。
3. 收支平衡。预期 TTL 内写入对应的读取次数；用数学公式说明净成本与未缓存对比。
4. 验证方案。CI 断言第二次相同请求时 `cache_read_input_tokens > 0`；仪表板按缓存和非缓存 token 细分。
5. 失败模式。列出缓存未命中的前三大原因（动态时间戳、工具顺序变化、文本近似重复）及防范措施。

拒绝发版包含动态字段置于断点以上。拒绝启用 1 小时 TTL 但没有令写入溢价 2 倍回本的复用计数。
```

## 练习

1. **简单。** 针对 Claude 运行一个 10 轮对话，system prompt 为 5,000 token。分别不使用和使用 `cache_control`，报告两次的输入 token 账单。
2. **中级。** 编写测试工具，给定提示模板和请求日志，估计四种方案（Anthropic 5m、Anthropic 1h、OpenAI 自动、Gemini 显式）的命中率和成本节省。
3. **困难。** 写一个布局优化器：输入带有标记 `stable=True/False` 字段列表的提示，重写提示将唯一缓存断点放置到最大缓存友好位置且不丢失信息。在真实 Anthropic 端点上验证。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| Prompt caching | “让长提示成本变低” | 重用服务端 KV-cache 缓存匹配前缀；重复输入 token 享 50-90% 折扣。 |
| `cache_control` | “Anthropic 的标记” | 内容块属性，声明“此处之前均可缓存”；`{"type": "ephemeral"}`。 |
| Cache write | “付写入溢价” | 首次写入缓存请求；Anthropic 按输入费约 1.25 倍计费，OpenAI 免费。 |
| Cache read | “享折扣” | 后续请求匹配前缀；按 10%（Anthropic），50%（OpenAI），约 25%（Gemini）计费。 |
| TTL | “缓存存活时间” | 缓存维持热态秒数；Anthropic 默认 5 分钟（可扩展至 1 小时），OpenAI 尽力保证最长期 1 小时，Gemini 用户配置。 |
| Extended TTL | “Anthropic 1 小时缓存” | `{"type": "ephemeral", "ttl": "1h"}`；写入溢价翻倍，但批次复用收益更多。 |
| Prefix match | “缓存未命中原因” | 只有前缀所有 token 字节完全相同缓存才命中。 |
| Context caching（Gemini） | “显式缓存” | Google 的命名、计存储费的缓存对象；适合多日多会话复用大语料。 |

## 延伸阅读

- [Anthropic — Prompt caching（提示缓存）](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) — `cache_control`，1小时 TTL，盈亏平衡表。
- [OpenAI — Prompt caching（提示缓存）](https://platform.openai.com/docs/guides/prompt-caching) — 自动前缀匹配。
- [Google — Context caching（上下文缓存）](https://ai.google.dev/gemini-api/docs/caching) — `CachedContent` API 和存储定价。
- [Anthropic engineering — Prompt caching（提示缓存）用于长上下文工作负载](https://www.anthropic.com/news/prompt-caching) — 原始发布帖，含延迟数据。
- Phase 11 · 05 (Context Engineering（上下文工程）) — 在哪里切分提示以便缓存命中。
- Phase 11 · 11 (Caching and Cost（缓存与成本）) — 将提示缓存与用户消息的语义缓存配对。
- [Pope 等, "Efficiently Scaling Transformer Inference（有效扩展Transformer推理）" (2022)](https://arxiv.org/abs/2211.05102) — KV-cache（KV缓存）内存模型是提示缓存向用户暴露的；解释了为什么缓存的前缀重读成本约是重新计算的 1/10。
- [Agrawal 等, "SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills（通过分块预填充搭载解码实现高效大模型推理）" (2023)](https://arxiv.org/abs/2308.16369) — 预填充是提示缓存的快捷阶段；本文解释了为什么缓存命中时 TTFT（首令牌延迟）显著下降而 TPOT（总处理时间）不受影响。
- [Leviathan 等, "Fast Inference from Transformers via Speculative Decoding（通过推测性解码实现Transformer快速推理）" (2023)](https://arxiv.org/abs/2211.17192) — 提示缓存与推测性解码（speculative decoding）、Flash Attention 和 MQA/GQA 一起，作为降低推理成本曲线的杠杆；阅读本文可了解其他三种方法。
