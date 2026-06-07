# Prompt 缓存与语义缓存经济学

> **定价快照日期为 2026-04。** 以下数值声称基于本课程发布时抓取的供应商费率表；在下游引用之前请验证链接文档。

> 缓存在两个层面发生。L2（提供商级别）prompt/prefix 缓存重用重复前缀的 attention KV——Anthropic 的 prompt 缓存文档宣传，在长提示上可节省高达 90% 成本和 85% 延迟；Claude 3.5 Sonnet缓存读取价格为 $0.30/M，刷新输入为 $3.00/M，5分钟 TTL，1小时 TTL 选项写入费用有 2 倍溢价（docs.anthropic.com，2026-04）。OpenAI prompt 缓存自动应用于提示≥1024 tokens，缓存输入价格大约比新输入便宜 90%（platform.openai.com，2026-04）；具体缓存费率依赖实时费率卡。L1（应用级别）语义缓存在嵌入相似性命中时完全跳过 LLM。供应商“95% 准确率”指匹配正确率，不是命中率——实际生产命中率从 10%（开放式聊天）到 70%（结构化 FAQ）不等；无供应商官方基线，视作社区遥测数据而非保证。生产坑点：并行化杀死缓存（在第一次缓存写入前发起 N 个并行请求会使费用膨胀数倍），前缀内动态内容则完全阻断缓存命中。ProjectDiscovery 报告通过将动态文本移出缓存前缀，命中率从 7% 提升到 74%（2025-11）。

**类型：** 学习  
**语言：** Python（标准库，简易两层缓存模拟器）  
**先决条件：** Phase 17 · 04 (vLLM Serving Internals), Phase 17 · 06 (SGLang RadixAttention)  
**时间：** 约 60 分钟

## 学习目标

- 区分 L2 prompt/prefix 缓存（提供商端 KV 重用）与 L1 语义缓存（相似提示绕过 LLM）。
- 阐释 Anthropic `cache_control` 显式标记及两种 TTL 选项（5分钟与 1小时）及其价格倍数。
- 根据命中率、prompt/response 组合和 token 价格计算预期月度节省。
- 说明导致账单膨胀 5-10 倍的并行化反模式及导致命中率崩溃的动态内容反模式。

## 问题描述

你为你的 RAG 服务添加了 prompt 缓存，账单却保持平稳。你测量命中率，只有 7%。看似静态的提示实际上并非如此——系统提示包括精确到分钟的当前日期、请求 ID 和为多样性做的随机示例重排序。每个请求写入新的缓存条目，读取为零。

另外，你的代理为每个用户问题并行调用十个工具。所有十个请求在第一次缓存写完成前同时到达提供商。十次写，零次读。账单是启用缓存预期费用的 5-10 倍。

缓存是一种协议，不是一个标记。有两层缓存，两个不同的失败模式。

## 概念讲解

### L2 — 提供商端 prompt/prefix 缓存

提供商保存可缓存前缀的 attention KV，并复用相同前缀的下一次请求。写入一次付费，读取几乎免费。

**Anthropic（Claude 3.5 / 3.7 / 4 系列）** ：请求中显式的 `cache_control` 标记。你指定哪些块可缓存。TTL：5分钟（写入成本为基础的 1.25 倍）或 1 小时（写入成本为基础的 2 倍）。缓存读取：Claude 3.5 Sonnet 为 $0.30/M，刷新输入 $3.00/M——便宜 10 倍（docs.anthropic.com，2026-04）。不同模型费率不同（Opus / Haiku 单独公布）；务必核对实时价格页面。

**OpenAI**：对≥1024 tokens 的提示自动缓存（platform.openai.com，2026-04）。无显式标志。缓存输入价格约为当前 gpt-4o/gpt-5 费率表新输入的 10%。文档和发布说明未公布官方命中率基线；社区报告在 30-60% 之间，依赖细致的提示设计。监测 `usage.cached_tokens` 以评估自身。

**Google（Gemini）**：通过显式 API 实现上下文缓存；1M token 上下文意味着缓存收益更高。

**自托管（vLLM，SGLang）**：Phase 17 · 06 涉及 RadixAttention——同样的模式在你自己的算力上运行。

### L1 — 应用级别语义缓存

调用 LLM 之前，对 prompt 进行哈希、嵌入，查找相似的缓存请求（余弦相似度高于阈值，通常为 0.95+）。命中则返回缓存响应，未命中调用 LLM 并缓存结果。

开源方案：Redis Vector Similarity、GPTCache、Qdrant。商业方案：Portkey Cache、Helicone Cache。

供应商准确率声明指的是返回的缓存响应在语义上合适的频率——不是命中率。实际生产命中率：

- 开放式聊天：10-15%。
- 结构化 FAQ / 支持：40-70%。
- 代码问题：20-30%（小变动会严重影响命中）。
- 语音代理重复请求：50-80%（语音标准化固定集合）。

### 并行化反模式

你的代理并行调用 10 个工具。所有 10 个请求使用同一 4K token 系统提示。Anthropic 缓存写入是每请求独立的，首次缓存写入在提供商接收到提示后约 300 毫秒完成。请求 2-10 与第 1 个请求几乎同一时间到达，看到全是缓存未命中。你付 10 次写入溢价，0 次读优惠。

修复：顺序优先批处理——先单独发请求 1，等其缓存写入完成后，并行触发请求 2-10。给第一个工具调用增加 300 毫秒，节省账单 5-10 倍。

### 动态内容反模式

你的系统提示类似：

```text
你是一个乐于助人的助手。当前时间是 14:32:17。
用户 ID: abc123。今天是星期二……
```

每次请求唯一，每次写入，无命中。

修复：将所有真正静态内容放入可缓存前缀，将动态内容放在缓存边界外：

```text
[cacheable]
你是一个乐于助人的助手。 [规则，示例，指令]
[/cacheable]
[dynamic, 不缓存]
当前时间：14:32:17。用户：abc123。
```

ProjectDiscovery 通过此法将缓存命中率从 7% 提升到 74%，并发布了细节。

### 对夜间任务批处理 + 缓存叠加

批处理 API（Phase 17 · 15）在 24 小时响应期内提供 50% 折扣。缓存输入叠加后进一步约 10 倍折扣。夜间分类、标注和报告生成等工作量可降低至同步未缓存成本的约 10%。

### 你应记住的数值

定价点于 2026-04 根据链接供应商文档抓取，几个月会有漂移——使用前请重新确认。

- Anthropic 缓存读取：Claude 3.5 Sonnet 上 $0.30/M，约为刷新输入的十分之一（docs.anthropic.com）。
- Anthropic 缓存写入溢价：5 分钟 TTL 时为 1.25 倍，1 小时 TTL 时为 2 倍。
- OpenAI 自动缓存：适用于≥1024 tokens 的提示，缓存输入价格约为新输入的 10%（platform.openai.com）。
- 语义缓存命中率（社区报告）：开放式聊天约 10%，结构化 FAQ 高达 70%。非官方供应商基线。
- ProjectDiscovery：通过将动态内容移出前缀，命中率从 7% 提升至 74%（项目博客，2025-11）。
- 并行化反模式：典型报告为当 N 个并行请求全部错失 第一次缓存写入时账单膨胀 5-10 倍。

## 使用示例

`code/main.py` 模拟 L1 + L2 缓存混合工作负载。报告命中率、账单并展示并行化惩罚。

## 发布使用

本课程生成 `outputs/skill-cache-auditor.md`。基于提示模版和流量，审计缓存性并推荐重构方案。

## 练习

1. 运行 `code/main.py`。切换并行化标志。账单变化多少？
2. 你的系统提示包含日期，将其移除。展示移除前后命中率计算。
3. 给定请求到达率，计算 1 小时 TTL（2 倍写入）和 5 分钟 TTL（1.25 倍写入）盈利平衡点。
4. 语义缓存阈值 0.95 命中率 20%，阈值 0.85 命中率 50%，但存在错误缓存响应。选定正确阈值并说明理由。
5. 你为每个用户问题批处理 10 个并行子查询。重写以兼容缓存，且不增加端到端延迟。

## 关键词

| 术语                  | 常见说法          | 实际含义                      |
|-----------------------|-------------------|------------------------------|
| L2 prompt cache       | “prefix cache”     | 提供商保存重复前缀的 KV       |
| `cache_control`        | “Anthropic cache 标记” | 显式标记可缓存的内容块         |
| Cache write premium    | “写入税”          | 第一次缓存写入的额外成本（1.25x 或 2x） |
| L1 semantic cache      | “embedding cache”  | 应用层哈希和嵌入后调用 LLM    |
| GPTCache               | “LLM 缓存库”      | 流行的开源 L1 缓存库          |
| Cache hit rate         | “命中率”          | 来自缓存的请求比例            |
| Parallelization anti-pattern | “N write 陷阱”    | N 个并行请求错失缓存导致重复写入 |
| Dynamic content trap   | “时间提示陷阱”    | 前缀中的动态字节导致命中率零  |
| RadixAttention         | “副本内缓存”      | SGLang 的前缀缓存实现          |

## 拓展阅读

- [Anthropic Prompt Caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) — 官方 `cache_control` 语义和 TTL 说明。  
- [OpenAI Prompt Caching](https://platform.openai.com/docs/guides/prompt-caching) — 自动缓存行为及资格。  
- [TianPan — LLM 语义缓存在生产环境](https://tianpan.co/blog/2026-04-10-semantic-caching-llm-production)  
- [ProjectDiscovery — 用 prompt 缓存降低 59% 语言模型成本](https://projectdiscovery.io/blog/how-we-cut-llm-cost-with-prompt-caching)  
- [DigitalOcean / Anthropic — Prompt 缓存](https://www.digitalocean.com/blog/prompt-caching-with-digital-ocean)
