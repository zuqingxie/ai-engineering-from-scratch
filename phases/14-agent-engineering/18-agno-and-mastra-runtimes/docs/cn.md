# Agno 和 Mastra：生产运行时

> Agno（Python）和 Mastra（TypeScript）是 2026 年的生产运行时组合。Agno 目标是实现微秒级的 agent 实例化和无状态的 FastAPI 后端。Mastra 在 Vercel AI SDK 底座上提供 agents、tools、workflows、统一模型路由和复合存储。

**类型：** 学习  
**语言：** Python、TypeScript  
**先决条件：** 第14阶段 · 01（Agent Loop）、第14阶段 · 13（LangGraph）  
**时长：** ~45分钟  

## 学习目标

- 识别 Agno 的性能目标以及它们何时重要。
- 说出 Mastra 的三大原语——Agents、Tools、Workflows，以及支持的服务器适配器。
- 解释为何无状态的会话范围 FastAPI 后端是推荐的 Agno 生产路径。
- 在给定技术栈中选择 Agno 还是 Mastra（Python优先 vs TypeScript优先）。

## 问题

LangGraph、AutoGen、CrewAI 框架较重。那些想要“只要 agent 循环、快速运行在我的运行时里”的团队会选择 Agno（Python）或 Mastra（TypeScript）。两者都放弃了一些框架级别的原语，以换取原始的速度和与周边技术栈更紧密的契合。

## 概念

### Agno

- Python 运行时，前称 Phi-data。
- “无图、无链、无复杂模式——只有纯粹的 python。”
- 官方文档中的性能目标：约 2μs agent 实例化，约 3.75 KiB 内存每个 agent，约支持 23 个模型供应商。
- 生产路径：无状态会话范围 FastAPI 后端。每个请求启动一个新的 agent；会话状态存储在数据库中。
- 原生支持多模态（文本、图像、音频、视频、文件）和 agentic 检索增强生成（RAG）。

当你每秒有成千上万个短生命周期 agent（如聊天合流、评测流水线）时，这些速度指标非常重要。当一个 agent 运行时间达 10 分钟时，重要性则降低。

### Mastra

- TypeScript，构建于 Vercel AI SDK 之上。
- 三大原语：**Agents**、**Tools**（Zod 类型定义）、**Workflows**。
- 统一模型路由器——截至 2026 年 3 月，支持 94 个供应商的 3,300+ 模型。
- 复合存储：内存、workflow、观察性数据分别对接不同后端；大规模观察建议用 ClickHouse。
- Apache 2.0 许可，`ee/` 目录在源码可得的企业许可下。
- 支持 Express、Hono、Fastify、Koa 服务器适配器；一流 Next.js 和 Astro 集成。
- 提供 Mastra Studio（localhost:4111）进行调试。
- GitHub 2.2 万星，npm 每周下载 30 万以上，1.0 版本发布于 2026 年 1 月。

### 定位

二者都不试图成为 LangGraph。它们竞争点在于：

- **语言契合度。** Agno 更适合 Python 优先团队；Mastra 更适合 TypeScript 优先。
- **运行时易用性。** Agno 近乎零开销；Mastra 与 Vercel 生态高度集成。
- **观察性。** 两者均能与 Langfuse/Phoenix/Opik（第24课）集成，但 Mastra Studio 是一手产品。

### 选择时机

- **Agno** — Python 后端，许多短生命周期 agent，强性能要求，FastAPI 团队。
- **Mastra** — TypeScript 后端，Next.js / Vercel 部署，统一的多供应商模型路由，Zod 类型化工具。
- **LangGraph**（第13课）— 当持久状态和明确图形推理比原始速度更重要时。
- **OpenAI / Claude Agent SDK** — 当你想要供应商产品化形态时（第16-17课）。

### 何时该模式出错

- **为了性能而性能。** 当工作负载是每次请求一个慢速 agent 调用，却选择 Agno 因为“2μs”听起来好。开销不是瓶颈。
- **生态锁定。** Mastra 针对 Vercel 的集成在 Vercel 上是优点，其他环境则是缺点。
- **企业许可混淆。** Mastra 的 `ee/` 目录是源码可得的，但非 Apache 2.0。如果准备 fork，请仔细阅读授权条款。

## 构建它

本课主要是比较——单一代码实例无法公平展示两框架。参见 `code/main.py` 了解一个并排示例：最简“运行 agent，流输出，持久会话”的流程两版本（Agno 风格与 Mastra 风格）。

运行：

```bash
python3 code/main.py
```

两条结构不同但功能等价的跟踪。

## 使用它

- **Agno** — 需要速度和 FastAPI 形态的 Python 后端。
- **Mastra** — 拥有多供应商和工作流原语的 TypeScript 后端。
- 两者都配备第一方观察性钩子，且均可与 Langfuse 集成。

## 交付它

`outputs/skill-runtime-picker.md` 根据技术栈、延迟预算和运营形态，选择 Agno、Mastra、LangGraph 或供应商 SDK。

## 练习

1. 阅读 Agno 文档。将标准库的 ReAct 循环（第01课）迁移到 Agno。哪些内容消失？哪些内容保留？
2. 阅读 Mastra 文档。同样迁移该循环到 Mastra。工具类型定义（Zod vs 无）有什么变化？
3. 基准测试：测量你栈中的 agent 实例化延时。Agno 的 2μs 对你工作负载重要吗？
4. 设计迁移方案：如果你之前用 Python 运行 CrewAI，迁移到 Agno 会出现哪些断裂？
5. 阅读 Mastra `ee/` 许可条款。哪些限制会影响开源分支？

## 关键术语

| 术语 | 口语表述 | 实际含义 |
|-------|---------|---------|
| Agno | “快速 Python agents” | 无状态的会话范围 agent 运行时 |
| Mastra | “运行于 Vercel AI SDK 的 TypeScript agents” | Agents + Tools + Workflows + 模型路由 |
| 统一模型路由器 | “多供应商访问” | 一客户端访问 94 供应商的 3,300+ 模型 |
| 复合存储 | “多个后端” | 内存/工作流/观察性分别对应不同存储 |
| Mastra Studio | “本地调试器” | localhost:4111 用于 agent 内省的 UI |
| 源码可得 | “非开源” | 许可允许查看源码但限制商业使用 |

## 延伸阅读

- [Agno Agent Framework 文档](https://www.agno.com/agent-framework) — 性能目标，FastAPI 集成
- [Mastra 文档](https://mastra.ai/docs) — 原语、服务器适配器、模型路由
- [LangGraph 概览](https://docs.langchain.com/oss/python/langgraph/overview) — 有状态图替代方案
- [Comet Opik](https://www.comet.com/site/products/opik/) — Mastra 集成引用的观察性比较
