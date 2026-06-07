# 生产运行时：队列、事件、定时任务

> 生产代理运行在六种运行时形态上：请求-响应、流式、持久执行、基于队列的后台、事件驱动和定时调度。选择框架之前先选定形态。可观测性在每种形态中都至关重要。

**类型：** 学习  
**语言：** Python（标准库）  
**先决条件：** 第14阶段 · 13（LangGraph）、第14阶段 · 22（语音）  
**时长：** ~60分钟

## 学习目标

- 命名六种生产运行时形态并对应到各自的框架/产品模式。  
- 解释为何持久执行（LangGraph）对于长时间任务重要。  
- 描述事件驱动运行时及其适用场景中的Claude托管代理。  
- 说明多步代理中可观测性作为承重因素的意义。

## 问题描述

生产代理失败的方式不会在Jupyter笔记本中显现：第37步骤的网络超时，用户在语音通话中途挂断，定时任务因机器重启而中断，后台工作者内存溢出。运行时形态决定了哪些失败是可承受的。

## 概念介绍

### 请求-响应（Request-response）

- 同步HTTP，用户等待完成。  
- 只适用于短任务（<30秒）。  
- 技术栈：Agno（Python + FastAPI）、Mastra（TypeScript + Express/Hono/Fastify/Koa）。  
- 可观测性：标准HTTP访问日志 + OTel跨度。

### 流式（Streaming）

- 通过SSE或WebSocket实现渐进输出。  
- LiveKit扩展至WebRTC支持语音/视频（第22课）。  
- 技术栈：任何支持流式的框架 + 支持SSE/WS的前端。  
- 可观测性：每块时间统计、首令牌延迟、尾端延迟。

### 持久执行（Durable execution）

- 每步后保存状态检查点；失败自动恢复。  
- AutoGen v0.4的actor模型将失败隔离到单个代理（第14课）。  
- LangGraph的核心差异化（第13课）。  
- 当步骤数未知且恢复代价高时至关重要。

### 基于队列/后台（Queue-based / background）

- 任务进入队列，工作者接取，结果通过Webhook或Pub/Sub回流。  
- 长时间任务必需（每任务几十至数百步，参见Anthropic的computer use公告）。  
- 技术栈：Celery（Python）、BullMQ（Node）、SQS + Lambda（AWS）、自定义。  
- 可观测性：队列深度、每任务延迟分布、死信队列（DLQ）大小。

### 事件驱动（Event-driven）

- 代理订阅触发器：新邮件、Pull Request创建、cron触发。  
- Claude托管代理天然支持该模式（第17课）。  
- CrewAI Flows结构化事件驱动的确定性工作流（第15课）。  
- 可观测性：触发源、事件到启动延迟、代理响应延迟。

### 定时调度（Scheduled）

- 类Cron的代理定期运行。  
- 配合持久执行，失败的夜间任务可在下一个周期继续。  
- 技术栈：Kubernetes CronJob + 持久框架；托管服务（Render cron, Vercel cron）。

### 2026年部署模式

- **CrewAI Flows** 用于事件驱动生产。  
- **Agno** 无状态FastAPI用于Python微服务。  
- **Mastra** 服务器适配器（Express, Hono, Fastify, Koa）用于嵌入式。  
- **Pipecat Cloud / LiveKit Cloud** 管理语音（第22课）。  
- **Claude 托管代理** 用于托管的长时异步。

### 可观测性是承重的

没有OpenTelemetry GenAI跨度（第23课）和Langfuse/Phoenix/Opik后端（第24课），你无法调试在第40步骤失败的多步代理。这对生产来说不是可选项。它决定了“快速调试”与“从头重放并增加日志”的差别。

### 生产运行时失败的来源

- **形态选择错误。** 为5分钟任务选择请求-响应。用户挂断，工作者积压，重试加剧。  
- **无死信队列。** 队列工作者无DLQ，失败任务消失无踪。  
- **后台工作不透明。** 后台代理无迹导出，失败只有用户举报后才知。  
- **跳过持久状态。** 任何运行时间超过30秒且无法承受重启的任务都需持久执行。

## 实践操作

`code/main.py` 是一个多形态标准库示例：

- 请求-响应端点（普通函数）。  
- 流式处理器（生成器）。  
- 带死信队列的队列工作者。  
- 事件触发注册表。  
- 类Cron的调度器。

运行它：

```bash
python3 code/main.py
```

输出：五个跟踪展示相同任务上各形态的表现。相同代理逻辑，不同外壳。持久执行（第六形态）由第13课的LangGraph检查点讲解。

## 使用建议

- **请求-响应** 用于聊天式用户体验。  
- **流式** 用于渐进响应。  
- **持久** 用于长时间任务。  
- **队列** 用于批量/异步/长运行。  
- **事件** 用于代理的响应性。  
- **定时** 用于系统维护（内存整理、评估、成本报告）。

## 部署建议

`outputs/skill-runtime-shape.md` 为任务选择运行时形态并连接可观测性需求。

## 练习

1. 将你的第01课ReAct循环移植到所有六种形态中。哪个形态适合哪个产品界面？  
2. 为基于队列的演示添加死信队列。模拟10%的任务失败；统计死信队列大小。  
3. 编写一个定时触发的评估代理，每晚针对当天的20条顶级跟踪运行。  
4. 实现带背压的流式：当客户端变慢时暂停代理。它如何影响回合预算？  
5. 阅读Claude托管代理文档。什么时候会将自托管的长时任务迁移到托管服务？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| Request-response（请求-响应） | “同步” | 用户等待；仅适合短任务 |
| Streaming（流式） | “SSE / WS” | 渐进输出；更佳用户体验；每块延迟可观测 |
| Durable execution（持久执行） | “失败恢复” | 状态检查点；从最后一步重启 |
| Queue-based（基于队列） | “后台任务” | 生产者 / 工作者池 / 死信队列 |
| Event-driven（事件驱动） | “触发型” | 代理响应外部事件 |
| DLQ（死信队列） | “死信队列” | 失败任务的存放区 |
| Claude Managed Agents（Claude托管代理） | “托管运行框架” | Anthropic托管的长时异步，支持缓存和压缩 |

## 拓展阅读

- [LangGraph 概览](https://docs.langchain.com/oss/python/langgraph/overview) — 持久执行细节  
- [Claude托管代理概览](https://platform.claude.com/docs/en/managed-agents/overview) — 托管长时异步  
- [Anthropic，介绍computer use](https://www.anthropic.com/news/3-5-models-and-computer-use) — “每任务几十至数百步”  
- [AutoGen v0.4（微软研究院）](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — actor模型故障隔离
