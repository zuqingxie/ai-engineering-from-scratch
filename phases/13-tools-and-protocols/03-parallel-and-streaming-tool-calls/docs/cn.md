# 并行工具调用和工具流式处理

> 三个独立的天气查询串行执行，需要三次往返。并行运行时，总耗时缩短为单次最长调用时间。每个前沿提供商现在都在单轮对话中发出多个工具调用。收益显著，但实现细节复杂。本课涵盖两个方面：并行扇出和流式参数重组，重点讲解 id 关联陷阱。

**类型：** 构建  
**语言：** Python（标准库，线程池 + 流式处理框架）  
**前置条件：** 第 13 阶段 · 02（函数调用深度解析）  
**时长：** 约 75 分钟

## 学习目标

- 解释为什么存在 `parallel_tool_calls: true` 以及何时禁用它。
- 在并行扇出时，将流式参数块关联到正确的工具调用 id。
- 将部分的 `arguments` 字符串重新组装成完整 JSON，避免提前解析错误。
- 运行三城市天气基准测试，演示串行与并行延迟差异。

## 问题描述

不进行并行调用时，代理回答“班加罗尔、东京和苏黎世的天气如何”时，执行过程如下：

```text
用户 -> LLM  
LLM -> 调用 get_weather(Bengaluru)  
主机 -> 执行器运行，回复结果  
LLM -> 调用 get_weather(Tokyo)  
主机 -> 执行器运行，回复结果  
LLM -> 调用 get_weather(Zurich)  
主机 -> 执行器运行，回复结果  
LLM -> 最终文本回答
```

共三次 LLM 往返，每次还需承担执行器的延迟。大约是理想时间的 4 倍。

开启并行调用时：

```text
用户 -> LLM  
LLM -> 调用 get_weather(Bengaluru); 调用 get_weather(Tokyo); 调用 get_weather(Zurich)  
主机 -> 并行运行三个执行器，回复三个结果  
LLM -> 最终文本回答
```

只需一次 LLM 往返。执行器时间为三者中最长的，不是总和。OpenAI、Anthropic 和 Gemini 的生产基准测试显示扇出工作负载的时钟时间减少了 60% 到 70%。

代价是关联复杂度。当三次调用乱序完成时，结果必须包含匹配的 `tool_call_id`，这样模型才能正确对应。流式结果时，必须在执行前将部分参数片段组装成完整 JSON。Gemini 3 添加了唯一 id，部分解决了现实问题——两个对同一工具的并行调用不易区分。

## 概念说明

### 启用并行

- **OpenAI。** 默认 `parallel_tool_calls: true`。设置为 `false` 强制串行。
- **Anthropic。** 通过 `disable_parallel_tool_use: false`（Claude 3.5 及以上版本默认开启）。设置为 `true` 禁用并行。
- **Gemini。** 始终支持并行；`tool_config.function_calling_config.mode = "AUTO"` 让模型决定。

当工具有执行顺序依赖（如先 `create_file` 再 `write_file`）、某个调用的输出为另一个调用的输入，或速率限制器无法承受扇出时，应禁用并行。

### Id 关联

模型发出的每个调用都有一个 `id`。主机返回的对应结果必须包含相同 id，否则结果将模糊不清。

- **OpenAI。** 每个工具角色消息内包含 `tool_call_id`。
- **Anthropic。** 每个 `tool_result` 块内含 `tool_use_id`。
- **Gemini。** 每个 `functionResponse` 内含 `id`（Gemini 3 及以上版本；Gemini 2 则通过名称匹配，导致同名的并行调用混淆）。

### 并发执行调用

主机会在独立线程、协程或远程工作者上运行每个调用的执行器。最简单的框架是线程池；生产中通常用 asyncio 的 `asyncio.gather` 或结构化并发。完成顺序不可预测——id 是唯一标识。

常见错误：按照调用列表顺序回复结果，而非根据完成顺序。这通常可行，因为模型只关心 `tool_call_id`，但若结果丢失或重复，乱序提交会增加调试难度。建议按照完成顺序回复，并明确携带 id。

### 流式工具调用

模型流式返回参数时，`arguments` 分片分多块到达。三个并行调用的三个参数流在传输线上交织，你需要为每个 id 单独累积。

按提供商区分：

- **OpenAI。** 每个分片位于 `choices[0].delta.tool_calls[i].function.arguments`（部分字符串），分片携带 `index`（在调用列表中的位置）。按索引累积，首次出现时读取 `id`，在 `finish_reason = "tool_calls"` 时解析 JSON。
- **Anthropic。** 流事件流程是 `message_start`，之后每个含 `tool_use` 类型块有一个 `content_block_start`（含 id、名称、空输入）。`content_block_delta` 事件携带 `input_json_delta` 分片，`content_block_stop` 结束每块。
- **Gemini。** `streamFunctionCallArguments`（Gemini 3 及以上）发出带 `functionCallId` 的分片，保证调用间流畅交织。Gemini 3 之前，流式只返回一次完整调用。

### 部分 JSON 和提前解析陷阱

不能在完整数据到齐之前解析 `arguments`，诸如 `{"city": "Beng` 等部分 JSON 是无效的，会抛错。正确的解析时机是提供商的调用结束信号：OpenAI 的 `finish_reason = "tool_calls"`，Anthropic 的 `content_block_stop`，或 Gemini 的流结束事件。只有到达这些时机，才执行 `json.loads`。更健壮的做法是用增量 JSON 解析器，随着结构完成产出事件；OpenAI 流教程推荐此方案，以实现实时“思考中”体验。用大括号计数判断完整性靠不住（字符串内或转义的括号易产生误判），只能做调试辅助手段。

### 乱序完成

```text
调用_A：快速 API，最先返回
调用_B：慢速 API，第二返回
调用_C：中速 API，最后返回
```

主机回复仍需引用 id：

```text
[{role: "tool", tool_call_id: "call_A", content: ...},  
 {role: "tool", tool_call_id: "call_B", content: ...},  
 {role: "tool", tool_call_id: "call_C", content: ...}]
```

回复顺序对 OpenAI 和 Anthropic 的正确性无影响。Gemini 允许任意顺序，只要 id 匹配。

### 基准测试：串行 vs 并行

`code/main.py` 框架模拟三个延迟分别为 400、600 和 800 毫秒的执行器。串行运行总耗时约 1800 毫秒。并行运行时间为 max(400, 600, 800) = 800 毫秒。差异为常量，与工具数量成正比，数量越多节省越明显。

实际情况：并行调用会对下游 API 施加压力。对受速率限制的服务进行十路扇出将导致失败。第 13 阶段 · 17 涵盖网关级别的背压；重试语义将在后续阶段加入。

### 流式扇出时钟优化

若模型本身流式返回，可在任一调用参数完成时即刻启动执行，无需等待所有调用完成。此为 OpenAI 文档中的优化，非所有 SDK 均支持。本课程框架实现了该功能：模拟流一旦产出完整参数对象，主机立刻触发对应调用。

## 使用方法

`code/main.py` 分两部分。第一部分使用 `concurrent.futures.ThreadPoolExecutor` 以串行和并行方式运行三个模拟天气调用，并打印时钟耗时。第二部分重播一个伪流响应——三个并行调用的 `arguments` 片段交织在一条流上——通过 `StreamAccumulator` 按 id 重组。无 LLM，无网络，仅实现重组逻辑。

重点关注：

- 串行计时约 1.8 秒，使用相同假延迟，并行计时约 0.8 秒。
- 累加器处理乱序分片，按 id 缓存，确保仅在完整 JSON 到达时才解析。
- 执行器在任一 id 参数完成时即触发，不必等所有流结束。

## 交付成果

本课生成文件 `outputs/skill-parallel-call-safety-check.md`。给定工具注册表，技能会审计哪些工具安全并行，哪些有顺序依赖，哪些会超出下游速率限制，最后返回修改后的注册表，带每工具的 `parallel_safe` 标志。

## 练习题

1. 运行 `code/main.py`，调整模拟延迟。确认并行与串行时间比约为 `max/sum`（实际运行因线程调度、序列化及框架开销略有偏差）。在哪种延迟分布下，并行优势不明显？

2. 扩展累加器，支持“调用在流中途被取消”情况，通过丢弃缓存并发出 `cancelled` 事件处理。哪个提供商文档明确说明了此情况？查看 Anthropic 的 `content_block_stop` 以及 OpenAI 的 `finish_reason: "length"` 行为。

3. 用 `asyncio.gather` 替换线程池，做性能基准比较。若执行器做真实 I/O，应该观察到异步的小幅优势，因为上下文切换成本低。

4. 选两个不应并行的工具（如先 `create_file` 后 `write_file`）。给注册表加入 `ordering_dependency` 图，基于该图限制并行扇出。这是最低限度的依赖感知调度机制，未来代理工程阶段将形式化。

5. 阅读 OpenAI 并行函数调用章节和 Anthropic 的 `disable_parallel_tool_use` 文档。找出 Anthropic 建议禁用并行的真实工具类型。（提示：同一资源上的关键性变更操作。）

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|------------|
| 并行工具调用 | “单轮扇出” | 模型在单条助手消息中发出多个工具调用 |
| `parallel_tool_calls` | “OpenAI 的开关” | 启用或禁用多调用发射 |
| `disable_parallel_tool_use` | “Anthropic 的否定” | 关闭并行的标签；默认启用并行 |
| 工具调用 id | “关联句柄” | 每次调用的标识，结果消息需回响相同 id |
| 累加器 | “流缓冲” | 每 id 的字符串缓冲区，用于部分 `arguments` 片段 |
| 乱序完成 | “最快先完成” | 并行调用完成顺序不可预测，id 是关联纽带 |
| 依赖图 | “顺序约束” | 工具之间有输入输出依赖，不能并行 |
| 提前解析陷阱 | “JSON.parse 出错” | 试图解析不完整的 `arguments` 字符串 |
| `streamFunctionCallArguments` | “Gemini 3 特性” | 带唯一 id 的流式参数分片 |
| 完成顺序回复 | “无需等全部完成” | 结果按到达顺序回复，按 id 键控 |

## 深入阅读

- [OpenAI — Parallel function calling](https://platform.openai.com/docs/guides/function-calling#parallel-function-calling) — 默认行为及可选关闭的开关  
- [Anthropic — Tool use: implementing tool use](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/implementing-tool-use) — `disable_parallel_tool_use` 与结果批处理  
- [Google — Gemini function calling parallel section](https://ai.google.dev/gemini-api/docs/function-calling) — Gemini 3 的 id 关联并行调用  
- [OpenAI — Streaming responses with tools](https://platform.openai.com/docs/api-reference/responses-streaming) — OpenAI 流式调用的参数分片重组  
- [Anthropic — Streaming messages](https://docs.anthropic.com/en/api/messages-streaming) — `content_block_delta` 携带 `input_json_delta`
