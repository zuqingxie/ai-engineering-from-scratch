# 功能调用深度解析 — OpenAI、Anthropic、Gemini

> 三个前沿供应商在2024年统一了工具调用循环，然后在其他方面各自不同。OpenAI使用`tools`和`tool_calls`。Anthropic使用`tool_use`和`tool_result`块。Gemini使用`functionDeclarations`和唯一ID关联。本课程并排对比三者，确保针对某一供应商编写的代码在移植到另一供应商时不会出错。

**类型:** 构建  
**语言:** Python（标准库，模式转换器）  
**先决条件:** 第13阶段 · 01（工具接口）  
**时长:** 约75分钟

## 学习目标

- 阐述OpenAI、Anthropic和Gemini功能调用负载（声明、调用、结果）的三种形态差异。
- 将一个工具声明翻译成三种供应商格式，并预测严格模式下约束的区别。
- 在每个供应商中使用`tool_choice`来强制、禁止或自动选择工具调用。
- 了解每个供应商的硬性限制（工具数量、模式深度、参数长度）及超限时的错误特征。

## 问题介绍

功能调用请求的形态因供应商而异。以下为2026年生产环境中的三个具体例子：

**OpenAI 聊天补全 / 响应API。** 传入`tools: [{type: "function", function: {name, description, parameters, strict}}]`。模型响应包含`choices[0].message.tool_calls: [{id, type: "function", function: {name, arguments}}]`，其中`arguments`是必须解析的JSON字符串。严格模式（`strict: true`）通过受限解码来强制模式合规。

**Anthropic 消息API。** 传入`tools: [{name, description, input_schema}]`。响应以`content: [{type: "text"}, {type: "tool_use", id, name, input}]`形式返回，`input`已解析为对象而非字符串。你回复一个包含`{type: "tool_result", tool_use_id, content}`块的新`user`消息。

**Google Gemini API。** 传入`tools: [{functionDeclarations: [{name, description, parameters}]}]`（嵌套在`functionDeclarations`下）。响应为`candidates[0].content.parts: [{functionCall: {name, args, id}}]`，`id`在Gemini 3及以后版本中唯一，用于并行调用关联。回复格式为`{functionResponse: {name, id, response}}`。

同样的循环。字段命名不同，嵌套结构不同，字符串与对象的约定不同，关联机制不同。在OpenAI上写一个天气代理，移植到Anthropic需要两天，移植到Gemini还需一天，仅仅是为了适配底层 plumbing。

本课程构建一个翻译器，将三种格式统一为一个规范工具声明，并在边缘路由。第13阶段 · 17将同样模式推广至LLM网关。

## 概念讲解

### 通用结构

每个供应商都需要五个元素：

1. **工具列表。** 每个工具的名称、描述和输入模式。
2. **工具选择。** 强制指定某工具，禁止工具，或让模型自由选择。
3. **调用发起。** 结构化输出，指定工具及参数。
4. **调用ID。** 将响应关联到正确调用（并行时尤为重要）。
5. **结果注入。** 消息或块将结果返回到对应调用。

### 逐字段形态差异

| 方面 | OpenAI | Anthropic | Gemini |
|--------|--------|-----------|--------|
| 声明包裹 | `{type: "function", function: {...}}` | `{name, description, input_schema}` | `{functionDeclarations: [{...}]}` |
| 模式字段名 | `parameters` | `input_schema` | `parameters` |
| 响应容器 | 助理消息中`tool_calls[]` | 类型为`tool_use`的`content[]` | 类型为`functionCall`的`parts[]` |
| 参数类型 | 字符串化的JSON | 已解析的对象 | 已解析的对象 |
| ID格式 | `call_...`（OpenAI生成） | `toolu_...`（Anthropic） | UUID（Gemini 3+） |
| 结果块 | 角色`tool`，含`tool_call_id` | `user`角色，含`tool_result`和`tool_use_id` | `functionResponse`，含匹配的`id` |
| 强制单工具 | `tool_choice: {type: "function", function: {name}}` | `tool_choice: {type: "tool", name}` | `tool_config: {function_calling_config: {mode: "ANY"}}` |
| 禁止工具 | `tool_choice: "none"` | `tool_choice: {type: "none"}` | `mode: "NONE"` |
| 严格模式 | `strict: true` | 模式即合约（始终强制） | 请求级别的`responseSchema` |

### 实际会遇到的限制

- **OpenAI。** 每次请求最多128个工具。模式深度限制5。参数字符串最大8192字节。严格模式下不允许`$ref`，禁止有重叠的`oneOf`/`anyOf`/`allOf`，`required`中必须列出每个属性。
- **Anthropic。** 每次请求最多64个工具。模式深度理论无限制，但实际有效上限10。无严格模式标志；模式是一种契约，模型倾向遵守。
- **Gemini。** 每次请求最多64个函数。模式类型为OpenAPI 3.0子集（相较JSON Schema 2020-12略有差异）。Gemini 3及以后版本支持并行调用唯一ID。

### `tool_choice` 行为

三种通用模式，不同名称：

- **自动。** 模型自由选择工具或文本，默认模式。
- **必选 / 任意。** 模型必须至少调用一个工具。
- **禁止。** 模型不得调用任何工具。

另有各自唯一模式：

- **OpenAI。** 按工具名强制指定工具。
- **Anthropic。** 按工具名强制指定工具；`disable_parallel_tool_use`标志区分单次与多次调用。
- **Gemini。** `mode: "VALIDATED"` 强制所有响应经过模式校验，无视模型意图。

### 并行调用

OpenAI默认`parallel_tool_calls: true`，可在一条助理消息中发起多次调用。你执行所有调用，回复一个包含各`tool_call_id`条目的批量工具角色消息。Anthropic历史上只支持单次调用（Claude 3.5起 默认`disable_parallel_tool_use: false`支持多次调用）。Gemini 2支持并行但无稳定ID，Gemini 3新增UUID，实现响应顺序无序时的准确关联。

### 流式传输

三者皆支持流式工具调用，但传输格式不同：

- **OpenAI。** 增量传输`tool_calls[i].function.arguments`的增量数据块，直到`finish_reason: "tool_calls"`。
- **Anthropic。** 以块起始 / 块增量 / 块结束事件传输，`input_json_delta`传递部分参数。
- **Gemini。** 新增`streamFunctionCallArguments`（Gemini 3），带`functionCallId`字段，可交错传输多个并行调用数据。

第13阶段 · 03深度探讨并行和流式组装，本课程专注声明和单次调用形态。

### 错误与修复

无效参数错误表现也不同：

- **OpenAI（非严格模式）。** 模型返回`arguments: "{坏的JSON}"`，你的JSON解析失败，注入错误信息后重试。
- **OpenAI（严格模式）。** 解码时即验证；无效JSON不可能出现，但可能有`refusal`拒绝块。
- **Anthropic。** `input`可能含非预期字段；模式只是参考，需服务器端验证。
- **Gemini。** OpenAPI 3.0特性：对象字段上的`enum`被静默忽略；需自行校验。

### 翻译器模式

代码中的规范工具声明示例（形态由你决定）：

```python
Tool(
    name="get_weather",
    description="Use when ...",
    input_schema={"type": "object", "properties": {...}, "required": [...]},
    strict=True,
)
```

三个小函数将其转换为三种供应商格式。`code/main.py`中按此执行，模拟工具调用通过各供应商响应形态回环。无需网络，本课聚焦形态而非HTTP调用。

生产团队将此翻译器封装在`AbstractToolset`（Pydantic AI），`UniversalToolNode`（LangGraph）或`BaseTool`（LlamaIndex）中。第13阶段 · 17发布了可暴露OpenAI形态API的网关，支持三大供应商。

## 使用方法

`code/main.py`定义了一个规范`Tool`数据类和三个翻译器，分别输出OpenAI、Anthropic和Gemini的声明JSON。随后解析手工构造的供应商响应，转换为同一规范调用对象，演示语义等价。运行并并排对比三种声明。

关注点：

- 三个声明块仅在包裹和字段名上不同。
- 三个响应块的调用所在位置不同（顶层`tool_calls`，`content[]`块，`parts[]`条目）。
- 一个`canonical_call()`函数从三种响应形态中提取`{id, name, args}`。

## 部署

本课程生成`outputs/skill-provider-portability-audit.md`。针对单一供应商的功能调用集成，生成移植性审核：依赖的供应商限制、需重命名字段、移植至其他供应商时会中断的项。

## 练习

1. 运行`code/main.py`，验证三个供应商声明JSON均序列化自同一`Tool`对象。修改规范工具，添加枚举参数，确认只有Gemini翻译器需处理OpenAPI特性。

2. 为每个供应商添加`ListToolsResponse`解析器，从模型对`list_tools`或发现调用的响应中提取工具列表。OpenAI无原生支持，注意这一不对称。

3. 实现`tool_choice`转换：将规范的`ToolChoice(mode="force", tool_name="x")`映射为三者格式；再映射`mode="any"`和`mode="none"`。参照本课差异表。

4. 选择一个供应商，通读其功能调用指南。找出其模式规格中其他两个不支持的字段。如OpenAI的`strict`，Anthropic的`disable_parallel_tool_use`，Gemini的`function_calling_config.allowed_function_names`。

5. 编写测试向量：一个工具调用，参数违反声明模式。分别经过三家供应商（第01课标准库验证器可作为代理）校验，记录错误发生情况。记录生产环境中你会选用哪个供应商以保证严格性。

## 关键词

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| 功能调用（Function calling） | “工具使用” | 供应商层级的结构化工具调用API |
| 工具声明（Tool declaration） | “工具规范” | 名称 + 描述 + JSON Schema输入负载 |
| `tool_choice` | “强制 / 禁止” | 自动 / 必选 / 禁止 / 指定名称模式 |
| 严格模式（Strict mode） | “模式强制” | OpenAI标志，限制解码符合模式 |
| `tool_use`块 | “Anthropic调用形态” | 内嵌内容块，含id、名称、输入 |
| `functionCall`条目 | “Gemini调用形态” | `parts[]`中包含名称、参数及id |
| 参数字符串化（Arguments-as-string） | “字符串化JSON” | OpenAI返回参数为JSON字符串非对象 |
| 并行工具调用 | “一轮多拨” | 助理消息中多次工具调用 |
| 拒绝（Refusal） | “模型拒绝调用” | 仅严格模式中拒绝块替代调用 |
| OpenAPI 3.0子集 | “Gemini模式特性” | Gemini采用JSON Schema类方言，有细微差异 |

## 拓展阅读

- [OpenAI — 功能调用指南](https://platform.openai.com/docs/guides/function-calling) — 官方参考，涵盖严格模式与并行调用  
- [Anthropic — 工具使用概览](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview) — `tool_use`和`tool_result`块语义  
- [Google — Gemini功能调用](https://ai.google.dev/gemini-api/docs/function-calling) — 并行调用、唯一ID和OpenAPI子集  
- [Vertex AI — 功能调用参考](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/multimodal/function-calling) — Gemini企业应用  
- [OpenAI — 结构化输出](https://platform.openai.com/docs/guides/structured-outputs) — 严格模式和模式强制详情
