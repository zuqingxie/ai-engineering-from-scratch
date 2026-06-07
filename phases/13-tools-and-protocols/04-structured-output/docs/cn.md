# 结构化输出 — JSON Schema、Pydantic、Zod、受限解码

> “礼貌地请求模型返回 JSON” 在前沿模型中失败率为 5% 到 15%，结构化输出通过受限解码（constrained decoding）弥补了这一差距：模型实际上被禁止生成会违反 schema 的 token。OpenAI 的严格模式（strict mode）、Anthropic 的 schema 类型工具调用、Gemini 的 `responseSchema`、Pydantic AI 的 `output_type` 和 Zod 的 `.parse` 是同一理念的五种表现形式。本课构建 schema 验证器和每个生产提取流程都会使用的严格模式合约学习器。

**类型:** 构建  
**语言:** Python（标准库、JSON Schema 2020-12 子集）  
**先修:** Phase 13 · 02（函数调用深度解析）  
**时间:** 约 75 分钟

## 学习目标

- 使用合适的约束（enum、min/max、required、pattern）编写 JSON Schema 2020-12 用于提取目标。  
- 解释严格模式和受限解码为何与“生成后验证”提供不同的保障。  
- 区分三种失败模式：解析错误（parse error）、schema 违规（schema violation）、模型拒绝（refusal）。  
- 发布一个带有类型化修复与类型化拒绝处理的提取流水线。

## 问题描述

一个代理读取采购订单邮件，需要将自由文本转成 `{customer, line_items, total_usd}`。有三种方案。

**方案一：请求 JSON。** “回复 JSON，字段包括 customer、line_items、total_usd。” 在前沿模型上 85% 到 95% 可靠。失败方式有六种：缺失括号、尾逗号、类型错误、幻觉字段、因 token 限制截断、输出带了“Here is your JSON:”之类的白话。

**方案二：生成后验证。** 自由生成，解析，校验 schema，失败则重试。可靠但昂贵——每次重试都要付费，截断 bug 造成每次多消耗一轮。

**方案三：受限解码。** 服务方在解码时强制 schema，禁止非法 token。输出保证可解析，保证校验通过。失败收敛为一种模式：拒绝（模型判断输入不符合 schema）。

2026 年所有前沿提供商都会包含某种方案三。

- **OpenAI。** `response_format: {type: "json_schema", strict: true}`，模型拒绝时响应带 `refusal` 字段。  
- **Anthropic。** 在 `tool_use` 输入层面强制 schema；无 `stop_reason: "refusal"`，没有调用工具且 `end_turn` 信号表示拒绝。  
- **Gemini。** 请求级别的 `responseSchema`；2026 年版中对特定类型支持 token 级语法约束。  
- **Pydantic AI。** `output_type=InvoiceModel` 输出结构化的 `RunResult`，类型化为 `InvoiceModel`。  
- **Zod（TypeScript）。** 运行时解析器，验证输出是否符合 Zod schema；配合 OpenAI 的 `beta.chat.completions.parse` 使用。

共同点：定义一次 schema，端到端强制执行。

## 概念详述

### JSON Schema 2020-12 — 通用语言

所有提供商均支持 JSON Schema 2020-12。常用构造：

- `type`：可选 `object`、`array`、`string`、`number`、`integer`、`boolean`、`null`。  
- `properties`：字段名到子 schema 的映射。  
- `required`：必须出现的字段名列表。  
- `enum`：允许值的闭合集合。  
- `minimum` / `maximum`（数字），`minLength` / `maxLength` / `pattern`（字符串）。  
- `items`：应用于所有数组元素的子 schema。  
- `additionalProperties`：`false` 禁止额外字段（默认因模式而异）。

OpenAI 严格模式额外要求：每个属性都必须列在 `required` 中，所有地方都必须 `additionalProperties: false`，且不得有未解析的 `$ref`。若违背，API 请求时返回 400 错误。

### Pydantic，Python 绑定

Pydantic v2 支持从类似数据类（dataclass）的模型通过 `model_json_schema()` 生成 JSON Schema。Pydantic AI 封装此功能，支持写如下代码：

```python
class Invoice(BaseModel):
    customer: str
    line_items: list[LineItem]
    total_usd: Decimal
```

代理框架会将其翻译成 OpenAI 严格模式、Anthropic `input_schema` 或 Gemini `responseSchema`。模型输出被转换为类型化的 `Invoice` 实例。验证错误抛出带路径的 `ValidationError`。

### Zod，TypeScript 绑定

Zod（如 `z.object({customer: z.string(), ...})`）是 TS 版本。OpenAI Node SDK 提供了 `zodResponseFormat(Invoice)`，它会转成 API 的 JSON Schema 负载。

### 拒绝（Refusals）

严格模式不强制模型一定回答。如果输入不符合 schema（例如“邮件是诗而非发票”），模型会输出 `refusal` 字段说明原因。你的代码必须把它作为首等结果处理，而非失败。拒绝还是安全信号：如提取受保护邮件中的信用卡号，模型返回带有安全理由的拒绝。

### 公开受限解码技术

开源权重实现通常用三种技术：

1. **基于语法的解码**（如 outlines、guidance、lm-format-enforcer）：根据 schema 构建确定性有限自动机（DFA），每步屏蔽会违背 FSM 的 token 概率。  
2. **结合 JSON 解析器的 logit 屏蔽**：与模型同步运行流式 JSON 解析器，每步计算合法的下一 token 集合。  
3. **带验证器的猜测性解码**：草稿模型快速生成候选，验证器执行 schema 校验。

商业提供商在后台任选一法。2026 年先进技术在短结构输出上比纯生成更快，长结构输出速度相差无几。

### 三种失败模式

1. **解析错误。** 输出不是有效 JSON。严格模式下不可能。非严格提供商仍会发生。  
2. **schema 违规。** 输出可解析但不符合 schema。严格模式下不发生，其他情况常见。  
3. **拒绝。** 模型拒绝回答。必须作为类型化结果处理。

### 重试策略

你若非严格模式（如 Anthropic 工具使用、非严格 OpenAI、旧版 Gemini），恢复流程：

```text
生成 -> 解析 -> 验证 -> 失败时注入错误信息并重试，最多三次
```

一次重试通常够用，三次捕获弱模型偶发失误。超过三次意味着 schema 有问题：模型无法满足，需修正提示或 schema。

### 小模型支持

受限解码适用于小模型。一个 30 亿参数的开源模型用语法强制，比一个 700 亿参数的纯提示模型在结构化任务上表现更好。这是结构化输出对生产重要的主要原因：可靠性不再依赖模型大小。

## 使用示例

`code/main.py` 提供了一个最小 JSON Schema 2020-12 校验器（支持 types、required、enum、min/max、pattern、items、additionalProperties），用来包裹 `Invoice` schema，并用伪造 LLM 输出演示解析错误、schema 违规和拒绝路径。可直接替换成任何提供商的真实响应。

关注点：

- 校验器返回带路径和消息的类型化 `[ValidationError]` 列表。这是你希望呈现给重试提示的错误形态。  
- 拒绝分支不执行重试，而是记录并返回类型化的拒绝。Phase 14 · 09 利用拒绝作为安全信号。  
- `additionalProperties: false` 检查在对抗测试输入时触发，展示严格模式为何封堵幻觉字段。

## 发布最佳实践

本课产物为 `outputs/skill-structured-output-designer.md`。给定自由文本提取目标（发票、客服单、简历等），该技能生成符合严格模式兼容的 JSON Schema 2020-12 和对应的 Pydantic 模型，并预设类型化拒绝与重试处理。

## 练习

1. 运行 `code/main.py`。新增第四个测试用例，其中 `total_usd` 为负数。确认校验器因 `minimum` 约束拒绝该输入。  
2. 扩展校验器以支持带判别器的 `oneOf`。常见场景：`line_item` 可为产品或服务，由 `kind` 标记。严格模式对此有细微规则；参考 OpenAI 结构化输出指南。  
3. 编写同一发票 Schema 的 Pydantic BaseModel，比较 `model_json_schema()` 输出与手写 Schema，找出 Pydantic 默认设置但手写版本缺失的字段。  
4. 测量拒绝率。构造十个无法提取的输入（歌曲歌词、数学证明、空白邮件），用真实提供商严格模式测试。统计拒绝和幻觉输出，作为拒绝感知重试的真实标准。  
5. 通读 OpenAI 结构化输出指南，找出严格模式明确禁止但普通 JSON Schema 允许的构造。然后设计一个使用该构造但非关键用途的 schema，改写为严格兼容版本。

## 关键词

| 术语 | 通俗说法 | 实际含义 |
|------|----------|----------|
| JSON Schema 2020-12 | “Schema 规范” | IETF 草案版 schema 方言，所有现代服务商支持 |
| 严格模式（Strict mode） | “保证 schema” | OpenAI 的标志，通过受限解码强制 schema |
| 受限解码（Constrained decoding） | “logit 屏蔽” | 解码时屏蔽所有非法下一个 token 的机制 |
| 拒绝（Refusal） | “模型拒绝” | 输入不合 schema 时的类型化输出结果 |
| 解析错误（Parse error） | “无效 JSON” | 输出未能解析为 JSON，严格模式不可能发生 |
| Schema 违规（Schema violation） | “格式错误” | JSON 合法但违背类型、必需字段、限定值区间 |
| `additionalProperties: false` | “不允许有多余字段” | 禁止未知字段，OpenAI 严格模式必需 |
| Pydantic BaseModel | “类型化输出类” | Python 类，生成且验证 JSON Schema |
| Zod schema | “TS 输出类型” | TypeScript 运行时 schema，用于验证提供商输出 |
| 语法强制（Grammar enforcement） | “开源受限解码” | 基于 FSM 的 logit 屏蔽，如 outlines、guidance |

## 拓展阅读

- [OpenAI — 结构化输出](https://platform.openai.com/docs/guides/structured-outputs) — 严格模式、拒绝和 schema 要求  
- [OpenAI — 介绍结构化输出](https://openai.com/index/introducing-structured-outputs-in-the-api/) — 2024 年 8 月发布，解释解码保证  
- [Pydantic AI — 输出](https://ai.pydantic.dev/output/) — 能序列化为各提供商的类型化 `output_type` 绑定  
- [JSON Schema — 2020-12 发布说明](https://json-schema.org/draft/2020-12/release-notes) — 官方规范  
- [微软 — Azure OpenAI 结构化输出](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/structured-outputs) — 企业部署和严格模式注意事项
