# 结构化输出与受限解码

> 向大型语言模型（LLM）请求 JSON。大多数情况下得到 JSON。但在生产环境中，“大多数”就是问题所在。受限解码通过采样前编辑 logits，将“大多数”变成“总是”。

**类型：** 构建  
**语言：** Python  
**先决条件：** 阶段 5 · 17（聊天机器人）、阶段 5 · 19（子词分词）  
**时间：** ~60 分钟  

## 问题

一个分类器提示 LLM：“返回 {positive, negative, neutral} 中之一。”模型返回了“The sentiment is positive — this review is overwhelmingly favorable because the customer explicitly states that they ...”。你的解析器崩溃了。分类器的 F1 分数变成了 0.0。

自由生成不是契约。它只是建议。生产系统需要一个契约。

2026 年存在三层。

1. **提示（Prompting）。** 礼貌请求。“只返回 JSON 对象。”前沿模型上大约 80% 成功率，较小模型更低。  
2. **原生结构化输出 API。** OpenAI 的 `response_format`，Anthropic 的工具使用，Gemini JSON 模式。对支持的 schema 可靠。供应商锁定。  
3. **受限解码。** 在每个生成步骤修改 logits，使模型*无法*输出无效标记。构造上 100% 有效。适用于任何本地模型。

本课旨在建立对这三者的直觉，并说明何时选择使用哪种方法。

## 概念

![受限解码在每步屏蔽无效标记](../assets/constrained-decoding.svg)

**受限解码的工作原理。** 在每个生成步骤，LLM 生成一个涵盖完整词汇表（约 10 万个标记）的 logit 向量。一个*logit 处理器*位于模型和采样器之间。它根据目标语法当前的位置——JSON Schema、正则表达式、上下文无关文法——计算哪些标记是有效的，并将所有无效标记的 logits 设为负无穷。对剩余 logits 的 softmax 只在有效续写上分配概率质量。

2026 年的实现：

- **Outlines。** 将 JSON Schema 或正则表达到有限状态机（FSM）。每个标记都可以通过 O(1) 查找得到是否有效。基于 FSM，因此递归 schema 需要扁平化处理。  
- **XGrammar / llguidance。** 上下文无关文法引擎。支持递归 JSON Schema。几乎无解码开销。OpenAI 在 2025 年结构化输出实现中提及 llguidance。  
- **vLLM 引导解码。** 内置 `guided_json`、`guided_regex`、`guided_choice`、`guided_grammar`，通过 Outlines、XGrammar 或 lm-format-enforcer 后端实现。  
- **Instructor。** 基于 Pydantic 的任意 LLM 包装器。在验证失败时重试。跨供应商，但不修改 logits —— 依赖重试加上结构化输出感知的提示。  

### 违背直觉的结果

受限解码往往比非受限生成*更快*。原因有两个。首先，它缩小了下一个标记的搜索空间。其次，巧妙的实现完全跳过强制标记的生成步骤（比如构架 `{"name": "` — 每个字节都是确定的）。

### 成本坑点

字段顺序很重要。将 `answer` 放在 `reasoning` 之前，模型会在思考前就做出答案承诺。JSON 是有效的，但答案是错的。没有验证能捕捉到。

```json
// 错误示范
{"answer": "yes", "reasoning": "because ..."}

// 正确示范
{"reasoning": "... therefore ...", "answer": "yes"}
```

Schema 字段顺序体现逻辑，而非格式化。

## 构建

### 第 1 步：从头实现正则受限生成

参见 `code/main.py`，Standalone FSM 实现。核心思想，30 行代码：

```python
def mask_logits(logits, valid_token_ids):
    mask = [float("-inf")] * len(logits)
    for tid in valid_token_ids:
        mask[tid] = logits[tid]
    return mask


def generate_constrained(model, tokenizer, prompt, fsm):
    ids = tokenizer.encode(prompt)
    state = fsm.initial_state
    while not fsm.is_accept(state):
        logits = model.next_token_logits(ids)
        valid = fsm.valid_tokens(state, tokenizer)
        logits = mask_logits(logits, valid)
        tok = sample(logits)
        ids.append(tok)
        state = fsm.transition(state, tok)
    return tokenizer.decode(ids)
```

FSM 负责跟踪迄今满足的语法部分。`valid_tokens(state, tokenizer)` 计算哪些词汇标记能让 FSM 继续进入接受路径。

### 第 2 步：Outlines 针对 JSON Schema

```python
from pydantic import BaseModel
from typing import Literal
import outlines


class Review(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float
    evidence_span: str


model = outlines.models.transformers("meta-llama/Llama-3.2-3B-Instruct")
generator = outlines.generate.json(model, Review)

result = generator("Classify: 'The wait staff was attentive and the food arrived hot.'")
print(result)
# Review(sentiment='positive', confidence=0.93, evidence_span='attentive ... hot')
```

零验证错误。永远。FSM 让无效输出无法触达。

### 第 3 步：Instructor 跨供应商 Pydantic

```python
import instructor
from anthropic import Anthropic
from pydantic import BaseModel, Field


class Invoice(BaseModel):
    vendor: str
    total_usd: float = Field(ge=0)
    line_items: list[str]


client = instructor.from_anthropic(Anthropic())
invoice = client.messages.create(
    model="claude-opus-4-7",
    max_tokens=1024,
    response_model=Invoice,
    messages=[{"role": "user", "content": "Extract from: 'Acme Corp $420. Widget, Gizmo.'"}],
)
```

机制不同。Instructor 不修改 logits。它将 schema 格式化到提示中，解析输出，并在验证失败时重试（默认 3 次）。支持所有供应商。重试会增加延迟和成本。跨供应商可移植性是卖点。

### 第 4 步：原生供应商 API

```python
from openai import OpenAI

client = OpenAI()
response = client.responses.create(
    model="gpt-5",
    input=[{"role": "user", "content": "Classify: 'The food was cold.'"}],
    text={"format": {"type": "json_schema", "name": "sentiment",
          "schema": {"type": "object", "required": ["sentiment"],
                     "properties": {"sentiment": {"type": "string",
                                                  "enum": ["positive", "negative", "neutral"]}}}}},
)
print(response.output_parsed)
```

服务器端受限解码。对于支持的 schema，可靠性与 Outlines 持平。无需本地模型管理。锁定供应商。

## 陷阱

- **递归 schema。** Outlines 将递归扁平化到固定深度。树形结构输出（嵌套评论、AST）需要使用 XGrammar 或 llguidance（基于 CFG）。  
- **超大枚举。** 1 万选项的枚举编译缓慢或超时。转为检索器机制：先预测 top-k 候选，再约束为这些。  
- **语法过严。** 强制 `date: "YYYY-MM-DD"` 正则表达式，模型无法输出 `"unknown"` 表示缺失日期，模型被迫虚构日期。允许使用 `null` 或哨兵值。  
- **过早承诺。** 参见上方字段顺序陷阱。务必先写推理，再写答案。  
- **无 schema 的供应商 JSON 模式。** 纯 JSON 模式只保证 JSON 语法有效，不保证*符合你用例的*有效性。一定要提供完整 schema。

## 使用建议

2026 年方案：

| 场景                             | 选择                         |
|---------------------------------|------------------------------|
| OpenAI/Anthropic/Google 模型，简单 schema | 原生供应商结构化输出             |
| 任意供应商，Pydantic 工作流，可接受重试    | Instructor                   |
| 本地模型，需要 100% 有效，扁平 schema     | Outlines（FSM）               |
| 本地模型，递归 schema               | XGrammar 或 llguidance        |
| 自托管推理服务器                   | vLLM 引导解码                 |
| 批处理，接受重试                   | Instructor + 最廉价模型       |

## 上线部署

保存为 `outputs/skill-structured-output-picker.md`：

```markdown
---
name: structured-output-picker
description: 选择结构化输出方案、schema 设计和验证计划。
version: 1.0.0
phase: 5
lesson: 20
tags: [nlp, llm, structured-output]
---

给定用例（供应商、延迟预算、schema 复杂度、失败容忍度），输出：

1. 机制。原生供应商结构化输出、Instructor 重试、Outlines FSM 或 XGrammar CFG。一句话理由。  
2. Schema 设计。字段顺序（推理优先，答案最后）、对 “unknown” 字段允许可为空、枚举 vs 正则表达式、必填字段。  
3. 失败策略。最大重试次数、备用模型、优雅的 `null` 处理、分布外拒绝。  
4. 验证计划。Schema 合规率（目标 100%）、语义有效性（LLM 评判）、字段覆盖率、延迟 p50/p99。  

拒绝任何将 `answer` 或 `decision` 放在推理字段之前的设计。拒绝无 schema 的纯 JSON 模式。对递归 schema 标记需要 FSM-only 库处理。
```

## 练习

1. **简单。** 针对小型开源权重模型（如 Llama-3.2-3B），不开启受限解码，提示生成 `Review(sentiment, confidence, evidence_span)`。在 100 个评论中测量有效 JSON 解析比例。  
2. **中等。** 同语料，使用 Outlines JSON 模式。对比合规率、延迟和语义准确率。  
3. **困难。** 从头实现一个针对于电话号码（`\d{3}-\d{3}-\d{4}`）的正则受限解码器。验证 1000 个样本中零无效输出。

## 关键词

| 术语               | 常见说法       | 实际含义                         |
|------------------|-------------|------------------------------|
| 受限解码（Constrained decoding） | 强制有效输出    | 在每步生成时掩码无效标记 logits。       |
| logit 处理器（Logit processor） | 限制功能的东西  | 函数 `(logits, state) -> masked_logits`。|
| FSM              | 有限状态机     | 编译的语法表示；O(1) 有效后续标记查找。    |
| CFG              | 上下文无关文法  | 处理递归的语法；比 FSM 慢但更具表现力。    |
| schema 字段顺序     | 重要吗？       | 重要——第一个字段即承诺；推理字段总是放前面。 |
| 引导解码（Guided decoding）  | vLLM 的称呼    | 同一概念，集成在推理服务器内。           |
| JSON 模式          | OpenAI 早期版本 | 保证 JSON 语法；不保证匹配 schema。       |

## 延伸阅读

- [Willard, Louf (2023). Efficient Guided Generation for LLMs](https://arxiv.org/abs/2307.09702) — Outlines 论文。  
- [XGrammar 论文 (2024)](https://arxiv.org/abs/2411.15100) — 快速基于 CFG 的受限解码。  
- [vLLM — 结构化输出](https://docs.vllm.ai/en/latest/features/structured_outputs.html) — 推理服务器集成。  
- [OpenAI — 结构化输出指南](https://platform.openai.com/docs/guides/structured-outputs) — API 参考与注意事项。  
- [Instructor 库](https://python.useinstructor.com/) — Pydantic + 跨供应商重试。  
- [JSONSchemaBench (2025)](https://arxiv.org/abs/2501.10868) — 6 种受限解码框架基准测试。
