# 对话状态跟踪（Dialogue State Tracking）

> “我想找一个北边的便宜餐厅……其实定个中等价位的……再来个意大利餐厅。” 三轮对话，三次状态更新。DST 保持槽-值字典同步，确保预订正确。

**类型：** 构建  
**语言：** Python  
**先决条件：** 第5阶段 · 17（聊天机器人），第5阶段 · 20（结构化输出）  
**时间：** 约75分钟

## 问题描述

在面向任务的对话系统中，用户的目标被编码为一组槽-值对：`{cuisine: italian, area: north, price: moderate}`。每轮用户对话都可能添加、更改或删除一个槽。系统必须读取完整对话并准确输出当前状态。

任一槽错误，系统就可能预订错误的餐厅、安排错误的航班，或扣错卡。DST 是用户输入与后端执行之间的枢纽。

即使在2026年大规模语言模型时代，DST仍重要的原因：

- 合规敏感领域（银行、医疗、航空预订）需要确定性的槽值，而非自由生成文本。
- 工具调用代理依然需要槽值解析才能调用API。
- 多轮纠错比看上去复杂：“其实不对，改成星期四。”

现代流程：经典DST概念 + 大型语言模型抽取器 + 结构化输出护栏。

## 核心概念

![DST: dialog history → slot-value state](../assets/dst.svg)

**任务结构。** 方案定义域（餐厅、酒店、出租车）及其槽（菜系、区域、价格、人数）。每个槽可以为空，从闭集取值（价格：{便宜，中等，贵}），或者自由文本（名称：“The Copper Kettle”）。

**两种DST表达方式。**

- **分类法。** 对每个（槽，候选值）对，预测是/否。适用于闭集槽。2020年前标准方法。
- **生成法。** 给定对话，生成槽值的自由文本。适用于开放词汇槽。现代默认。

**评估指标。** 联合目标准确率（Joint Goal Accuracy, JGA）——指每轮对话中*所有*槽均正确的比例。全有或全无。MultiWOZ 2.4 排行榜顶峰至2026年约83%。

**架构。**

1. **基于规则（槽正则 + 关键词）。** 窄域强基线。易调试。
2. **TripPy / BERT-DST。** 结合BERT编码的基于复制的生成。大模型前标准。
3. **LDST（LLaMA + LoRA）。** 具领域-槽提示的指令微调大模型。MultiWOZ 2.4 上达到ChatGPT级质量。
4. **无本体（2024–26）。** 跳过方案，直接生成槽名和值。支持开放域。
5. **提示 + 结构化输出（2024–26）。** 大模型 + Pydantic方案 + 受限解码。五行代码，生产级。

### 经典失败场景

- **跨轮指代。** “还是第一个选项吧。”需要解析指代哪个选项。
- **覆盖与追加。** 用户说“加意大利菜”。是替换菜系还是追加？
- **隐式确认。** “好的，很棒”——是接受了提供的预订吗？
- **纠正。** “其实改到晚上7点。”必须更新时间但不清除其他槽。
- **指代之前系统发言。** “是的，就那个。”哪个“那个”？

## 构建步骤

### 第1步：基于规则的槽抽取器

见 `code/main.py`。正则+词义字典覆盖窄域70%典型话语：

```python
CUISINE_SYNONYMS = {
    "italian": ["italian", "pasta", "pizza", "italy"],
    "chinese": ["chinese", "chow mein", "noodles"],
}


def extract_cuisine(utterance):
    for canonical, synonyms in CUISINE_SYNONYMS.items():
        if any(syn in utterance.lower() for syn in synonyms):
            return canonical
    return None
```

局限于典型词汇外表现脆弱。适合确定性槽确认。

### 第2步：状态更新循环

```python
def update_state(state, utterance):
    new_state = dict(state)
    for slot, extractor in SLOT_EXTRACTORS.items():
        value = extractor(utterance)
        if value is not None:
            new_state[slot] = value
    for slot in NEGATION_CLEARS:
        if is_negated(utterance, slot):
            new_state[slot] = None
    return new_state
```

三个不变量：

- 不重置用户未修改的槽。
- 显式否定（“不管菜系了”）必须清空对应槽。
- 用户纠正（“其实……”)必须覆盖，不是追加。

### 第3步：结构化输出的LLM驱动DST

```python
from pydantic import BaseModel
from typing import Literal, Optional
import instructor

class RestaurantState(BaseModel):
    cuisine: Optional[Literal["italian", "chinese", "indian", "thai", "any"]] = None
    area: Optional[Literal["north", "south", "east", "west", "center"]] = None
    price: Optional[Literal["cheap", "moderate", "expensive"]] = None
    people: Optional[int] = None
    day: Optional[str] = None


def llm_dst(history, llm):
    prompt = f"""You track the slot values of a restaurant booking across turns.
Dialogue so far:
{render(history)}

Update the state based on the latest user turn. Output only the JSON state."""
    return llm(prompt, response_model=RestaurantState)
```

Instructor + Pydantic保证输出是有效状态对象。无正则，无方案不匹配，无虚构槽。

### 第4步：JGA评估

```python
def joint_goal_accuracy(predicted_states, gold_states):
    correct = sum(1 for p, g in zip(predicted_states, gold_states) if p == g)
    return correct / len(predicted_states)
```

校准：系统在多少轮中*所有*槽都正确？MultiWOZ 2.4 2026顶级系统约80%-83%。你自己的领域系统应超越相应闭域词典或LLM基线。

### 第5步：处理纠错

```python
CORRECTION_CUES = {"actually", "no wait", "on second thought", "change that to"}


def is_correction(utterance):
    return any(cue in utterance.lower() for cue in CORRECTION_CUES)
```

检测到纠正时，应覆盖最近更新槽，而非追加。无LLM辅助难以精准处理。现代模式是——总让LLM根据历史重新生成全状态，而非增量更新，自然支持纠正。

## 常见坑

- **全历史重生成成本。** 让LLM每轮都重生状态，代价为总token数O(n²)。限制历史长度或摘要旧轮。
- **方案漂移。** 新增槽会破坏旧训练集。方案请版本化。
- **大小写敏感。** “Italian” vs “italian” vs “ITALIAN”——全处归一。
- **隐式继承。** 若用户之前说“4人”，新时间请求不应清除人数。始终传递完整历史。
- **自由格式 vs 闭集。** 名称、时间、地址用自由格式槽；菜系、区域用闭集。混合方案支持。

## 选用指南

2026年技术栈：

| 情况 | 方案 |
|-----------|----------|
| 窄域（1-2意图） | 基于规则 + 正则表达式 |
| 广域，有标注数据 | LDST（基于LLaMA + LoRA，多领域数据） |
| 广域，无标注，生产环境 | 大模型 + Instructor + Pydantic方案 |
| 语音 | ASR + 归一化器 + LLM驱动DST |
| 多领域预订流程 | 方案引导LLM，按域定义Pydantic模型 |
| 合规敏感场景 | 规则优先，LLM备援带确认流程 |

## 发布指南

保存为 `outputs/skill-dst-designer.md`：

```markdown
---
name: dst-designer
description: Design a dialogue state tracker — schema, extractor, update policy, evaluation.
version: 1.0.0
phase: 5
lesson: 29
tags: [nlp, dialogue, task-oriented]
---

Given a use case (domain, languages, vocab openness, compliance needs), output:

1. Schema. Domain list, slots per domain, open vs closed vocabulary per slot.
2. Extractor. Rule-based / seq2seq / LLM-with-Pydantic. Reason.
3. Update policy. Regenerate-whole-state / incremental; correction handling; negation handling.
4. Evaluation. Joint Goal Accuracy on a held-out dialogue set, slot-level precision/recall, confusion on the hardest slot.
5. Confirmation flow. When to explicitly ask the user to confirm (destructive actions, low-confidence extractions).

Refuse LLM-only DST for compliance-sensitive slots without a rule-based secondary check. Refuse any DST that cannot roll back a slot on user correction. Flag schemas without version tags.
```

## 练习

1. **简单。** 在 `code/main.py` 实现3个槽（菜系、区域、价格）的基于规则状态追踪器。用10个手工对话测试。测量JGA。
2. **中等。** 用同数据集结合Instructor + Pydantic + 小型LLM。对比JGA。排查最难轮。
3. **困难。** 两者皆实现并路由：规则为主，规则置信度低(<2槽)时LLM备援。测综合JGA和每轮推理成本。

## 关键词

| 术语 | 通常说法 | 实际意义 |
|------|---------|-----------|
| DST | 对话状态跟踪 | 管理对话中跨轮的槽-值字典。 |
| Slot（槽） | 用户意图单位 | 后端需命名参数（如菜系、日期）。 |
| Domain（领域） | 任务范围 | 餐厅、酒店、出租车等一组槽。 |
| JGA | 联合目标准确率 | 每轮所有槽全对的比例。全有或全无。 |
| MultiWOZ | 评测基准 | 多领域仿真助手数据集；标准DST评测。 |
| Ontology-free DST | 无方案 | 直接生成槽名和值，无固定列表。 |
| Correction（纠正） | “其实……” | 覆盖之前填充槽的对话轮。 |

## 深入阅读

- [Budzianowski et al. (2018). MultiWOZ — A Large-Scale Multi-Domain Wizard-of-Oz](https://arxiv.org/abs/1810.00278) — 经典评测基准。
- [Feng et al. (2023). Towards LLM-driven Dialogue State Tracking (LDST)](https://arxiv.org/abs/2310.14970) — LLaMA + LoRA 指令微调DST。
- [Heck et al. (2020). TripPy — A Triple Copy Strategy for Value Independent Neural Dialog State Tracking](https://arxiv.org/abs/2005.02877) — 复制机制DST骨干。
- [King, Flanigan (2024). Unsupervised End-to-End Task-Oriented Dialogue with LLMs](https://arxiv.org/abs/2404.10753) — 基于EM的无监督端到端TOD。
- [MultiWOZ leaderboard](https://github.com/budzianowski/multiwoz) — 经典DST结果排行榜。
