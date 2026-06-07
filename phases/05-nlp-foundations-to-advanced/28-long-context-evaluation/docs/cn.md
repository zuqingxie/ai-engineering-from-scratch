# 长上下文评估 — NIAH, RULER, LongBench, MRCR

> Gemini 3 Pro 宣称支持 1000 万个 token 的上下文。实际在 100 万 token 时，8 针 MRCR 的准确率降至 26.3%。宣称的容量 ≠ 实际可用容量。长上下文评估告诉你交付模型的真实容量。

**类型：** 学习  
**语言：** Python  
**前置知识：** 阶段 5 · 13（问答），阶段 5 · 23（分块策略）  
**时长：** ~60 分钟

## 问题

你有一份 200 页的合同。模型声称支持 100 万 token 的上下文。你将合同粘贴进去并问：“终止条款是什么？”模型给出了答案 —— 但答案来自封面页，因为终止条款位于 12 万 token 以后，超出了模型实际关注的范围。

这就是 2026 年的上下文能力差距。规格说明上说 100 万或 1000 万。现实中可用的只有 60-70%，而且“可用”还取决于任务。

- **检索（大海捞针）：** 在先进模型上接近规格最大值时表现几乎完美。
- **多跳推理 / 聚合：** 大多数模型在超过 ~128k 后准确率急剧下降。
- **分散事实推理：** 最容易失败的任务。

长上下文评估测量这些维度。本课介绍各测试集，测量内容，以及如何为你的领域构建定制的“针测试”。

## 概念

![NIAH baseline, RULER multi-task, LongBench holistic](../assets/long-context-eval.svg)

**Needle-in-a-Haystack (NIAH，2023)。** 在长上下文中控制深度植入一个事实（如“魔术词是 pineapple”），让模型检索。扫不同深度和长度组合。原版长上下文基准。前沿模型已趋于饱和，是必要但不充分的基线。

**RULER (Nvidia，2024)。** 包含 13 种任务类型，分为 4 类：检索（单键/多键/多值）、多跳跟踪（变量追踪）、聚合（常用词频）、问答。上下文长度可调（4k 到 128k+）。揭示了饱和 NIAH 但多跳失败的模型。2024 版本中，17 个声称支持 32k+ 上下文的模型，仅半数在 32k 上保持质量。

**LongBench v2 (2024)。** 503 道多选题，涵盖 8k-2M 字的上下文，六大任务类别：单文档问答、多文档问答、长上下文学习、长对话、代码仓库、长结构化数据。代表真实世界长上下文行为的生产基准。

**MRCR (Multi-Round Coreference Resolution)。** 大规模多轮指代消解。支持 8 针、24 针、100 针版本。暴露模型在注意力衰减前可处理的事实数量。

**NoLiMa。** “非字面针”。针与查询无文字重叠；检索需一步语义推理。难度高于 NIAH。

**HELMET。** 将多文档串联，提问来自任一文档。测试选择性注意力。

**BABILong。** 在无关的稻草堆中嵌入 bAbI 推理链。测试稻草堆中的推理，而非仅检索。

### 具体报告内容

- **宣称的上下文窗口大小。** 规格说明书数字。
- **有效检索长度。** NIAH 在某阈值（如 90%）时的通过长度。
- **有效推理长度。** 多跳或聚合任务在相同阈值下的通过长度。
- **性能下降曲线。** 各任务类型准确率随上下文长度变化的曲线。

规格书请提供两个数据：检索有效长度和推理有效长度。通常推理有效长度是宣称窗口的 25-50%。

## 构建步骤

### 第 1 步：为你的领域构建定制 NIAH

查看 `code/main.py`。代码骨架：

```python
def build_haystack(filler_text, needle, depth_ratio, total_tokens):
    if not (0.0 <= depth_ratio <= 1.0):
        raise ValueError(f"depth_ratio must be in [0, 1], got {depth_ratio}")
    if total_tokens <= 0:
        raise ValueError(f"total_tokens must be positive, got {total_tokens}")

    filler_tokens = tokenize(filler_text)
    needle_tokens = tokenize(needle)
    if not filler_tokens:
        raise ValueError("filler_text produced no tokens")

    # 重复 filler 直到足够长以填充稻草堆主体。
    body_len = max(total_tokens - len(needle_tokens), 0)
    while len(filler_tokens) < body_len:
        filler_tokens = filler_tokens + filler_tokens
    filler_tokens = filler_tokens[:body_len]

    insert_at = min(int(body_len * depth_ratio), body_len)
    haystack = filler_tokens[:insert_at] + needle_tokens + filler_tokens[insert_at:]
    return " ".join(haystack)


def score_niah(model, haystack, question, expected):
    answer = model.complete(f"Context: {haystack}\nQ: {question}\nA:", max_tokens=50)
    return 1 if expected.lower() in answer.lower() else 0
```

对 `depth_ratio` ∈ {0, 0.25, 0.5, 0.75, 1.0} × `total_tokens` ∈ {1k, 4k, 16k, 64k} 进行扫表。绘制热力图，这即是你的目标模型的 NIAH 卡。

### 第 2 步：多针版本

```python
def build_multi_needle(filler, needles, total_tokens):
    depths = [0.1, 0.4, 0.7]
    chunks = [filler[:int(total_tokens * 0.1)]]
    for depth, needle in zip(depths, needles):
        chunks.append(needle)
        next_chunk = filler[int(total_tokens * depth): int(total_tokens * (depth + 0.3))]
        chunks.append(next_chunk)
    return " ".join(chunks)
```

类似 “三个魔幻词是什么？”的问题需要检索所有三个。单针成功不代表多针成功。

### 第 3 步：多跳变量追踪（RULER 风格）

```python
haystack = """X1 = 42. ... (filler) ... X2 = X1 + 10. ... (filler) ... X3 = X2 * 2."""
question = "What is X3?"
```

答案需要连接三个赋值。前沿模型在 128k 上准确率通常降至 50-70%。

### 第 4 步：在你体系上评测 LongBench v2

```python
from datasets import load_dataset
longbench = load_dataset("THUDM/LongBench-v2")

def eval_model_on_longbench(model, subset="single-doc-qa"):
    tasks = [x for x in longbench["test"] if x["task"] == subset]
    correct = 0
    for x in tasks:
        answer = model.complete(x["context"] + "\n\nQ: " + x["question"], max_tokens=20)
        if normalize(answer) == normalize(x["answer"]):
            correct += 1
    return correct / len(tasks)
```

报告各类别准确率。聚合得分会掩盖任务级别的巨大差异。

## 注意事项

- **仅用 NIAH 评测。** 100 万 token 能通过 NIAH 并不代表多跳表现良好。必须运行 RULER 或自定义多跳测试。
- **均匀采样深度。** 许多实现只测试深度 0.5。测试深度 0, 0.25, 0.5, 0.75, 1.0 —— “中间迷失”效应是真实存在的。
- **针与填充文本的词汇重叠。** 若针含关键词与填充文本重叠，检索将变得简单。使用 NoLiMa 风格的无重叠针。
- **忽略延迟。** 100 万 token 的 prompt 预填充耗时 30-120 秒。测量首次输出时间与准确率。
- **厂商自报数据。** OpenAI、Google、Anthropic 等都发布自己的成绩。应始终在你的用例中独立复测。

## 何时使用

2026 年方案：

| 场景           | 评测基准                        |
| -------------- | ----------------------------- |
| 快速合理性检查 | 自定义 NIAH，于 3 深度 × 3 长度 |
| 选模投产       | RULER（13 任务）于目标上下文长度 |
| 真实问答质量   | LongBench v2 单文档问答子集     |
| 多跳推理       | BABILong 或自定义变量追踪任务    |
| 会话 / 对话    | MRCR 8 针于目标上下文长度       |
| 模型升级回归   | 固化的内部 NIAH + RULER 测试框架，每次新模型均运行 |

生产规则：未获得 NIAH + 1 个推理任务在目标长度上验证的上下文窗口，永不信任。

## 交付示范

保存为 `outputs/skill-long-context-eval.md`：

```markdown
---
name: long-context-eval
description: Design a long-context evaluation battery for a given model and use case.
version: 1.0.0
phase: 5
lesson: 28
tags: [nlp, long-context, evaluation]
---

Given a target model, target context length, and use case, output:

1. Tests. NIAH depth × length grid; RULER multi-hop; custom domain task.
2. Sampling. Depths 0, 0.25, 0.5, 0.75, 1.0 at each length.
3. Metrics. Retrieval pass rate; reasoning pass rate; time-to-first-token; cost-per-query.
4. Cutoff. Effective retrieval length (90% pass) and effective reasoning length (70% pass). Report both.
5. Regression. Fixed harness, rerun on every model upgrade, surface deltas.

Refuse to trust a context window from the model card alone. Refuse NIAH-only evaluation for any multi-hop workload. Refuse vendor self-reported long-context scores as independent evidence.
```

## 练习

1. **简单。** 构建一个 3 深度（0.25、0.5、0.75）× 3 长度（1k、4k、16k）的 NIAH。对任意模型运行，绘制 3×3 通过率热力图。
2. **中等。** 添加三针版本。测量各长度下三针均被检索的成功率，并与同长度单针成功率对比。
3. **困难。** 构造一个嵌入 64k 填充文本中的变量追踪任务（X1 → X2 → X3，三跳）。测量三个前沿模型的准确率，报告各模型的有效推理长度。

## 关键词

| 术语        | 通俗说法          | 实际含义                          |
|-------------|-------------------|----------------------------------|
| NIAH        | 大海捞针          | 在填充文本中植入事实，问模型检索。   |
| RULER       | NIAH 升级版       | 包含 13 种检索/多跳/聚合/问答任务。  |
| 有效上下文   | 真实能力          | 准确率保持在阈值以上的上下文长度。    |
| 中间迷失    | 深度偏差          | 模型对长输入中间内容的关注度不足。     |
| 多针        | 多事实同时        | 多个事实植入；考察注意力调度，不仅是检索。 |
| MRCR        | 多轮指代          | 8、24、100 针指代，暴露注意力饱和。      |
| NoLiMa      | 非字面针          | 针和查询不共享字面词，需要推理。        |

## 延伸阅读

- [Kamradt (2023). Needle in a Haystack analysis](https://github.com/gkamradt/LLMTest_NeedleInAHaystack) — 原始 NIAH 仓库。  
- [Hsieh et al. (2024). RULER: What's the Real Context Size of Your Long-Context LMs?](https://arxiv.org/abs/2404.06654) — 多任务基准。  
- [Bai et al. (2024). LongBench v2](https://arxiv.org/abs/2412.15204) — 真实长上下文评测。  
- [Modarressi et al. (2024). NoLiMa: Non-lexical needles](https://arxiv.org/abs/2404.06666) — 更难的针。  
- [Kuratov et al. (2024). BABILong](https://arxiv.org/abs/2406.10149) — 稻草堆中推理。  
- [Liu et al. (2024). Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172) — 深度偏差论文。
