# LLM 评估 — RAGAS、DeepEval、G-Eval

> 精确匹配（Exact-match）和 F1 无法捕捉语义等价。人工审核无法扩展。LLM 作为裁判（LLM-as-judge）是生产环境中的答案 —— 只需经过充分校准即可信任结果数值。

**类型：** 构建  
**语言：** Python  
**先决条件：** 第5阶段 · 13（问答），第5阶段 · 14（信息检索）  
**时间：** 约75分钟

## 问题

你的 RAG 系统回答：“2007年6月29日。”  
黄金参考答案是：“2007年6月29日。”  
Exact Match 得分为 0，F1 得分约为 75%。人工评分是 100%。

现在将问题扩展到 10,000 个测试用例，再乘以检索器、分块、提示词或模型的每一次更改。你需要一个能理解语义、廉价且可扩展运行、不隐瞒回归错误、并能突出正确失败模式的评估器。

2026 年有三个框架专注解决此问题。

- **RAGAS。** Retrieval-Augmented Generation ASsessment。四个 RAG 指标（可靠性faithfulness、答案相关性answer-relevance、上下文精确度context-precision、上下文召回context-recall），基于 NLI + LLM 裁判后端。研究支撑，轻量级。
- **DeepEval。** LLM 的 pytest。包含 G-Eval、任务完成度、幻觉检测、偏见指标。原生支持 CI/CD。
- **G-Eval。** 一种方法（及 DeepEval 的指标）：带有链式思维的 LLM 裁判，定制标准，0-1 评分。

三者均依赖于 LLM 作为裁判。课程构建方法直觉及其信任层。

## 概念

![四个评估维度，LLM 作为裁判架构](../assets/llm-evaluation.svg)

**LLM 作为裁判（LLM-as-judge）。** 用 LLM 替代静态指标，基于评分标准对输出进行评分。给定 `(query, context, answer)`，提示裁判 LLM：“对可靠性评分 0-1。”返回分数。

为何有效：LLM 在极低成本下近似人工判断。GPT-4o-mini 每例评分约 0.003 美元，使得千例回归 eval 运行成本低于5美元。

为何无声失败：

1. **裁判偏见。** 裁判偏好更长答案、自家模型家族的答案、匹配提示风格的答案。
2. **JSON 解析失败。** 错误 JSON → NaN 分数 → 静默排除。RAGAS 用户对此十分熟悉。用 try/except 和显式失败模式进行防护。
3. **模型版本漂移。** 升级裁判模型改变每个指标。冻结裁判模型及版本。

**RAG 四指标。**

| 指标           | 问题                           | 后端                                |
|--------------|------------------------------|-----------------------------------|
| 可靠性（Faithfulness）    | 答案中的每个断言是否源自检索上下文？          | 基于 NLI 的蕴含判定（entailment）             |
| 答案相关性（Answer relevance） | 答案是否回答了问题？                         | 从答案生成假设问题；与真实问题比较             |
| 上下文精确度（Context precision） | 检索到的分块中，实际相关的比例是多少？            | LLM 裁判                               |
| 上下文召回（Context recall）     | 检索是否返回了所有必要内容？                   | 用 LLM 裁判与黄金答案比对                    |

**G-Eval。** 自定义标准：“答案是否引用了正确的来源？” 框架自动展开链式思维评估步骤，再给出 0-1 评分。适合 RAGAS 之外的领域特定质量维度。

**校准（Calibration）。** 未获得与人工标签的相关性前，绝不信任原始裁判分数。运行100个人工标注样本。绘制裁判与人工分数散点图。计算 Spearman 相关系数 (rho)。若 rho < 0.7，则裁判规则需改进。

## 实现

### 第1步：利用 NLI 进行可靠性评估（RAGAS 风格）

```python
from typing import Callable
from transformers import pipeline

nli = pipeline("text-classification",
               model="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
               top_k=None)

# `llm` 是任意可调用对象：prompt str -> 生成的 str。
# 示例： llm = lambda p: client.messages.create(model="claude-haiku-4-5", ...).content[0].text
LLM = Callable[[str], str]


def atomic_claims(answer: str, llm: LLM) -> list[str]:
    prompt = f"""将此答案拆解为简单事实断言（一行一个）：
{answer}
"""
    return llm(prompt).splitlines()


def faithfulness(answer: str, context: str, llm: LLM) -> float:
    claims = atomic_claims(answer, llm)
    if not claims:
        return 0.0
    supported = 0
    for claim in claims:
        result = nli({"text": context, "text_pair": claim})[0]
        entail = next((s for s in result if s["label"] == "entailment"), None)
        if entail and entail["score"] > 0.5:
            supported += 1
    return supported / len(claims)
```

将答案分解为原子断言。用 NLI 对每个断言与检索上下文比对。可靠性为支持断言的比例。

### 第2步：答案相关性

```python
import numpy as np
from sentence_transformers import SentenceTransformer

# encoder：任意实现了 .encode(texts, normalize_embeddings=True) -> ndarray 接口的模型
# 例如 encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")

def answer_relevance(question: str, answer: str, encoder, llm: LLM, n: int = 3) -> float:
    prompt = f"为此答案写出 {n} 个可能对应的问题：\n{answer}"
    generated = [line for line in llm(prompt).splitlines() if line.strip()][:n]
    if not generated:
        return 0.0
    q_emb = np.asarray(encoder.encode([question], normalize_embeddings=True)[0])
    g_embs = np.asarray(encoder.encode(generated, normalize_embeddings=True))
    sims = [float(q_emb @ g_emb) for g_emb in g_embs]
    return sum(sims) / len(sims)
```

若答案暗示的问题与所提问题不符，答案相关性会降低。

### 第3步：G-Eval 自定义指标

```python
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams, LLMTestCase

metric = GEval(
    name="Correctness",
    criteria="答案应事实准确并符合预期输出。",
    evaluation_steps=[
        "阅读预期输出。",
        "阅读实际输出。",
        "列举实际输出中的事实断言。",
        "对每个断言，标记预期输出是否支持。",
        "返回分数 = 支持的断言比例。",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
)

test = LLMTestCase(input="When was the first iPhone released?",
                   actual_output="June 29th, 2007.",
                   expected_output="June 29, 2007.")
metric.measure(test)
print(metric.score, metric.reason)
```

评估步骤即评分标准。显式步骤比暗示的“评分0-1”提示更稳定。

### 第4步：CI 门控

```python
import deepeval
from deepeval.metrics import FaithfulnessMetric, ContextualRelevancyMetric


def test_rag_system():
    cases = load_regression_cases()
    faith = FaithfulnessMetric(threshold=0.85)
    rel = ContextualRelevancyMetric(threshold=0.7)
    for case in cases:
        faith.measure(case)
        assert faith.score >= 0.85, f"faithfulness regression on {case.id}"
        rel.measure(case)
        assert rel.score >= 0.7, f"relevancy regression on {case.id}"
```

作为 pytest 文件发布。每个 PR 运行。出现回归阻止合并。

### 第5步：从零开始的玩具评估

查看 `code/main.py`。仅使用标准库实现可靠性（答案断言与上下文重叠）和相关性（答案与问题词的重叠）近似。非生产环境，仅演示形态。

## 陷阱

- **无校准。** 裁判与人工标签相关系数为0.3即为噪声。发布前必须运行校准。
- **自评。** 用相同 LLM 生成和评判，会导致分数虚高 10-20%。评判模型应属于不同家族。
- **配对评判中的位置偏见。** 裁判偏好第一个选项。始终随机顺序，并两次评判。
- **原始汇总隐藏失败。** 平均得分 0.85 常掩盖 5% 灾难性失败。务必查看底层分位。
- **黄金数据集腐烂。** 未版本化的评估集随时间漂移，破坏纵向比较。每次变更均要打标签。
- **LLM 成本。** 大规模时，裁判调用是成本主导。选用满足校准阈值的最廉价模型，如 GPT-4o-mini、Claude Haiku、Mistral-small。

## 应用

2026 堆栈：

| 用例               | 框架                   |
|------------------|----------------------|
| RAG 质量监控          | RAGAS（4指标）          |
| CI/CD 回归门控       | DeepEval + pytest      |
| 自定义领域标准         | DeepEval 中的 G-Eval      |
| 线上实时流量监控        | RAGAS 参考无关模式        |
| 人工巡查Spot Check | LangSmith 或 Phoenix 注释 UI |
| 红队测试 / 安全评估      | Promptfoo + DeepEval    |

典型堆栈：监控用 RAGAS，CI 用 DeepEval，新增维度用 G-Eval。三者一起运行，争议促进有益洞察。

## 部署

保存为 `outputs/skill-eval-architect.md`：

```markdown
---
name: eval-architect
description: 使用校准裁判和 CI 门控设计 LLM 评估方案。
version: 1.0.0
phase: 5
lesson: 27
tags: [nlp, evaluation, rag]
---

给定用例（RAG / agent / 生成任务），输出：

1. 指标。可靠性 / 相关性 / 上下文精确度 / 上下文召回 + 任何带标准的自定义 G-Eval 指标。
2. 裁判模型。命名模型 + 版本，成本与准确性权衡理由。
3. 校准。人工标注样本数量，目标 Spearman rho > 0.7。
4. 数据集版本控制。标签策略，变更日志，分层方法。
5. CI 门控。每指标阈值，回归窗口逻辑，低分段警报。

拒绝使用未与≥50个人工标注样本测试的裁判。拒绝自评（同一模型生成与评判）。拒绝仅汇总报告而不揭露底10%表现。拒绝无平行基线评估的裁判升级管线。
```

## 练习

1. **简单。** 用 RAGAS 评估 10 个已知有幻觉的 RAG 示例。验证可靠性指标均能检出。
2. **中等。** 手工标注 50 个问答答案的正确性（0-1）。用 G-Eval 评分。计算裁判与人工的 Spearman 相关系数。
3. **困难。** 利用 DeepEval 构建 pytest CI 门控。有意回归检索器。验证门控失败。通过最低10%阈值检查添加底分段警报。

## 关键词

| 术语            | 大众说法                 | 实际含义                                   |
|---------------|----------------------|----------------------------------------|
| LLM-as-judge | 用 LLM 打分              | 提示裁判模型根据评分标准对输出给出0-1分数。               |
| RAGAS         | RAG 指标库             | 开源评估框架，包含4个无参考的 RAG 指标。                    |
| Faithfulness  | 答案是否有依据            | 答案断言被检索上下文蕴含的比例。                           |
| Context precision | 检索分块是否相关           | 前 K 个检索分块中实际相关的比例。                           |
| Context recall | 检索是否找齐全部内容       | 检索分块支持黄金答案断言的比例。                            |
| G-Eval        | 定制 LLM 裁判           | 评分标准 + 链式思维评估步骤 + 0-1 分数。                     |
| Calibration   | 信但需验证              | 裁判分数与人工分数之间的 Spearman 相关系数。                 |

## 延伸阅读

- [Es et al. (2023). RAGAS: Automated Evaluation of Retrieval Augmented Generation](https://arxiv.org/abs/2309.15217) — RAGAS 论文。  
- [Liu et al. (2023). G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment](https://arxiv.org/abs/2303.16634) — G-Eval 论文。  
- [DeepEval 文档](https://deepeval.com/docs/metrics-introduction) — 开源生产级堆栈。  
- [Zheng et al. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685) — 偏见、校准、局限。  
- [MLflow GenAI Scorer](https://mlflow.org/blog/third-party-scorers) — 集成 RAGAS、DeepEval、Phoenix 的统一框架。
