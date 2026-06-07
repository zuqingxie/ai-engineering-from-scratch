# STaR、V-STaR、Quiet-STaR — 自我学习推理

> 最小的自我改进循环就藏在推理链中。模型生成思维链，保留那些得出正确答案的链条，并基于它们进行微调。这就是 STaR（Self-Taught Reasoner，自我学习推理器）。V-STaR 增加了一个验证器，使推理时的选择更好。Quiet-STaR 将推理细化到每个 token 上。这三种方法都有效。它们都不是魔法——循环会保留任何偶然到达正确答案的捷径。

**类型:** 学习  
**语言:** Python（标准库，引导循环模拟器）  
**先决条件:** 阶段 13 · 01-03（推理和 CoT），阶段 15 · 01（长远视角框架）  
**时长:** 约 60 分钟

## 问题

教模型推理的直接方法是收集人类写的推理过程。这既昂贵又缓慢，而且受限于高质量Chain-of-Thought（思维链）的人类写作量。

STaR（Zelikman 等，2022）提出：如果模型自己写推理链，并根据已知答案评分呢？循环为：

1. 采样一个推理链和答案。  
2. 如果最终答案正确，保留该推理链。  
3. 对保留的推理链微调模型。  
4. 重复。

这确实有效。GSM8K 和 CommonsenseQA 在无新人类标注情况下均有提升。但循环有内在偏差：任何得出正确答案的推理链都会被保留，无论推理本身是否合理。V-STaR（Hosseini 等，2024）通过学习验证器改进该点；Quiet-STaR（Zelikman 等，2024）将该思想推广到每个 token 的内部推理。

## 概念

### STaR：基于有效结果引导学习

从具有一定弱推理能力的基础模型开始。每题采样推理链及答案。如果答案与标签匹配，则保留（问题，推理链，答案）三元组。基于保留集合微调模型。反复进行。

有一个关键点。如果模型从未正确解决某题，循环无法学习。STaR 引入 **rationalization（推理合理化）**：对模型未解题目，注入正确答案提示，重新提示模型生成通往该答案的推理链。合理化推理链加入训练集。

原论文结果（Zelikman 等，2022）：GPT-J 基础模型在多轮带合理化的 STaR 循环后，GSM8K 准确率从 5.8% 提升至 10.7%，绝对提升约 5 个百分点。CommonsenseQA 上，STaR 训练的 GPT-J 6B 达到 72.5%，接近微调的 GPT-3 175B (~73%)——后者是约 30 倍大型的模型，且训练于人工标注推理链。

### V-STaR：用 DPO 训练验证器

STaR 舍弃错误推理链。Hosseini 等（2024）注意到这也是数据：每对（推理链，“是否正确”）可用于训练验证器。他们用 Direct Preference Optimization（直接偏好优化，DPO）基于正确及错误推理构建排序器。推理时采样 N 条候选，挑选验证器评分最高的。

报告增益：在 GSM8K 和 MATH 上比原自我改进基线提升 +4 到 +17 个百分点，大部分收益来自用验证器做推理时选择，而非额外微调生成器。

### Quiet-STaR：每 token 内部推理

Zelikman 等（2024）提出：如果模型在每个 token 位置都学会生成短内在推理会怎样？Quiet-STaR 训练模型在预测每个 token 前产生隐藏“思考”向量，再用学习到的权重将该思考感知预测与基础预测混合。

结果：Mistral 7B 在 GSM8K 零-shot 任务上准确率从 5.9% 提升至 10.9%，CommonsenseQA 从 36.3% 提升到 47.2%，无需任务特定微调。模型学会“何时思考”——难的 token 得到更长内部推理；简单的几乎没有。

### 三者共享的安全隐患

三种方法均以最终答案作为梯度信号。通过有缺陷推理、走捷径、猜测或非泛化模式得出的正确答案推理，都会被正向强化。在训练分布内，这些捷径有效；分布外则默默失败。

V-STaR 的验证器通过排名缓解该问题，但验证器训练用的仍是相同标签集。它可能偏好格式良好的错误推理胜过诚实的不确定。更安全的设计是把 STaR 数据与（a）基于流程监督的奖励模型（奖励中间步骤，而非仅答案）和（b）持出OOD评价（破坏简单捷径）结合。

### 比较

| 方法 | 训练信号 | 推理成本 | 数据浪费 | 已知失败模式 |
|---|---|---|---|---|
| STaR | 正确时保留（推理链，答案） | 1x | 丢弃所有错误推理链 | 捷径推理链 |
| STaR + 推理合理化 | 以上 + 注入正确答案重试 | 1x | 更少 | 合理化推理链可能不可信 |
| V-STaR | STaR + 对正误推理链的 DPO 验证器 | Nx（best-of-N） | 最小 | 验证器可能强化自信错误 |
| Quiet-STaR | 每 token 推理链 + 混合权重 | 1.5-3x | 最小 | 依然是答案条件梯度 |

### 2026 技术栈中的位置

STaR 是早期方法，但该模式在 2025-2026 年广泛出现。可验证数学题上的 RL（DeepSeek-R1、Kimi-k1.5、o1）是 STaR 的答案条件梯度信号放大。流程奖励模型（Lightman 等，2023；OpenAI 的“逐步验证”）是流程监督替代。AlphaEvolve（第3课）是代码版 STaR，用程序评估器取代标签。Darwin Godel 机器（第4课）是代理自建的 STaR。

理解 STaR 让这些方法变得通透。它是最低可行自我改进循环。

## 使用

`code/main.py` 在玩具算术任务上运行模拟 STaR 循环。你可以观察：

- 准确率如何随着 bootstrap 循环上升。  
- 捷径推理如何渗入：模拟器含“懒惰”推理类别，40% 得对但泛化差。观察 STaR 是否保留它们。  
- 验证器（V-STaR 方式）如何在推理时帮助，但无法完全剔除训练阶段引入的捷径。

## 上线

`outputs/skill-star-loop-reviewer.md` 帮你审查拟议的自学推理流水线，训练前进行把关。

## 练习

1. 运行模拟器。将捷径频率设为 0，然后改为 0.4。尽管两次训练分布准确率均 >90%，最终准确率差距有多大？

2. 给模拟器增添持出OOD测试。分别从不同分布采样测试题，评估自举模型在训练分布和OOD集上的表现。量化差距。

3. 阅读 Quiet-STaR 论文（arXiv:2403.09629）第3节，分别用三句话解释“end-of-thought” token 和混合权重头。

4. 比较 STaR 的“答对即保留”过滤与过程监督的替代方案——对每一步推理单独奖励。分析标注成本差异及潜在质量差异。

5. 设计一个可用于上线模型检测捷径推理的评估。无需完美，只需能识别 STaR 循环易强化的简单捷径。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|---|---|---|
| STaR | “Self-Taught Reasoner” 自学推理器 | 基于模型生成、正确答案的推理链微调模型；循环进行 |
| Rationalization | “Hinted retry” 推理合理化 | 对失败题注入正确答案提示，重新生成推理链 |
| V-STaR | “Verifier STaR” 验证器 STaR | 用 DPO 训练验证器区分对错推理链，推理时选择 |
| Quiet-STaR | “Per-token rationales” 每 token 推理 | 每个 token 位置生成隐藏思考，混合预测输出 |
| Answer-conditioned gradient | “Outcome-based signal” 答案条件梯度 | 训练信号基于最终答案，而非推理步骤 |
| Process reward model | “Step-level verifier” 步骤级奖励模型 | 基于每步正误训练奖励模型，与 STaR 目标不同 |
| Shortcut rationale | “Right answer, wrong reasoning” 答对错推理 | 通过非泛化模式达到正确答案的推理链，STaR 保留 |

## 延伸阅读

- [Zelikman 等（2022）。STaR: Bootstrapping Reasoning With Reasoning](https://arxiv.org/abs/2203.14465) — 原始论文。  
- [Hosseini 等（2024）。V-STaR: Training Verifiers for Self-Taught Reasoners](https://arxiv.org/abs/2402.06457) — 增加推理时选择的 DPO 验证器。  
- [Zelikman 等（2024）。Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking](https://arxiv.org/abs/2403.09629) — 每 token 内部推理。  
- [Lightman 等（2023）。Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) — 流程奖励模型，替代梯度信号。  
- [DeepSeek-R1 论文（arXiv:2501.12948）](https://arxiv.org/abs/2501.12948) — 可验证任务上 RL，STaR 扩展至前沿训练。
