# LLM 功能的 A/B 测试 — GrowthBook、Statsig 及直觉（vibes）问题

> 传统的 A/B 测试并非为非确定性（non-deterministic）的大语言模型（LLM）设计。关键区别是：evals 回答“模型能否完成任务？”，A/B 测试回答“用户是否在乎？”。两者缺一不可；靠直觉（vibe checks）发布已经过时。2026年测试重点：提示工程（措辞）、模型选择（GPT-4 vs GPT-3.5 vs 开源 OSS；准确性 vs 成本 vs 延迟），生成参数（temperature，top-p）。真实案例：一个聊天机器人奖励模型变量提升了 70% 会话长度，留存提升 30%；Nextdoor AI 的主题行实验经过奖励函数优化后点击率提升 1%；Khan Academy 的 Khanmigo 在延迟与数学准确性之间反复迭代。平台选择：**Statsig**（2025年9月被 OpenAI 以11亿美元收购）——顺序测试、CUPED、全功能一体化。**GrowthBook** —— 开源、仓库原生、支持贝叶斯（Bayesian）+ 频率学派（Frequentist）+ 顺序引擎、CUPED、SRM 检查、Benjamini-Hochberg+Bonferroni 校正。根据你团队对仓库 SQL 的偏好及对“是否被 OpenAI 收购”因素的在意程度进行选择。

**类型：** 学习  
**语言：** Python（标准库，顺序测试模拟器）  
**先决条件：** 第17阶段·13（可观测性）、第17阶段·20（渐进部署）  
**时间：** 约60分钟

## 学习目标

- 区分 evals（“模型能否完成任务”）与 A/B 测试（“用户是否在乎”）。
- 列举三个可测试轴（提示、模型、参数），并为每个选择度量指标。
- 解释 CUPED、顺序测试、Benjamini-Hochberg 多重比较校正。
- 根据仓库 SQL 使用习惯和公司收购态度选择 Statsig 或 GrowthBook。

## 问题描述

你手动调整了系统提示，感觉更好了，于是发布了它。转换率变动只是噪声，你怪指标不准。或者你发布了新模型，转换率没有变化——是模型退化了还是变化太小无法检测？你不知道，因为没做 A/B 测试。

Evals 回答模型能否在标注集上完成任务，但不回答用户是否更偏好模型输出。只有控制的在线实验能回答后者，并且实验需要足够的统计功效、控制非确定性因素，且校正多重比较。

## 概念

### Evals 与 A/B 测试

**Evals** —— 离线、标注集、评判者（评分标准／LLM 作为评判者／人工）。问题是：“在这个固定分布上，输出是否正确/有用/安全？”

**A/B 测试** —— 在线、真实用户、随机分配。问题是：“新版本是否提升了关键用户指标？”

两者缺一不可。Evals 用来捕捉模型质量回退；A/B 确认产品影响。

### 测试什么

1. **提示工程** —— 措辞、系统提示结构、示例。指标：任务成功率、用户留存、单次请求成本。
2. **模型选择** —— GPT-4 vs GPT-3.5-Turbo vs Llama-OSS。指标：准确率（任务）+ 每次请求成本 + 99百分位延迟，多目标。
3. **生成参数** —— temperature、top-p、最大 tokens。指标：任务相关（输出多样性与确定性权衡）。

### CUPED —— 方差降低

Controlled-experiments Using Pre-Experiment Data。对比后期指标前先利用前期数据回归，减少后期方差。通常能降低 30-70% 方差，等同免费提升有效样本量。

实现：Statsig 和 GrowthBook 都支持。

### 顺序测试

经典 A/B 固定样本大小，顺序测试（“边看边决定”）能控制多次检查中的假阳性率。始终有效的顺序统计方法（mSPRT、Howard 的置信序列）允许在明显赢家出现时提前停止。

### 多重比较校正

同时做 20 个 95% 置信区间的 A/B 测试，预计有一个假阳性。Bonferroni 通过 α 除以测试数收紧阈值；Benjamini-Hochberg 控制假发现率，较宽松。GrowthBook 支持两者。

### SRM — 样本比例不匹配

通过分配散列随机分配用户到版本。如果 50/50 分配结果是 47/53，说明某处出错，SRM 检测能发现问题。两平台均实现。

### Statsig 与 GrowthBook 对比

**Statsig**：  
- 2025年9月，OpenAI 以11亿美元收购。托管 SaaS。  
- 支持顺序测试、CUPED、持出人群对照。  
- 一体化方案：特征标识 + 实验 + 可观测性。  
- 适合：团队需要集成产品，不在意 OpenAI 归属。

**GrowthBook**：  
- 开源（MIT）；仓库原生，直接读取 Snowflake/BigQuery/Redshift 数据。  
- 多引擎：贝叶斯、频率学派、顺序测试。  
- 支持 CUPED、SRM、Bonferroni、BH 校正。  
- 可自托管或云托管。  
- 适合：专注仓库 SQL 的数据团队，喜欢开源。

### 非确定性增加功效复杂度

相同提示产生不同输出。传统功效计算假设样本独立同分布（IID）。LLM 非确定性导致有效样本量低于名义样本量。留出约 1.3-1.5 倍样本数作为安全边际。

### 真实案例成果

- 聊天机器人奖励模型版本：提高 70% 会话长度，留存提升 30%。  
- Nextdoor 主题行：奖励函数调整后点击率提升 1%。  
- Khan Academy Khanmigo：在延迟和数学准确性之间迭代权衡。

### 反模式：凭感觉直接发布

每个资深工程师都能说出“凭感觉感觉更好”却没做 A/B 测试直接发布的功能，多数导致团队数月未察觉的产品指标倒退。A/B 是强制函数。

### 你应该记住的数字

- Statsig 被 OpenAI 收购价格：11亿美元，2025年9月。  
- GrowthBook：开源 MIT；贝叶斯 + 频率学派 + 顺序测试。  
- CUPED 方差降低：30-70%。  
- LLM 非确定性 → 额外 30-50% 样本缓冲。

## 使用示例

`code/main.py` 模拟一个固定边界和顺序检测边界的顺序 A/B 测试。演示顺序方法如何让你提前停止。

## 交付产物

本课输出文件 `outputs/skill-ab-plan.md`，根据特性变动、工作负载、基准选平台、关卡数和样本量。

## 练习

1. 运行 `code/main.py`。基准转化率3%，期望提升5%，达到80%统计功效需要多少样本？
2. 为一家受医疗监管的自托管客户选择 Statsig 还是 GrowthBook。  
3. 设计一个测试 GPT-4 与 GPT-3.5 在单解决工单成本上的 A/B 测试。主要指标、护栏指标、次要指标分别是什么？  
4. 你的金丝雀测试通过，但 A/B 显示转化率下降1.2%。你发布吗？写出升级标准。  
5. 对前期方差为后期60%的数据应用 CUPED，计算有效样本提升。

## 关键词

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| Eval | “离线测试” | 基于标注集的模型能力评估 |
| A/B test | “实验” | 真实用户的在线随机对比 |
| CUPED | “方差降低” | 利用前期数据做回归削减方差 |
| Sequential test | “边看边测” | 始终有效的顺序检测允许提前停止 |
| Multiple comparison | “多重错误率” | 多测试导致假阳性增多 |
| Bonferroni | “严格校正” | α 除以测试个数 |
| Benjamini-Hochberg | “BH 假发现率” | 假发现率控制，更宽松 |
| SRM | “样本比例失配” | 样本分配异常，存在分配逻辑错误 |
| Statsig | “OpenAI 所有” | 2025年被 OpenAI 收购的商业一体化平台 |
| GrowthBook | “开源平台” | MIT 协议仓库原生平台 |
| mSPRT | “顺序比率检验” | 经典顺序统计程序 |

## 参考资料

- [GrowthBook — 如何对 AI 做 A/B 测试](https://blog.growthbook.io/how-to-a-b-test-ai-a-practical-guide/)  
- [Statsig — 超越提示词：数据驱动的 LLM 优化](https://www.statsig.com/blog/llm-optimization-online-experimentation)  
- [Statsig vs GrowthBook 对比](https://www.statsig.com/perspectives/ab-testing-feature-flags-comparison-tools)  
- [Deng 等 — CUPED](https://www.exp-platform.com/Documents/2013-02-CUPED-ImprovingSensitivityOfControlledExperiments.pdf)  
- [Howard — 置信序列](https://arxiv.org/abs/1810.08240)
