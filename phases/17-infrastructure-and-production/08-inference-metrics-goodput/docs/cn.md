# 推理指标 — TTFT、TPOT、ITL、Goodput、P99

> 有四个指标决定推理部署是否正常工作。TTFT 是预填充时间（prefill）加排队时间（queue）加网络时间（network）。TPOT（等同于 ITL）是每个 token 的内存受限解码耗时。端到端延迟（E2E）是 TTFT 加上 TPOT 乘以输出长度。吞吐率是整个集群每秒处理的 token 数。但产品真正关心的是 goodput —— 同时满足所有 SLO 的请求比例。高吞吐率但低 goodput 意味着你处理了大量无法按时到达用户的 token。2026 年 TRT-LLM 上 Llama-3.1-8B-Instruct 的参考指标：平均 TTFT 162 ms、平均 TPOT 7.33 ms、平均 E2E 1093 ms。永远报告 P50、P90、P99 —— 切勿只报告均值。并且注意测量陷阱：GenAI-Perf 从 ITL 计算中排除 TTFT，而 LLMPerf 包含它；两个工具对同一次运行的 TPOT 产生不同结果。

**类型：** 学习  
**语言：** Python（标准库，小型百分位计算器和 goodput 报告器）  
**先修知识：** 第 17 阶段·04（vLLM 服务内部）  
**时长：** 约 60 分钟

## 学习目标

- 精确定义 TTFT、TPOT、ITL、E2E、吞吐率和 goodput，并指出各自测量的组成部分。
- 解释为什么均值是 LLM 服务中的错误统计值，以及如何解读 P50/P90/P99。
- 构建一个多约束 SLO（如 TTFT<500 ms 且 TPOT<15 ms 且 E2E<2 s），并计算 goodput。
- 说出两个对同一次运行的 TPOT 测量结果存在分歧的基准工具，并解释原因。

## 问题描述

“我们的吞吐率是 15,000 token/秒。”那又怎么样？如果 40% 的请求端到端超过 2 秒，用户就会放弃会话。单看吞吐率无法判断产品是否正常。

推理延迟有多个维度，各自的失效方式不同。预填充是计算受限，随 prompt 长度线性增长。解码是内存受限，随批量大小线性增长。排队延迟是运维问题。网络是物理距离问题。你需要不同的指标来衡量每个维度，需要百分位数统计，还需要一个综合指标来判断“用户是否得到了他们预期的体验”——这就是 goodput。

## 概念说明

### TTFT — 到第一个 token 的时间

`TTFT = queue_time + network_request + prefill_time`

当 prompt 很长时，预填充时间占主导。以 Llama-3.3-70B FP8 在 H100 上为例，32k 长度的 prompt 纯预填充时间约为 800 ms。排队时间由调度器负载行为决定。网络请求时间包括 TLS 的线缆传输时间。TTFT 是用户看到输出流开始之前的延迟。

### TPOT / ITL — Token 间延迟

同一个量的多个名称。`TPOT`（每个输出 token 的时间）、`ITL`（token 间延迟）、`每 token 的解码延迟` — 均指相同含义。指的是除第一个 token 之后连续两 token 之间的时间。

`TPOT = (decode_forward_time + scheduler_overhead) / tokens_produced`

在同一 Llama-3.3-70B H100 堆栈启用分块预填充时，TPOT 平均约为 7 ms。没有分块预填充时，若邻近序列正长时间预填充，TPOT 可能暴涨至 50 ms。关注 P99，不要只看均值。

### E2E 延迟

`E2E = TTFT + TPOT * output_tokens + network_response`

对于长输出（>500 token），E2E 由 TPOT 主导。对于带长 prompt 的短输出，E2E 由 TTFT 主导。需要按输出长度分条件报告 E2E。

### 吞吐率

`throughput = total_output_tokens / elapsed_time`

聚合指标，反映集群效率。不能反映单次请求的健康度。

### Goodput — 你真正关心的指标

`goodput = 满足 (TTFT <= a) 且 (TPOT <= b) 且 (E2E <= c) 的请求比例`

SLO 是多约束组合。只有所有约束同时成立的请求才被视为“好”。goodput 是这部分请求的占比。高吞吐率但 goodput 仅为 60% 表示失败。低吞吐率且 goodput 达 99% 是目标。

2026 年，goodput 是 MLPerf Inference v6.0 提交和 AI 平台内部 SLA 跟踪使用的指标。

### 为什么均值是错误的统计量

LLM 延迟分布呈右偏态。一个解码批次，如果旁边有一个长时间的预填充邻居，可能连发 500 token 的 TPOT ~7 ms，同时也有 20 token 的 TPOT ~60 ms。均值 TPOT 是 9 ms，但 P99 TPOT 达 65 ms。用户经常遇到 P99，正是他们离开的原因。

务必报告 (P50, P90, P99) 三元组。对用户体验来说，P99 是你优化的重点。

### 参考数值 — 2026 年 TRT-LLM 上的 Llama-3.1-8B-Instruct

- 平均 TTFT：162 ms  
- 平均 TPOT：7.33 ms  
- 平均 E2E：1,093 ms  
- P99 TPOT：依分块预填充配置变动，10-25 ms 不等。

这些是 NVIDIA 发布的参考基准。指标会随模型规模（70B 约是 3-5 倍）、硬件（H100 VS B200 约 3 倍）和负载变化。

### 测量陷阱

两个 2026 年常用基准工具对同一次运行的 TPOT 测量结果不一致：

- **NVIDIA GenAI-Perf**：从 ITL 计算中排除 TTFT，ITL 从第 2 个 token 开始计算。  
- **LLMPerf**：包含 TTFT，ITL 从第 1 个 token 开始计算。

对于一个 TTFT 500 ms 且 100 个输出 token 总解码时间 700 ms 的请求，GenAI-Perf 报告 `ITL = 700/99 = 7.07 ms`，LLMPerf 报告 `ITL = 1200/100 = 12.00 ms`。工具选择影响结果数值。

务必说明使用哪个工具，并公开定义。

### 构建 SLO

2026 年一个面向消费者的 70B 聊天模型合理 SLO：

- TTFT P99 ≤ 800 ms  
- TPOT P99 ≤ 25 ms  
- 对于少于 300 token 的输出，E2E P99 ≤ 3 秒  
- Goodput 目标 ≥ 99%

企业级 SLO 会收紧 TTFT（200-400 ms）并放宽 E2E。关键是写下来，测量这三项，追踪 goodput 作为一个综合指标。

### 如何测量

- 运行真实流量或真实感知合成流量（如 LLMPerf 参数 `--mean-input-tokens 800 --stddev-input-tokens 300 --mean-output-tokens 150`）。  
- 基准测试运行目标设定在峰值并发的 2 倍。  
- 运行 30-50 次迭代，取合并样本的百分位。  
- 公布时注明工具名称、版本、模型、硬件、并发、prompt 分布。

## 使用它

`code/main.py` 是一个简单的 goodput 计算器。生成合成延迟分布，应用 SLO 计算 goodput。还展示了同一轨迹上 GenAI-Perf 与 LLMPerf 的 TPOT 差异。

## 交付它

本课生成 `outputs/skill-slo-goodput-gate.md`。给定工作负载和 SLO，它生成一个 CI/CD 就绪的基准测试方案，以 goodput 而非吞吐率作为部署门槛。

## 练习

1. 运行 `code/main.py`。生成带 1% 尾部尖峰的分布。当你将 P99 TPOT 从 30 ms 收紧到 15 ms 时，goodput 会如何变化？  
2. 某供应商声称“Llama 3.3 70B H100 下吞吐率 15,000 tok/s”，请列出三个你会询问的问题以判断可信度。  
3. 为什么分块预填充保护了 P99 TPOT，却不保护均值 TPOT？  
4. 为语音助手构建一个消费者 SLO（第一个 token 是听到的，不是读取的），哪个指标对用户体验最明显？  
5. 阅读 LLMPerf README 和 GenAI-Perf 文档，找出两个工具在其他三个指标上的分歧。

## 关键词

| 术语               | 常用说法              | 实际含义                                          |
|--------------------|-----------------------|---------------------------------------------------|
| TTFT               | “time to first token” | 排队 + 网络 + 预填充；长 prompt 时由预填充主导     |
| TPOT               | “time per output token” | 内存受限的单 token 解码耗时，从第一个 token 之后算起 |
| ITL                | “inter-token latency”  | 多数工具与 TPOT 相同（部分工具不同——见 GenAI-Perf）  |
| E2E                | “end to end”           | TTFT + TPOT*输出长度 + 响应端网络延迟             |
| 吞吐率（Throughput）| “tok/s”                | 集群效率；无延迟百分位时无用                        |
| Goodput            | “SLO-met rate”         | 同时满足所有 SLO 约束的请求比例                     |
| P99                | “tail”                 | 1/100 最差延迟；用户体验的关键指标                  |
| SLO 多约束         | “the joint”            | 三个延迟限制的逻辑与；只要有一项违规请求即失败       |
| GenAI-Perf vs LLMPerf | “the tool trap”        | 两工具对 ITL 是否包含 TTFT 存分歧                    |

## 深入阅读

- [NVIDIA NIM — LLM 基准指标](https://docs.nvidia.com/nim/benchmarking/llm/latest/metrics.html) — TTFT、ITL、TPOT 的权威定义。  
- [Anyscale — LLM 服务基准指标](https://docs.anyscale.com/llm/serving/benchmarking/metrics) — 另一套定义和测量方案。  
- [BentoML — LLM 推理指标](https://bentoml.com/llm/inference-optimization/llm-inference-metrics) — 实际部署中的应用测量。  
- [LLMPerf](https://github.com/ray-project/llmperf) — 基于 Ray 的开源基准工具。  
- [GenAI-Perf](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/client/src/c++/perf_analyzer/genai-perf/README.html) — NVIDIA 官方基准工具。  
- [MLPerf Inference](https://mlcommons.org/benchmarks/inference-datacenter/) — 行业内认可的基于 goodput 的基准。
