# 负载测试大语言模型（LLM）API — 为什么 k6 和 Locust 会误导你

> 传统负载测试工具并未针对流式响应、可变输出长度、逐令牌指标或 GPU 饱和设计。大多数团队会陷入两个陷阱。GIL 陷阱：Locust 在 Python GIL（全局解释器锁）下对令牌进行测量，令牌化与请求生成争抢资源，高并发时令牌化积压，导致报告的令牌间延迟被放大——瓶颈在客户端而非服务器。提示一致性陷阱：循环测试中的相同提示只覆盖令牌分布的一个点，真实流量具有可变长度和多样的前缀匹配。LLMPerf 使用 `--mean-input-tokens` + `--stddev-input-tokens` 解决该问题。2026 工具映射：LLM 专用工具（GenAI-Perf、LLMPerf、LLM-Locust、guidellm）用于令牌级准确性；**k6 v2026.1.0** + **k6 Operator 1.0 GA（2025年9月）** — 支持流式，Kubernetes 原生，分布式通过 TestRun/PrivateLoadZone CRDs，适合 CI/CD 门禁；Vegeta 用于 Go 常速饱和测试；Locust 2.43.3 仅支持 LLM-Locust 扩展做流式。负载模式包括：稳定状态、渐增、突发（自动扩缩容测试）、浸泡（内存泄漏检测）。

**类型：** 构建  
**语言：** Python（标准库、真实提示生成器 + 延迟收集器）  
**前置知识：** 第 17 阶段 · 08（推理指标），第 17 阶段 · 03（GPU 自动扩缩容）  
**时长：** 约 75 分钟

## 学习目标

- 解释使通用负载测试工具在 LLM API 上测得虚假结果的两个反模式（GIL 陷阱、提示一致性陷阱）。
- 根据目标选择合适的工具：LLMPerf（基准运行）、k6 + 流式扩展（CI 门禁）、guidellm（大规模合成测试）、GenAI-Perf（NVIDIA 参考）。
- 设计四种负载模式（稳定、渐增、突发、浸泡），并明确各自能捕获的故障类型。
- 使用输入令牌的均值 + 标准差构建真实提示分布，而非固定长度。

## 问题描述

你用 k6 测试 LLM 端点，500 并发请求，稳定。上线了。生产时实际只有 200 用户，服务却崩溃了——P99 TTFT（首次标记时间）暴涨，GPU 占用满载。

原因有二。第一，k6 发送了 500 条完全相同的提示——请求合并和前缀缓存让你误以为支持 500 并发解码，其实只是在处理一条。第二，k6 不能像人眼那样追踪流式响应的令牌间延迟；它只看到一个 HTTP 连接，而非 500 个令牌以不同间隔到达。

LLM 负载测试是一项独立学科。

## 核心概念

### GIL 陷阱（Locust）

Locust 使用 Python，在全局解释器锁（GIL）下在客户端执行令牌化。高并发时，令牌化排队等待请求生成完成，令牌间延迟报告包含了客户端令牌化堆积的时间。你误判是服务器性能瓶颈，实则是测试工具自身导致。

解决方法：LLM-Locust 扩展将令牌化移到独立进程，或使用编译语言实现的工具（如 k6、LLMPerf 中的 tokenizers.rs）。

### 提示一致性陷阱

所有已知负载测试工具只允许配置一个提示。在 10,000 次循环测试中，每次发送完全相同的提示。服务器每次看到相同前缀，前缀缓存命中率几乎 100%，吞吐率看起来很好。

解决方法：从提示分布抽样。LLMPerf 使用 `--mean-input-tokens 500 --stddev-input-tokens 150`，生成多样长度、多样内容。

### 四种负载模式

1. **稳定状态（Steady-state）** — 持续恒定请求率 30-60 分钟。捕获：基线性能退化。
2. **渐增（Ramp）** — 请求率线性从 0 增加到目标值，时长 15 分钟。捕获：容量拐点，预热异常。
3. **突发（Spike）** — 突然请求率提升 3-10 倍，持续 2 分钟后恢复。捕获：自动扩缩容延迟、队列饱和、冷启动影响。
4. **浸泡（Soak）** — 持续稳定请求 4-8 小时。捕获：内存泄漏、连接池漂移、观测数据过载。

### 2026 工具映射

**LLMPerf**（Anyscale） — Python 脚本，Rust 支撑的令牌化。使用均值/标准差提示。支持流式。是性能跑测的默认首选。

**NVIDIA GenAI-Perf** — NVIDIA 官方参考，基于 Triton 客户端，指标覆盖全面。其 ITL 不包括 TTFT，LLMPerf 包含。两工具对同一服务器产生不同 TPOT（总处理时间）。

**LLM-Locust**（TrueFoundry）— Locust 扩展，解决 GIL 陷阱。使用熟悉的 Locust DSL+流式指标。

**guidellm** — 大规模合成基准工具。

**k6 v2026.1.0** + **k6 Operator 1.0 GA（2025年9月）**：  
- k6 本体（Go 实现，编译型，无 GIL）增加流式感知指标。  
- k6 Operator 使用 TestRun / PrivateLoadZone CRDs，实现 Kubernetes 原生分布式测试。  
- 适用于 CI/CD 门禁和 SLA 测试。

**Vegeta** — Go 实现，功能简洁，固定速率 HTTP 饱和测试。不支持 LLM，适合网关/限流测试。

**Locust 2.43.3 原版** — 存在 GIL 陷阱。需配合 LLM-Locust 扩展使用。

### CI 中的 SLA 门禁

在 PR 上运行 k6，配置：

- 30-50 次迭代，均在基线 RPS 下。
- 门禁：P50/P95 TTFT，5xx 错误率 < 5%，TPOT 低于阈值。
- 违反则构建失败。

### 真实提示分布

基于真实流量样本（若有）或公开分布（如 ShareGPT 聊天提示、HumanEval 代码提示）构建。将均值+标准差输入 LLMPerf。务必避免单提示循环。

### 应记住的数字

- k6 Operator 1.0 GA：2025年9月。  
- k6 v2026.1.0：增加流式指标。  
- 典型 LLMPerf 运行：并发 X 下 100-1000 请求。  
- 典型 CI 门禁：每 PR 30-50 次迭代。  
- 四种负载模式：稳定、渐增、突发、浸泡。

## 使用指南

`code/main.py` 模拟带有真实提示分布的负载测试，测量有效 TPOT，演示提示一致性陷阱。

## 交付物

本课产生 `outputs/skill-load-test-plan.md` 文件。根据工作负载和 SLA，选择工具并设计四种负载模式。

## 练习题

1. 运行 `code/main.py`。比较一致与真实分布，差距在哪里？  
2. 编写 k6 脚本，满足 CI 门禁要求：P95 TTFT < 800 ms，100 并发，运行 5 分钟。  
3. 浸泡测试显示内存每小时增长 50 MB。列举三个可能原因及所需的检测手段。  
4. 突发测试从 10 RPS 瞬间提升至 100 RPS。有 Karpenter + vLLM 生产栈保障时，预期恢复时间是多少（参见第17阶段·03和18）？  
5. 在同一服务器上，GenAI-Perf 报告 TPOT=6ms，LLMPerf 报告 TPOT=11ms。解释原因。

## 关键术语

| 术语               | 常见说法                              | 实际含义                              |
|--------------------|-------------------------------------|-------------------------------------|
| LLMPerf            | “LLM 测试工具”                      | Anyscale 基准工具，支持流式            |
| GenAI-Perf         | “NVIDIA 工具”                       | NVIDIA 官方参考测试工具               |
| LLM-Locust         | “LLM 专用 Locust”                   | Locust 扩展，修复 GIL 陷阱           |
| guidellm           | “合成基准”                         | 大规模合成测试工具                   |
| k6 Operator        | “Kubernetes k6”                     | 基于 CRD 的分布式 k6                 |
| GIL 陷阱           | “Python 客户端开销”                 | 令牌化积压使报告延迟被放大           |
| 提示一致性陷阱      | “单提示谎言”                       | 单一提示循环击中缓存，虚假提升吞吐率  |
| 稳定状态（Steady-state） | “恒定负载”                        | N 分钟恒定请求率                     |
| 渐增（Ramp）       | “线性增长”                         | 在指定时长内由 0 线性增至目标值        |
| 突发（Spike）      | “突发测试”                         | 突然倍率提升后恢复                   |
| 浸泡（Soak）       | “长时间测试”                       | 数小时用于检测泄漏                   |

## 拓展阅读

- [TianPan — 负载测试 LLM 应用](https://tianpan.co/blog/2026-03-19-load-testing-llm-applications)  
- [PremAI — 2026 年 LLM 负载测试](https://blog.premai.io/load-testing-llms-tools-metrics-realistic-traffic-simulation-2026/)  
- [NVIDIA NIM — LLM 推理基准介绍](https://docs.nvidia.com/nim/large-language-models/1.0.0/benchmarking.html)  
- [TrueFoundry — LLM-Locust](https://www.truefoundry.com/blog/llm-locust-a-tool-for-benchmarking-llm-performance)  
- [LLMPerf](https://github.com/ray-project/llmperf)  
- [k6 Operator](https://github.com/grafana/k6-operator)
