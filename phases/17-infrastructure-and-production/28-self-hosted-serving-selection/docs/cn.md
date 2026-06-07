# 自托管服务选择 — llama.cpp、Ollama、TGI、vLLM、SGLang

> 2026 年自托管推理领域由四个引擎主导。根据硬件、规模和生态系统选择。**llama.cpp** 在 CPU 上最快——支持模型最广，量化和线程控制完全自主。**Ollama** 是开发笔记本上“一条命令”安装的方案，比 llama.cpp 慢约 15-30%（Go + CGo + HTTP 序列化），在生产级负载下吞吐量差距达 3 倍。**TGI 于 2025 年 12 月 11 日进入维护模式**——仅进行漏洞修复，原始吞吐量略慢于 vLLM 约 10%，但历来监控能力和 Hugging Face（HF）生态集成最佳。该维护状态使其成为长期项目的风险选项——新项目更建议选择 SGLang 或 vLLM。**vLLM** 是通用生产默认方案——v0.15.1（2026 年 2 月）新增支持 PyTorch 2.10，RTX Blackwell SM120，H200 优化。**SGLang** 专注于具代理性的多轮对话和前缀密集型工作负载——已在 40 万+ GPU 上生产（xAI、LinkedIn、Cursor、Oracle、GCP、Azure、AWS）。硬件限制：仅 CPU → 仅 llama.cpp；AMD / 非 NVIDIA → 仅 vLLM（TRT-LLM 仅限 NVIDIA）。2026 年管线模式：开发用 Ollama，预发布用 llama.cpp，生产用 vLLM 或 SGLang。全程使用相同 GGUF/HF 权重。

**类型：** 学习  
**语言：** Python（标准库，决策树遍历）  
**前置条件：** 包含引擎内容的所有第17阶段课程（04、06、07、09、18）  
**时长：** 约 45 分钟

## 学习目标

- 根据硬件（CPU / AMD / NVIDIA Hopper / Blackwell）、规模（1 用户 / 100 / 10,000）和工作负载（通用对话 / 代理 / 长上下文）选择引擎。
- 说明 2026 年 TGI 进入维护模式的时间（2025 年 12 月 11 日）及其为何使新项目偏向 vLLM 或 SGLang。
- 描述开发、预发布、生产使用相同 GGUF 或 HF 权重的管线。
- 解释为何“仅 CPU”必选 llama.cpp，而“AMD”排除 TRT-LLM。

## 问题描述

团队启动新的自托管大语言模型（LLM）项目。一个工程师推荐 Ollama，另一个说 vLLM，第三个人说“难道 TGI 不开箱即用吗？”三者在不同情况下均有道理，但没有单一方案适合所有场景。

2026 年的决策树关键：先看硬件，再看规模，最后看工作负载。并且 2025 年 12 月 11 日 TGI 进入维护模式这一事件改变了新项目默认选项。

## 概念解析

### 五个引擎

| 引擎       | 适用场景                 | 说明                   |
|------------|--------------------------|------------------------|
| **llama.cpp** | CPU / 边缘 / 最少依赖 / 模型支持最广 | CPU 上最快，完全自主控制   |
| **Ollama**   | 开发笔记本、单用户、一键安装    | 比 llama.cpp 慢 15-30%；生产吞吐差距 3 倍 |
| **TGI**     | HF 生态系统、受规管行业          | **2025 年 12 月 11 日起进入维护模式** |
| **vLLM**    | 通用生产、100+ 用户            | 广泛生产默认方案；2026 年 2 月 v0.15.1 发布 |
| **SGLang**  | 具代理性的多轮对话，前缀密集型负载 | 生产环境部署 40 万 + GPU   |

### 硬件优先决策

**仅 CPU** → 选 llama.cpp。Ollama 也可用，但速度较慢。其他引擎在仅 CPU 上无竞争力。

**AMD GPU** → 选 vLLM（支持 AMD ROCm）。SGLang 也可用。TRT-LLM 为 NVIDIA 专属，不可用。

**NVIDIA Hopper（H100 / H200）** → vLLM、SGLang 或 TRT-LLM 都是顶级选择。

**NVIDIA Blackwell（B200 / GB200）** → TRT-LLM 是吞吐量冠军（第 17 阶段·07）。vLLM 和 SGLang 紧随其后。

**Apple Silicon（M 系列）** → 选 llama.cpp（Metal 支持）。Ollama 在其基础上封装。

### 规模次序决策

**1 用户 / 本地开发** → Ollama。一条命令，秒启动。

**10-100 用户 / 小团队** → vLLM 单 GPU。

**100-10,000 用户 / 生产** → vLLM 生产栈（第 17 阶段·18）或 SGLang。

**超过 10,000 用户 / 企业级** → vLLM 生产栈 + 分布式部署（第 17 阶段·17）+ LMCache（第 17 阶段·18）。

### 工作负载第三决策

**通用对话 / 问答** → vLLM 以广泛默认取胜。

**具代理性的多轮（工具、规划、记忆）** → SGLang 的 RadixAttention（第 17 阶段·06）优势明显。

**RAG（基于检索的生成）且前缀高度复用** → SGLang。

**代码生成** → vLLM 也不错；SGLang 在缓存方面略优。

**长上下文（128K+）** → vLLM + 分块预置；SGLang + 分层 KV。

### TGI 维护陷阱

Hugging Face 的 TGI 于 2025 年 12 月 11 日入维护模式——只修复漏洞。TGI 历来监控能力顶尖，HF 生态集成最佳（模型卡、安全工具），但原始吞吐量略逊于 vLLM。

2026 年新项目默认应避免使用 TGI。已有 TGI 部署可继续使用，但建议迁移。SGLang 和 vLLM 是更安全的默认选项。

### 管线模式

开发用 Ollama → 预发布用 llama.cpp → 生产用 vLLM。全程使用相同 GGUF 或 HF 权重。工程师快速在笔记本迭代，预发布环境镜像生产中的量化，生产环境为推理目标。

### Ollama 注意事项

Ollama 适合开发环境。一旦进入共享生产环境表现不佳：Go HTTP 序列化带来开销， 并发管理比 vLLM 简单，OpenTelemetry 支持滞后。应在 Ollama 优势场景（一用户、一命令）使用，生产环境切换至 vLLM。

### 自托管与托管是独立决策

第 17 阶段·01（托管超大规模）、·02（推理平台）涵盖托管方案。本课程假定你已决策自托管。自托管理由：数据驻留、本地微调、规模成本完全掌控、域模型托管端无提供。

### 必记关键数字

- TGI 维护模式开始时间：2025 年 12 月 11 日。
- vLLM v0.15.1：2026 年 2 月，支持 PyTorch 2.10、Blackwell SM120。
- SGLang 生产部署规模：40 万+ GPU。
- Ollama 相较 llama.cpp 吞吐差距：慢 15-30%，生产负载下三倍吞吐差距。

## 实操

`code/main.py` 是决策树遍历器：输入硬件 + 规模 + 负载，选择引擎并解释原因。

## 发布

本课生成 `outputs/skill-engine-picker.md`。基于限制条件，选择引擎并写出迁移方案。

## 练习

1. 用你的硬件 / 规模 / 负载运行 `code/main.py`。输出是否符合直觉？
2. 你的基础设施有 12 块 H100 与 8 块 MI300X AMD。选什么引擎？为何排除 TRT-LLM？
3. 一个团队想在 2026 年用 TGI 因为“我们熟悉它”。请论证迁移方案。
4. Ollama 开发切换到 vLLM 生产：量化、配置和监控有哪些变化？
5. RAG 产品，P99 前缀长度 8K，跨租户高度复用。选哪个引擎并结合第 17 阶段·11 + 18 堆栈。

## 关键词汇

| 术语        | 俗称         | 实际含义                           |
|-------------|--------------|----------------------------------|
| llama.cpp   | “CPU 方案”    | 模型支持最广，CPU 上最快            |
| Ollama      | “笔记本方案”  | 一条命令安装，开发级吞吐量          |
| TGI         | “HF 的服务”  | 自 2025 年 12 月起维护模式          |
| vLLM        | “默认方案”   | 2026 年广泛生产基线               |
| SGLang      | “具代理性方案” | 前缀密集，支持 RadixAttention     |
| TRT-LLM     | “NVIDIA 专属” | Blackwell 吞吐量冠军，仅限 NVIDIA  |
| GGUF        | “llama.cpp 格式” | 包含 K-quant 变体                  |
| Production-stack | “vLLM K8s” | 第 17 阶段·18 参考部署              |
| Pipeline pattern | “开发→预发布→生产” | Ollama → llama.cpp → vLLM 同权重    |

## 深入阅读

- [AI Made Tools — 2026 年 vLLM vs Ollama vs llama.cpp vs TGI 比较](https://www.aimadetools.com/blog/vllm-vs-ollama-vs-llamacpp-vs-tgi/)
- [Morph — 2026 年 llama.cpp vs Ollama](https://www.morphllm.com/comparisons/llama-cpp-vs-ollama)
- [n1n.ai — 全面 LLM 推理引擎对比](https://explore.n1n.ai/blog/llm-inference-engine-comparison-vllm-tgi-tensorrt-sglang-2026-03-13)
- [PremAI — 2026 年十大最佳 vLLM 替代方案](https://blog.premai.io/10-best-vllm-alternatives-for-llm-inference-in-production-2026/)
- [TGI 维护公告](https://github.com/huggingface/text-generation-inference) — 发行说明。
- [vLLM v0.15.1 发行说明](https://github.com/vllm-project/vllm/releases)
