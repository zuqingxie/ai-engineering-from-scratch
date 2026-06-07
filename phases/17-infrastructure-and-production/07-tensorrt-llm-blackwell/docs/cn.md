# TensorRT-LLM 在 Blackwell 上支持 FP8 和 NVFP4

> TensorRT-LLM 是 NVIDIA 独有，但在 Blackwell 上表现最佳。在 GB200 NVL72 上，结合 Dynamo 编排，SemiAnalysis InferenceX 在 2026 年 Q1-Q2 期间测得 120B 模型每百万 token 成本为 $0.012，相比 H100 + vLLM 的 $0.09/M，经济效益提升了 7 倍。该技术栈包含三个复合的浮点计算方案：FP8 对于 KV 缓存和注意力核保持关键地位，因为其提供了所需的动态范围；NVFP4（4 位微缩放）用于权重和激活；多 token 预测（MTP）和解耦预填充/解码进一步提升 2-3 倍。Day-0 模型支持直接加载 FP4 权重，无需后训练转换。2026 年工程团队面临的挑战是：TRT-LLM 是一个封闭的 NVIDIA 技术栈，采用它意味着以更高的吞吐量换取可移植性的丧失。在做决定前，请根据你的模型和硬件组合计算成本效益。

**类型：** 学习  
**语言：** Python（标准库，玩具级 FP8/NVFP4 内存和成本计算器）  
**先决条件：** 第 17 阶段 · 04（vLLM 服务内部）、第 10 阶段 · 13（量化）  
**时间：** 约 75 分钟

## 学习目标

- 解释为什么即使权重使用 NVFP4，FP8 仍对 KV 缓存和注意力至关重要。  
- 计算 BF16、FP8 和 NVFP4 下前沿模型的 HBM 占用，并分析节省来自何处。  
- 了解 TRT-LLM 利用的 Blackwell 特有特性（Day-0 FP4、多 token 预测、解耦服务、全对全通信原语）。  
- 判断 TRT-LLM 的 NVIDIA 锁定是否值得，相较于 Hopper 上的 vLLM 实现 7 倍成本差异。

## 问题描述

2026 年推理经济的前沿问题是“每美元能处理多少 token”。答案取决于四层叠加的选择：硬件代际（Hopper H100/H200 vs Blackwell B200/GB200）、精度（BF16 → FP8 → NVFP4）、服务引擎（vLLM vs SGLang vs TRT-LLM）和编排方式（简单 vs 解耦 vs Dynamo）。

在 Hopper 上用 vLLM 运行 120B MoE 的成本约为每百万 token $0.09；在 Blackwell 上用 TRT-LLM + Dynamo，则相同模型的成本约为 $0.012，便宜 7 倍。这一差异部分源于硬件（Blackwell 每 GPU LLM 吞吐比 Hopper 高 11-15 倍），部分源于软件栈：FP4 权重、多 token 预测草案、解耦预填充/解码及 MoE 专家通信的 NVLink 5 全对全通信。

这一优势无法在 NVIDIA 生态系统外复制。权衡点即是 —— 以经济效益换取平台锁定。理解各栈选项贡献的成本差异是本课重点。

## 核心概念

### 为什么 FP8 仍是 KV 缓存的底线

2026 年一个常见错误是误认为 NVFP4 可无处不适用。实际上不是。KV 缓存需用 FP8（8 位浮点）存储注意力的键值对，它们跨越宽广的动态范围。若将 KV 缓存量化为 FP4，会造成毁灭性准确率丢失 —— 分布尾部信息消失注意力得分崩溃。FP8 的指数位为 KV 缓存提供必需的范围。

NVFP4（2025-2026 年）应用于权重和激活。微缩放（Microscaling）：每个权重块拥有独有的缩放因子，使得小块权重可以独立适应不同动态范围，而不会出现单一张量缩放带来的性能损失。激活值则因其层内动态范围较小，FP4 依然可用。

典型 Blackwell 配置：

- 权重：NVFP4（4 位微缩放）。  
- 激活：NVFP4。  
- KV 缓存：FP8。  
- 注意力累加器：FP32（保证 softmax 稳定性）。

### TRT-LLM 利用的 Blackwell 特定原语

- **Day-0 FP4 权重**：模型提供方直接发布 FP4 权重；TRT-LLM 加载时无需后训练转换，省去了 AWQ / GPTQ 步骤。  
- **多 token 预测（MTP）**：与 EAGLE（第 17 阶段 · 05）理念相同，但集成于 TRT-LLM 构建流程。  
- **解耦服务**：预填充和解码分布在不同 GPU 池；KV 缓存通过 NVLink 或 InfiniBand 传输，与 Dynamo（第 17 阶段 · 20）类似。  
- **全对全通信原语**：NVLink 5 将 MoE 专家通信延迟相比 Hopper 降低 3 倍，TRT-LLM 的 MoE 核心针对其调优。  
- **NVFP4 + MXFP8 微缩放**：Blackwell Tensor Core 支持的硬件加速缩放因子处理。

### 必背数据

- HGX B200 通过 TRT-LLM 在 GPT-OSS-120B 上达 $0.02/M token。  
- GB200 NVL72 通过 Dynamo（编排 TRT-LLM）达 $0.012/M token。  
- H100 + vLLM 约 $0.09/M token。  
- 2026 年 TRT-LLM 更新 3 个月内吞吐量提升 2.8 倍。  
- Blackwell 相对 Hopper 单 GPU LLM 吞吐提升 11-15 倍。  
- MLPerf Inference v6.0（2026 年 4 月）：Blackwell 在所有提交项目中占优。

### FP4 品质折损

NVFP4 精度要求高。在推理密集型任务（链式思维、数学、长上下文代码生成）中，FP4 权重明显影响准确率。分块校准有缓和作用但无法完全消除。发布推理模型团队常用 FP8 权重 + FP4 激活作为折中，或全程使用 FP8（H200 平台）。

经验法则：提交 NVFP4 权重前，总在评测集上严密验证任务质量。

### 为什么这是一个 NVIDIA 锁定决策

TRT-LLM 是 C++ + CUDA + 封闭源码核函数。模型必须针对特定 GPU SKU 编译。不支持 AMD、Intel 或 ARM。如果你的基础设施要多供应商，TRT-LLM 作为 serving 层不可行 —— 你只能在混合硬件上用 vLLM 服务。如果纯 NVIDIA，7 倍的经济优势足以弥补锁定。

### 2026 年实务建议

对于年度推理预算超过 1 亿美元的用户，继续在 Hopper + vLLM 上运行相当于放弃 7-10 倍成本优势。应将成本主导任务迁移到 Blackwell + TRT-LLM + Dynamo。实验环境保留 H100 + vLLM，保持模型迭代速度。每次 NVFP4 转换模型部署前必须验证质量。

### 解耦服务的额外收益

TRT-LLM 解耦服务（分开预填充与解码 GPU 池）详见第 17 阶段 · 20。Blackwell 上各项技术乘积叠加：FP4 权重 × MTP 提速 × 解耦部署 × 缓存感知路由，7 倍提升包括了完整技术栈。

## 实践使用

`code/main.py` 计算模型在三种技术栈下的 HBM 占用、解码吞吐（带宽瓶颈模式）及 $/M-tokens：H100 + BF16 + vLLM，H100 + FP8 + vLLM，以及 B200 + NVFP4/FP8 + TRT-LLM。运行它可以看到叠加效应及各变更对成本差距的贡献。

## 交付成果

本课生成 `outputs/skill-trtllm-blackwell-advisor.md`。给定一个工作负载、模型大小和年 token 量，判断是否值得采用 Blackwell + TRT-LLM 技术栈及其 NVIDIA 锁定。

## 练习题

1. 运行 `code/main.py`。对 120B MoE（30% 活跃参数），计算 H100 BF16、H100 FP8 和 B200 NVFP4/FP8 的内存带宽限制解码吞吐。哪个环节提升最大？  
2. 客户在 H100 + vLLM 上年花费 200 万美元。为了 12 个月内摊销迁移 TRT-LLM 的成本，需要购买多少 Blackwell GPU 以达成成本平衡？  
3. NVFP4 权重转换后 MATH 题测试精度下降 3 点。请给出两个恢复方案：一个质量优先（保留 FP8 权重），一个成本优先（使用领域内数据校准）。  
4. 阅读 MLPerf v6.0 推理结果。哪个任务上 Blackwell 相较 Hopper 差距最小，为什么？  
5. 计算 405B 模型在 NVFP4 权重 + FP8 KV 缓存、128k 上下文下所需 HBM。是否能在单个 GB200 NVL72 节点内装下？

## 关键术语

| 术语            | 通俗说法           | 实际含义                           |
|-----------------|--------------------|----------------------------------|
| FP8             | “八位浮点”          | 8 位浮点；用于 KV 缓存和注意力，因需动态范围 |
| NVFP4           | “四位微缩”          | NVIDIA 的 4 位微缩放浮点格式；Blackwell 上用作权重和激活 |
| MXFP8           | “MX 八位”           | 微缩放 FP8 变体；Blackwell Tensor Core 硬件加速 |
| Day-0 FP4       | “发布 FP4 权重”     | 模型提供方直接发布 FP4 权重，无需后训练转换步骤 |
| MTP             | “多 token 预测”      | TRT-LLM 集成的推测解码草案（第 17 阶段 · 05） |
| 解耦服务        | “拆分预填/解码”      | 预填充与解码在不同 GPU 池；KV 通过 NVLink/InfiniBand 传输 |
| 全对全通信      | “MoE 专家通信”       | 将 token 路由至专家 GPU 的通信模式；NVLink 5 降低延迟 3 倍 |
| InferenceX      | “SemiAnalysis 推理基准” | 2026 年行业认可的每 token 成本基准          |

## 延伸阅读

- [NVIDIA — Blackwell Ultra MLPerf Inference v6.0](https://developer.nvidia.com/blog/nvidia-blackwell-ultra-sets-new-inference-records-in-mlperf-debut/) — 2026 年 4 月 MLPerf 结果。  
- [NVIDIA — Blackwell 上的 MoE 推理](https://developer.nvidia.com/blog/delivering-massive-performance-leaps-for-mixture-of-experts-inference-on-nvidia-blackwell/) — NVLink 5 全对全与 MoE 核心。  
- [TensorRT-LLM 概览](https://nvidia.github.io/TensorRT-LLM/overview.html) — 官方引擎文档。  
- [NVIDIA — 介绍 Dynamo](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/) — TRT-LLM 上的解耦编排。  
- [MLPerf 推理](https://mlcommons.org/benchmarks/inference-datacenter/) — 发布 Blackwell 测试成绩的基准套件。
