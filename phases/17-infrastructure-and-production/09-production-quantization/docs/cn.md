# 生产量化 — AWQ, GPTQ, GGUF K-quant, FP8, MXFP4/NVFP4

> 量化格式不是通用选择——它取决于硬件、推理引擎和工作负载。GGUF Q4_K_M 或 Q5_K_M 占CPU和边缘设备领域，通过 llama.cpp 和 Ollama 提供。GPTQ 在需要多-LoRA 同时作用于同一基础模型时，在 vLLM 中胜出。AWQ 结合 Marlin-AWQ 内核，在7B级模型上实现约741令牌每秒且在 INT4 精度中 Pass@1 最佳——2026年数据中心生产默认选择。FP8 在 Hopper、Ada 和 Blackwell 上保持中间地带——接近无损且广泛支持。NVFP4 和 MXFP4（Blackwell微缩）激进且需要逐块验证。两个陷阱困扰团队：校准数据集必须匹配部署领域，且 KV 缓存与权重量化分开——AWQ 的“我的模型现在4GB”忽略了生产批量大小下10-30GB的 KV 缓存。

**类型：** 学习  
**语言：** Python（标准库，跨格式内存和吞吐量简单对比）  
**先决条件：** 第10·13阶段（量化基础），第17·04阶段（vLLM推理内部）  
**时长：** 约75分钟

## 学习目标

- 说出六种生产量化格式及其2026年的适用场景。
- 根据硬件（CPU vs GPU，Hopper vs Blackwell）、引擎（vLLM, TRT-LLM, llama.cpp）和工作负载（日常聊天、推理、多-LoRA）选择合适格式。
- 计算所选格式节省的权重内存和未触及的 KV 缓存大小。
- 说出导致量化模型在领域流量表现下降的校准数据集坑点。

## 问题描述

量化减少了内存和高带宽内存（HBM）带宽消耗，正是解码器所需。一个 FP16 精度的 70B 模型有 140 GB 权重，量化成 INT4（AWQ 或 GPTQ）后模型为 35 GB——可以装入一块 H100 并有余地容纳 KV 缓存，关键是128个并发序列、2k上下文下单纯 KV 缓存就达20-30 GB。

但量化不是白送的。激进量化会降低质量，尤其在推理密集任务中表现明显。不同格式适配不同推理引擎。不同硬件对精度有原生支持的差异。2026的格式生态真实存在，你不能盲抄别人选择——必须根据自身技术栈来定。

## 概念

### 六种格式

| 格式 | 位数 | 适用场景 | 引擎 |
|--------|------|-----------|---------|
| GGUF Q4_K_M / Q5_K_M | 4-5 | CPU，边缘设备，笔记本 | llama.cpp，Ollama |
| GPTQ | 4-8 | vLLM 多-LoRA | vLLM，TGI |
| AWQ | 4 | 数据中心 GPU 生产 | vLLM (Marlin-AWQ)，TGI |
| FP8 | 8 | Hopper/Ada/Blackwell 数据中心 | vLLM，TRT-LLM，SGLang |
| MXFP4 | 4 | Blackwell 多用户环境 | TRT-LLM |
| NVFP4 | 4 | Blackwell 多用户环境 | TRT-LLM |

### GGUF — CPU/边缘默认

GGUF 是一种文件格式，不是量化方案本身——它捆绑了 K-quant 变种（Q2_K，Q3_K_M，Q4_K_M，Q5_K_M，Q6_K，Q8_0）于单一容器内。Q4_K_M 和 Q5_K_M 是生产默认——在4-5位中接近 BF16 质量。是 CPU 或边缘推理最快的选择，因为 llama.cpp 是迄今最快的 CPU 推理引擎。

vLLM中的吞吐量罚分：7B模型约93令牌每秒——该格式未针对 GPU 内核优化。仅当部署目标是 CPU/边缘时使用 GGUF，其它场景不推荐。

### GPTQ — vLLM 多-LoRA

GPTQ 是带校准过程的训练后量化算法。Marlin 内核使其 GPU 速度提升 2.6 倍相较未用 Marlin 的 GPTQ。7B 模型约712令牌每秒。

独特优势：GPTQ-Int4 支持 vLLM 中的 LoRA 适配器。若需服务一个基础模型加10-50个微调变体（均作为 LoRA），GPTQ 是路径。截止2026年初，NVFP4 尚不支持 LoRA。

### AWQ — 数据中心 GPU 默认

Activation-aware Weight Quantization（感知激活的权重量化）。量化时保护约1%的最关键权重。Marlin-AWQ 内核相比朴素实现加速10.9倍。7B模型约741令牌每秒，INT4 版本中 Pass@1 最佳。

除非需要多-LoRA（选 GPTQ）或激进的 Blackwell FP4（选 NVFP4），否则新GPU服务优选 AWQ。

### FP8 — 可靠中间选

8位浮点。几乎无损。广泛支持。Hopper Tensor Core 原生加速 FP8，Blackwell 继承此能力。2026年当质量不可妥协（推理、医疗、代码生成）时 FP8 是安全默认。节省的内存约为 INT4 的一半，但质量风险低很多。

### MXFP4 / NVFP4 — Blackwell 激进选择

Microscaling FP4（微缩FP4）。每个权重块都有独立的缩放因子。激进，但在 Blackwell Tensor Core 上硬件加速。比 FP8 每令牌字节数减半——这是第17·07阶段的经济优势。

注意：
- 2026年初尚无 LoRA 支持。
- 在推理密集负载下质量下降明显。
- 需对评测集逐模型验证。

### 校准陷阱

AWQ 与 GPTQ 需要校准数据集——通常用 C4 或 WikiText。对领域模型（代码、医疗、法律），用通用网页文本做校准会导致算法错误决策保护哪些权重。HumanEval Pass@1 分数可能跌落数点。

修正方法：用领域内数据校准。几百条领域样本通常足够。发货前务必在评测集测试。

### KV 缓存陷阱

AWQ权重缩减至4位。KV缓存独立，保持 FP16/FP8 精度。以 70B AWQ 模型为例：

- 权重约35 GB（从140 GB INT4）。
- 128并发×2k上下文的KV缓存约20 GB。
- 激活约5 GB。
- 总计约60 GB——可放入80GB H100。

轻率地说“我把模型量化到4 GB”会忽视另外30-50 GB的KV缓存。HBM预算整体考虑。

KV缓存量化（FP8 KV 或 INT8 KV）是另一选择，有自身权衡——它直接影响注意力准确度，不是免费提升。

### AWQ INT4 对推理不友好

链式思维、数学、代码生成等长上下文推理，受激进量化影响明显。AWQ INT4 在 MATH 测试中失分约3-5点。推理密集任务应发放 FP8 或 BF16，接受内存成本。

### 2026选择指南

- CPU/边缘服务：GGUF Q4_K_M。完成。
- GPU 服务、常规聊天、不用 LoRA：AWQ。
- GPU 服务、多-LoRA：GPTQ 加 Marlin 内核。
- 推理负载：FP8。
- Blackwell数据中心、需验证质量：NVFP4 + FP8 KV。
- 不确定时：对每个候选格式跑1000样本评测。

## 使用示例

`code/main.py` 计算一系列模型大小下六种格式的内存占用（权重+KV+激活）和相对吞吐量。展示哪里KV缓存主导，哪里权重压缩收益明显，哪里FP8最安全。

## 交付成果

本课题生成 `outputs/skill-quantization-picker.md`。基于硬件、模型大小、工作负载类型及质量容忍度，选定格式并制定校准/验证计划。

## 练习

1. 运行 `code/main.py`。对于7B，128并发，2k上下文，计算每种格式总HBM占用。哪个格式可以装入一块80GB H100？
2. 有一个7B代码模型。选格式并说明理由。如果质量容忍判断错，如何补救？
3. 计算为医疗领域模型校准 AWQ 需要多大校准数据集。为何更多数据不总是更好？
4. 阅读 Marlin-AWQ 内核论文或发布说明。用三句话解释为何 AWQ 在7B达到741令牌每秒，而原始 GPTQ 约712。
5. 何时同时使用 AWQ 权重与 FP8 KV 缓存合适，何时保持 KV BF16 更佳？

## 关键词

| 术语 | 常说 | 实际含义 |
|------|------|----------|
| GGUF | “llama.cpp 格式” | 打包 K-quant 变种的文件格式；CPU/边缘默认 |
| Q4_K_M | “Q4 K M” | 4位 K-quant 中等精度；生产 GGUF 默认 |
| GPTQ | “gee pee tee q” | 带校准的训练后 INT4 量化；支持 vLLM LoRA |
| AWQ | “a w q” | 感知激活的 INT4；Marlin 内核；INT4 Pass@1 最优 |
| Marlin kernels | “快速 INT4 内核” | Hopper 自定 CUDA 内核；10倍加速 |
| FP8 | “8位浮点” | Hopper/Ada/Blackwell 上安全默认精度 |
| MXFP4 / NVFP4 | “微缩四位” | Blackwell 4位浮点，逐块缩放因子 |
| 校准数据集 | “cal data” | 用于计算量化参数的输入文本；需匹配领域 |
| KV缓存量化 | “KV INT8” | 权重之外的选择；影响注意力准确性 |

## 延伸阅读

- [VRLA Tech — LLM Quantization 2026](https://vrlatech.com/llm-quantization-explained-int4-int8-fp8-awq-and-gptq-in-2026/) — 比较基准。
- [Jarvis Labs — vLLM Quantization Complete Guide](https://jarvislabs.ai/blog/vllm-quantization-complete-guide-benchmarks) — 按格式统计吞吐量。
- [PremAI — GGUF vs AWQ vs GPTQ vs bitsandbytes 2026](https://blog.premai.io/llm-quantization-guide-gguf-vs-awq-vs-gptq-vs-bitsandbytes-compared-2026/) — 格式对比选择指南。
- [vLLM docs — Quantization](https://docs.vllm.ai/en/latest/features/quantization/index.html) — 支持格式和参数。
- [AWQ paper (arXiv:2306.00978)](https://arxiv.org/abs/2306.00978) — AWQ 原始论文。
- [GPTQ paper (arXiv:2210.17323)](https://arxiv.org/abs/2210.17323) — GPTQ 原始论文。
