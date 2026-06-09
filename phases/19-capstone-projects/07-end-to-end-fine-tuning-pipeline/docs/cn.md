# Capstone 07 — 端到端微调流水线（数据到SFT到DPO到服务）

> 一个在您自己的数据上训练的8B模型，基于您的偏好进行DPO对齐，完成量化、投机解码，并以可度量的每百万tokens成本（$/1M tokens）提供服务。2026年的开源技术栈是Axolotl v0.8、TRL 0.15、用于迭代的Unsloth、用于量化的GPTQ/AWQ/GGUF、使用EAGLE-3的vLLM 0.7进行服务。这个capstone任务是可复现地运行整个流水线——YAML输入，服务端点输出——并根据2026模型开放度框架发布模型卡。

**类型：** Capstone  
**语言：** Python（流水线）、YAML（配置）、Bash（脚本）  
**前置知识：** 阶段2（机器学习）、阶段3（深度学习）、阶段7（transformers）、阶段10（从零构建LLM）、阶段11（LLM工程）、阶段17（基础设施）、阶段18（安全）  
**涉及阶段：** P2 · P3 · P7 · P10 · P11 · P17 · P18  
**时长：** 35小时

## 问题

2026年的每个严肃AI团队都会保持一个随时可用的微调流水线。不是因为他们直接发布最前沿的基础模型，而是因为下游适配——领域SFT、针对标注偏好的DPO、用于投机解码的蒸馏草稿、基于EAGLE-3的服务——才是真正能带来可度量收益的环节。Axolotl v0.8支持多GPU的SFT配置。TRL 0.15支持DPO和GRPO。Unsloth提供快速的单GPU迭代。vLLM 0.7和EAGLE-3实现了2-3倍的解码吞吐量提升且无质量损失。工具链已成熟，关键在于YAML配置、数据质量和评估纪律。

你将运行一个8B基础模型（Llama 3.3、Qwen3 或 Gemma 3）进行任务特定数据上的SFT和DPO，完成量化后提供服务，并通过lm-evaluation-harness、RewardBench-2、MT-Bench-v2和MMLU-Pro衡量性能提升。最终产出根据2026模型开放度框架编写的模型卡。关键是可复现性——一条命令即可重跑整个流水线。

## 概念

流水线包含五个阶段。**数据**：去重（MinHash / Datatrove）、质量过滤（Nemotron-CC风格分类器）、PII（个人身份信息）清理，拆分卫生检查防止公开基准数据污染。**SFT**：Axolotl YAML，8xH100上的ZeRO-3，余弦调度，序列打包，2-3个epoch。**DPO或GRPO**：TRL配置，1个epoch，偏好对来自人工标注或模型判断，beta参数调优。**量化**：GPTQ+AWQ+GGUF以保证部署灵活性。**服务**：vLLM 0.7配合EAGLE-3投机解码头（或SGLang加SpecForge），K8s部署，基于队列等待的HPA（Horizontal Pod Autoscaler）。

交付的对比实验包括：仅SFT vs SFT+DPO vs SFT+GRPO，基于三个任务特定基准。服务性能指标：batch大小为1/8/32时tokens/s，EAGLE-3接受率，每百万tokens成本。安全评估：Llama Guard 4通过率。模型卡：偏差评估、复现用随机种子、数据授权说明。

## 架构

```text
原始数据（HF数据集 + 内部数据）
    |
    v
Datatrove去重 + Nemotron-CC质量过滤 + PII清理
    |
    v
拆分卫生（MMLU-Pro污染检测）
    |
    v
Axolotl SFT配置（YAML）  ---> 8xH100, ZeRO-3
    |
    v
TRL DPO / GRPO配置         ---> 4xH100, 1个epoch
    |
    v
GPTQ + AWQ + GGUF量化
    |
    v
vLLM 0.7 + EAGLE-3投机解码
    |
    v
K8s部署，基于队列等待的HPA
    |
    v
lm-eval-harness + RewardBench-2 + MT-Bench-v2 + MMLU-Pro
    |
    v
模型卡（2026 MOF） + 安全评估（Llama Guard 4）
```

## 技术栈

- 数据：Datatrove去重，Nemotron-CC分类器进行质量过滤，Presidio处理PII  
- 基础模型：Llama 3.3 8B，Qwen3 14B，或 Gemma 3 12B  
- SFT：Axolotl v0.8配合ZeRO-3，Flash Attention 3，序列打包  
- 偏好微调：TRL 0.15支持DPO或GRPO；Unsloth用于单GPU迭代  
- 量化：GPTQ（Marlin）、AWQ、GGUF通过llama.cpp  
- 服务：vLLM 0.7结合EAGLE-3投机解码（或SGLang 0.4加SpecForge）  
- 评测：lm-evaluation-harness，RewardBench-2，MT-Bench-v2，MMLU-Pro  
- 安全评估：Llama Guard 4，ShieldGemma-2  
- 基础设施：Kubernetes + NVIDIA设备插件，基于队列等待的HPA  
- 可观测性：W&B用于训练监控，Langfuse用于推理诊断  

## 搭建流程

1. **数据流水线。** 对原始语料执行Datatrove去重。应用Nemotron-CC风格质量分类器。Presidio清理PII。使用固定随机种子写入训练/验证拆分。

2. **污染检测。** 对每个验证拆分，使用MinHash检查与MMLU-Pro、MT-Bench-v2、RewardBench-2测试集的重叠。删除任何重叠数据。

3. **Axolotl SFT。** YAML配置ZeRO-3，Flash Attention 3，序列打包。8xH100上训练2-3个epoch。日志传输至W&B。

4. **TRL DPO / GRPO。** 使用SFT检查点，基于偏好对运行1个epoch的DPO（或带有可验证奖励的数学/代码任务上的GRPO）。调优beta参数。

5. **量化。** 产生三个量化模型：GPTQ-INT4-Marlin，AWQ-INT4，GGUF-Q4_K_M用于llama.cpp。记录大小和标称吞吐量。

6. **投机解码服务。** 配置vLLM 0.7和EAGLE-3草稿头（通过Red Hat Speculators训练）。测量批量大小为1/8/32时的接受率和尾部延迟。对比Anthropic和OpenAI同样评测的$/1M tokens成本。

7. **评测矩阵。** 在基础模型、仅SFT、SFT+DPO、SFT+GRPO上运行lm-evaluation-harness、RewardBench-2、MT-Bench-v2、MMLU-Pro。输出性能表。

8. **安全评估。** 在验证集上运行Llama Guard 4通过率检测。ShieldGemma-2输出过滤。

9. **模型卡。** 使用2026 MOF模板编写：数据、训练、评测、安全、授权，附带复现用的YAML和提交SHA。

## 使用方式

```text
$ ./pipeline.sh config/llama3.3-8b-domainX.yaml
[data]    30万去重，1.2万过滤，28万通过（seed=7）
[SFT]     3个epoch，8xH100，6小时12分钟，验证损失 1.42 -> 1.03
[DPO]     1个epoch，beta=0.08，4xH100，1小时40分钟
[quant]   GPTQ-INT4 4.6 GB，AWQ-INT4 4.8 GB，GGUF-Q4_K_M 5.1 GB
[serve]   vLLM 0.7，EAGLE-3接受率0.74，p99延迟126ms @ 批量8
[eval]    MMLU-Pro +3.2，MT-Bench-v2 +0.41，RewardBench-2 +0.08
[card]    根据2026 MOF生成了model-card.md
```

## 发布结果

`outputs/skill-finetuning-pipeline.md`描述交付内容。一条命令即可完成从数据处理到SFT，再到DPO、量化、服务部署和评测，并输出模型卡和服务端点。

| 权重 | 评判标准 | 测量方式 |
|:-:|---|---|
| 25 | 目标任务相较基础模型的性能提升 | Measured gains on MMLU-Pro, MT-Bench-v2, and task-specific benchmarks |
| 20 | 流水线复现性 | One-command end-to-end rerun with the same random seed |
| 20 | 数据卫生状况 | Deduplication rate, PII cleanup coverage, and contamination-check pass status |
| 20 | 服务效率 | tokens/s at batch sizes 1/8/32, EAGLE-3 acceptance rate, and $/1M tokens |
| 15 | 模型卡及安全评估 | Complete 2026 MOF template plus Llama Guard 4 pass rate |
| **100** | | |

## 练习

1. 在同一个任务特定基准上运行仅SFT、SFT+DPO和SFT+GRPO，报告哪种偏好调优方法表现最好及提升幅度。

2. 用Qwen3 14B替换Llama 3.3 8B，比较相同质量下的$/1M tokens成本。

3. 测量领域数据与通用ShareGPT上EAGLE-3接受率差异，分析其对延迟预算的影响。

4. 注入1%的污染（向训练数据泄露MMLU-Pro答案），重新评测。观察MMLU-Pro准确度异常飙升。基于此构建污染检测CI门控。

5. 添加LoRA微调作为全量微调的替代方案，测量在内存需求降低10倍时的性能差距。

## 关键术语

| 术语 | 业界说法 | 实际含义 |
|------|----------|----------|
| Axolotl | “SFT训练器” | 一个支持SFT、DPO和蒸馏的统一YAML驱动训练器 |
| TRL | “偏好调优器” | Hugging Face提供的LLM DPO、GRPO、PPO库 |
| GRPO | “组相对策略优化” | DeepSeek R1的带可验证奖励的强化学习方案 |
| EAGLE-3 | “投机解码草稿头” | 预测未来N个tokens的草稿头，vLLM用目标模型确认 |
| MOF | “模型开放度框架” | 2026年模型发布的数据、代码、授权评级标准 |
| 污染检测 | “拆分卫生” | 利用MinHash检测测试集数据在训练集中的泄露 |
| 接受率 | “EAGLE / MTP指标” | 目标模型接受投机草稿token的比例 |

## 深入阅读

- [Axolotl文档](https://axolotl-ai-cloud.github.io/axolotl/) — SFT / DPO参考训练器  
- [TRL文档](https://huggingface.co/docs/trl) — DPO和GRPO参考实现  
- [Unsloth](https://github.com/unslothai/unsloth) — 单GPU迭代参考实现  
- [DeepSeek R1论文 (arXiv:2501.12948)](https://arxiv.org/abs/2501.12948) — GRPO方法论  
- [vLLM + EAGLE-3文档](https://docs.vllm.ai) — 参考服务栈  
- [SGLang SpecForge](https://github.com/sgl-project/SpecForge) — 另一种投机解码训练器  
- [模型开放度框架2026](https://isocpp.org/) — 开放发布评级标准  
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) — 权威评测执行器
