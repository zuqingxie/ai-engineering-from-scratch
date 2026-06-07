# Capstone 14 — 猜测解码推理服务器（Speculative-Decoding Inference Server）

> vLLM 0.7 中的 EAGLE-3 在真实流量上实现了 2.5-3 倍的吞吐量。P-EAGLE（AWS 2026）将并行猜测推理推进得更远。SGLang 的 SpecForge 实现了大规模草稿头训练。Red Hat 的 Speculators hub 发布了针对常见开源模型的对齐草稿。TensorRT-LLM 将猜测解码作为 NVIDIA 的一级功能。2026 年的生产推理栈是 vLLM 或 SGLang，配合 EAGLE 系列草稿模型、FP8 或 INT4 量化，以及基于排队等待的 HPA。本 Capstone 任务旨在以 2.5 倍以上基线吞吐量服务两个开源模型，并提供完整的尾部延迟报告。

**类型：** Capstone  
**语言：** Python（推理服务）、C++ / CUDA（内核调试）、YAML（配置）  
**先决条件：** 第3阶段（深度学习）、第7阶段（Transformer（Transformer 架构））、第10阶段（从零构建 LLM）、第17阶段（基础设施）  
**涉及阶段：** P3 · P7 · P10 · P17  
**时间：** 30 小时

## 问题

猜测解码在 2026 年成为了标配。EAGLE-3 草稿头基于目标模型的隐藏状态训练，预测未来 N 个标记；目标模型在单次调用中验证这些预测。60-80% 的接受率带来 2-3 倍的端到端吞吐提升。vLLM 0.7 原生集成了此机制。SGLang + SpecForge 提供训练流水线。Red Hat 的 Speculators 发布了针对 Llama 3.3 70B、Qwen3-Coder-30B MoE、GPT-OSS-120B 的对齐草稿。

核心在于推理/服务操作，而非模型本身。接受率随流量分布变化（ShareGPT、代码、特定领域数据）。基于拒绝的尾部延迟比无猜测时更差——必须针对多个批大小报告 p99 尾延迟，而非仅稳定态 Tokens/Sec。与 Anthropic/OpenAI API 的 100 万 Tokens 成本比较是可信度的关键。

## 概念

猜测解码有两层：一个 **草稿模型**（EAGLE-3 头、ngram，或者对齐的较小目标模型）每步提出 k 个候选标记；**目标模型**单次调用验证这 k 个标记中的所有前缀，接受后取代贪心路径。接受率依赖于草稿与目标的对齐程度及输入分布。

EAGLE-3 在多数流量上优于 ngram 草稿。P-EAGLE 支持并行猜测，构建更深的草稿树。折中点是：拒绝时的 p99 延迟更高，因为验证调用内容更多。服务配置需按批大小统计延迟，以体现这一点。

部署环境为 Kubernetes。vLLM 0.7 每个 GPU 或张量并行分片运行一个副本。HPA 基于排队等待自动扩缩容，而非 CPU 利用率。FP8（Marlin）和 INT4（AWQ）量化保证 GPU 内存占用符合 H100 / H200 限制。端到端报告包括吞吐量、接受率、p50/p99 延迟（批大小 1/8/32），及每百万 Tokens 成本。

## 架构

```text
请求入口
    |
    v
vLLM 服务器（0.7）或 SGLang（0.4）
    |
    +-- 草稿：EAGLE-3 头 | P-EAGLE 并行 | ngram 兜底
    +-- 目标：Llama 3.3 70B | Qwen3-Coder-30B | GPT-OSS-120B
    |     FP8-Marlin 或 INT4-AWQ 量化
    |
    v
验证调用：目标模型批量验证 k 个草稿标记
    |
    v （接受前缀；对拒绝后缀重采样）
    v
标记流发送回客户端
    |
    v
Prometheus 指标：吞吐量、接受率、队列等待、p50/p99 延迟
    |
    v
基于队列等待的 HPA
```

## 技术栈

- 推理服务：vLLM 0.7 或 SGLang 0.4  
- 猜测方法：EAGLE-3 草稿头、P-EAGLE 并行猜测、ngram 兜底  
- 草稿训练：SpecForge（SGLang）或 Red Hat Speculators  
- 目标模型：Llama 3.3 70B、Qwen3-Coder-30B MoE、GPT-OSS-120B  
- 量化：FP8（Marlin）、INT4（AWQ）  
- 部署：Kubernetes + NVIDIA 设备插件；基于队列等待的 HPA  
- 评测：ShareGPT、MT-Bench-v2、GSM8K、HumanEval 用于领域分布接受率测量  
- 参考：TensorRT-LLM 猜测解码供应商基线  

## 动手实践

1. **目标模型准备。** 选定 Llama 3.3 70B，使用 Marlin 量化为 FP8。部署于 vLLM 0.7，运行于 1xH100（或 2 个张量并行分片）。

2. **草稿模型来源。** 从 Red Hat Speculators 获取对齐的 EAGLE-3 草稿头（或使用 SpecForge 训练）。加载到 vLLM 猜测解码配置中。

3. **基线数据。** 启用猜测前，测量批大小 1/8/32 的 tokens/s、p50/p99 延迟、GPU 利用率。发布。

4. **启用 EAGLE-3。** 切换配置；复测同一基准。报告加速比、接受率、p99 尾延迟变化。

5. **P-EAGLE。** 启用并行猜测；衡量更深草稿树与串行 EAGLE-3 的区别。报告 P-EAGLE 何时带来优势、何时不利。

6. **领域流量。** 针对 ShareGPT、HumanEval 和领域专用流量测量接受率。识别草稿漂移发生时机。

7. **第二目标模型。** 在 Qwen3-Coder-30B MoE 上运行相同流程。草稿训练更复杂（MoE 路由噪声）。报告结果。

8. **K8s HPA。** 部署于 K8s，HPA 监控 `queue_wait_ms`。展示负载三倍时自动扩展能力。

9. **成本对比。** 计算每百万 Tokens 费用，与 Anthropic Claude Sonnet 4.7 和 OpenAI GPT-5.4 在同一评测下对比。发布。

## 使用示例

```text
$ curl https://infer.example.com/v1/chat/completions -d '{"messages":[...]}'
[serve]     vLLM 0.7，Llama 3.3 70B FP8，EAGLE-3 启用
[decode]    bs=8，accepted_tokens_per_step=3.2，acceptance_rate=0.76
[latency]   首标记 42ms，完整响应 980ms（620 标记）
[cost]      持续吞吐下每百万输出标记 $0.34
```

## 交付物

`outputs/skill-inference-server.md` 描述交付物：测量的推理栈，包含猜测解码、完整基准报告以及 K8s 部署。

| 权重 | 评价标准 | 测量方式 |
|:-:|---|---|
| 25 | 相较基线的实际加速 | 两个模型上匹配质量时吞吐 ≥2.5 倍 |
| 20 | 真实流量的接受率 | 按分布报告接受率 |
| 20 | p99 尾延迟指标 | 批大小 1/8/32 不同猜测状态下的 p99 延迟 |
| 20 | 运营能力 | K8s 部署，基于队列等待的 HPA，平滑滚展 |
| 15 | 文档与方法论 | 清晰解释变化与原因 |
| **100** | | |

## 练习

1. 测量当草稿版本比目标版本落后一代（如 Llama 3.3 ➔ 3.4 漂移）时接受率下降。构建监控与告警。

2. 实现 ngram 兜底机制：当 EAGLE-3 接受率低于阈值时，切换到 ngram 草稿。报告稳定性改进。

3. 进行受控 MoE 实验：对同一 Qwen3-Coder-30B，分别注入和不注入路由噪声。测量草稿接受率敏感度。

4. 扩展部署到 H200（141GB 内存）。报告每副本可用模型大小增益，以及是否能服务未经量化的 Llama 3.3 70B。

5. 在同一 H100 硬件上基准测试 TensorRT-LLM 猜测解码。报告其优于 vLLM 的场景。

## 关键词

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| 草稿模型 | “猜测模型” | 小模型提出 N 个标记供目标验证 |
| EAGLE-3 | “2026 草稿架构” | 基于目标隐藏状态训练草稿头；约 75% 接受率 |
| P-EAGLE | “并行猜测” | 目标模型一次调用验证整棵草稿树 |
| 接受率 | “命中率” | 草稿标记被接受无需重采样的比例 |
| 量化 | “FP8 / INT4” | 低精度权重，节省 GPU 内存 |
| 队列等待 | “HPA 指标” | 请求在队列中等待启动推理的时长 |
| Speculators hub | “对齐草稿中心” | Red Hat Neural Magic 发布的常见开源模型草稿 |

## 进一步阅读

- [vLLM EAGLE 和 P-EAGLE 文档](https://docs.vllm.ai) — 参考推理栈  
- [P-EAGLE (AWS 2026)](https://aws.amazon.com/blogs/machine-learning/p-eagle-faster-llm-inference-with-parallel-speculative-decoding-in-vllm/) — 并行猜测论文及集成  
- [SGLang SpecForge](https://github.com/sgl-project/SpecForge) — 草稿头训练流水线  
- [Red Hat Speculators](https://github.com/neuralmagic/speculators) — 对齐草稿中心  
- [TensorRT-LLM 猜测解码](https://nvidia.github.io/TensorRT-LLM/) — 供应商方案  
- [Fireworks.ai 推理架构](https://fireworks.ai/blog) — 商业参考  
- [EAGLE-3 论文 (arXiv:2503.01840)](https://arxiv.org/abs/2503.01840) — 方法论文  
- [vLLM 代码库](https://github.com/vllm-project/vllm) — 代码与基准测试
