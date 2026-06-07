# EAGLE-3 在生产中的投机解码（Speculative Decoding）

> 投机解码将一个快速的草稿模型与目标模型配对。草稿模型提出 K 个候选令牌；目标模型单次前向验证；被接受的令牌免费。2026 年，EAGLE-3 是生产级变体——它在目标模型的隐藏状态上训练草稿头，而非在原始令牌上，从而将接受率 alpha 推高到通用聊天的 0.6-0.8 区间。关键问题不是“草稿有多快”，而是“我的流量上 alpha 是多少？”如果 alpha 低于约 0.55，投机解码在高并发下是负收益，因为每个被拒绝的草稿需要额外一次目标前向。这一课程教你先测量 alpha，再开关投机解码开关。

**类型：** 学习  
**语言：** Python（标准库，自制接受率模拟器）  
**先决条件：** 第 17 阶段 · 04（vLLM 服务内部实现）、第 10 阶段 · 18（多令牌预测）  
**时间：** 约 60 分钟

## 学习目标

- 说出三代投机解码，解释 EAGLE-3 相对于 EAGLE-2 和经典草稿模型的变化。
- 定义接受率 alpha，根据 alpha 和 K（草稿长度）计算预期加速，找出你的目标并发下的盈亏平衡 alpha。
- 解释为什么 vLLM 2026 版本中的投机解码是显式选择（非默认），以及为何不测量 alpha 就开启是生产反模式。
- 制定测量计划：选择哪个基准，哪个提示分布，哪个并发点，哪个指标做为门控。

## 问题

解码过程受内存带宽限制。在 H100 上运行 Llama 3.3 70B FP8，每解码一个令牌约读取 140 GB/s 权重并输出一个令牌。解码时 GPU 计算几乎闲置——瓶颈是 HBM 带宽，而非矩阵乘法吞吐。

投机解码利用这种差距。先用廉价草稿模型生成 K 个候选令牌，再让目标模型单次前向验证全部 K 个。每个被验证的令牌实际上是“免费”的（摊销到目标模型本来就要做的一批 K 个前向中）。

经典草稿模型方法用同一系列更小模型（Llama 3.2 1B 为 Llama 3.3 70B 草拟）。此方法可行，但接受率一般——因为小模型分布与目标分布偏离。EAGLE、EAGLE-2 及 EAGLE-3 都直接在目标模型内部状态上训练轻量草稿头，使草稿分布更贴近目标分布。这也是 alpha 从草稿模型的 0.4 提升到 EAGLE-3 的 0.6-0.8 的原因。

注意：EAGLE-3 在 vLLM 2026 版本中为显式开启。必须设置 `speculative_config`。不开启则无加速。不测量真实流量上的 alpha 直接开启的团队，经常遭遇尾时延变差。

## 概念

### 投机解码真正带来的收益

无投机解码，每个令牌成本是一轮目标前向。投机解码以草稿长度 K 和接受率 alpha 计，目标模型每次前向期望生成令牌数为 `1 + K * alpha`。加速比为 `(1 + K * alpha) / (1 + epsilon)`，其中 epsilon 是草稿加验证的额外开销。举例 K=5，alpha=0.7：`(1 + 5*0.7) / (1 + 0.1) = 4.5 / 1.1 = 4.1x`。实际加速一般 2-3 倍，因为生产流量上 alpha 很少这么高且高批次时 epsilon 会增大。

### 为什么 alpha 是唯一关键指标

被拒绝的令牌不会消失——它们会导致对第一个被拒令牌进行第二次目标前向。在 alpha 降至 0.4 的工作负载下，既要付出草稿开销、验证开销，还要额外重试。高并发（如 256 并发）时，目标模型的批次已足够大，内存带宽瓶颈变窄。在大多数 2026 年硬件上，alpha 低于 0.55 就会使投机解码成为负收益。

Alpha 会根据工作负载变化。在 ShareGPT 风格通用聊天上，用共享训练的 EAGLE-3 可达到 0.6-0.8。特定领域流量（代码、医疗、法律）中通用草稿头训练下的 alpha 降至 0.4-0.6。训练专有领域草稿头能恢复 alpha——相较于目标微调，这是轻量且快速的训练。

### EAGLE 各代一览

- **经典草稿模型**：同系列小模型。Alpha 0.3-0.5。架构简单——加载两个模型，草稿每次前向 K 次，目标前向一次。
- **EAGLE-1（2024）**：单个草稿头在目标隐藏层（最后一层）训练。Alpha 约 0.5-0.6。对目标带来小参数开销。
- **EAGLE-2（2025）**：自适应草稿长度和基于树的草稿（单次目标前向验证多条分支）。Alpha 约 0.6-0.7。草稿调度器更复杂。
- **EAGLE-3（2025-2026）**：草稿头在多个目标层训练（非仅最后一层），对齐更好。通用聊天 alpha 约 0.6-0.8。

### 2026 年生产方案

1. 先无修改部署目标模型。测 baseline 的 TTFT（首令牌时间）、ITL（内尾延迟）、目标并发吞吐。
2. 通过 vLLM `speculative_config` 开启 EAGLE-3 草稿。重新跑基准。
3. 记录接受率 alpha。vLLM V1 报告为 `spec_decode_metrics.accepted_tokens_per_request`，除以请求的草稿长度得 alpha。
4. 若生产流量上 alpha 小于 0.55，关闭投机解码或训练专门领域的 EAGLE-3 草稿。
5. 在生产并发下复测，确认 P99 ITL 未恶化。

### 生产中的陷阱：P99 尾延迟

投机解码平均 ITL 降低。若不调优，P99 可能上升。被拒草稿触发两轮执行（草稿 + 验证失败 + 重试）。满批情况下，这两轮过程序列化。应关注 P99 ITL，而非 P50。

### EAGLE-3 已部署场景

Google 在 2025 年于 AI Overviews 中部署了投机解码（质量相同，响应更快）。vLLM V1 提供文档化接口 `speculative_config`；V1 版本中的 N-gram GPU 投机解码兼容 chunked prefill。SGLang 推荐 EAGLE-3 作为重前缀任务的草稿路径。

### 盈亏平衡数学公式

预期加速：`S(alpha, K) = (1 + K*alpha) / (1 + verify_overhead)`。设定 `S = 1`，解出 alpha：`alpha_breakeven = verify_overhead / K`。典型验证开销约 0.15，K=5 时：`alpha_breakeven = 0.03`。这是假设基础解码时序。高并发时验证开销上升，且批量序列共享内存带宽，实际有效 alpha 盈亏平衡升到约 0.45-0.55。

### 何时不使用投机解码

- 单批离线生成，时延不敏感。用纯目标模型。
- 超短输出（50 令牌以下）。草稿和验证开销占主导。
- 无领域训练草稿头的专用领域。alpha 太低。
- vLLM v0.18.0 版本加草稿模型投机解码及 `--enable-chunked-prefill` 组合无法编译。文档中唯一例外是 V1 版本的 N-gram GPU 投机解码。

## 使用方法

`code/main.py` 模拟在不同 alpha 和 K 草稿长度下，使用与不使用投机解码的解码循环。它会打印盈亏平衡 alpha 、测得加速率和尾时延行为。可以在多个 (alpha, K) 组合上运行，精准把握投机解码何时开始带来收益。

## 推出方案

此课程生成 `outputs/skill-eagle3-rollout.md`。输入目标模型、流量分布说明和并发目标，生成分阶段的 EAGLE-3 推出计划——基准测试基线、启用配置、测量 alpha、alpha >= 0.55 作为门控，关注 P99 ITL。

## 练习

1. 运行 `code/main.py`。K=5 时，要实现 2 倍和 3 倍加速，alpha 分别需要多少？对 verify_overhead 的敏感性怎样？
2. 假设生产流量 70% 通用聊天，30% 代码。通用聊天用 ShareGPT 训练的 EAGLE-3 达 alpha 0.7；代码达 alpha 0.4。混合后的 alpha是多少？投机解码是否净收益？
3. 阅读 vLLM `speculative_config` 文档。列出三种模式（草稿模型、EAGLE、N-gram），说明哪种兼容 chunked prefill。
4. 你发现启用 EAGLE-3 后平均 ITL 降低 25%，但 P99 ITL 上升 15%。诊断原因并提出缓解方案。
5. 计算 Llama 3.3 70B 的 EAGLE-3 草稿头内存开销。和用 Llama 3.2 1B 作为经典草稿模型相比如何？

## 关键词

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| Speculative decoding（投机解码） | “草稿加验证” | 用廉价模型提出 K 个令牌，再用一次目标前向验证全部 K |
| Acceptance rate alpha（接受率 alpha） | “投机接受率” | 草稿令牌被目标接受的比例；唯一重要的指标 |
| Draft length K（草稿长度 K） | “spec k” | 草稿模型每次目标前向提出的令牌数；典型 4-8 |
| Verify overhead epsilon（验证开销 epsilon） | “投机开销” | 相较纯目标前向，验证和重试增加的额外开销；随批量增大 |
| EAGLE-3 | “最新 EAGLE” | 2025-2026 版本；草稿头训练多层隐藏状态；通用聊天 alpha 0.6-0.8 |
| `speculative_config` | “vLLM 投机配置” | vLLM V1 中的显式开启项；无默认不开加速 |
| N-gram spec decode（N-gram 投机解码） | “N-gram 草稿” | GPU 侧利用提示中的 N-gram 查找进行草稿；支持 chunked prefill |
| Break-even alpha（盈亏平衡 alpha） | “无收益 alpha” | 投机解码加速为零时的 alpha；需注意生产并发时的值 |
| Rejected-draft two-pass（拒绝草稿两轮） | “重试成本” | 草稿拒绝时造成两次目标前向；加剧 P99 尾延迟 |

## 延伸阅读

- [vLLM — 投机解码文档](https://docs.vllm.ai/en/latest/features/spec_decode/) — 关于 `speculative_config` 和 V1 版本 chunked-prefill 兼容的权威来源。  
- [vLLM 投机配置 API](https://docs.vllm.ai/en/latest/api/vllm/config/speculative/) — 具体字段集。  
- [EAGLE 论文（arXiv:2401.15077）](https://arxiv.org/abs/2401.15077) — EAGLE 草稿头最初的提出。  
- [EAGLE-2 论文（arXiv:2406.16858）](https://arxiv.org/abs/2406.16858) — 自适应草稿和树形草稿。  
- [加州大学伯克利校区 EECS-2025-224 报告](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2025/EECS-2025-224.html) — 带投机解码的高效 LLM 系统。  
- [BentoML — 投机解码](https://bentoml.com/llm/inference-optimization/speculative-decoding) — 生产部署检查清单。
