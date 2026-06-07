# Janus-Pro：用于统一多模态模型的解耦编码器

> 统一的多模态模型存在不可避免的矛盾。理解任务需要语义特征——SigLIP 或 DINOv2 输出向量富含概念层信息。生成任务需要便于重构的编码——VQ 令牌可重新组合成清晰像素。这两种目标在单一编码器中无法兼得。Janus（DeepSeek，2024年10月）和 Janus-Pro（DeepSeek，2025年1月）提出解决方案：停止硬撑，解耦两个编码器。在任务间共享 transformer 主体，但理解走 SigLIP，生成走 VQ 分词器。7B 模型规模下，Janus-Pro 在 GenEval 上击败 DALL-E 3，同时在 MMMU 上匹配 LLaVA。此课讲解为何两个编码器能成功而单一失败。

**类型：** 实战  
**语言：** Python（标准库，双编码器路由 + 共享主体信号）  
**先决条件：** 第12阶段·13（Transfusion），第12阶段·14（Show-o）  
**时长：** ~120分钟  

## 学习目标

- 解释为何单一共享编码器会牺牲理解或生成质量。  
- 描述 Janus-Pro 的路由：理解输入采用 SigLIP 特征，生成输入和输出均采用 VQ 令牌。  
- 梳理 Janus-Pro 在数据混合和规模上的提升如何造成成功。  
- 比较解耦（Janus-Pro）、耦合连续（Transfusion）和耦合离散（Show-o）架构。  

## 问题

统一模型在理解和生成之间共享 transformer 主体。以往尝试（Chameleon、Show-o、Transfusion）都用单一视觉分词器应对双向。此分词器存在折中：

- 优化重构（生成）：VQ-VAE 捕捉细粒度像素细节，但令牌语义连贯性较弱。  
- 优化语义（理解）：SigLIP 嵌入使“猫”图像聚在“猫”令牌附近，但无法良好重构。  

Show-o 和 Transfusion 的代价是单向质量明显受限。Janus-Pro 问：任务需求不同，为什么必须用同一分词器？

## 概念

### 解耦视觉编码

Janus-Pro 架构分离两个编码器：

- 理解路径。输入图像 → SigLIP-SO400m → 两层 MLP → transformer 主体。  
- 生成路径。输入图像（条件已有图像时）→ VQ 分词器 → 令牌ID → transformer 主体。  
- 输出生成。transformer 预测图像令牌 → VQ 解码器 → 像素。  

transformer 主体共享。主体上下游部分针对任务单独设计。

输入通过提示格式区分：`<understand>` 标签走 SigLIP，`<generate>` 走 VQ。或根据任务隐式路由。

### 原理

理解损失得到 SigLIP 特征，CLIP 风格预训练使其调校成语义相似度高。因输入特征更契合任务，模型感知基准胜过 Show-o / Transfusion。

生成损失得到 VQ 令牌，分词器针对重构做优化。图像质量优于 Show-o，因为 VQ 码可清晰还原像素。

共享 transformer 主体见两种输入分布（SigLIP 和 VQ），能学习适配。论点：足够数据 + 充足参数，主体能吸收切换。

### 数据规模扩展 —— Janus 与 Janus-Pro

Janus（原始，arXiv 2410.13848）首次提出解耦但规模小（1.3B 参数，数据有限）。Janus-Pro（arXiv 2501.17811）扩展：

- 7B 参数（对比 1.3B）。  
- 阶段1（对齐）用 9000万图文对，较之前的 7200万增加。  
- 阶段2（统一）用 7200万，较之前 2600万增加。  
- 阶段3新增 20万图像生成指令样本。  

结果：Janus-Pro-7B 在 MMMU 上匹配 LLaVA（60.3 对约 58），在 GenEval 上击败 DALL-E 3（0.80 对 0.67）。开源模型同时实现统一光谱两端的竞争力。

### JanusFlow —— 纠正流变体

JanusFlow（arXiv 2411.07975）将 VQ 生成路径改为纠正流（continuous flow）生成路径。分割变为理解用 SigLIP + 生成用纠正流。质量上限进一步提升。架构依旧是解耦编码器 + 共享主体。

### 共享主体的职责

transformer 主体处理统一序列，但面对两种输入分布。职责是：

- 理解任务：消费 SigLIP 特征 + 文本令牌 → 自回归产生文本。  
- 生成任务：消费文本令牌 + （可选）图像 VQ 令牌 → 自回归生成图像 VQ 令牌。  

主体无模态特定权重。它是你在 Qwen 或 Llama 内见到的典型文本风 transformer，加上两种输入适配器。

有趣的是，Janus-Pro 的主体可从预训练 LLM 初始化。Janus-Pro 确实从 DeepSeek-MoE-7B 初始化。此选择重要：LLM 为纯从零训练的统一模型难以达到的推理能力。

### 与 InternVL-U 比较

InternVL-U（第12.10课）是2026年后续，结合：

- 原生多模态预训练（InternVL3 主干）。  
- 解耦编码器路由（SigLIP 输入，VQ + diffusion 头输出）。  
- 统一的理解 + 生成 + 编辑。  

InternVL-U 将 Janus-Pro 架构方案纳入更大框架中。解耦编码器已成大规模统一模型默认做法。

### 局限性

解耦编码器增加架构复杂度。需训练两个分词器、维护两条输入路径和两套失败模式。无生成需求产品，Janus-Pro 过度设计——选用 LLaVA 系列理解模型。无理解需求产品，Janus-Pro 资格过高——选用 Stable Diffusion 3 / Flux 模型。需两者产品，Janus-Pro 是当前参考开源架构。

## 使用它

`code/main.py` 模拟 Janus-Pro 路由：

- 两个模拟编码器：SigLIP 风（产出 256 维语义向量）和 VQ 风（产出整数码）。  
- 基于任务标签选择编码器的提示路由器。  
- 共享主体（模拟）处理任一编码器产出令牌序列。  
- 从阶段1（对齐）切换到阶段3（指令微调）加权采样调度。  

打印3个示例的路由路径：图像问答、文本到图像、图像编辑。

## 发布它

本课生成 `outputs/skill-decoupled-encoder-picker.md`。对希望获得统一生成与理解尖端质量的产品，推荐 Janus-Pro、JanusFlow 或 InternVL-U，并附具体数据规模建议。

## 练习

1. Janus-Pro-7B 在 GenEval 上击败 DALL-E 3。解释为何一个7B开源模型能匹配前沿专有模型生成能力，但理解能力上达不到。  

2. 实现一个路由函数：给定提示文本，分类为 `understand` 或 `generate`。如何处理“describe and then sketch”这类歧义提示？  

3. JanusFlow 用纠正流替换了 VQ 路径。现在 transformer 主体输出什么，损失有何变化？  

4. 提出 Janus-Pro 架构可通过另一个独立编码器支持的第四任务。示例：图像分割（DINO 风格）、深度估计（MiDaS 风格）。  

5. 阅读 Janus-Pro 数据扩规模第4.2节。哪个数据阶段对比 Janus 对文本生成质量提升贡献最大？  

## 关键词

| 术语 | 大众说法 | 实际含义 |
|------|----------|----------|
| 解耦编码 | “两个视觉编码器” | 每个方向独立分词器或编码器：理解用语义，生成用重构 |
| 共享主体 | “一个 transformer” | 单一 transformer 处理任一编码器输出，无模态特定权重 |
| 理解用 SigLIP | “语义特征” | CLIP 系列视觉塔，提供丰富概念特征但重构差 |
| 生成用 VQ | “重构编码” | 向量量化令牌，可清晰还原像素 |
| JanusFlow | “纠正流变体” | Janus-Pro 换成连续流生成头替代 VQ |
| 路由标签 | “任务标签” | 提示标记（`<understand>` / `<generate>`）选择输入编码器 |

## 相关阅读

- [Wu 等 — Janus（arXiv:2410.13848）](https://arxiv.org/abs/2410.13848)  
- [Chen 等 — Janus-Pro（arXiv:2501.17811）](https://arxiv.org/abs/2501.17811)  
- [Ma 等 — JanusFlow（arXiv:2411.07975）](https://arxiv.org/abs/2411.07975)  
- [InternVL-U（arXiv:2603.09877）](https://arxiv.org/abs/2603.09877)  
- [Dong 等 — DreamLLM（arXiv:2309.11499）](https://arxiv.org/abs/2309.11499)
