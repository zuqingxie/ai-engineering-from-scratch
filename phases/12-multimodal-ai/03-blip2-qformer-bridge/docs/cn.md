# 从 CLIP 到 BLIP-2 — Q-Former 作为模态桥梁

> CLIP 对齐图像和文本，但无法生成标题、回答问题或进行对话。BLIP-2（Salesforce，2023）通过一个小型可训练桥梁解决了这个问题：32 个可学习查询向量通过交叉注意力（cross-attention）关注冻结的 ViT 特征，然后直接插入冻结的 LLM 输入流。188M 参数的桥梁连接了一个 11B LLM 和一个 ViT-g/14。所有基于适配器（adapter）的 VLM 到 2026 年——MiniGPT-4、InstructBLIP、LLaVA 的亲戚——都是其后代。本课题解析 Q-Former 体系结构，解释其两阶段训练，并构建一个玩具版本，将视觉标记输入冻结文本解码器。

**类型：** 构建  
**语言：** Python（标准库，交叉注意力 + 可学习查询演示）  
**先修：** 第 12 阶段 · 02（CLIP），第 7 阶段（Transformer）  
**时间：** ~180 分钟

## 学习目标

- 解释为何在冻结的视觉编码器和冻结 LLM 之间设置可训练瓶颈，相较端到端微调，在成本和稳定性上更胜一筹。
- 实现一个交叉注意力模块，其中固定数量的可学习查询关注外部图像特征。
- 讲解 BLIP-2 的两阶段预训练：表示学习（ITC + ITM + ITG）然后生成式训练（带冻结解码器的语言模型损失）。
- 比较 Q-Former 与 LLaVA 中使用的简单 MLP 投影器，并论证每种选择的优劣时机。

## 问题描述

你有一个冻结的 ViT，它对每张图像输出 256 个维度为 1408 的 patch 标记。你还有一个冻结的 7B LLM，期望的标记嵌入维度为 4096。明显的桥梁是：用一个线性层从 1408 投影到 4096，这可行，但将所有 256 个 patch 标记输入 LLM 的上下文会额外消耗 256 个标记。对于一批 32 张图像，仅视觉模态就消耗了 8192 个标记。

BLIP-2 的问题是：能否将 256-token 的图像表示压缩为更少的标记（比如 32 个），同时保留足够的信息以让 LLM 生成标题、回答问题和推理？而且能否只训练这个桥梁，不动冻结的骨干，实现仅训练桥梁参数的低成本？

答案是：Q-Former。32 个可学习查询向量通过交叉注意力访问 ViT 的 patch 标记，产生 32 个视觉摘要标记供 LLM 消费。共计 1.88 亿参数。训练时，先用对比、匹配和生成目标训练，不触碰 LLM。

## 概念解析

### 可学习查询（Learnable queries）

Q-Former 的核心妙招是：不让 LLM 的文本标记去关注图像 patch，而是引入一组 32 个可学习查询向量 `Q`，让它们去关注图像 patch。查询是模型的参数——在训练中学习，且同一组查询用于所有图像。

交叉注意力后，每个查询包含图像的压缩摘要——“描述主要物体”、“描述背景”、“计数物体数量”等。查询并不专门对应具体语义标签；它们学习任何能降低下游损失的编码方式。

### 架构

Q-Former 是一个小型 Transformer（12 层，约 1 亿参数），有两条路径：

1. 查询路径：32 查询向量先经自注意力（相互之间），再通过冻结 ViT patch 标记的交叉注意力，最后经过前馈网络（FFN）。
2. 文本路径：类似 BERT 的文本编码器，与查询路径共享自注意力和 FFN 权重。文本路径禁用交叉注意力。

训练时两条路径同时运行。查询和文本通过共享的自注意力交互，因此查询可以根据文本调整（用于 ITM、ITG 任务）。推理时只让查询通行，得到 32 个视觉标记。

### 两阶段训练

BLIP-2 分两阶段预训练：

阶段 1：表示学习（不涉及 LLM）。三个损失：
- ITC（图文对比）：CLIP 风格的对比损失，针对查询池化向量和文本 CLS 向量。
- ITM（图文匹配）：二分类器，判定图文对是否匹配，采用困难负样本挖掘。
- ITG（基于图像的文本生成）：条件于查询的因果语言模型头，逼迫查询编码可生成文本的内容。

仅训练 Q-Former，ViT 冻结，不涉及 LLM。

阶段 2：生成式训练。接入冻结 LLM（OPT-2.7B 或 Flan-T5-XL 等）。将 32 个查询输出通过小线性层映射到 LLM 嵌入维度，作为文本提示的前缀。仅训练线性投影和 Q-Former，针对拼接的提示 + 图像 + 标题序列计算语言模型损失。

阶段 2 结束后，Q-Former 加投影层就是完整视觉适配器。推理流程：图像 → ViT → Q-Former → 线性投影 → 置于文本前缀 → 冻结 LLM 生成输出。

### 参数经济性

BLIP-2 配置 ViT-g/14（11 亿参数，冻结）+ OPT-6.7B（67 亿参数，冻结）+ Q-Former（1.88 亿，训练中）= 总共约 80 亿参数，仅训练 1.88 亿，即 ~2.4%。训练成本反映这一点：几天用少量 A100 对比端到端微调几周。

质量方面：BLIP-2 在零样本视觉问答中匹配或超越 Flamingo-80B，同时体积小 50 倍。桥梁方案行之有效。

### InstructBLIP 和指令感知 Q-Former

InstructBLIP（2023）将输入扩展为额外的指令文本。交叉注意力时，查询同时访问图像 patch 和指令。查询可针对具体指令“数汽车”、“描述氛围”等专项调整，而不是固定摘要。在未见指令的任务中获得基准提升。

### MiniGPT-4 与仅投影器方案

MiniGPT-4 保留 Q-Former，本体训练只动输出线性投影，其余冻结。成本低，但代价是质量下降——查询来自 BLIP-2，不再专属。适合快速迭代，但非最佳架构。

### LLaVA 采用更简单的原因

LLaVA（2023，第 12.05 课）将 Q-Former 替换为简单的两层 MLP，将每个 ViT patch token 投影进 LLM 空间——因 24x24 网格图像带来 576 个 token，全数送入 LLM。压缩率低，但允许 LLM 直接关注原始 patch。当时颇有争议；2023 年末，由于视觉指令数据（LLaVA-Instruct-150k）证明 MLP 可训练保留足够信号，该方案主流化。权衡是：LLaVA 上下文消耗快但自然支持多图像和视频。

到 2026 年，领域分化：Q-Former 在受限 token 预算场景（长视频、多图）存活；MLP 投影器在每 token 质量优先场景占优。

### 门控交叉注意力：Flamingo 祖先

Flamingo（第 12.04 课）早于 BLIP-2，采用同样交叉注意力理念，但在冻结 LLM 每层引入，而非单桥。BLIP-2 表明对输入层压缩即可成功。Gemini 和 Idefics 结合两者：交替使用输入 token 加可选门控交叉注意力实现上下文内少样本学习。

### 2026 年的后代们

- Q-Former：BLIP-2、InstructBLIP、MiniGPT-4 及多数视频语言模型，皆因 token 预算限制。
- Perceiver resampler：Flamingo 变体（第 12.04 课）；Idefics 系列，Eagle，OmniMAE。
- MLP 投影器：LLaVA，LLaVA-NeXT，LLaVA-OneVision，Cambrian-1。
- 注意力池（Attention pool）：VILA，PaliGemma。

四者皆有效。关键取决于你是受 token 预算约束，还是优先单 token 质量。

## 使用方法

`code/main.py` 构建一个标准库 Q-Former 样式交叉注意力：

1. 模拟 256 个图像 patch 标记（128 维）。
2. 实例化 32 个可学习查询（128 维）。
3. 执行缩放点积交叉注意力（查询来自 queries，键/值来自 patches）。
4. 通过线性层投影到 LLM 维度（512）。
5. 输出 32 个 LLM 就绪视觉标记。

所有数学计算用纯 Python（嵌套循环向量操作）。玩具模型但形状正确。会打印注意力权重矩阵，展示每个查询关注哪些 patch。

## 发布成果

本课产出 `outputs/skill-modality-bridge-picker.md`。根据目标 VLM 配置（视觉编码器标记数、LLM 上下文预算、部署限制、质量目标）推荐 Q-Former、MLP 或 Perceiver resampler，并附简短理由和参数量估计。

## 练习

1. 用 PyTorch 实现交叉注意力模块。验证 32 个查询和 256 个键/值时，注意力权重矩阵大小为 32x256，且每行 softmax 后和为 1。

2. BLIP-2 阶段 1 中 Q-Former 同时运行三种损失：ITC、ITM、ITG。写出它们的前向签名伪代码。哪一个需要激活文本编码路径？

3. 比较参数量：Q-Former（12 层，隐藏 768）与两层 MLP 投影器（1408 → 4096，两层）。188M 参数的 Q-Former 在何种 LLM 规模下训练效率回本？

4. 阅读 BLIP-2 论文（arXiv:2301.12597）第 3.2 节关于 Q-Former 初始化。解释为何选用 BERT-base 初始化（非随机）可加速收敛。

5. 对一段 10 分钟视频，1 FPS 采样共 60 帧，计算（Q-Former → 32 token/帧）与（MLP 投影器 → 576 token/帧）方案的每帧 token 成本。哪个更适合 128k token LLM 上下文窗口？

## 关键词

| 术语                     | 大众说法           | 实际含义                                                         |
|--------------------------|--------------------|------------------------------------------------------------------|
| Q-Former                 | “Querying transformer”（查询 Transformer） | 拥有 32 个可学习查询向量的小型 Transformer，交叉关注冻结 ViT 特征。           |
| Learnable queries        | “Soft prompt for vision”（视觉软提示） | 一组固定参数，作为交叉注意力查询端；针对模型训练，共享于所有输入。             |
| Cross-attention          | “Q from here, K/V from there”（这里取 Q，那里取 K/V） | 查询、键、值来自不同源的注意力机制；查询从 ViT patch 中抽取信息。              |
| ITC                      | “Image-text contrastive”（图文对比） | 对 Q-Former 聚合查询与文本 CLS 之间的 CLIP 风格对比损失。                     |
| ITM                      | “Image-text matching”（图文匹配） | 针对困难负样本的二分类匹配器，迫使查询区分细粒度不匹配。                       |
| ITG                      | “Image-grounded text generation”（基于图像的文本生成） | 条件于查询生成文本的因果语言模型损失，推动查询编码可解码文本的内容。           |
| Two-stage pretraining    | “Representation then generative”（先表示后生成） | 阶段 1 仅训练 Q-Former（ITC/ITM/ITG）；阶段 2 接入冻结 LLM 训练投影和 Q-Former。  |
| Frozen backbone          | “Do not finetune”（不微调） | 视觉编码器和 LLM 权重冻结，仅训练桥梁。                                     |
| Projection head          | “Linear to LLM dim”（线性映射至 LLM 维度） | 最终线性层，将 Q-Former 输出映射至 LLM 嵌入维度。                             |
| Perceiver resampler      | “Flamingo's version”（Flamingo 版本） | 类似可学习查询交叉注意力，由 Flamingo 在每层使用，而非单一桥梁。              |

## 延伸阅读

- [Li 等人 — BLIP-2（arXiv:2301.12597）](https://arxiv.org/abs/2301.12597) — 核心论文。  
- [Li 等人 — BLIP（arXiv:2201.12086）](https://arxiv.org/abs/2201.12086) — 预先版本，包含 ITC/ITM/ITG 三合一。  
- [Li 等人 — ALBEF（arXiv:2107.07651）](https://arxiv.org/abs/2107.07651) — “先对齐再融合”，阶段 1 训练的概念祖先。  
- [Dai 等人 — InstructBLIP（arXiv:2305.06500）](https://arxiv.org/abs/2305.06500) — 指令感知 Q-Former。  
- [Zhu 等人 — MiniGPT-4（arXiv:2304.10592）](https://arxiv.org/abs/2304.10592) — 仅投影器方案。  
- [Jaegle 等人 — Perceiver IO（arXiv:2107.14795）](https://arxiv.org/abs/2107.14795) — 通用可学习查询交叉注意力架构。
