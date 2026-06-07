# Mixture of Experts (MoE) 混合专家模型

> 一个稠密的 700 亿参数 Transformer（Transformer 架构）对每个 token 激活所有参数。一个 6710 亿参数的 MoE 每个 token 只激活 370 亿参数，但在所有基准测试中均超越前者。稀疏性是本十年最重要的规模扩展理念。

**类型：** 构建  
**语言：** Python  
**先修知识：** 第7阶段 · 05（完整 Transformer），第7阶段 · 07（GPT）  
**时间：** 约45分钟

## 问题

稠密 Transformer 的推理 FLOPs 约等于其参数量（正向传播乘以2）。当稠密模型规模扩大时，每个 token 的计算成本也成倍增加。到2024年，前沿技术面临计算瓶颈：要有实质性提升，单个 token 需要指数级更多 FLOPs。

混合专家模型（MoE）突破了这一限制。将每个前馈网络（FFN）替换为 `E` 个独立专家和一个路由器，后者为每个 token 选择 `k` 个专家。总参数量 = `E × FFN_size`，每个 token 激活参数 = `k × FFN_size`。2026年的典型配置：`E=256`，`k=8`。存储大小随 `E` 线性增长，计算成本随 `k` 线性增长。

2026年技术前沿几乎全部是 MoE：DeepSeek-V3（总参数6710亿，激活370亿），Mixtral 8×22B，Qwen2.5-MoE，Llama 4，Kimi K2，gpt-oss。人工分析网站（Artificial Analysis）的独立排行榜上，前10的开源模型均为 MoE。

## 概念

![MoE层：路由器为每个token从E个专家中选择k个](../assets/moe.svg)

### FFN替换

稠密 Transformer 模块：

```text
h = x + attn(norm(x))
h = h + FFN(norm(h))
```

MoE 模块：

```text
h = x + attn(norm(x))
scores = router(norm(h))              # (N_tokens, E)
top_k = argmax_k(scores)              # 每个token从E个专家中选k个
h = h + sum_{e in top_k}(
        gate(scores[e]) * Expert_e(norm(h))
    )
```

每个专家是独立的 FFN（通常是 SwiGLU）。路由器是一个线性层。每个 token 选择其 `k` 个专家并获得它们输出的加权混合。

### 负载均衡问题

如果路由器将90%的 token都分配给专家3，其他专家将得不到数据。主要有三种解决方案：

1. **辅助负载均衡损失**（Switch Transformer，Mixtral）。添加与专家使用量方差成比例的惩罚项。有效但增加了超参数和第二个梯度信号。  
2. **专家容量限制 + token丢弃**（早期Switch）。每个专家最多处理 `C × N/E` 个 token，超出则跳过该层。降低质量。  
3. **无辅助损失的负载均衡**（DeepSeek-V3）。为每个专家添加一个可学习偏置，调整路由器的 top-k 选择。偏置在训练损失之外更新，对主目标无惩罚。2024年的重要突破。

DeepSeek-V3方案：每次训练步骤后，检查每个专家的使用率是否超标，针对超载或欠载的专家，将偏置上下调节 `±γ`。选择时使用 `scores + bias`，但门控概率使用原始 `scores`，解耦路由选择和输出表达。

### 共享专家

DeepSeek-V2/V3 将专家划分为**共享专家**和**路由专家**。每个 token 通过所有共享专家，路由专家使用 top-k 选择。共享专家负责提取通用知识，路由专家负责专门化。V3 配置是1个共享专家加256个路由专家中的top-8。

### 细粒度专家

经典 MoE（GShard，Switch）中，每个专家宽度等于整个 FFN。`E` 小（8~64），`k` 小（1~2）。

现代细粒度 MoE（DeepSeek-V3，Qwen-MoE）中，每个专家更窄（约是 FFN 的1/8宽度）。`E` 大（256以上），`k` 也大（8以上）。总参数量相同，但组合数爆炸式增长。`C(256, 8) = 4×10^14` 种不同“专家”组合每个 token。质量提升，延迟不变。

### 成本剖面

每个 token，每层：

| 配置 | 每token激活参数量 | 总参数量 |
|--------|-------------------|----------|
| Mixtral 8×22B | 约39B | 141B |
| Llama 3 70B（稠密） | 70B | 70B |
| DeepSeek-V3 | 37B | 671B |
| Kimi K2（MoE） | 约32B | 1T |

DeepSeek-V3在几乎所有基准上均优于Llama 3 70B（稠密），但**每token激活FLOPs更少**。更多参数＝更多知识；更多激活FLOPs＝更多计算成本。MoE解耦了它们。

### 限制：内存

所有专家权重全驻留GPU，无论哪些被激活。6710亿参数模型需要约1.3 TB fp16显存。前沿MoE部署需要专家并行——专家分片到不同GPU，token分布通过网络路由。延迟主要由全对全通信决定，不是矩阵乘法。

## 构建

参见 `code/main.py`。一个纯标准库的小型 MoE 层，具有:

- `n_experts=8` 个类似 SwiGLU 的专家（每个一个线性层，方便演示）  
- top-k=2 路由  
- softmax 归一化的门控权重  
- 通过每专家偏置实现无辅助损失的均衡  

### 第1步：路由器

```python
def route(hidden, W_router, top_k, bias):
    scores = [sum(h * w for h, w in zip(hidden, W_router[e])) for e in range(len(W_router))]
    biased = [s + b for s, b in zip(scores, bias)]
    top_idx = sorted(range(len(biased)), key=lambda i: -biased[i])[:top_k]
    # 对所选择专家的原始scores做softmax
    chosen = [scores[i] for i in top_idx]
    m = max(chosen)
    exps = [math.exp(c - m) for c in chosen]
    s = sum(exps)
    gates = [e / s for e in exps]
    return top_idx, gates
```

偏置影响的是选择，不影响门控权重。这是DeepSeek-V3的诀窍——偏置修正负载不均但不影响模型输出。

### 第2步：对100个token进行路由

追踪不同专家的激活频率。无偏置时使用率倾斜。加入偏置更新循环（对过载专家减 `γ`，欠载专家增 `γ`）后，使用率在若干迭代内收敛到均匀分布。

### 第3步：参数量比较

打印一个MoE配置对应的“稠密等效”参数量。DeepSeek-V3配置为256路由 + 1共享，激活8个，d_model=7168。总参数量惊人，但激活参数量只有稠密 Llama 3 70B 的七分之一。

## 使用

HuggingFace加载示例：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x22B-v0.1")
```

2026年生产推理：vLLM原生支持 MoE 路由。SGLang 具备最快专家并行路径。两者自动处理 top-k 选择和专家并行。

**何时选用 MoE：**
- 你希望以较低推理成本获得最前沿质量。  
- 你具备足够显存/专家并行基础设施。  
- 你的工作负载以token数为主（聊天、代码）而非上下文长度（长文档）。

**何时不选 MoE：**
- 边缘部署——任何活跃FLOP都要付出全部存储成本。  
- 延迟敏感的单用户服务——专家路由增加额外开销。  
- 小模型（<70亿参数）——MoE 的质量优势只在计算量达到一定阈值后出现（约6亿激活参数）。

## 部署

参见 `outputs/skill-moe-configurator.md`。该技能根据参数预算、训练token数和部署目标，选择 E, k 和共享专家布局。

## 练习

1. **简单。** 运行 `code/main.py`。观察无辅助损失偏置更新如何在50次迭代内均衡专家使用率。  
2. **中等。** 用基于哈希的路由替换学习路由器（确定性，无需训练）。比较质量与均衡性。为什么学习路由器更好？  
3. **困难。** 实现GRPO风格的“rollout-matched routing”（DeepSeek-V3.2技巧）：记录推理时哪些专家激活，反向传播时强制使用相同路由。衡量对玩具策略梯度任务的影响。

## 关键词

| 术语 | 常用说法 | 实际含义 |
|------|-----------|---------|
| Expert（专家） | “多个FFN之一” | 独立的前馈网络；其参数专门处理稀疏的FFN计算片段。 |
| Router（路由器） | “门控” | 一个小型线性层，为每个token和专家打分；选择top-k专家。 |
| Top-k routing（Top-k路由） | “每token激活k个专家” | 每个token的前馈计算通过且仅通过k个专家，按门控权重加权。 |
| Auxiliary loss（辅助损失） | “负载均衡惩罚” | 用于惩罚专家激活不均的额外损失项。 |
| Auxiliary-loss-free（无辅助损失） | “DeepSeek-V3的诀窍” | 只通过路由器偏置平衡负载，无需额外梯度。 |
| Shared expert（共享专家） | “始终激活” | 所有token都会经过的额外专家，负责学习通用知识。 |
| Expert parallelism（专家并行） | “按专家分片” | 将不同专家分配到不同GPU，token跨网络路由。 |
| Sparsity（稀疏性） | “激活参数<总参数” | 计算比率 `k × expert_size / (E × expert_size)`；DeepSeek-V3约为5.5%。 |

## 延伸阅读

- [Shazeer 等 (2017)《极大规模神经网络：稀疏门控混合专家层》](https://arxiv.org/abs/1701.06538) — 基础理念。  
- [Fedus, Zoph, Shazeer (2022)《Switch Transformer：通过简单高效稀疏性实现万亿参数模型》](https://arxiv.org/abs/2101.03961) — Switch，经典MoE。  
- [Jiang 等 (2024)《Mixtral专家混合》](https://arxiv.org/abs/2401.04088) — Mixtral 8×7B。  
- [DeepSeek-AI (2024)《DeepSeek-V3技术报告》](https://arxiv.org/abs/2412.19437) — MLA + 无辅助损失 MoE + MTP。  
- [Wang 等 (2024)《无辅助损失的专家混合负载均衡策略》](https://arxiv.org/abs/2408.15664) — 基于偏置的负载平衡论文。  
- [Dai 等 (2024)《DeepSeekMoE：迈向专家混合语言模型的极致专家专门化》](https://arxiv.org/abs/2401.06066) — 本课程路由器采用的细粒度+共享专家拆分。  
- [Kim 等 (2022)《DeepSpeed-MoE：推进混合专家推理与训练》](https://arxiv.org/abs/2201.05596) — 最初的共享专家论文。
