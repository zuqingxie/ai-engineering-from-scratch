# 梯度检查点（Gradient Checkpointing）与激活重计算（Activation Recomputation）

> 反向传播会保留每一个中间激活值。在70B参数和128K上下文中，每个rank需保存3 TB的激活。检查点方法通过用计算换内存：重计算而非保存。关键是哪些部分应丢弃，答案绝非“全部”。

**类型：** 构建  
**语言：** Python（含numpy，可选torch）  
**先修课程：** 阶段10第04课（预训练Mini-GPT），阶段10第05课（扩展与分布式）  
**时间：** 约70分钟

## 问题

训练transformer时，每层的反向传播需要保存所有被微分的操作输入：注意力输入、Q/K/V投影、softmax输出、前馈网络输入、归一化输出和残差流。对于隐藏维度为`d`、序列长度`L`、批大小`B`的层，每层大约需要保存 `12 * B * L * d` 个浮点数。

对于 `d=8192, L=8192, B=1`，每层BF16激活约800 MB。64层模型的激活量为51 GB——这还不包括微批大小的乘积，不包括注意力softmax的中间值（每个头部 `L^2`），也不包括张量并行的部分复制。

二者成本：BF16权重加优化器状态大概占80GB内存，但激活超出限制。梯度检查点（亦称激活重计算）是标准的解决方案。丢弃大部分激活；反向计算时重做前向，重新生成它们。成本是额外的浮点运算。好处是内存减少量约为检查点段数与层数比值。

直接使用时，检查点使得每步的前向计算多消耗约33%。精心设计时——比如Korthikanti等人的“智能选择”方法——内存节省5倍，而浮点运算开销不足5%。加上FP8矩阵乘法、FSDP卸载及专家并行MoE时，这极其关键：你既节省不了内存，也浪费不起计算资源。

## 概念

### 反向传播真实需求

`output = layer(input)`。反向传播需要 `grad_input` 和 `grad_params`。计算它们需：

- `input`（线性层中 `grad_params = input.T @ grad_output` 需要用到）
- 一些激活的导数中间值（ReLU/GELU/softmax的导数依赖于激活值本身）

前向传播通过自动求导图自动保存这些。每个 `tensor.retain_grad()` 和每个需要保留输入的操作都会引用它们。

### 原始全层检查点

将网络拆成`N`段，前向时只保存每段的*输入*。当反向传播需要中间变量时，重新运行该段的前向以重新生成，然后求导。

示例：32层transformer分成32个1层段。

- 内存：只保存32个层输入（很小）vs保存32个层的激活量（很大）。
- 额外计算：每段多1次前向，总前向多33% FLOPs（反向约是前向两倍，整个步骤变成 1 + 1 + 2 = 4 单位计算，原来是 1 + 2 = 3）。

这就是Chen et al. 2016的原始方法：每 `sqrt(L)` 层设一个检查点以平衡内存与计算。L=64时即8检查点。

### 选择性检查点（Korthikanti 2022）

不同激活成本不同。注意力softmax输出是 `B*L*L*heads`，随序列长度平方增长。FFN隐藏层激活是 `B*L*4d`，线性增长。长序列时，softmax占主导。

选择性检查点保留存储成本低的激活（线性投影，残差），只重计算高成本部分（注意力）。仅付出小额FLOPs开销，省下O(L^2)内存。

Megatron-Core实现了此“选择性”激活重计算，2024年及以后多数前沿训练采用。

### 卸载（Offload）

重计算的替代方案：把激活从GPU转到CPU内存等待反向传播。需PCIe带宽；当空闲带宽大于重新计算代价时有益。混合策略常见：一些层checkpoint，另一些层卸载。

FSDP2直接支持卸载方案。卸载适用于GPU受内存瓶颈、CPU-GPU传输带宽充裕的场景。

### 重计算成本模型

每步使用每 `k` 层检查点的 FLOPs：

```text
flops_fwd_normal = L * f_layer
flops_bwd_normal = 2 * L * f_layer
flops_total_normal = 3 * L * f_layer

flops_fwd_ckpt = L * f_layer
flops_recompute = L * f_layer  # 每层段额外一次前向
flops_bwd_ckpt = 2 * L * f_layer
flops_total_ckpt = 4 * L * f_layer
overhead = 4 / 3 - 1 = 0.33 = 33%
```

选择性检查点仅重算注意力部分，不重算全层：

```text
flops_recompute_selective = L * f_attention ~= L * f_layer * 0.15
overhead_selective = (3 + 0.15) / 3 - 1 = 0.05 = 5%
```

### 内存节省模型

每层激活量：`A`。`L`层时总激活：`L * A`。

全检查点（段长1）：只保存层输入，即约 `L * 1/10 A`（标准transformer）。省约 `9 * L * A * 1/10`。

每 `k` 层设置一个检查点：保存 `L/k * A`，加上活动段中 `k-1` 层的激活。

当 `k = sqrt(L)`，内存与重计算成本都随 `sqrt(L)` 增长——是均匀成本层的最佳折中。

### 不适合检查点的情况

- 正在流水线最内层处理的层，不得不完成。
- 第1和最后1层，若计算占比极大（transformer中罕见）。
- 已用FlashAttention的注意力核——Flash本身快速重算softmax，额外检查点成本不明显。

### 实现模式

1. **函数包裹器：** 用 `torch.utils.checkpoint.checkpoint(fn, input)` 包裹一个段。PyTorch只保存输入，反向时重算其他。
2. **装饰器基：** 标记层为可检查点，训练脚本配置时决定哪些段包裹。
3. **手动重计算：** 自写反向传播，调用自定义 `recompute_forward`，传入保存的输入，重做前向。

三者功能相同。包裹器是标准用法。

### 与TP/PP/FP8的交互

- **张量并行（TP）：** 检查点输入需聚合或重新分散，需考虑通信成本。
- **流水线并行（PP）：** 一般对每个流水线阶段前向执行检查点，以便反序微批复用激活内存。
- **FP8重计算：** 重新计算时amax历史需与正向一致，防止FP8比例漂移。大多数框架会快照scale。

## 构建步骤

### 第1步：带段的简易模型

```python
import numpy as np


def linear_forward(x, w, b):
    return x @ w + b


def relu(x):
    return np.maximum(x, 0)


def layer_forward(x, w1, b1, w2, b2):
    h = relu(linear_forward(x, w1, b1))
    return linear_forward(h, w2, b2)


def model_forward(x, params):
    activations = [x]
    h = x
    for w1, b1, w2, b2 in params:
        h = layer_forward(h, w1, b1, w2, b2)
        activations.append(h)
    return h, activations
```

### 第2步：需要所有激活的朴素反向传播

```python
def model_backward(grad_output, activations, params):
    grads = [None] * len(params)
    g = grad_output
    for i in range(len(params) - 1, -1, -1):
        w1, b1, w2, b2 = params[i]
        x_in = activations[i]
        h_pre = linear_forward(x_in, w1, b1)
        h = relu(h_pre)
        gh = g @ w2.T
        gw2 = h.T @ g
        gb2 = g.sum(axis=0)
        g_pre = gh * (h_pre > 0)
        gx = g_pre @ w1.T
        gw1 = x_in.T @ g_pre
        gb1 = g_pre.sum(axis=0)
        grads[i] = (gw1, gb1, gw2, gb2)
        g = gx
    return g, grads
```

### 第3步：每k层检查点实现内存优化

```python
def model_forward_checkpointed(x, params, k=4):
    saved_inputs = [x]
    h = x
    for i, (w1, b1, w2, b2) in enumerate(params):
        h = layer_forward(h, w1, b1, w2, b2)
        if (i + 1) % k == 0:
            saved_inputs.append(h)
    return h, saved_inputs


def model_backward_checkpointed(grad_output, saved_inputs, params, k=4):
    grads = [None] * len(params)
    g = grad_output
    segments = [(j * k, min((j + 1) * k, len(params))) for j in range(len(saved_inputs))]
    for seg_idx in range(len(saved_inputs) - 1, -1, -1):
        start, end = segments[seg_idx]
        if start >= end:
            continue
        x_in = saved_inputs[seg_idx]
        _, seg_acts = model_forward(x_in, params[start:end])
        g, seg_grads = model_backward(g, seg_acts, params[start:end])
        for j, gr in enumerate(seg_grads):
            grads[start + j] = gr
    return g, grads
```

### 第4步：成本模型

```python
def checkpoint_cost(n_layers, segment_size, flops_per_layer=1.0):
    fwd = n_layers * flops_per_layer
    recompute = n_layers * flops_per_layer
    bwd = 2 * n_layers * flops_per_layer
    return {
        "fwd": fwd,
        "recompute": recompute,
        "bwd": bwd,
        "total": fwd + recompute + bwd,
        "overhead_vs_no_ckpt": (fwd + recompute + bwd) / (fwd + bwd) - 1.0,
    }


def selective_checkpoint_cost(n_layers, attention_fraction=0.15,
                              flops_per_layer=1.0):
    fwd = n_layers * flops_per_layer
    recompute = n_layers * attention_fraction * flops_per_layer
    bwd = 2 * n_layers * flops_per_layer
    return {
        "fwd": fwd,
        "recompute": recompute,
        "bwd": bwd,
        "total": fwd + recompute + bwd,
        "overhead_vs_no_ckpt": (fwd + recompute + bwd) / (fwd + bwd) - 1.0,
    }
```

### 第5步：内存估算器

```python
def activation_memory_mb(n_layers, hidden=8192, seq=8192,
                        batch=1, bytes_per_value=2):
    per_layer = 12 * batch * seq * hidden * bytes_per_value
    return n_layers * per_layer / 1e6


def memory_after_checkpoint(n_layers, segment_size, hidden=8192,
                           seq=8192, batch=1, bytes_per_value=2):
    n_seg = max(1, n_layers // segment_size)
    saved = (n_seg + segment_size) * 1 * batch * seq * hidden * bytes_per_value
    return saved / 1e6
```

### 第6步：最优段大小函数

```python
def optimal_segment(n_layers):
    return int(round(np.sqrt(n_layers)))
```

### 第7步：选择性重计算判断

```python
def should_recompute(layer_type, activation_bytes, recompute_flops_ratio):
    if layer_type == "attention" and activation_bytes > 100 * 1e6:
        return True
    if layer_type == "ffn" and activation_bytes > 500 * 1e6:
        return recompute_flops_ratio < 0.1
    return False
```

## 使用方式

- **torch.utils.checkpoint**：`from torch.utils.checkpoint import checkpoint` —— PyTorch中的标准包装器。包装函数，保存输入，反向时重计算。
- **Megatron-Core激活重计算**：支持`selective`、`full`和`block`模式。为2024年及以后最前沿训练的标准配置。
- **FSDP2卸载**：使用 `module.to_empty(device="cpu")` 与 FSDP2 的 `offload_policy` 将激活转移到CPU，替代重计算。
- **DeepSpeed ZeRO-Offload**：CPU卸载优化器状态和激活，与检查点互补。

## 交付成果

本课生成 `outputs/prompt-activation-recompute-policy.md` —— 一个提示（prompt），它接收你的模型配置（层数、隐藏维度、序列长度、批量大小）和可用GPU内存，并输出每层的重新计算策略（无 / 选择性 / 完全 / 卸载）。

## 练习

1. 验证正确性。运行 `model_forward` + `model_backward`（完全激活）与 `model_forward_checkpointed` + `model_backward_checkpointed`（分段）进行对比。参数梯度必须达到机器精度级别一致。

2. 扫描分段大小 `k` 从 1 到 `L`。绘制 FLOP 额外开销和内存使用曲线。找到曲线的拐点。

3. 实现选择性检查点（selective checkpointing）：只存储注意力模块的输入，不存储其中间激活。测量序列长度为8192且有32层模型的 FLOP 额外开销，相较于全层检查点。

4. 添加卸载（offload）。将分段输入保存到模拟的“CPU 缓冲区”（独立列表）。测量“PCIe 带宽”即字节数/时间，找到卸载与重新计算的平衡点。

5. 基准测试一个真实 PyTorch Transformer，分别用和不用 `torch.utils.checkpoint`。测量内存（通过 `torch.cuda.max_memory_allocated`）和步长时间。

## 关键术语

| 术语 | 人们说法 | 实际含义 |
|------|----------|---------|
| Gradient checkpointing（梯度检查点） | “通过重做前向节省内存” | 只存储分段输入；反向时重新计算中间激活以获得梯度所需的张量 |
| Activation recomputation（激活重新计算） | “和检查点一样” | 高性能计算（HPC）领域对同一技术的称呼 |
| Segment size (k)（分段大小） | “每个检查点包含多少层” | 其间的中间激活被丢弃并一起重计算的层数 |
| Selective checkpointing（选择性检查点） | “Korthikanti的技巧” | 只重新计算代价高的激活（注意力 softmax）；保留便宜激活 |
| Full checkpointing（完全检查点） | “基础版本” | 每个分段中重新计算所有层的中间激活 |
| Block checkpointing（块级检查点） | “粗粒度” | 检查点整个 Transformer 块；最大粒度 |
| FLOP overhead（FLOP额外开销） | “计算税” | 每步额外 FLOPs = (重计算 FLOPs) / (前向 + 反向 FLOPs)；朴素为33%，选择性为5% |
| Activation offload（激活卸载） | “转移到CPU” | 将激活在前向到反向过程中移至CPU内存；替代重新计算 |
| sqrt-L rule（√L规则） | “经典最优” | 对均匀成本层，最优检查点间隔为 sqrt(L) 层 |
| Attention-softmax volume（注意力softmax体积） | “O(L^2)问题” | L² × heads × batch的浮点数；长上下文中激活内存的主导 |

## 拓展阅读

- [Chen et al., 2016 -- "Training Deep Nets with Sublinear Memory Cost"](https://arxiv.org/abs/1604.06174) —— 首个提出梯度检查点的论文
- [Korthikanti et al., 2022 -- "Reducing Activation Recomputation in Large Transformer Models"](https://arxiv.org/abs/2205.05198) —— 选择性激活重计算及成本分析
- [Pudipeddi et al., 2020 -- "Training Large Neural Networks with Constant Memory using a New Execution Algorithm"](https://arxiv.org/abs/2002.05645) —— 通过反向模式重材料化实现的常量内存方案
- [Ren et al., 2021 -- "ZeRO-Offload: Democratizing Billion-Scale Model Training"](https://arxiv.org/abs/2101.06840) —— 大规模激活卸载技术
- [PyTorch torch.utils.checkpoint 文档](https://pytorch.org/docs/stable/checkpoint.html) —— 标准API
- [Megatron-Core 激活重新计算文档](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/features/memory_optimizations.html) —— 选择性、完全与块模式介绍
