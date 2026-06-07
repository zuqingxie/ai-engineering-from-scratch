# KV Cache，Flash Attention 与推理优化

> 训练是并行且受 FLOP（浮点运算数）限制的。推理是串行且受内存限制的。瓶颈不同，优化技巧也不同。

**类型：** 构建  
**语言：** Python  
**先决条件：** 第7阶段 · 02（Self-Attention（自注意力））、第7阶段 · 05（Full Transformer（完整 Transformer））、第7阶段 · 07（GPT）  
**时间：** 约75分钟

## 问题

一个简单的自回归解码器生成 `N` 个 token 需要 `O(N²)` 工作量：每一步都重新计算对整个前缀的注意力。对于 4K-token 的回复，就是 1600 万次注意力操作，其中大多数是冗余的。每个前缀 token 的隐藏状态一旦计算完成就是确定性的——你只需要用新 token 的 query 去与之前缓存的 keys 和 values 计算即可。

此外，注意力本身传输大量数据。标准注意力物化一个 `N×N` 的分数矩阵，`N×d` 的 softmax 输出，`N×d` 的最终输出——对 HBM（高带宽内存）的读写过多。当 `N≥2K` 时，注意力变成受内存限制而非 FLOP 限制。经典注意力内核在现代 GPU 上效率低，利用率只有 4–10 倍。

Dao 等人提出的两种优化，将推理速度从“慢”推到了“快”的前沿：

1. **KV cache（缓存键值）。** 存储每个前缀 token 的 K 和 V 向量。每个新 token 的注意力只需用 query 对缓存的 keys 做一次计算。推理从 `O(N²)` 降为每步 `O(N)`。
2. **Flash Attention（闪电注意力）。** 采用切片（tiling）方式计算注意力，保证整个 `N×N` 矩阵不进入 HBM。全部 softmax 和矩阵乘法在 SRAM（静态随机存取内存）中完成。在 A100 上提升 2–4 倍，H100 上使用 FP8 量化可提升 5–10 倍。

到 2026 年，这两者都会成为标准。每个生产推理栈（如 vLLM、TensorRT-LLM、SGLang、llama.cpp）都默认支持它们。每个前沿模型都会启用 Flash Attention。

## 概念

![KV cache growth and Flash Attention tiling](../assets/kv-cache-flash-attn.svg)

### KV cache 数学

每层解码器，每个 token，每个头：

```text
bytes_per_token_per_layer = 2 * d_head * dtype_size
                          ^
                          包含 K 和 V
```

对于 7B 模型，32 层，32 头，d_head=128，fp16：

```text
每层每 token = 2 * 128 * 2 = 512 bytes
每个 token（32 层）= 16 KB
32K 上下文 = 512 MB
```

对于 Llama 3 70B（80 层，d_head=128，使用 8 个 KV 头的 GQA）：

```text
每层每 token = 2 * 8 * 128 * 2 = 4096 bytes（4 KB）
32K 上下文 = 10.4 GB
```

这 10GB 就是 Llama 3 70B 在 128K 上下文下，批大小为 1 时，KV cache 需要 40 GB A100 大部分显存的原因。

**GQA 是 KV cache 的优势。** 多头注意力（MHA）64 头可能需要 32 GB，混合头注意力（MLA）则进一步压缩。

### Flash Attention——切片技巧

标准注意力：

```text
S = Q @ K^T          （HBM 读取，N×N，HBM 写入）
P = softmax(S)       （HBM 读取，HBM 写入）
O = P @ V            （HBM 读取，HBM 写入）
```

共需要三次 HBM 往返。在 H100 上，HBM 带宽约 3 TB/s，SRAM 带宽约 30 TB/s。每次 HBM 访问相比在片上处理都会慢 10 倍。

Flash Attention：

```text
对于每块 Q（切片大小约 128 × 128）：
    加载 Q_tile 到 SRAM
    对每块 K、V：
        加载 K_tile、V_tile 到 SRAM
        计算 S_tile = Q_tile @ K_tile^T     （SRAM）
        进行 running softmax 聚合          （SRAM）
        累积到 O_tile                       （SRAM）
    将 O_tile 写回 HBM
```

每个切片一次 HBM 访问，整体内存消耗从 `O(N²)` 降到 `O(N)`。反向传播时重新计算一部分前向计算值，节省存储——再次节省显存。

**数值技巧。** running softmax 维护跨切片的 `(max, sum)`，保证最终归一化准确。不是近似——Flash Attention 产生的输出与标准注意力位级相同（除了 fp16 计算的非结合性误差）。

**版本演进：**

| 版本    | 年份 | 关键改进                     | 参考硬件上的加速     |
|---------|-------|----------------------------|--------------------|
| Flash 1 | 2022  | 基于 SRAM 的切片内核           | 在 A100 上 2×        |
| Flash 2 | 2023  | 改进并行性，先因果顺序排序      | 在 A100 上 3×        |
| Flash 3 | 2024  | Hopper 异步计算，FP8 量化      | 在 H100 (~740 TFLOPs FP16) 上 1.5–2× |
| Flash 4 | 2026  | Blackwell 五阶段流水线，软件 exp2 优化 | 优先用于推理（初步只支持前向）|

Flash 4 上线时仅支持前向传递，训练仍用 Flash 3。Flash 4 对 GQA 和可变长度支持预计 2026 年中完成。

### 猜测性解码——另一个延迟瓶颈突破

廉价模型预测 N 个 token。大型模型并行验证这 N 个 token。如果验证通过了其中 k 个，那么你只用一次大模型前向计算就得到了 k 个预测。代码和文本任务中，典型 k=3–5。

2026 年默认方案：
- **EAGLE 2 / Medusa。** 集成的草稿头共享验证器隐藏状态，无精度损失下加速 2–3 倍。
- **草稿模型猜测性解码。** 在消费级硬件上加速 2–4 倍。
- **预见解码。** 使用 Jacobi 迭代，无需草稿模型，小众但免费。

### 持续批处理

经典批处理推理：等最慢序列执行完，才开启下一批，短序列提前结束时浪费 GPU 时间。

持续批处理（首发于 Orca，现在在 vLLM、TensorRT-LLM、SGLang）：老请求结束后立刻替换进新请求。对典型聊天工作负载提升 5–10 倍吞吐量。

### PagedAttention——KV Cache 作为虚拟内存

vLLM 的核心特性。KV cache 按 16-token 块分配，页表映射逻辑位置到物理块。支持多样本并行共享 KV（beam search，采样并行）、热交换前缀实现提示缓存，以及减少内存碎片。吞吐提升 4 倍，相比简单连续分配。

## 构建它

参考 `code/main.py`。实现内容：

1. 一个简单的 `O(N²)` 递增解码器。
2. 一个 `O(N)` 的 KV cache 解码器。
3. 一个切片 softmax，模拟 Flash Attention 的 running-max 算法。

### 第 1 步：KV cache

```python
class KVCache:
    def __init__(self, n_layers, n_heads, d_head):
        self.K = [[[] for _ in range(n_heads)] for _ in range(n_layers)]
        self.V = [[[] for _ in range(n_heads)] for _ in range(n_layers)]

    def append(self, layer, head, k, v):
        self.K[layer][head].append(k)
        self.V[layer][head].append(v)

    def read(self, layer, head):
        return self.K[layer][head], self.V[layer][head]
```

简单：保存每个 token 的 K、V 向量，按层和头存储的列表持续增长。

### 第 2 步：切片 softmax

```python
def tiled_softmax_dot(q, K, V, tile=4):
    """Flash-attention 风格的 softmax(qK^T)V，带 running max/sum。"""
    m = float("-inf")
    s = 0.0
    out = [0.0] * len(V[0])
    for start in range(0, len(K), tile):
        k_block = K[start:start + tile]
        v_block = V[start:start + tile]
        scores = [sum(qi * ki for qi, ki in zip(q, k)) for k in k_block]
        new_m = max(m, *scores)
        exp_old = math.exp(m - new_m) if m != float("-inf") else 0.0
        exp_new = [math.exp(sc - new_m) for sc in scores]
        s = s * exp_old + sum(exp_new)
        for j in range(len(out)):
            out[j] = out[j] * exp_old + sum(e * v[j] for e, v in zip(exp_new, v_block))
        m = new_m
    return [o / s for o in out]
```

输出位级相同于一次性计算的 `softmax(qK) V`，但运行时的活动集只需 `tile × d_head` 大小，而非完整 `N × d_head`。

### 第 3 步：比较简单与缓存解码器，生成 100 个 token

统计注意力计算次数。简单解码器为 `O(N²)` = 5050，缓存解码器为 `O(N)` = 100。代码打印两者。

## 使用它

```python
# HuggingFace transformers 自动启用 decoder-only generate() 的 KV cache
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B",
    attn_implementation="flash_attention_2",  # Hopper 需用 FA3
    torch_dtype="bfloat16",
)
# generate() 自动使用 KV cache
```

vLLM 生产部署示例：

```bash
pip install vllm
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --enable-prefix-caching \
    --kv-cache-dtype fp8
```

请求间前缀缓存是 2026 年大胜利——相同系统提示、少量示例或长文本上下文调用间复用 KV。代理工作负载中重复调用工具提示时，缓存提升吞吐通常达 5 倍。

## 部署它

参见 `outputs/skill-inference-optimizer.md`。该技能为新推理部署选择注意力实现、KV 缓存策略、量化和猜测性解码方案。

## 练习

1. **简单。** 运行 `code/main.py`。确认简单和缓存解码器输出相同，注意操作次数差异。
2. **中等。** 实现前缀缓存：给定提示 P 和多个完成样本，单次前向传递填充 KV cache，然后分支生成。对比每次重新编码 P 的加速效果。
3. **困难。** 实现一个简易 PagedAttention：KV cache 固定分配 16-token 块，带空闲链表。序列结束时归还数据块。模拟 1000 轮不同长度聊天，比较内存碎片和连续分配。

## 术语

| 术语          | 常说含义               | 实际含义                                               |
|--------------|-----------------------|------------------------------------------------------|
| KV cache     | “让解码加速的技巧”      | 保存每个前缀 token 的 K、V；新查询直接用缓存键值计算。       |
| HBM          | “GPU 主内存”          | 高带宽内存，H100 80 GB，B200 192 GB，带宽约 3 TB/s。           |
| SRAM         | “片上内存”             | 每个 SM 的高速缓存，H100 上约 256 KB / SM，带宽约 30 TB/s。     |
| Flash Attention | “切片注意力内核”       | 不将 N×N 矩阵写入 HBM，优化注意力计算。                      |
| Continuous batching | “无限等待批处理”      | 旧序列完结马上替换进新序列，无需清空批次，提升利用率和吞吐。          |
| PagedAttention | “vLLM 旗舰特色”        | KV cache 使用固定大小块和页表，消除内存碎片。                  |
| Prefix caching | “复用长提示”           | 不同请求共享同样前缀的 KV，显著降低成本，代理应用特别显著。          |
| Speculative decoding | “草稿+验证策略”      | 便宜模型先预测，主模型批量验证多个 token。                    |

## 深入阅读

- [Dao 等（2022）。FlashAttention: 快速且内存高效的精确注意力，关注 IO 性能](https://arxiv.org/abs/2205.14135) — Flash 1。  
- [Dao（2023）。FlashAttention-2: 更好的并行和工作划分提高速度](https://arxiv.org/abs/2307.08691) — Flash 2。  
- [Shah 等（2024）。FlashAttention-3: 异步和低精度实现快速精准注意力](https://arxiv.org/abs/2407.08608) — Flash 3。  
- [FlashAttention-4 发布日志（Dao-AILab，2026）](https://github.com/Dao-AILab/flash-attention) — Blackwell 五阶段流水线与软件 exp2 优化；详见仓库 README 中本课程提及的前向仅实现说明。  
- [Kwon 等（2023）。大语言模型服务的高效内存管理：PagedAttention 提案](https://arxiv.org/abs/2309.06180) — vLLM 论文。  
- [Leviathan 等（2023）。通过猜测性解码实现变压器快速推理](https://arxiv.org/abs/2211.17192) — 猜测性解码。  
- [Li 等（2024）。EAGLE: 猜测采样需要重新思考特征不确定性](https://arxiv.org/abs/2401.15077) — EAGLE-1/2 论文，介绍集成草稿模型方法。  
- [Cai 等（2024）。Medusa: 多解码头简单的 LLM 推理加速框架](https://arxiv.org/abs/2401.10774) — 与 EAGLE 一起引用的 Medusa 方法。  
- [vLLM 文档 — PagedAttention](https://docs.vllm.ai/en/latest/design/kernel/paged_attention.html) — 关于 16-token 块及页表设计的权威深度解读。
