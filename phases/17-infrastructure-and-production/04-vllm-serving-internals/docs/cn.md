# vLLM 服务内部机制：PagedAttention、连续批处理、分块预填充

> vLLM 在 2026 年的主导地位建立在三个叠加的默认设置上，而非单一技巧。PagedAttention（分页注意力）始终开启。连续批处理（Continuous batching）在解码迭代之间向活跃批次注入新请求。分块预填充（Chunked prefill）将长提示切片，使得解码 token 永不饥饿。开启这三者后，一台搭载 H100 SXM5 的 Llama 3.3 70B FP8 能在 128 并发下推动 2,200-2,400 tok/s，约为 vLLM 自身默认性能的 25% 提升，也是朴素 PyTorch 循环的 3-4 倍。本教程将从可绘制图解的层面解读调度器和注意力内核，最后给出一个玩具版连续批处理器 `code/main.py`，它以 vLLM 相同方式调度预填充和解码。

**类型：** 学习  
**语言：** Python（标准库，玩具连续批处理调度器）  
**先修知识：** 阶段 17 · 01（模型服务），阶段 11（LLM 工程）  
**时长：** ~75 分钟

## 学习目标

- 解释 PagedAttention 作为 KV 缓存分配器：块（blocks）、块表和为何生产负载下碎片率保持在 4% 以下。
- 绘制迭代级连续批处理示意图：完成序列如何从批次移出、新序列如何加入且不中断流程。
- 用一句话描述分块预填充，并指出其保护的延迟指标（提示：是 TTFT 尾部延迟，而非平均吞吐率）。
- 说出 2026 年 vLLM v0.18.0 中启用所有优化同时生效导致的问题。

## 问题描述

朴素 PyTorch 服务循环一次只处理一个请求：分词、预填充、直到 EOS 解码、返回。单个用户时无问题，但百人时就成了耐心排队。显而易见的解决方案——静态批处理，会将所有请求填充到窗口中的最长提示，解码也填充到最长预期输出，全批次受最慢序列拖累。你为未使用的填充买单，快速请求需等待慢请求。

vLLM 同时解决了三个问题。PagedAttention 阻止了经典连续内存分配导致的 KV 缓存占用 GPU 内存 60-80% 的碎片浪费。连续批处理允许请求在每次解码迭代间进出批次，使批次始终满载真实工作。分块预填充将 32k token 长提示拆成大约 512 token 的切片和解码交错，避免长提示冻结 GPU 上每个解码 token。

2026 年生产环境默认开启这三者。你需要了解每个机制的工作原理，因为所有失败模式都出现在调度器，而非模型本身。

## 概念详解

### PagedAttention 作为虚拟内存系统

单序列 KV 缓存大小是 `num_layers × 2 × num_heads × head_dim × seq_len × bytes_per_element`。以 Llama 3.3 70B 8192 token 为例，在 BF16 精度下大约 1.25 GB。如果为每个请求预留 8192 个位置，而平均请求仅使用 1500 token，则大约浪费 82% 的已分配 HBM。经典批处理就承受这种浪费。

PagedAttention 借鉴操作系统虚拟内存分配思想。KV 缓存不再为每条序列连续分配，而是固定大小块（默认 16 token）分配。每条序列维护一个块表，将逻辑 token 位置映射到物理块 ID。序列增长超过已分配块时增加一个块，结束时释放所有块归还池子。

碎片率从经典方案的 60-80% 降至 PagedAttention 下不到 4%。PagedAttention 没有开关选项，是 vLLM 唯一使用的分配器。调整参数是 `--gpu-memory-utilization`（默认 0.9），告诉 vLLM 载入权重和激活后预留多少 HBM 用于 KV 块。

### 迭代级连续批处理

旧的“动态批处理”是在一个窗口期（如 10 毫秒）等待批次满，运行预填充 + 解码直到所有序列完成。快速序列提前退出但坐等 GPU 处理慢序列。

连续批处理作用于每次解码步骤之间。用 `RUNNING` 表示当前运行序列集合。每次迭代：

1. 移除 `RUNNING` 中刚达 EOS 或 max_tokens 的序列。
2. 调度器查看等待队列，若有空闲 KV 块，接收入批新序列（预填充或续跑）。
3. 正向传播运行当前 `RUNNING` 中所有序列，逐序列产生一个新 token。

批次大小不填充到固定数。处于不同输出位置的序列共享一次融合的前向传播。在 2026 年 vLLM 中称为 `V1 scheduler`。关键不变量是调度器按解码迭代运行，不是按请求运行。

### 分块预填充保护 TTFT 尾部延迟

预填充算力密集。Llama 3.3 70B 32k token 提示预填充单次耗时约 800 ms（单 H100）。预填充运行时，批次中其它序列的解码 token 需等待。服务循环中，一条长提示的首 token 延迟（TTFT）成为数十个其他用户的 token 之间延迟（ITL）尖峰。

分块预填充将预填充拆分为固定大小块（默认 512 token），按块调度。块间调度器可以推进解码序列一个 token。你用少量的绝对预填充延迟增量（每块几毫秒）换取了明显更低的解码时抖动。混合负载下 P99 ITL 从 ~50 ms 降至 ~15 ms。

### 三个默认特性相互作用

三者相辅相成。PagedAttention 为调度器提供细粒度的 KV 资源交易。连续批处理依赖此细粒度资源，以免加入序列时引发全局重排。分块预填充是调度器在同一个 `RUNNING` 列表上的另一策略，而非独立系统。

无需掌握所有参数，只需明白调度器优化目标：在 KV 块预算内最大良品输出率（goodput），受限于分块预填切片。

### 2026 v0.18.0 的坑点

vLLM v0.18.0 中，不能同时启用 `--enable-chunked-prefill` 和草稿模型的假设解码 (`--speculative-model`)。官方文档中例外的是在 V1 调度器中的 N-gram GPU 假设解码。整开所有旗号没看发行说明的团队开机时会遇到运行时错误，不是软性能退化。如果你启用分块预填充为获取假设模型增益，就得重新考量——2026 年正确做法通常是无分块预填充的 EAGLE-3，而非不编译的草稿模型加分块预填充。

### 重要数字回顾

- Llama 3.3 70B FP8，H100 SXM5，128 并发，三者全开：2,200-2,400 tok/s。
- 同模型，vLLM 默认（无分块预填充）：约 1,800 tok/s。
- 同模型，朴素 PyTorch 前向循环：约 600 tok/s。
- PagedAttention 下生产负载 KV 碎片浪费：<4%。
- 混合负载下 P99 ITL：分块预填充约 15 ms，无分块预填充约 50 ms。

### 调度器示例伪代码

```python
while True:
    finished = [s for s in RUNNING if s.is_done()]
    for s in finished: release_blocks(s); RUNNING.remove(s)

    while WAITING and have_free_blocks_for(WAITING[0]):
        s = WAITING.pop(0)
        allocate_initial_blocks(s)
        RUNNING.append(s)

    # 调度预填充切片 + 解码批次
    batch = []
    for s in RUNNING:
        if s.in_prefill:
            batch.append(next_prefill_chunk(s))   # 如 512 token
        else:
            batch.append(decode_one_token(s))     # 1 token

    run_forward(batch)                            # 一次融合 GPU 调用
```

`code/main.py` 就是这样一个标准库 Python 循环，带假 token 计数和假前向延迟。运行它能演示分块预填充如何在长预填充期间维持解码序列活跃。

## 使用指南

`code/main.py` 模拟了可切换功能的 vLLM 风格调度器。运行它观察：

- `NAIVE` 模式：一次请求，无批处理。
- `STATIC` 模式：填充等待，经典批处理。
- `CONTINUOUS` 模式：迭代级入批与释放。
- `CONTINUOUS + CHUNKED` 模式：预填充切片与解码交织。

输出包含总体吞吐率（每虚拟秒 token 数）、TTFT 平均值及 P99 ITL。混合流量下 `CONTINUOUS + CHUNKED` 行应领先。

## 实践部署

本课生成 `outputs/skill-vllm-scheduler-reader.md`。基于服务配置（批次大小、KV 内存利用率、分块预填充大小、假设配置），输出调度器诊断，指出三个默认中的瓶颈和调优方向。

## 练习题

1. 运行 `code/main.py`。对比混合短长请求负载下 `STATIC` 和 `CONTINUOUS` 的吞吐差异。瓶颈在哪——预填充效率、解码效率还是尾部延迟？
2. 修改玩具调度器，添加 `--max-num-batched-tokens` 参数。对 H100 上运行 Llama 3.3 70B FP8 的正确值应是多少？（提示：它是 KV 块大小与空闲块数的函数，而非单纯 HBM 大小。）
3. 重读 vLLM v0.18.0 发行说明。哪些参数组合互斥？列出。
4. 计算 1,000 请求轨迹下 KV 缓存碎片浪费，平均输出 1,500 token，标准差 600 tokens，分别对比 (a) 8192 最大的连续单请求分配和 (b) 使用 16 token 块的 PagedAttention。
5. 以一段文字说明分块预填充为何仅改善 P99 ITL 而非单独提升吞吐率。实际中的吞吐率提升源于何处？

## 术语表

| 术语 | 常说 | 实际含义 |
|------|------|----------|
| PagedAttention | “KV 技巧” | KV 缓存固定大小块分配器；碎片率 <4% |
| Block table（块表） | “页表” | 每序列逻辑 token 位置到物理 KV 块的映射 |
| Continuous batching | “正确的动态批处理” | 每次解码迭代做入批/释放决策 |
| Chunked prefill | “预填拆分” | 将长预填拆分成 512 token 切片，与解码交错 |
| TTFT | “首 token 时间” | 预填 + 排队 + 网络；长提示时以预填主导 |
| ITL | “token 间延迟” | 连续解码 token 间时间；受批次大小主导 |
| Goodput | “达 SLO 的吞吐” | 每秒生成 token，所有请求满足 TTFT 和 ITL 目标 |
| V1 scheduler | “新调度器” | vLLM 2026 版调度器；N-gram 规格解码支持分块预填充 |
| `--gpu-memory-utilization` | “内存旋钮” | 载入权重激活后预留给 KV 块的 HBM 占比 |

## 延伸阅读

- [vLLM 文档 —— 假设解码（Speculative Decoding）](https://docs.vllm.ai/en/latest/features/spec_decode/) —— 官方关于分块预填充和假设解码兼容性的资料。  
- [vLLM 发行说明（NVIDIA）](https://docs.nvidia.com/deeplearning/frameworks/vllm-release-notes/index.html) —— 2026 版本发布节奏与版本特性。  
- [vLLM 博客 —— PagedAttention](https://blog.vllm.ai/2023/06/20/vllm.html) —— 最初的阐述，仍然指导对分配器的理解。  
- [PagedAttention 论文 (arXiv:2309.06180)](https://arxiv.org/abs/2309.06180) —— 碎片率分析与调度器设计。  
- [Aleksa Gordic —— 解读 vLLM](https://www.aleksagordic.com/blog/vllm) —— V1 调度器详解及火焰图分析。
