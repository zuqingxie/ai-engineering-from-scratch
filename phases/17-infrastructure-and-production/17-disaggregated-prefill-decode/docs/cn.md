# 解耦预填充/解码 — NVIDIA Dynamo 和 llm-d

> 预填充（Prefill）是计算负载密集型；解码（Decode）是内存带宽密集型。在同一 GPU 上同时运行两者会浪费其中一种资源。解耦将它们拆分到独立的资源池，并通过 NIXL（RDMA/InfiniBand 或 TCP 回退）在池间传输 KV 缓存。NVIDIA Dynamo（GTC 2025 发布，1.0 版本 GA）位于 vLLM/SGLang/TRT-LLM 之上——其 Planner Profiler + SLA Planner 可自动匹配预填充与解码的比率以满足 SLO。NVIDIA 发布了大致类似的吞吐量提升 —— developer.nvidia.com（2025-06）显示 DeepSeek-R1 MoE 在 GB200 NVL72 + Dynamo 中等延迟场景提升约 6 倍吞吐量，Dynamo 产品页（developer.nvidia.com，未标日期）表示 GB300 NVL72 + Dynamo 相较 Hopper 可达 50 倍 MoE 吞吐。社区“30 倍”数据是基于 Blackwell + Dynamo + DeepSeek-R1 全栈的汇总，我们未找到单一来源确切说明为 30 倍，应视为趋势性数据。llm-d（Red Hat + AWS）是 Kubernetes 原生：预填充 / 解码 / 路由器作为独立服务，带有针对角色的 HPA。llm-d 0.5 版本增加了分层 KV 卸载、缓存感知的 LoRA 路由、UCCL 网络和规模归零（scale-to-zero）。经济学方面：多个客户披露的内部汇总显示，在 $200 万量级推理开销下，从同机部署切换到基于 Dynamo 的解耦部署可节省 30–40%（即每年 $600-800K）；具体 $200 万 → $600-800K 是内部复合数据，非单一公开案例——仅供数量级参考。短提示（<512 标记，短输出）不值得承担传输开销。

**类型：** 学习  
**语言：** Python（标准库，玩具解耦与同机模拟器）  
**先决条件：** 第 17 阶段 · 04（vLLM 服务内部），第 17 阶段 · 08（推理指标）  
**时长：** 约 75 分钟

## 学习目标

- 解释为何预填充和解码需要不同的 GPU 配额，并量化同机部署下资源浪费。
- 绘制解耦架构图：预填充池、解码池、通过 NIXL 转移 KV 缓存、路由器。
- 指出何种情况下解耦不划算（短提示、短输出）。
- 分辨 NVIDIA Dynamo（跨堆栈）与 llm-d（Kubernetes 原生），并匹配不同的运维场景。

## 问题

你使用 8 个 H100 运行 Llama 3.3 70B。在混合负载（长提示 + 短输出）下，解码时 GPU 闲置，因为大部分计算在预填充阶段。当负载反转（短提示 + 长输出）时情况相反。同机预填充和解码意味着资源超配。

预算影响：20-40% 的 GPU 时间浪费在错误资源上。你买了用于计算密集型的 H100 来跑内存瓶颈的解码，或者买了高带宽 HBM 来跑计算密集型的预填充，都是昂贵的浪费。

解耦将预填充和解码拆到独立池，分别针对瓶颈尺寸配置。KV 缓存通过高速互连从预填充池传输到解码池。

## 概念

### 瓶颈为何不同

**预填充** — 在完整输入提示上执行 Transformer 的一次前向传播。以矩阵乘法为主，计算瓶颈。H100 FP8 计算能力约 2000 TFLOPS 有效吞吐量。批处理效率高——一次前向处理多标记。

**解码** — 一次生成一个标记，每次迭代均读取全部权重。受内存带宽限制。HBM3 约 3 TB/s。仅高并发时批处理效率好——权重读取成本可摊销。

同机部署：需买适合两者的 GPU。H100 兼顾计算和内存，但成本相同。规模化时，理想是预填充池用 H100（计算密集型），解码池用 H200（内存密集型），或采用激进量化。

### 架构示意

```text
            ┌──────────────┐
  请求 → │    路由器    │ ───────────────────────┐
            └──────┬───────┘                        │
                   │                                │
                   ▼ （仅提示）                      │
            ┌──────────────┐    KV 缓存    ┌───────▼──────┐
            │ 预填充池     │ ─── NIXL ────► │ 解码池      │
            │ （计算）     │                │ （内存）    │
            └──────────────┘                └──────┬───────┘
                                                   │ 标记
                                                   ▼
                                                 客户端
```

NIXL 是 NVIDIA 的节点间传输协议。优先使用 RDMA/InfiniBand，不可用时回退到 TCP。传输时延真实存在——70B FP8 4K 标记提示的 KV 缓存传输通常为 20-80 毫秒。这就是短提示不划算解耦的原因：传输成本超过收益。

### Dynamo 与 llm-d 区别

**NVIDIA Dynamo**（GTC 2025 发布，1.0 GA）：  
- 位于 vLLM、SGLang、TRT-LLM 之上，作为协调器。  
- Planner Profiler 测量负载，SLA Planner 自动调整预填充与解码比率。  
- 核心用 Rust 实现，支持 Python 扩展。  
- 吞吐量提升：NVIDIA 报告 DeepSeek-R1 MoE 在 GB200 NVL72 + Dynamo 中等延迟提升约 6 倍（developer.nvidia.com，2025-06）；社区关于 Blackwell + Dynamo + DeepSeek-R1 全栈“最高达 30 倍”提升无单一权威来源，应视作趋势数据。  
- GB300 NVL72 + Dynamo 可达 Hopper 的 50 倍 MoE 吞吐（developer.nvidia.com，未标日期）  

**llm-d**（Red Hat + AWS，Kubernetes 原生）：  
- 预填充 / 解码 / 路由器均为独立 Kubernetes 服务。  
- 每角色带有 HPA，根据队列深度（预填充）/ KV 利用率（解码）调节。  
- `topologyConstraint packDomain: rack` 可在同机架打包预填充+解码群组以实现高速 KV 传输。  
- llm-d 0.5（2026）：支持分层 KV 卸载、缓存感知 LoRA 路由、UCCL 网络和规模归零。  

若需托管式跨栈协调器，选择 Dynamo；若希望 Kubernetes 原生并扎根 CNCF 生态，选 llm-d。

### 经济效益

内部综合数据（非单一公开案例，仅数量级参考）：

- $200 万/年的推理支出，采用同机部署。  
- 切换至 Dynamo 解耦部署。  
- 请求量和 P99 延迟 SLO 无变。  
- 报告节省：$60-80 万/年（节省 30–40%）。  
- 无新增硬件。  

此数据由多客户披露合成，非可引用单例。公开最接近的数据点是 Baseten 的 Dynamo KV 路由实现的 2 倍更快 TTFT / 61% 更高吞吐量（baseten.co，2025-10），以及 VAST + CoreWeave 预测的 40–60% KV 命中率下带来 60–130% 令牌成本优势（vastdata.com，2025-12）。节省主要源于合理配置池容量，预填重载（带 8K+ 前缀的 RAG）受益更大。

### 何时不该解耦

- 提示 < 512 标记，输出 < 200 标记：传输成本超过收益。  
- 小型集群（< 4 GPUs）：资源池不够多样化。  
- 团队无法管理两套 GPU 池及按角色扩缩：Dynamo 有帮助但不简单。  
- 无 RDMA 互联：TCP 传输成本更高。  

### 路由器与第 17 阶段 · 11 集成

解耦路由器具备 KV 缓存感知能力（第 17 阶段 · 11）。请求会落在持有其前缀缓存的解码池；若无缓存命中，则流程走预填充 → 解码。命中率与解耦互相影响——缓存感知路由器决定是否需要新预填充。

### Blackwell 上 MoE 是真实数据来源

GB300 NVL72 + Dynamo 显示对比 Hopper 可实现 50 倍 MoE 吞吐。MoE 专家路由在预填充阶段计算密集，而在解码阶段内存密集（专家缓存），解耦双赢。2026 年前沿模型服务主要为 MoE（DeepSeek-V3、未来 GPT-5 变体）。

### 你应该记住的数字

基准数据波动较大 — NVIDIA 和推理栈每季度更新结果。引用前请复查。

- DeepSeek-R1 在 GB200 NVL72 + Dynamo 中等延迟下约 6 倍吞吐（developer.nvidia.com，2025-06）；社区“最高 30 倍”声称为趋势汇总，缺少单一权威。  
- GB300 NVL72 + Dynamo：对比 Hopper 达 50 倍 MoE 吞吐（developer.nvidia.com，未标日期）。  
- 节省基准（内部综合，非单例案例）：$600-800K/年，基于 $200 万年支出及同 SLO。  
- 解耦阈值：提示 >512 标记，输出 >200 标记。  
- 通过 NIXL 传输：70B FP8 4K 提示 KV 缓存传输延迟 20-80 毫秒。

## 使用方法

`code/main.py` 模拟同机与解耦部署。报告吞吐量、每请求成本及提示长度临界点。

## 交付物

本课生成 `outputs/skill-disaggregation-decider.md`。给定负载和集群，判断是否解耦。

## 练习

1. 运行 `code/main.py`。提示长度多少时解耦优于同机？  
2. 设计一个 P99 8K 前缀、输出 300 的 RAG 服务预填充池和解码池。  
3. Dynamo 与 llm-d：为纯 Kubernetes 团队无 Python 运行时偏好选择一个。  
4. 计算 KV 传输开销：70B FP8 4K 预填充约 500 MB KV。RDMA 100 GB/s 则传输时间约 5 毫秒，TCP 10 GB/s 则 50 毫秒。哪个对你的 SLO 更关键？  
5. MoE 专家路由改变 KV 访问模式。MoE 启动不同专家时，解耦如何表现？

## 关键词汇

| 术语 | 常见说法 | 实际含义 |
|------|---------|----------|
| 解耦部署（Disaggregated serving） | “拆分预填充/解码” | 为各阶段单独分配 GPU 池 |
| NIXL | “NVIDIA 传输” | Dynamo 的节点间 KV 转移（RDMA/TCP） |
| NVIDIA Dynamo | “协调器” | vLLM/SGLang/TRT-LLM 以上的调度器 |
| llm-d | “Kubernetes 原生” | Red Hat + AWS 的 Kubernetes 解耦栈 |
| Planner Profiler | “Dynamo 自动配置” | 测量负载，配置池比率 |
| SLA Planner | “Dynamo 策略” | 自动匹配预填充与解码以满足 SLO |
| `packDomain: rack` | “llm-d 拓扑” | 在同机架打包预填充+解码以加速 KV |
| UCCL | “统一集合通信” | llm-d 0.5 的网络层，用于规模归零 |
| MoE 专家路由 | “每标记专家” | DeepSeek-V3 模式；解耦助力 |

## 进一步阅读

- [NVIDIA — 介绍 Dynamo](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/)
- [NVIDIA — Kubernetes 上的解耦大语言模型推理](https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/)
- [TensorRT-LLM 解耦服务博客](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog5_Disaggregated_Serving_in_TensorRT-LLM.html)
- [llm-d GitHub](https://github.com/llm-d/llm-d)
- [llm-d 0.5 发行说明](https://github.com/llm-d/llm-d/releases)
