# Kubernetes 上的 GPU 自动扩缩容 — Karpenter、KAI Scheduler、群组调度（Gang Scheduling）

> 三层架构，而非单层。Karpenter 动态提供节点（耗时约一分钟，比 Cluster Autoscaler 快 40%）。KAI Scheduler 处理群组调度、拓扑感知和分层队列 —— 它避免了 7/8 部分分配陷阱，即七个节点等待并浪费资源，仅因缺少一个 GPU。应用层自动扩缩容器（NVIDIA Dynamo Planner、llm-d Workload Variant Autoscaler）基于特定推理信号扩缩容 —— 队列深度、KV 缓存利用率 —— 而不是 CPU/DCGM 占空比。经典的 HPA 陷阱在于 `DCGM_FI_DEV_GPU_UTIL` 是占空比指标：100% 利用率可能是 10 个请求，也可能是 100 个。vLLM 预分配 KV 缓存内存，因此内存永远不会触发缩容。本课将教你如何组合这三层，避免默认的 Karpenter `WhenEmptyOrUnderutilized` 策略，该策略会在推理中途终止运行中的 GPU 作业。

**类型：** 学习  
**语言：** Python（标准库、简单的队列深度自动扩缩容模拟器）  
**前提知识：** 阶段 17·02（推理平台经济学）、阶段 17·04（vLLM 服务内部原理）  
**时长：** 约 75 分钟

## 学习目标

- 绘制三层自动扩缩容架构图（节点提供、群组调度、应用层），并标明每层使用的工具。
- 解释为何 `DCGM_FI_DEV_GPU_UTIL` 是 vLLM 错误的 HPA 信号，并列举两个替代信号（队列深度、KV 缓存利用率）。
- 描述群组调度及 KAI Scheduler 防止的部分分配失败模式（7个 GPU 空闲中的 8 个）。
- 说出会终止运行中的 GPU 作业的 Karpenter 合并策略（`WhenEmptyOrUnderutilized`），并列出 2026 年推荐的安全替代方案。

## 问题背景

你的团队在 Kubernetes 上交付了一个 LLM 服务。HPA 使用 `DCGM_FI_DEV_GPU_UTIL` 作为信号。服务在工作时间达到 100% 利用率。HPA 从未扩展——它已经认为满负载了。你手动添加了副本，TTFT（首字响应时间）降低，但 HPA 依然没扩容。信号在误导你。

另外，你使用 Cluster Autoscaler 管理节点。凌晨 2 点时，来了一个 100 万令牌的请求；集群花费 3 分钟才创建节点，导致请求超时。

再者，你部署了一个需要 8 个 GPU、跨 2 个节点的 70B 模型。集群有 7 个 GPU 空闲，1 个分散在 3 个节点。Cluster Autoscaler 为缺少的 1 个 GPU 提供节点。七个节点等待 4 分钟，浪费资金，而 Kubernetes 在等待最后一张 GPU 上线。

三层架构，三种不同失败模式。2026 年的 GPU 感知自动扩缩容不是“开启 HPA”，而是组合节点提供、群组调度与应用层信号的自动扩缩容。

## 核心理念

### 第一层 — 节点提供（Karpenter）

Karpenter 监控待处理的 Pod 并在约 45-60 秒内提供节点（Cluster Autoscaler 创建 GPU 节点通常需要 90-120 秒）。它根据 `NodePool` 约束动态选择实例类型 —— 如果 pod 需要 8 个 H100，而集群没有匹配的节点，Karpenter 会直接开新节点，而不会扩展现有节点组。

**合并陷阱**：Karpenter 默认的 `consolidationPolicy: WhenEmptyOrUnderutilized` 对 GPU 池很危险。它会终止正在运行的 GPU 节点，将 Pod 迁移至更便宜且合适大小的实例。对于推理工作负载，这意味着驱逐正在运行的请求，并在新节点重新加载 70B 模型。损失是数分钟的容量及请求失败。

GPU 池的安全配置：

```yaml
disruption:
  consolidationPolicy: WhenEmpty
  consolidateAfter: 1h
```

允许 Karpenter 在节点真正空闲 1 小时后合并，但永远不会驱逐正在运行的作业。

### 第二层 — 群组调度（KAI Scheduler）

KAI Scheduler（前身项目名“Karp”）负责默认 kube-scheduler 不能做的工作：

**群组调度**——要么全体调度成功，要么全部推迟。一个分布式推理 Pod 需要 8 个 GPU，要么这 8 个 GPU 全部同时启动，要么全部不启动。否则会出现部分分配陷阱：7 个 Pod 启动，永久等待且浪费资金。

**拓扑感知**——知道哪些 GPU 共享 NVLink，哪些在同一机架，同一机架间有哪些 InfiniBand 连接。根据这些放置 Pod。DeepSeek-V3 67B 的张量并行任务必须保持在同一 NVLink 域内，KAI Scheduler 会尊重这一点。

**分层队列**——多个团队同时竞争同一 GPU 池，带优先级和配额。团队 A 的生产紧急任务只有在优先级规则允许时，才会被团队 B 的训练任务抢占。

KAI 作为 kube-scheduler 的辅助调度器部署，需要给工作负载添加注解以使用。Ray 和 vLLM 的生产栈均已集成。

### 第三层 — 应用层信号

**HPA 陷阱**：`DCGM_FI_DEV_GPU_UTIL` 是占空比指标——测量 GPU 在采样时间内是否工作。100% 利用率可以是 10 个并发请求，也可以是 100 个，GPU 都处于繁忙。基于占空比扩缩容是盲目的。

更糟的是，vLLM 等引擎预分配 KV 缓存内存（最多为 `--gpu-memory-utilization` 设置值）。内存使用即使只有一个请求时，也近 90%。基于内存的 HPA 永远不会缩容。

**2026 年替代信号：**

- 队列深度（等待预加载请求数）。
- KV 缓存利用率（为活动序列分配的缓存块比例）。
- 每副本 P99 TTFT（你的 SLA 指标）。
- 有效吞吐率（满足所有 SLO 的请求数/秒）。

NVIDIA Dynamo Planner 和 llm-d Workload Variant Autoscaler 使用这些信号进行副本扩缩容。它们完全替代 LLM 服务的 HPA。

### 何时用哪层

| 扩缩容决策           | 工具                         |
|---------------------|------------------------------|
| 增减节点             | Karpenter                    |
| 调度多 GPU 作业      | KAI Scheduler                |
| 增减副本             | Dynamo Planner / llm-d WVA（或基于队列深度的自定义 HPA） |
| 选择 GPU 类型        | Karpenter NodePool           |
| 抢占低优先级作业     | KAI Scheduler 队列           |

### 分离的预填充/解码令情况更复杂

如果运行分离的预填充/解码（阶段 17·17），你有两类 Pod，触发扩缩容信号不同：预填充 Pod 基于队列深度扩容，解码 Pod 基于 KV 缓存压力扩容。llm-d 以不同 `Services` 暴露这些 Pod，分别使用不同的 HPA。不要试图用一个 HPA 同时驱动两者。

### 冷启动也非常重要

冷启动缓解（阶段 17·10）是节点提供时延出现在用户可感知层面的关键。Karpenter 的 45-60 秒启动加 20GB 模型加载和引擎初始化，导致零请求启动时耗时 2-5 分钟。保持一个热节点池（`min_workers=1`）用于关键路径，或在应用层采用 Modal 风格的检查点技术。

### 应记住的数字

- Karpenter 节点提供时间约 45-60 秒，Cluster Autoscaler 为 90-120 秒（GPU 节点）。
- KAI Scheduler 防止部分分配浪费 —— 7/8 陷阱。
- `DCGM_FI_DEV_GPU_UTIL` 作为 HPA 信号：错误，使用队列深度或 KV 利用率替代。
- Karpenter `WhenEmptyOrUnderutilized`：会终止正在运行的 GPU 作业。推理使用建议 `WhenEmpty + consolidateAfter: 1h`。

## 试用

`code/main.py` 模拟一个三层自动扩缩容器，处理突发 GPU 负载。比较简单的基于占空比 HPA、基于队列深度的 HPA，以及 KAI-群组调度扩缩容。报告未满足请求数量、GPU 空闲时间和综合评分。

## 实践

本课生成 `outputs/skill-gpu-autoscaler-plan.md`。结合集群拓扑、负载形态和 SLO，设计三层自动扩缩容方案。

## 练习题

1. 运行 `code/main.py`。在突发负载下，基于占空比的简单 HPA 会丢失多少请求，基于队列深度的 HPA 能捕获多少？差异来源于哪里？
2. 设计一个 Karpenter NodePool，用于服务 Llama 3.3 70B FP8，运行在 H100 SXM5 集群。指定 `capacity-type`、`disruption.consolidationPolicy`、`consolidateAfter`，以及让非 GPU 工作负载不能调度到这些节点的污点。
3. 你团队报告部署一直处于 Pending 状态，原因是“有 GPU 可用但 Pod 无法调度”。请诊断——是 Karpenter、kube-scheduler 还是 KAI Scheduler 导致？有哪些指标能确认？
4. 选一个扩缩容信号用于分离的预填充 Pod，选另一个用于解码 Pod，并说明理由。
5. 计算在一个 24x7 生产服务中，`WhenEmptyOrUnderutilized` 合并陷阱的成本，该服务平均每天有 60 次请求丢失事件，P99 TTFT 超过 10 秒。

## 关键术语

| 术语                 | 流行说法             | 实际含义                         |
|----------------------|----------------------|----------------------------------|
| Karpenter            | “节点提供器”         | Kubernetes 节点自动扩缩容器；亚分钟级节点创建 |
| Cluster Autoscaler   | “老版扩缩容器”       | Kubernetes 节点自动扩缩容前代；速度较慢，基于节点组 |
| KAI Scheduler        | “GPU 调度器”         | 次级调度器，支持群组调度、拓扑感知、队列管理 |
| 群组调度（Gang scheduling） | “全有或全无”         | 原子性调度 N 个 Pod，要么都调度，要么都推迟 |
| 拓扑感知（Topology awareness） | “机架感知”          | 基于 NVLink / InfiniBand / 机架位置调度 Pod |
| `DCGM_FI_DEV_GPU_UTIL` | “GPU 利用率”         | 占空比指标；不是 LLM 扩缩容信号                |
| 队列深度（Queue depth）   | “等待请求数”         | 预填充绑定的正确 HPA 信号                       |
| KV 缓存利用率（KV cache utilization） | “内存压力”          | 解码绑定的正确 HPA 信号                         |
| 合并（Consolidation）   | “Karpenter 合并”    | 终止节点以迁移到更便宜实例                        |
| `WhenEmpty + 1h`       | “安全合并”           | 不驱逐运行中 GPU 作业的策略                      |

## 深入阅读

- [KAI Scheduler GitHub](https://github.com/kai-scheduler/KAI-Scheduler) — 设计文档和配置示例。  
- [Karpenter 断电控制](https://karpenter.sh/docs/concepts/disruption/) — 合并策略语义及 GPU 安全默认配置。  
- [NVIDIA — Kubernetes 上的分离式 LLM 推理](https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/) — Dynamo Planner 扩缩容信号。  
- [Ray 文档 — 针对 RayClusters 的 KAI Scheduler](https://docs.ray.io/en/latest/cluster/kubernetes/k8s-ecosystem/kai-scheduler.html) — Ray 集成模式。  
- [AWS EKS 计算与自动扩缩容最佳实践](https://docs.aws.amazon.com/eks/latest/best-practices/aiml-compute.html) — 特定于托管 Kubernetes 的指南。  
- [llm-d GitHub](https://github.com/llm-d/llm-d) — Workload Variant Autoscaler 设计。
