# 带 LMCache KV 卸载的 vLLM 生产栈

> vLLM 的生产栈是参考 Kubernetes 部署 —— 路由器、引擎和可观测性组件联合工作。LMCache 是 KV 卸载层，将 KV 缓存从 GPU 内存中提取出来，并跨查询和引擎复用（先在 CPU DRAM，然后是磁盘/Ceph）。vLLM 0.11.0 的 KV 卸载连接器（2026 年 1 月）通过连接器 API（v0.9.0+）使卸载过程异步且可插拔。卸载延迟对用户不可见。即使没有共享前缀，LMCache 也很有价值 —— 当 GPU KV 插槽耗尽时，抢占的请求可以从 CPU 端恢复，而无需重新计算预填充。16x H100（80GB HBM）跨 4 台 a3-highgpu-4g 的已发布基准显示：当 KV 缓存超过 HBM 时，原生 CPU 卸载和 LMCache 都显著提升吞吐量；低 KV 占用时，所有配置与基线相符，仅带有少量开销。

**类型：** 学习  
**语言：** Python（标准库，简易 KV 溢出模拟器）  
**先决条件：** 第17阶段 · 04（vLLM 服务内部实现），第17阶段 · 06（SGLang/RadixAttention）  
**时间：** 约 60 分钟

## 学习目标

- 绘制 vLLM 生产栈层次结构图：路由器、引擎、KV 卸载、可观测性。
- 讲解 KV 卸载连接器 API（v0.9.0+）以及 0.11.0 版本中异步路径如何隐藏卸载延迟。
- 量化 LMCache CPU-DRAM 何时有帮助（KV 大于 HBM）何时带来开销（KV 足够小可容纳于 HBM）。
- 根据部署限制，在原生 vLLM CPU 卸载与 LMCache 连接器之间做出选择。

## 问题描述

你的 vLLM 服务中 GPU HBM 使用率达 100%，并且在并发量提升时出现抢占事件。请求被驱逐、重新排队，你却在一分钟内对相同的 2K 令牌提示重新预填充四次。GPU 计算资源花费在了冗余预填充上；有效吞吐率远低于原始吞吐率。

增加更多 GPU 成本线性增长。增加更多 HBM 不可行。但 CPU DRAM 便宜 —— 单个插槽容量超过 512 GB，虽然延迟比 HBM 高几个数量级，但对于“暂时热”的 KV 缓存足够用。

LMCache 将 KV 缓存提取到 CPU DRAM，使抢占的请求能快速恢复，且不同引擎间的重复前缀共享缓存，无需每个引擎都重新预填充。

## 概念

### vLLM 生产栈

`github.com/vllm-project/production-stack` 是参考的 Kubernetes 部署：

- **路由器** — 支持缓存感知（第17阶段 · 11）。消费 KV 事件。
- **引擎** — vLLM 工作节点。每个 GPU 或 TP/PP 组一个。
- **KV 缓存卸载** — LMCache 部署或原生连接器。
- **可观测性** — Prometheus 抓取，Grafana 仪表板，OTel 跟踪。
- **控制平面** — 服务发现、配置管理、滚动更新。

以 Helm chart + operator 形式发布。

### KV 卸载连接器 API（v0.9.0+）

vLLM 0.9.0 引入了连接器 API，实现可插拔的 KV 缓存后端。你的引擎将块卸载给连接器，连接器负责存储（RAM、磁盘、对象存储、LMCache）。请求需要块时，连接器会加载回该块。

vLLM 0.11.0（2026 年 1 月）添加了异步卸载路径 —— 卸载可在后台进行，通常情况下引擎不会阻塞等待卸载完成。整体延迟和吞吐率依赖于工作负载形态、KV 缓存命中率和系统压力；vLLM 笔记指出，定制内核卸载在低命中率时可能降低吞吐量，且异步调度与投机解码存在已知交互问题。

### 原生 CPU 卸载 vs LMCache

**原生 vLLM CPU 卸载**：局部于引擎。将 KV 块存储在主机 RAM 中。实现快速，无网络跳转。不能跨引擎共享。

**LMCache 连接器**：集群级别。存储块于共享 LMCache 服务器（CPU DRAM + Ceph/S3 分层存储）。块对所有引擎可访问。已发布 16x H100 基准。

单引擎 HBM 压力时选用原生。多引擎共享前缀时（带通用系统提示的 RAG，多租户共享模板）选用 LMCache。

### 基准行为

16x H100（80 GB HBM）分布在 4 台 a3-highgpu-4g 测试：

- 低 KV 负载（短提示、低并发）：所有配置均与基线持平，LMCache 开销约 3-5%。
- 中等负载：LMCache 在跨引擎前缀复用上开始显著帮助。
- KV 超过 HBM：原生 CPU 卸载和 LMCache 均显著提升吞吐量，LMCache 增益更大因支持跨引擎共享。

### LMCache 关键适用场景

- 多租户服务，多租户间共享系统提示。
- RAG（检索增强生成），跨查询重用文档块。
- 在相同基础模型上进行微调变体（LoRA），通过基础模型 KV 复用减少冗余计算。
- 抢占频繁工作负载：从 CPU 恢复成本低于重新预填充。

### 不建议启用场景

- HBM 压力较小——仅付出开销无收益。
- 短上下文（<1K 令牌）——传输时间超过重新预填充时间。
- 单租户、单提示工作负载——无缓存复用价值。

### 与解耦式服务集成

第17阶段 · 17 的解耦式服务与 LMCache 结合使用：KV 从预填充池传输到解码池时，到达 LMCache（若未使用）；后续查询从 LMCache 获取。第17阶段 · 11 的缓存感知路由器可路由至本地或 LMCache 共享缓存匹配的引擎。

### 需记住的数字

- vLLM 0.9.0：连接器 API 发布。
- vLLM 0.11.0（2026 年 1 月）：异步卸载路径；端到端延迟影响依赖工作负载、KV 命中率和系统压力（非绝对保证）。
- 16x H100 基准：KV 占用超 HBM 时 LMCache 有效。
- HBM 压力小：3-5% 开销无收益。

## 使用方法

`code/main.py` 模拟含或不含 LMCache 的抢占重度工作负载。报告避免的重新预填充次数、吞吐率提升以及平衡点的 HBM 利用率。

## 交付成果

本课件生成 `outputs/skill-vllm-stack-decider.md`。根据工作负载形态和 vLLM 部署情况，决定使用原生 CPU 卸载、LMCache 还是两者皆无。

## 练习

1. 运行 `code/main.py`。LMCache 从何种 HBM 利用率开始产生收益？
2. 某租户跨 200 次查询/小时共享 6K 令牌系统提示。计算此租户预期的 LMCache 节省量。
3. LMCache 服务器为单点故障。设计高可用策略（副本，回退原生卸载）。
4. LMCache 存储于旋转磁盘上的 Ceph。4K 令牌 KV（70B FP8，约500 MB）的读取时间与重新预填充相比如何？
5. 论证 vLLM 0.11.0 异步路径是否“免费” —— 开销隐藏在哪里？

## 关键词汇

| 术语            | 常用说法                      | 实际含义                                   |
|-----------------|-------------------------------|-------------------------------------------|
| Production-stack | “参考部署”                   | vLLM 的 Kubernetes Helm chart + operator |
| Connector API    | “KV 后端接口”                | vLLM 0.9.0+ 可插拔 KV 存储接口           |
| Native CPU offload | “本地引擎溢出”              | 同引擎主机 RAM 中存储 KV                  |
| LMCache         | “集群 KV 缓存”                | 跨引擎 KV 缓存服务器，CPU DRAM + 磁盘层  |
| 0.11.0 async    | “非阻塞卸载”                  | 卸载过程隐藏在引擎流后                     |
| Preemption      | “驱逐让位”                    | HBM 满时 KV 缓存置换                       |
| Prefix reuse    | “相同系统提示”                | 多查询共享开头，命中缓存                   |
| Ceph tier      | “磁盘层”                      | 缓存层次结构中 DRAM 之下的持久存储         |

## 深入阅读

- [vLLM 博客 —— KV 卸载连接器（2026 年 1 月）](https://blog.vllm.ai/2026/01/08/kv-offloading-connector.html)
- [vLLM Production Stack GitHub](https://github.com/vllm-project/production-stack) — Helm chart + operator。
- [LMCache 企业级大规模 LLM 推理（arXiv:2510.09665）](https://arxiv.org/html/2510.09665v2)
- [LMCache GitHub](https://github.com/LMCache/LMCache) — 连接器实现。
- [vLLM 0.11.0 发布说明](https://github.com/vllm-project/vllm/releases) — 异步路径细节。
