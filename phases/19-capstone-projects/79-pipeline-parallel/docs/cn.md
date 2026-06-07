# Pipeline Parallel and Bubble Analysis

> Tensor parallelism（张量并行）将矩阵乘法拆分到不同rank。Pipeline parallelism（流水线并行）将模型拆分到不同rank，每个rank一阶段。微批次（microbatches）沿流水线流动。开始和结束的空闲时间即为bubble（气泡）；最小化它是核心技术。

**类型：** 构建  
**语言：** Python  
**前置知识：** 第19阶段 C轨道 课程42-49  
**时间：** 约90分钟

## 学习目标

- 将顺序模型拆分为N个阶段，并模拟跨N个rank的前向流水线。
- 使用GPipe调度（先正向填充，再反向排空）调度M个微批次通过流水线，并计算bubble比例。
- 与Megatron-LM和PipeDream中使用的交错1F1B调度比较bubble。
- 论证阶段分配：每阶段相等的计算量比每阶段相等的参数量更重要。

## 问题描述

一个70B参数的fp16模型，仅参数就需要140 GB。没有消费级GPU能容纳。ZeRO-3将参数切分到不同的rank，但每个前向步骤仍需所有rank allgather整层参数，每层需log(N)次跳转。流水线并行采用不同路径：将模型切成N个阶段，每个rank放一个阶段。层1的正向在rank 0完成后传递激活张量给rank 1；rank 1运行层2，再传给rank 2；依此类推。反向传播顺序相反。内存线性减少，因为每个rank只持有一个阶段；计算是顺序的，这就是bubble问题。

bubble是流水线开始的空闲时间（等待首个微批次到达最后阶段）和结束的空闲时间（等待最后微批次返回）。有M个微批次和N个阶段时，每阶段的bubble比例为 (N-1)/(M+N-1)。当M=8，N=4时，比例为27%。M=64，N=4时，比例为4.5%。bubble随着微批次数增加而减少，这意味着每个微批次的batch size要小，这是设计微批次的限制。

## 概念

```mermaid
flowchart LR
  R0[rank 0: stage 0 / layer 0] --> R1[rank 1: stage 1 / layer 1]
  R1 --> R2[rank 2: stage 2 / layer 2]
  R2 --> R3[rank 3: stage 3 / loss]
  R3 -.backward.-> R2
  R2 -.backward.-> R1
  R1 -.backward.-> R0
```

### GPipe 调度

先用所有M个微批次正向填满流水线，然后才开始反向排空。每个微批次的激活都得保留直到其反向完成，因此内存随M线性增长。正向耗时M+N-1周期，反向同样耗时M+N-1周期。每阶段有效工作周期是2M，bubble是2(N-1)。当正向和反向各耗时单位时间时，bubble比例为 (N-1)/(M+N-1)。取M远大于N可以掩盖bubble。

### 1F1B调度

交错执行：一旦某微批次的正向抵达最后阶段，立即开始反向传播，让其流回。该调度每阶段交替执行一个正向和一个反向。bubble仍是N-1，但激活内存受流水线深度限制，而非微批次数限制。生产流水线使用1F1B（如Megatron, PipeDream）。本课先实现GPipe因其更简单，再通过练习实现1F1B。

### 为什么每阶段均等计算量重要

如果阶段0耗时50ms，阶段1耗时100ms，每个周期都会被阶段1限速。其他阶段每周期空闲50ms等待阶段1释放。每阶段相等参数数目是错误指标：Transformers中计算主要由attention加MLP决定，而embedding层参数多但计算少。阶段分配应均衡每阶段FLOPs，而非权重数量。

### 微批次与批次

流水线执行M个大小为B的微批次。有效批次大小为M*B。流水线结束时的梯度即为这M*B样本的梯度。bubble比例依赖于M，优化器看到的批次大小是M*B。调整M是bubble（M大则低）和每微批次内存（GPipe中M大则激活内存高）间的权衡。

## 构建实现

`code/main.py` 实现：

- `PipelineStage`：一个小的 `nn.Module` 存储一个阶段的参数，暴露 `forward(activation)`。
- `Pipeline(stages, num_microbatches)`：使用模拟的阶段和模拟的每阶段时钟管理GPipe调度。
- `bubble_fraction(num_stages, num_microbatches)`：闭式解 (N-1)/(M+N-1)。
- 一个4阶段演示，输出每微批次的时间轨迹和测量的bubble比例。

运行：

```bash
python3 code/main.py
```

输出：每阶段每微批次的甘特图和bubble比例与闭式解的对比。

## 生产实践

三种模式使流水线并行足够稳定用于生产。

**激活检查点与流水线配合。** GPipe中激活内存是M倍单微批次内存，激活检查点在反向时重新计算正向，换算计算换内存；二者结合使得流水线适合长序列。

**阶段均衡由测量决定，不是假设。** 生产团队进行性能分析，测量目标硬件上每层实际计算（FLOPs和时钟时间），然后据此分区。Megatron-LM的 `--num-layers-per-stage` 允许不均匀层数以匹配不同层计算成本。

**发送接收调度避免死锁。** 如果流水线中所有阶段都先发送再接收，会导致线缆死锁。常用修复是交错：偶数rank阶段先发送后接收，奇数rank阶段先接收后发送。本课中调度显式安排阶段顺序，模式清晰。

## 应用示例

生产模式：

- **Megatron-LM。** 大规模流水线并行的参考实现。采用1F1B，支持张量+流水线+数据并行组合。
- **DeepSpeed Pipeline。** 与ZeRO集成；ZeRO-1 + 流水线是最大开放模型的常用组合。
- **PyTorch Pipe。** PyTorch原生流水线封装，基于 `torch.distributed.pipeline.sync.Pipe` 构建。

## 交付

第80课将各阶段参数切片存储于分片检查点。第81课组合DDP + ZeRO + 流水线并行于端到端演示（保持流水线模拟以保障运行时轻量）。

## 练习

1. 实现1F1B调度，并验证bubble比与GPipe一致，但激活内存有上限。  
2. 在更深模型上分析真实每阶段时间，并基于实测时钟重新均衡阶段。  
3. 在流水线微批次间加入梯度累积，验证梯度与等效全批次前向梯度一致。  
4. 将流水线配合激活检查点，并测量内存下降和计算开销。  
5. 将流水线与DDP结合（流水线rank复制于数据并行组上），分析二维调度。

## 关键术语

| 术语 | 常用说法 | 实际含义 |
|------|----------|----------|
| Pipeline | “沿深度做模型并行” | 每rank一个阶段，激活在阶段间流动 |
| Bubble | “流水线空闲时间” | 开始、结束时 (N-1) 步无工作阶段 |
| Microbatch | “批次的切片” | 一组正向/反向单元；M越大bubble越小 |
| GPipe | “先填后排空” | 所有M个正向先完成再反向，激活内存大 |
| 1F1B | “交错调度” | 每个阶段交替一个正向一个反向，激活内存有限 |

## 拓展阅读

- [Huang et al, GPipe: Efficient Training of Giant Neural Networks](https://arxiv.org/abs/1811.06965)  
- [Narayanan et al, PipeDream: Generalized Pipeline Parallelism for DNN Training](https://arxiv.org/abs/1806.03377)  
- [Megatron-LM pipeline parallel docs](https://github.com/NVIDIA/Megatron-LM)  
- 第19阶段 课程76 - schedule使用的send/recv原语  
- 第19阶段 课程78 - ZeRO与流水线正交且常结合使用
