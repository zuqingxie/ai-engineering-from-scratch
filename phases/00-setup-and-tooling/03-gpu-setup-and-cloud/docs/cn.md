# GPU 设置与云端

> 训练时使用 CPU 学习是可以的。真正训练需要 GPU。

**类型：** 构建  
**语言：** Python  
**先决条件：** 阶段 0，课程 01  
**时间：** ~45 分钟

## 学习目标

- 使用 `nvidia-smi` 和 PyTorch 的 CUDA API 验证本地 GPU 可用性
- 配置配备 T4 GPU 的 Google Colab，实现免费云端实验
- 比较 CPU 与 GPU 上矩阵乘法的基准测试并测量加速比
- 使用 fp16 经验法则估算你显存中能容纳的最大模型大小

## 问题背景

阶段 1-3 的大多数课程可以在 CPU 上正常运行。但一旦开始训练 CNN、Transformer（Transformer 架构）或大型语言模型（LLM，阶段 4 及以后）时，就需要 GPU 加速。CPU 训练运行 8 小时的任务，在 GPU 上只需 10 分钟。

你有三个选择：本地 GPU、云端 GPU，或者 Google Colab（免费）。

## 概念介绍

```text
你的选择：

1. 本地 NVIDIA GPU
   费用: $0（你已经有了）
   配置: 安装 CUDA + cuDNN
   适合: 常规使用、大型数据集

2. Google Colab（免费层）
   费用: $0
   配置: 无需配置
   适合: 快速实验、没有本地 GPU

3. 云端 GPU（Lambda、RunPod、Vast.ai）
   费用: $0.20-2.00/小时
   配置: SSH + 安装
   适合: 严肃训练、大型模型
```

## 实践操作

### 选项 1：本地 NVIDIA GPU

检查你是否有 GPU：

```bash
nvidia-smi
```

安装带 CUDA 支持的 PyTorch：

```python
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

### 选项 2：Google Colab

1. 访问 [colab.research.google.com](https://colab.research.google.com)  
2. 运行时 > 更改运行时类型 > 选择 T4 GPU  
3. 运行 `!nvidia-smi` 进行验证

将本课程的笔记本直接上传至 Colab。

### 选项 3：云端 GPU

针对 Lambda Labs、RunPod 或 Vast.ai：

```bash
ssh user@your-gpu-instance

pip install torch torchvision torchaudio
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### 没有 GPU？没关系。

大多数课程可在 CPU 上运行。需要 GPU 的课程会注明，并附带 Colab 链接。

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")
```

## 实践操作：GPU 与 CPU 基准测试

```python
import torch
import time

size = 5000

a_cpu = torch.randn(size, size)
b_cpu = torch.randn(size, size)

start = time.time()
c_cpu = a_cpu @ b_cpu
cpu_time = time.time() - start
print(f"CPU: {cpu_time:.3f}s")

if torch.cuda.is_available():
    a_gpu = a_cpu.to("cuda")
    b_gpu = b_cpu.to("cuda")

    torch.cuda.synchronize()
    start = time.time()
    c_gpu = a_gpu @ b_gpu
    torch.cuda.synchronize()
    gpu_time = time.time() - start
    print(f"GPU: {gpu_time:.3f}s")
    print(f"加速比: {cpu_time / gpu_time:.0f}x")
```

## 练习

1. 运行上述基准测试，比较 CPU 和 GPU 的运行时间  
2. 如果没有 GPU，试试在 Google Colab 上运行并比较结果  
3. 查看你的 GPU 显存大小，并估算能容纳的最大模型（经验法则：fp16 时每个参数占用 2 字节）

## 关键词汇

| 术语            | 人们说            | 实际含义                                       |
|-----------------|-------------------|-----------------------------------------------|
| CUDA            | “GPU 编程”        | NVIDIA 的并行计算平台，能让你在 GPU 上运行代码   |
| VRAM            | “GPU 内存”        | GPU 上的视频内存，独立于系统 RAM，限制模型大小   |
| fp16            | “半精度”          | 16 位浮点数，内存占用为 fp32 的一半，精度损失较小 |
| Tensor Core     | “快速矩阵硬件”    | 专门用于矩阵乘法的 GPU 核心，比普通核快 4-8 倍   |
