# 边缘推理 — Apple Neural Engine，Qualcomm Hexagon，WebGPU/WebLLM，Jetson

> 边缘的核心限制是内存带宽，而非计算能力。移动DRAM带宽为50-90 GB/s；数据中心的HBM3达到了2-3 TB/s——相差30-50倍。解码受限于内存带宽，因此这个差距决定性。到2026年，格局分为四块。Apple M4/A18 Neural Engine峰值38 TOPS，采用统一内存（无CPU↔NPU拷贝开销）。Qualcomm Snapdragon X Elite / 8 Gen 4 Hexagon达到45 TOPS。WebGPU + WebLLM 在M3 Max上运行Llama 3.1 8B（第四象限Q4）约41 tok/s（大约是原生性能的70-80%）；GitHub星标1.76万，兼容OpenAI API，覆盖约70-75%移动端。NVIDIA Jetson Orin Nano Super（8GB）能运行Llama 3.2 3B / Phi-3；AGX Orin通过vLLM运行gpt-oss-20b，速度约40 tok/s；Jetson T4000（JetPack 7.1）性能是AGX Orin的2倍。TensorRT Edge-LLM 支持EAGLE-3，NVFP4，分块预填充——由Bosch、ThunderSoft、MediaTek在CES 2026展示。

**类型：** 学习  
**语言：** Python（标准库，简易带宽受限解码模拟器）  
**先决条件：** 第17阶段·04（vLLM服务内部机制），第17阶段·09（生产环境量化）  
**时长：** 约60分钟

## 学习目标

- 解释为何移动端LLM推理受限于内存带宽，计算能力次要。
- 枚举四个边缘目标（Apple ANE，Qualcomm Hexagon，WebGPU/WebLLM，NVIDIA Jetson）并对应使用案例。
- 说明2026年WebGPU覆盖缺口（Firefox Android渐趋完善）及Safari iOS 26的正式落地。
- 针对不同目标选择量化格式（ANE用Core ML INT4 + FP16，Hexagon用QNN INT8/INT4，浏览器用WebGPU Q4，Jetson用NVFP4）。

## 问题描述

客户需要一款设备端聊天机器人：语音优先、默认隐私保护、支持离线工作。MacBook Pro M3 Max上，Llama 3.1 8B Q4约55 tok/s——表现良好。iPhone 16 Pro上，同模型只有3 tok/s——表现不佳。中端Android手机搭载Snapdragon 8 Gen 3，约7 tok/s。Chrome Android v121+通过WebGPU浏览器中，视设备不同，4-8 tok/s。

吞吐量差异不是迁移问题，而是带宽差×量化格式×NPU在用户空间访问性的结果。2026年的边缘推理实则是四种不同问题，各自解决。

## 概念解释

### 带宽才是真正瓶颈

解码每个token都需读取完整权重集。一个7B模型在第四象限（Q4）格式约3.5 GB。以50 GB/s速度读取3.5 GB需要70 ms——理论瓶颈约14 tok/s。高端移动DRAM 90 GB/s，理论约25 tok/s。低于此，算力再高也无补于事。

数据中心HBM3带宽3 TB/s，能在1.2 ms完成同样读取，瓶颈达830 tok/s。相同模型，相同权重，不同内存子系统。

### Apple Neural Engine（M4 / A18）

- 峰值38 TOPS。统一内存（CPU与ANE共享同一内存池），无拷贝开销。
- 通过Core ML + `.mlmodel`编译模型访问，或通过Metal Performance Shaders（MPS）经PyTorch使用。
- Llama.cpp Metal后端使用MPS而非直接ANE，原生ANE需通过Core ML转换。
- 2026年iOS最佳实战路径：Core ML配合INT4权重 + FP16激活。

### Qualcomm Hexagon（Snapdragon X Elite / 8 Gen 4）

- 峰值45 TOPS。与CPU和GPU集成于SoC但有单独内存域。
- QNN（Qualcomm Neural Network）SDK和AI Hub支持PyTorch/ONNX转换。
- 聊天模板、Llama 3.2及Phi-3均作为一等公民在AI Hub发布。

### Intel / AMD NPU（Lunar Lake，Ryzen AI 300）

- 40-50 TOPS。软件相较Apple/Qualcomm稍显滞后；OpenVINO持续改进但市场小众。
- 适合Windows ARM copilot应用；AMD/Intel桌面端原生支持本地优先。

### WebGPU + WebLLM

- 通过WebGPU计算着色器在浏览器中运行模型，无需安装。
- Llama 3.1 8B Q4在M3 Max约41 tok/s——大致是原生后端性能70-80%。
- GitHub星标1.76万；OpenAI兼容JS API；采用Apache 2.0许可。
- 2026年覆盖范围：Chrome Android v121+，Safari iOS 26正式版，Firefox Android仍在赶超中。整体移动覆盖率约70-75%。

### NVIDIA Jetson系列

- Orin Nano Super（8GB）：可运行Llama 3.2 3B、Phi-3，性能良好。
- AGX Orin：通过vLLM执行gpt-oss-20b，约40 tok/s。
- Thor / T4000（JetPack 7.1）：性能是AGX Orin的2倍，支持EAGLE-3和NVFP4。
- TensorRT Edge-LLM（2026年）支持EAGLE-3预测性解码，NVFP4权重，分块预填充——将数据中心优化移植到边缘。

### 各目标量化格式选择

| 目标 | 格式 | 说明 |
|--------|--------|-------|
| Apple ANE | INT4权重 + FP16激活 | Core ML转换路径 |
| Qualcomm Hexagon | QNN INT8 / INT4 | AI Hub转换器 |
| WebGPU / WebLLM | Q4 MLC（q4f16_1） | 使用`mlc_llm convert_weight` + 编译的`.wasm`；不支持GGUF |
| Jetson Orin Nano | Q4 GGUF 或 TRT-LLM INT4 | 受限内存带宽 |
| Jetson AGX / Thor | NVFP4 + FP8 KV | Edge-LLM路径 |

### 边缘设备的长上下文陷阱

Llama 3.1的128K上下文为数据中心功能。手机8 GB内存，4 GB模型 + 2 GB KV缓存（32K token）+操作系统开销= 超出内存限制（OOM）。边缘部署通常保持上下文在4K-8K，除非接受极端量化（Q4 KV）。

### 语音是杀手级应用

语音代理对延迟敏感（首token<500 ms）。本地推理完全消除网络延迟。结合边缘可运行的语音转文本（Whisper Turbo变体），边缘推理成为生产级语音循环。

### 必须牢记的数据

- Apple M4 / A18 ANE：38 TOPS。
- Qualcomm Hexagon SD X Elite：45 TOPS。
- WebLLM M3 Max：Llama 3.1 8B Q4约41 tok/s。
- AGX Orin：vLLM运行gpt-oss-20b约40 tok/s。
- 数据中心与边缘带宽差距：30-50倍。
- WebGPU移动覆盖率：约70-75%（Firefox Android落后）。

## 使用方式

`code/main.py`基于带宽受限数学模型计算不同边缘目标的理论解码吞吐极限。对比实际基准测量，突出带宽而非计算是瓶颈的情况。

## 交付内容

本课产出`outputs/skill-edge-target-picker.md`。给定平台（iOS/Android/浏览器/Jetson）、模型及延迟/内存预算，选择量化格式和转换流程。

## 练习

1. 运行`code/main.py`。针对Snapdragon 8 Gen 3（约77 GB/s带宽）上7B Q4模型，计算解码峰值。对比实际6-8 tok/s，判断运行效率。
2. WebGPU在Android上要求Chrome v121+。设计老版本浏览器回退方案——通过相同兼容OpenAI的API在服务器端执行。
3. 你开发的iOS应用需要4K上下文流式推理。哪种模型/格式组合可以保证iPhone 16上活动内存低于4 GB？
4. Jetson AGX Orin以40 tok/s运行gpt-oss-20b，Jetson Nano只支持3B型号。如果产品同时支持两者，如何统一推理栈？
5. 论述“WebLLM在2026年是否可用于生产环境”。结合覆盖范围、性能及Firefox Android的差距。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| ANE | “Apple neural engine” | M系列和A系列设备内置NPU，统一内存 |
| Hexagon | “Qualcomm NPU” | Snapdragon SoC内置NPU；QNN SDK访问 |
| WebGPU | “浏览器GPU” | W3C标准的浏览器GPU接口；2026年Chrome/Safari支持 |
| WebLLM | “浏览器LLM运行时” | MLC-LLM项目；Apache 2.0许可；兼容OpenAI的JS API |
| Jetson | “NVIDIA边缘设备” | Orin Nano / AGX / Thor / T4000 系列 |
| TRT Edge-LLM | “边缘TensorRT” | 2026年TensorRT-LLM的边缘版；支持EAGLE-3 + NVFP4 |
| 统一内存 | “共享内存池” | CPU与NPU访问相同RAM，无拷贝开销 |
| 带宽受限 | “内存带宽限制” | 解码受限于加载权重的字节/秒速率 |
| Core ML | “Apple转换工具” | Apple框架，用以生成ANE原生模型 |
| QNN | “Qualcomm生态” | Qualcomm Neural Network SDK |

## 进阶阅读

- [On-Device LLMs State of the Union 2026](https://v-chandra.github.io/on-device-llms/) — 生态格局与基准测试。  
- [NVIDIA Jetson Edge AI](https://developer.nvidia.com/blog/getting-started-with-edge-ai-on-nvidia-jetson-llms-vlms-and-foundation-models-for-robotics/) — Orin / AGX / Thor介绍。  
- [NVIDIA TensorRT Edge-LLM](https://developer.nvidia.com/blog/accelerating-llm-and-vlm-inference-for-automotive-and-robotics-with-nvidia-tensorrt-edge-llm/) — 2026年边缘版本公告。  
- [WebLLM (arXiv:2412.15803)](https://arxiv.org/html/2412.15803v2) — 设计与基准测试。  
- [Apple Core ML](https://developer.apple.com/documentation/coreml) — ANE原生模型转换。  
- [Qualcomm AI Hub](https://aihub.qualcomm.com/) — Hexagon预转换模型。
