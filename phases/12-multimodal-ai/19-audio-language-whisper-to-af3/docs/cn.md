# Audio-Language Models: 从 Whisper 到 Audio Flamingo 3 的演变

> Whisper（Radford 等，2022年12月）解决了语音识别问题——680k 小时弱监督多语种语音，一个简单的 encoder-decoder Transformer（变换器架构），一个让每个后续 ASR（自动语音识别）发布都引用的基准。但识别不是推理。问“这段录音中有哪些乐器”或“说话者表达了什么情感”或“第3分钟发生了什么”需要音频理解，而非转录。Qwen-Audio、SALMONN、LTU 和 NVIDIA 的 Audio Flamingo 3（AF3，2025年7月）逐步构建了这一技术栈：保留 Whisper 级别的编码器，添加 Q-formers，利用音频-文本指令数据训练，加入链式思维推理。本课将讲解这条演进路径。

**类型：** 构建  
**语言：** Python（标准库，log-Mel 频谱图 + 音频 Q-former 骨架）  
**前置知识：** 第6阶段（语音与音频），第12阶段 · 03（Q-Former）  
**时长：** ~180分钟

## 学习目标

- 从波形计算 log-Mel 频谱图：窗口化、FFT（快速傅里叶变换）、滤波器组、对数变换。  
- 比较编码器选项：Whisper 编码器、BEATs、AF-Whisper 混合体，何时选用哪种。  
- 构建音频 Q-former：N 个可学习查询对频谱图补丁进行交叉注意力。  
- 解释级联（Whisper-then-LLM）与端到端音频-LLM 训练的差异：为何端到端更适合推理任务的扩展。

## 问题背景

语音识别被 Whisper 解决，音频 OCR（光学字符识别）已成为商品。可“商品化”止步于转录。如果模型不能对听到内容进行推理——包括时间、说话者、情感、音乐结构、环境声音等信息——仅凭转录无法驱动产品功能。

三条明显路径：

1. 级联：Whisper 转录文本，LLM（大语言模型）基于转录文本推理。适合纯语音场景。对音乐、环境音、多说话人重叠、情感等处理较差。  
2. 端到端音频-LLM：音频编码器直接将音频 token 输入 LLM，跳过转录过程。保持声学信息（情感、说话人、环境音）。需要新训练数据。  
3. 混合型：音频编码器 + 可进行转录和推理的文本解码器。Qwen-Audio 和 Audio Flamingo 采用此路。

## 核心概念

### Log-Mel 频谱图：输入特征

所有音频编码器都以相同特征开始：log-Mel 频谱图。

1. 重采样到 16kHz。  
2. 采用 25ms 窗口、10ms 步长的短时傅里叶变换（STFT）。  
3. 取 FFT 结果的幅度。  
4. 施加 Mel 滤波器组（通常为80个滤波器，对数间隔，0-8000Hz），转换为感知频率尺度。  
5. 对幅度进行对数压缩（log(1 + x)）促进动态范围缩放。

结果是一个二维数组，形状为 (T, 80)，其中 T 是时间帧数。以一个持续30秒、帧率100Hz的音频为例，形状为 (3000, 80)。

### Whisper 的编码器

Whisper 编码器是一个12层 ViT 风格的 Transformer，把 log-Mel 频谱图作为时间帧序列处理。输出是每个时间帧对应的隐藏状态向量。

对 ASR，Whisper 的解码器是一个对编码器输出进行条件交叉注意力生成文本 token 的 Transformer。是标准的编码器-解码器架构。

对 ALM（音频-LLM），使用编码器输出作为另一个 LLM 的输入。惯例为：Whisper 编码器冻结，Q-former 可训练，LLM 按需冻结或微调。

### BEATs 和音频专用编码器

Whisper 训练数据语音为主。音乐和环境音较弱。

BEATs（Chen 等，2022）是一个自监督 Transformer，在 AudioSet 数据集上训练。对音乐和环境声捕捉优于同等参数规模的 Whisper。

AF-Whisper（Audio Flamingo 3 的混合体）：拼接 Whisper 与 BEATs 的特征作为音频输入。Whisper 提供语言信号，BEATs 负责声学信号。

### 音频 Q-former

与 BLIP-2 的视觉 Q-former 同理。固定数量的可学习查询（通常为32或64）对音频编码器输出帧进行交叉注意力。查询转化为 LLM 消费的音频 token。

训练对齐阶段：仅训练 Q-former，使用音频-文本对（AudioCaps、Clotho）上的对比和说明损失。指令阶段：端到端，解冻 LLM，在指令数据上训练。

### 演变历程 — SALMONN、Qwen-Audio、AF3

SALMONN（Tang 等，2023）：Whisper + BEATs + Q-former + LLaMA。首个具备严肃推理能力的开放音频-LLM。MMAU 基准约为0.55综合分。

Qwen-Audio（Chu 等，2023）：类似架构，训练数据更丰富，针对多轮对话优化。MMAU 达0.60。

LTU — Listen, Think, Understand（Gong 等，2023）：引入显式推理数据，聚焦音频片段链式思维。规模较小，目标聚焦。

Audio Flamingo 3（Goel 等，2025年7月）：现行开放 SOTA。8B LLM 骨干（Qwen2 7B），Whisper-large 编码器拼接 BEATs，64 查询 Q-former，训练于百万级音频-文本指令对。MMAU 0.72，在部分子任务匹配商业前沿水平。

AF3 引入了可选链式思维推理：模型可先发出思考 token（“让我先识别乐器：...”）然后才给出最终答案。启用思考时，复杂推理任务准确率提升3-5个百分点。

### 级联与端到端对比

级联流程：

1. Whisper 将音频转录为文本。  
2. LLM 对文本进行推理。

适合“为这期播客做总结”。不适合：

- “这首歌的情绪是什么？”——情绪存在声音中，非文字。  
- “是谁在说话，Alice 还是 Bob？”——需说话人识别。  
- “爆炸在第几秒发生？”——时间定位在文本丢失。  
- “这是生成音频还是实录？”——深度伪造检测需要声学特征。

端到端保留声学信号，Qwen-Audio 和 AF3 原生处理音乐、环境和情感。

### 2026 生产方案

新建音频理解产品：

- 若目标是转录，无音乐、无情感推断，选级联方案。  
- 若需求包含音乐、情感、多说话人或复杂音频推理，选 AF3 / Qwen-Audio 家族。

级联成本低、实现简单；端到端更强大。

### MMAU — 音频推理基准

MMAU（Massive Multimodal Audio Understanding）为2024-2025年音频推理基准：

- 10,000条横跨语音、音乐、环境音的音频-文本问答对。  
- 涵盖分类、时间推理、因果推理、开放式问答。  
- 测试级联流水线系统性缺失。

开源 SOTA（AF3）得分0.72；商业前沿约0.78（Gemini 2.5 Pro、Claude Opus 4.7）。差距小于 VideoMME 的开源与封闭差距，显示音频-LLM日渐成熟。

## 使用说明

`code/main.py`：

- 实现标准库中的 log-Mel 频谱图计算：窗口化、朴素 DFT、Mel 滤波器组。  
- 音频 Q-former 骨架：给定编码器输出帧，计算 Q、K、V，注意力，输出 N 个 token。  
- 级联与端到端在示例任务上的对比。

## 交付产物

本课生成文件 `outputs/skill-audio-llm-pipeline-picker.md`。输入音频任务（转录、音乐标注、情感推断、多说话人分离、环境分类），输出所选级联、端到端 AF3 或混合方案。

## 课后习题

1. 计算一个30秒音频剪辑的 log-Mel 频谱图尺寸，采样率16kHz，25ms 窗口，10ms 步长，80个 Mel 滤波器。采样率48kHz时尺寸如何变化？  

2. 为什么 Whisper 在音乐任务上表现不佳？BEATs 捕获了 Whisper 不具备的哪些音频特征？  

3. 64 查询与 32 查询的音频 Q-former：在何种任务复杂度下64查询更有优势？32查询又在哪些情况节省计算？  

4. 阅读 AF3 第4节关于按需链式思维。提出三个链式思维极大帮助的音频任务。  

5. 用 AF3 输出实现一个简易的说话人分离流水线。如何标记说话人切换？

## 术语表

| 术语 | 常用表述 | 实际含义 |
|------|-----------|-----------|
| Log-Mel 频谱图 | “Mel 特征” | Mel 滤波器组处理后，对数幅值的二维（时间×频率）数组 |
| 音频 Q-former | “音频感知器（Audio Perceiver）” | 音频编码器输出到固定长度查询的交叉注意力瓶颈，输入 LLM |
| 级联 | “ASR然后LLM” | Whisper 转录后，文本 LLM 推理；丢失声学信息 |
| 端到端 | “音频-LLM” | 音频特征通过 Q-former 直接进入 LLM；保留声学信号 |
| BEATs | “AudioSet 音频编码器” | 自监督 Transformer，训练于 AudioSet，擅长音乐及环境声 |
| MMAU | “音频推理基准” | 包含语音、音乐、环境共10k问答对；2024评测标准 |
| 按需链式思维 | “音频 CoT” | 模型可选先发推理 token 后给出最终答案，提升准确率3-5点 |

## 深入阅读

- [Radford 等 — Whisper（arXiv:2212.04356）](https://arxiv.org/abs/2212.04356)  
- [Chu 等 — Qwen-Audio（arXiv:2311.07919）](https://arxiv.org/abs/2311.07919)  
- [Goel 等 — Audio Flamingo 3（arXiv:2507.08128）](https://arxiv.org/abs/2507.08128)  
- [Tang 等 — SALMONN（arXiv:2310.13289）](https://arxiv.org/abs/2310.13289)  
- [Gong 等 — LTU（arXiv:2305.10790）](https://arxiv.org/abs/2305.10790)
