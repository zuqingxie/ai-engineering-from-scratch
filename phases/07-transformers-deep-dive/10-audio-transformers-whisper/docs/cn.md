# Audio Transformers — Whisper 架构

> 音频是时间上的频率图像。Whisper 是一个吃 mel 频谱图并输出语音的 ViT（视觉Transformer）。

**类型：** 学习  
**语言：** Python  
**前置知识：** 第7阶段·05（完整Transformer）、第7阶段·08（编码器-解码器）、第7阶段·09（ViT）  
**时长：** 约45分钟

## 问题背景

在 Whisper（OpenAI，Radford 等，2022）之前，最先进的自动语音识别（ASR）依赖于 wav2vec 2.0 和 HuBERT——自监督特征提取器加上微调的分类头。高质量但昂贵的数据管道，领域适应性差。多语言语音识别需要针对每个语系单独建模。

Whisper 做了三项押注：

1. **训练数据全覆盖。** 从网上抓取的 97 种语言共计68万小时弱标记音频。无干净学术语料，无音素标签。
2. **多任务单模型。** 通过任务令牌联合训练用于转录、翻译、语音活动检测（VAD）、语言识别和时间戳。
3. **标准编码器-解码器 Transformer（Transformer 架构）**。编码器处理 log-mel 频谱图。解码器自回归生成文本 tokens。无声码器（vocoder）、无 CTC、无隐马尔可夫模型（HMM）。

结果：Whisper large-v3 在口音、噪声以及零净标签数据的语言上表现稳定。它是每个开源语音助手和大多数商业助手在2026年的默认语音前端。

## 原理概述

![Whisper 流程：音频 → mel → 编码器 → 解码器 → 文本](../assets/whisper.svg)

### 第1步 — 重采样 + 窗口处理

音频采样率为16kHz。裁剪或填充为30秒。计算 log-mel 频谱图：80个mel频带，10毫秒步长 → 约3000帧 × 80特征。这即是 Whisper 看到的“输入图像”。

### 第2步 — 卷积干线（convolutional stem）

使用两个一维卷积（Conv1D）层，核大小3，步长2，将3000帧序列降到1500帧。序列长度减半，参数量不大。

### 第3步 — 编码器

一个24层（large版）Transformer编码器，处理1500个时间步。使用正弦位置编码，自注意力，GELU前馈网络（FFN）。输出 1500 × 1280 维隐藏态。

### 第4步 — 解码器

24层Transformer解码器。自回归生成基于BPE词汇表的token，词汇表是GPT-2的超集，增加了一些音频特有的特殊token。

### 第5步 — 任务令牌

解码器的提示词以控制tokens开头，指示模型执行的任务：

```text
<|startoftranscript|>  <|en|>  <|transcribe|>  <|0.00|>
```

或

```text
<|startoftranscript|>  <|fr|>  <|translate|>  <|0.00|>
```

模型采用这种约定训练。通过前缀控制任务。这是2026年语音版的instruction-tuning。

### 第6步 — 输出

采用宽度为5的束搜索（beam search）加对数概率阈值。若不存在 `<|notimestamps|>` token，则每0.02秒预测一次时间戳。

### Whisper 模型规模

| 模型 | 参数量 | 层数 | d_model | 头数 | 显存需求（fp16） |
|-------|--------|--------|---------|-------|-------------|
| Tiny | 39M | 4 | 384 | 6 | ~1 GB |
| Base | 74M | 6 | 512 | 8 | ~1 GB |
| Small | 244M | 12 | 768 | 12 | ~2 GB |
| Medium | 769M | 24 | 1024 | 16 | ~5 GB |
| Large | 1550M | 32 | 1280 | 20 | ~10 GB |
| Large-v3 | 1550M | 32 | 1280 | 20 | ~10 GB |
| Large-v3-turbo | 809M | 32 | 1280 | 20 | ~6 GB（4层解码器） |

Large-v3-turbo（2024）将解码器从32层减至4层。解码速度提升8倍，性能损失不足1个WER点。高速解码使 Whisper-turbo 成为2026年前实时语音助手的默认选择。

### Whisper 不做的事

- 不做说话人分离。可配合 pyannote 使用。
- 天生不支持实时流式识别——固定30秒窗口。现代封装（`faster-whisper`、`WhisperX`）通过VAD+窗口重叠实现流式。
- 无30秒外的长上下文支持。因人类语音转录极少需要长距离上下文，实用性良好。

### 2026 年相关产品格局

| 任务 | 模型 | 备注 |
|------|-------|-------|
| 英语自动语音识别 | Whisper-turbo、Moonshine | Moonshine在边缘设备上快4倍 |
| 多语言自动语音识别 | Whisper-large-v3 | 支持97种语言 |
| 流式语音识别 | faster-whisper + VAD | 可达到150毫秒延迟目标 |
| 语音合成（TTS） | Piper, XTTS-v2, Kokoro | 编码器-解码器模式，但类似Whisper架构 |
| 音频+语言多模态 | AudioLM、SeamlessM4T | Transformer统一处理文本和音频tokens |

## 实现步骤

查看 `code/main.py`。我们不训练 Whisper，而是实现log-mel频谱图管道和任务令牌提示格式器。这是生产环境中你会使用的部分。

### 第1步: 合成音频

生成一个1秒钟的440 Hz正弦波，采样率16 kHz，总16000样本。

### 第2步: log-mel频谱图（简化版）

完整mel频谱图计算依赖FFT。这里用简化帧分割+每帧能量替代，展示整体流水线且无需 `librosa`：

```python
def frame_signal(x, frame_size=400, hop=160):
    frames = []
    for start in range(0, len(x) - frame_size + 1, hop):
        frames.append(x[start:start + frame_size])
    return frames
```

帧大小25毫秒，帧移10毫秒。对应 Whisper 的窗口参数。每帧能量替代mel频带用于教学演示。

### 第3步: 填充到30秒

Whisper始终处理30秒片段。对频谱图填充（或裁剪）至3000帧。

### 第4步: 构建任务令牌

```python
def whisper_prompt(lang="en", task="transcribe", timestamps=True):
    tokens = ["<|startoftranscript|>", f"<|{lang}|>", f"<|{task}|>"]
    if not timestamps:
        tokens.append("<|notimestamps|>")
    return tokens
```

这即是整个任务控制面。4个token作前缀。

## 使用示例

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe("meeting.wav", language="en", task="transcribe")
print(result["text"])
print(result["segments"][0]["start"], result["segments"][0]["end"])
```

更快，兼容OpenAI接口：

```python
from faster_whisper import WhisperModel
model = WhisperModel("large-v3-turbo", compute_type="int8_float16")
segments, info = model.transcribe("meeting.wav", vad_filter=True)
for s in segments:
    print(f"{s.start:.2f} - {s.end:.2f}: {s.text}")
```

**2026年选用Whisper时机：**

- 多语言自动语音识别单模型方案。
- 对嘈杂、多样音频的稳健转录。
- 研究/原型阶段的快速ASR起点。

**不选用Whisper时机：**

- 极低延迟流式识别任务——Moonshine在相同性能下更快。
- 需要 <200毫秒响应的实时对话AI——采用专门流式ASR。
- 说话人分离需求——Whisper不支持，可结合 pyannote。

## 部署说明

参见 `outputs/skill-asr-configurator.md`。该技能选择ASR模型、解码参数和预处理流水线，启动新的语音应用。

## 练习

1. **简单。** 运行 `code/main.py`。确认16 kHz采样、10毫秒帧移下1秒音频约100帧，30秒约3000帧。
2. **中等。** 使用 `numpy.fft` 实现完整log-mel频谱图。验证80个mel频带与 `librosa.feature.melspectrogram(n_mels=80)` 结果数值基本一致。
3. **困难。** 实现流式推理：将音频拆成10秒窗口，2秒重叠，分别调用 Whisper，合并转录结果。用5分钟播客样本与全长单次推理对比词错误率。

## 关键词

| 术语 | 通俗说法 | 实际含义 |
|------|--------|---------|
| Mel spectrogram（梅尔频谱图） | “音频图像” | 频率bins和时间帧构成的二维表示，每格为对数能量。 |
| Log-mel（对数梅尔） | “Whisper看到的” | Mel频谱经对数变换，近似人类的响度感知。 |
| Frame（帧） | “一时间切片” | 25毫秒样本的窗口，10毫秒步长重叠。 |
| Task token（任务令牌） | “语音提示前缀” | 解码器提示里特殊token如 `<|transcribe|>` / `<|translate|>`。 |
| Voice activity detection（语音活动检测, VAD） | “找到说话部分” | 去除静默段，减少计算成本。 |
| CTC（Connectionist Temporal Classification） | “典型ASR损失” | 无需对齐的训练目标；Whisper不采用。 |
| Whisper-turbo | “小解码器，全编码器” | large-v3的编码器＋4层解码器，解码提速8倍。 |
| Faster-whisper | “生产级封装” | 用CTranslate2重实现，int8量化，比官方快4倍。 |

## 参考资料

- [Radford 等（2022）。大规模弱监督实现稳健语音识别](https://arxiv.org/abs/2212.04356) — Whisper论文。  
- [OpenAI Whisper 代码库](https://github.com/openai/whisper) — 参考代码和模型权重。阅读 `whisper/model.py`，约400行涵盖 Conv1D干线+编码器+解码器全流程。  
- [OpenAI Whisper — `whisper/decoding.py`](https://github.com/openai/whisper/blob/main/whisper/decoding.py) — 束搜索和任务令牌逻辑代码，约500行，完全可读。  
- [Baevski 等（2020）。wav2vec 2.0：自监督语音表示学习框架](https://arxiv.org/abs/2006.11477) — Whisper前驱；某些场景仍然是SOTA特征提取器。  
- [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper) — 生产级封装，比官方快4倍。  
- [Jia 等（2024）。Moonshine: 用于实时转录和语音命令的语音识别](https://arxiv.org/abs/2410.15608) — 2024年边缘设备友好ASR，小巧且类似Whisper。  
- [HuggingFace 博客 — “用 🤗 Transformers 微调 Whisper 实现多语言 ASR”](https://huggingface.co/blog/fine-tune-whisper) — 经典微调方案，含mel频谱预处理和token时间戳处理。  
- [HuggingFace `modeling_whisper.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/modeling_whisper.py) — 完整实现（编码器、解码器、交叉注意、生成），对应本课架构图。
