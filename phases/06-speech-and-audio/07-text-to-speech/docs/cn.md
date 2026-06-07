# 文本转语音（Text-to-Speech, TTS）— 从 Tacotron 到 F5 和 Kokoro

> 自动语音识别（ASR）是语音到文本的反向过程；文本转语音（TTS）则是文本到语音的反向过程。2026 年的技术栈分为三个部分：文本 → 词元(token)，词元 → 梅尔频谱图(mel)，梅尔频谱图 → 波形。每个部分都有适合笔记本电脑运行的默认模型。

**类型：** 构建  
**语言：** Python  
**先决条件：** 第6阶段 · 02（频谱图与梅尔频谱图），第5阶段 · 09（序列到序列，Seq2Seq），第7阶段 · 05（完整 Transformer（Transformer 架构））  
**时间：** ~75 分钟

## 问题

你有一段文本字符串："Please remind me to water the plants at 6 pm." 你需要一段持续3秒的音频剪辑，听起来自然，韵律（停顿、重音）正确，“plants”的发音元音正确，并且在 CPU 上实时（300 毫秒内）生成，用于实时语音助手。你还需要能够切换声音，应对代码切换输入（"remind me at 6 pm, daijoubu?"），并且不会在人名发音上出错。

现代 TTS 流水线如下：

1. **文本前端（Text frontend）**：文本归一化（日期、数字、邮箱）、转成音素(phonemes)或子词词元(subword tokens)、预测韵律特征。
2. **声学模型（Acoustic model）**：文本 → 梅尔频谱图。包括 Tacotron 2（2017）、FastSpeech 2（2020）、VITS（2021）、F5-TTS（2024）、Kokoro（2024）。
3. **声码器（Vocoder）**：梅尔频谱图 → 波形。WaveNet（2016）、WaveRNN、HiFi-GAN（2020）、BigVGAN（2022）、2024 年以后出现的神经编解码器神经声码器。

到2026年，声学模型和声码器的界限因端到端扩散模型和流匹配模型而变得模糊，但三部分的思维模型依旧适用于调试。

## 概念

![Tacotron, FastSpeech, VITS, F5/Kokoro 并列图](../assets/tts.svg)

**Tacotron 2 (2017)。** 序列到序列（seq2seq）模型：字符嵌入 → 双向 LSTM 编码器 → 位置敏感注意力机制 → 自回归 LSTM 解码器输出梅尔频谱帧。速度慢（自回归），处理长文本时不够稳定。仍被引用作为基线模型。

**FastSpeech 2 (2020)。** 非自回归模型。时长预测器预测每个音素对应多少梅尔频谱帧。单通道推理，比 Tacotron 快 10 倍。自然度有一定损失（单调对齐），但广泛应用。

**VITS (2021)。** 联合训练编码器、基于流的时长预测和 HiFi-GAN 声码器，端到端变分推断。高质量单模型，2022–2024 年主流开源 TTS。变体有 YourTTS（多说话人零样本）、XTTS v2（2024，Coqui）。

**F5-TTS (2024)。** 基于流匹配的扩散 Transformer。自然韵律，5 秒参考音频实现零样本语音克隆。2026 年开源 TTS 排行榜领先。335M 参数。

**Kokoro (2024)。** 小型（82M）、CPU 可运行、实时使用的顶级英语 TTS。仅限固定词汇表和英语，Apache-2.0 许可。

**OpenAI TTS-1-HD、ElevenLabs v2.5、Google Chirp-3。** 商业顶尖技术。ElevenLabs v2.5 支持情感标签（如"[whispered]"，"[laughing]"）和角色语音，2026 年主导有声书制作。

### 声码器演进

| 时代 | 声码器 | 延迟 | 质量 |
|-----|---------|---------|---------|
| 2016 | WaveNet | 离线只能使用 | 发布时顶级 |
| 2018 | WaveRNN | 近实时 | 优良 |
| 2020 | HiFi-GAN | 100 倍实时 | 近乎真人 |
| 2022 | BigVGAN | 50 倍实时 | 跨说话人/语言泛化 |
| 2024 | SNAC, DAC（神经编解码器） | 与自回归模型集成 | 离散词元，高效比特利用 |

截至2026年，大部分“TTS”模型已经实现端到端从文本到波形，梅尔频谱图成为内部表示。

### 评估方法

- **MOS（平均意见分Mean Opinion Score）**，1–5 评分，众包。仍是金标准，但速度极慢。
- **CMOS（对比MOS）**，A 对 B 的偏好比较。单次标注置信区间更紧凑。
- **UTMOS、DNSMOS。** 无参考神经网络 MOS 预测器。常用于排行榜。
- **CER（字符错误率）通过 ASR。** 将 TTS 输出通过 Whisper 转录，计算与输入文本的 CER。语音可懂度的代理指标。
- **SECS（说话人嵌入余弦相似度）。** 语音克隆质量指标。

2026 年 LibriTTS test-clean 数据集指标：

| 模型 | UTMOS | CER（通过 Whisper） | 参数量 |
|-------|-------|-------------------|---------|
| 真实语音（Ground truth） | 4.08 | 1.2% | — |
| F5-TTS | 3.95 | 2.1% | 335M |
| XTTS v2 | 3.81 | 3.5% | 470M |
| VITS | 3.62 | 3.1% | 25M |
| Kokoro v0.19 | 3.87 | 1.8% | 82M |
| Parler-TTS Large | 3.76 | 2.8% | 2.3B |

## 构建流程

### 步骤 1：对输入文本音素化（phonemize）

```python
from phonemizer import phonemize
ph = phonemize("Hello world", language="en-us", backend="espeak")
# 'həloʊ wɜːld'
```

音素是跨语言桥梁。避免在 VITS 时代之前的模型中直接输入纯文本。

### 步骤 2：运行 Kokoro（2026 CPU 默认）

```python
from kokoro import KPipeline
tts = KPipeline(lang_code="a")  # "a" = 美式英语
audio, sr = tts("Please remind me to water the plants at 6 pm.", voice="af_bella")
# audio: float32 张量, sr=24000
```

离线运行，单文件，82M 参数。

### 步骤 3：用 F5-TTS 进行语音克隆

```python
from f5_tts.api import F5TTS
tts = F5TTS()
wav = tts.infer(
    ref_file="my_voice_5s.wav",
    ref_text="The quick brown fox jumps over the lazy dog.",
    gen_text="Please remind me to water the plants.",
)
```

传入5秒参考音频及其文本，F5 同时克隆韵律和音色。

### 步骤 4：从头实现 HiFi-GAN 声码器

教程脚本无法涵盖全部细节，框架如下：

```python
class HiFiGAN(nn.Module):
    def __init__(self, mel_channels=80, upsample_rates=[8, 8, 2, 2]):
        super().__init__()
        # 4 个上采样块，总共 256x，从梅尔频率到音频采样率
        ...
    def forward(self, mel):
        return self.blocks(mel)  # -> 波形
```

训练目标：对抗训练（判别器作用于短窗口）+ 梅尔频谱重建损失 + 特征匹配损失。已商品化——建议使用 `hifi-gan` 仓库或 nvidia-NeMo 的预训练检查点。

### 步骤 5：全流水线（伪代码）

```python
text = "Please remind me at 6 pm."
phones = phonemize(text)
mel = acoustic_model(phones, speaker=alice)      # [T, 80]
wav = vocoder(mel)                                # [T * 256]
soundfile.write("out.wav", wav, 24000)
```

## 使用指南

2026 年技术栈：

| 场景 | 方案选择 |
|-----------|------|
| 实时英语语音助手 | Kokoro（CPU）或 XTTS v2（GPU） |
| 从5秒参考音频进行语音克隆 | F5-TTS |
| 商业角色声音 | ElevenLabs v2.5 |
| 有声书朗读 | ElevenLabs v2.5 或者 XTTS v2 微调 |
| 低资源语言 | 在目标语言5-20小时数据上训练 VITS |
| 表现力 / 情感标签 | ElevenLabs v2.5 或 StyleTTS 2 微调 |

2026 年开源领导者：**F5-TTS 代表质量，Kokoro 代表效率**。除了历史研究，不建议使用 Tacotron。

## 常见问题

- **缺乏文本归一化器。** “Dr. Smith” 是读成“Doctor”还是“Drive”？“2026” 是“二零二六”还是“二千零二十六”？归一化必须先于音素化。
- **未登录专有名词。** “Ghumare” 读作 “ghyu-mair”？备用字符到音素模型可处理未知词。
- **削波（Clipping）。** 声码器输出很少削波，但推理时梅尔频谱缩放不匹配时会超出 ±1.0。务必 `np.clip(wav, -1, 1)`。
- **采样率不匹配。** Kokoro 输出24 kHz；下游流水线期望16 kHz，需要重采样避免混叠。

## 上线发布

保存为 `outputs/skill-tts-designer.md`。设计一个针对特定声音、延迟和语言目标的 TTS 流水线。

## 练习

1. **简单。** 运行 `code/main.py`。构建一个玩具词汇的音素字典，估计每个音素的时长，打印假的“梅尔”时间表。
2. **中等。** 安装 Kokoro，合成同一句话，分别用 `af_bella` 和 `am_adam` 语音。比较音频时长和主观质量。
3. **困难。** 录制一段5秒的参考音频，使用 F5-TTS 克隆该声音。报告参考音频和克隆输出的 SECS。

## 关键词

| 术语 | 俗称 | 实际含义 |
|------|-----------------|-----------------------|
| Phoneme | 音素 | 抽象声音类别；英语中有39个（ARPABet）。 |
| Duration predictor | 持续时长预测 | 非自回归模型输出；每个音素的整数帧数。 |
| Vocoder | 声码器 | 将梅尔频谱映射到原始音频采样的神经网络。 |
| HiFi-GAN | 标准声码器 | 基于生成对抗网络；2020–2024 年主流。 |
| MOS | 主观质量评分 | 来自人工评审者的1–5分平均意见分。 |
| SECS | 语音克隆度量 | 目标与输出说话人嵌入余弦相似度。 |
| F5-TTS | 2024 开源顶尖 | 流匹配扩散；零样本克隆。 |
| Kokoro | CPU 英语领导者 | 82M 参数模型，Apache 2.0 许可。 |

## 深入阅读

- [Shen et al. (2017). Tacotron 2](https://arxiv.org/abs/1712.05884) — 序列到序列基线模型。
- [Kim, Kong, Son (2021). VITS](https://arxiv.org/abs/2106.06103) — 端到端流式模型。
- [Chen et al. (2024). F5-TTS](https://arxiv.org/abs/2410.06885) — 当前开源顶尖技术。
- [Kong, Kim, Bae (2020). HiFi-GAN](https://arxiv.org/abs/2010.05646) — 2026 年仍广泛使用的声码器。
- [Kokoro-82M on HuggingFace](https://huggingface.co/hexgrad/Kokoro-82M) — 2024 年 CPU 友好英语 TTS。
