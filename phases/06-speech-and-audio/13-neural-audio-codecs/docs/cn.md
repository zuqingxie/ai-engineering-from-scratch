# 神经音频编解码器 — EnCodec、SNAC、Mimi、DAC 及语义-声学分离

> 2026 年的音频生成几乎全部基于离散令牌。EnCodec、SNAC、Mimi 和 DAC 将连续波形转换为 Transformers 可预测的离散序列。语义-声学令牌拆分 — 第一个码本为语义，剩余为声学 — 是继 Transformer（Transformer 架构）之后音频领域最重要的架构变革。

**类型：** 学习  
**语言：** Python  
**先决条件：** 第6阶段 · 02（频谱图 Spectrograms）、第10阶段 · 11（量化 Quantization）、第5阶段 · 19（子词令牌化 Subword Tokenization）  
**时间：** 约60分钟

## 问题

语言模型处理离散令牌，音频是连续的。如果你想要一个类似大语言模型（LLM）的语音/音乐模型 — 如 MusicGen、Moshi、Sesame CSM、VibeVoice、Orpheus — 首先需要一个**神经音频编解码器**：一个学习到的编码器，将音频离散化为少量令牌词汇；以及一个匹配的解码器，重构波形。

出现了两大家族：

1. **优先重构的编解码器** — EnCodec、DAC。优化感知音频质量。令牌是“声学”的 — 捕捉包括说话人身份、音色、背景噪声等全部信息。  
2. **优先语义的编解码器** — Mimi (Kyutai)、SpeechTokenizer。强制第一个码本编码语言/音素内容（通常通过从 WavLM 蒸馏获得）。后续码本是声学细节。

2024-2026 年的洞见是：**纯重构编解码器在文本生成时产生模糊语音。** 语言模型必须在同一码本中同时学习语言结构和声学结构，难以扩展。通过将它们分离—语义码本 0，声学码本 1-N — 使 Moshi 和 Sesame CSM 成功工作。

## 概念

![四大编解码器地图: EnCodec, DAC, SNAC（多尺度）, Mimi（语义+声学）](../assets/codec-comparison.svg)

### 核心技巧：残差矢量量化（Residual Vector Quantization，RVQ）

不是一个巨大的码本（需要数百万码字以保证高质量），所有现代音频编解码器采用**残差矢量量化（RVQ）**：一串小码本级联。第一个码本对编码器输出进行量化；第二个量化残差；依此类推。每个码本有 1024 个码字。8 个码本相当于1024^8 = 10^24的有效词汇量。

推理时，解码器对每帧选中的码字求和进行重构。

### 2026 年重要的四个编解码器

**EnCodec（Meta，2022 年）。** 基线。波形上的编码器-解码器，RVQ 瓶颈。24 kHz，最多32码本，默认4码本 @ 1.5 kbps。使用 `1D conv + transformer + 1D conv` 架构。被 MusicGen 采用。

**DAC（Descript，2023 年）。** 采用带 L2 归一化码本的 RVQ，周期性激活函数，改进的损失函数。是所有开源编解码器中最高的重构保真度 — 有时12码本下与原始语音无法区分。全频带44.1 kHz。

**SNAC（Hubert Siuzdak，2024 年）。** 多尺度 RVQ — 粗码本运行在比细码本更低的帧率。有效地对音频进行分层建模：粗略“草图”约12 Hz + 细节50 Hz。被 Orpheus-3B 采用，因为层次结构很好地映射到基于大语言模型的生成。

**Mimi（Kyutai，2024 年）。** 2026 年的革命者。12.5 Hz 帧率（极低），8 码本 @ 4.4 kbps。码本 0 **通过 WavLM 蒸馏** — 训练以预测 WavLM 的语音内容特征。码本 1-7 是声学残差。此分离技术支撑了 Moshi（第15课）和 Sesame CSM。

### 帧率对语言模型的重要性

帧率低 = 序列短 = 语言模型更快。

| 编解码器 | 帧率 | 1秒 = N 帧 | 适用领域 |
|----------|------|-------------|----------|
| EnCodec-24k | 75 Hz | 75 | 音乐、通用音频 |
| DAC-44.1k | 86 Hz | 86 | 高清音乐 |
| SNAC-24k (粗码本) | ~12 Hz | 12 | 自回归语言模型高效 |
| Mimi | 12.5 Hz | 12.5 | 流式语音 |

12.5 Hz 下，一段10秒语音仅有125帧 — Transformer 轻松处理。

### 语义令牌与声学令牌

```text
frame_t → [semantic_token_t, acoustic_token_0_t, acoustic_token_1_t, ..., acoustic_token_6_t]
```

- **语义令牌（Mimi 的码本 0）。** 编码语义内容 — 音素、单词、内容。通过辅助预测损失从 WavLM 蒸馏而来。  
- **声学令牌（码本 1-7）。** 编码音色、说话人身份、韵律、背景噪声、细节。

自回归语言模型先预测语义令牌（基于文本条件），然后预测声学令牌（基于语义及说话人参考）。这使得现代 TTS 能够零样本克隆声音：语义模型处理内容，声学模型处理音色。

### 2026 年重构质量（单位比特率，码率越低越好）

| 编解码器 | 码率 | PESQ | ViSQOL |
|----------|------|------|--------|
| Opus-20kbps | 20 kbps | 4.0 | 4.3 |
| EnCodec-6kbps | 6 kbps | 3.2 | 3.8 |
| DAC-6kbps | 6 kbps | 3.5 | 4.0 |
| SNAC-3kbps | 3 kbps | 3.3 | 3.8 |
| Mimi-4.4kbps | 4.4 kbps | 3.1 | 3.7 |

传统编解码器如 Opus 在单位码率的感知质量上仍占优势。神经编解码器胜在**产生离散令牌**（Opus 不具备）和**生成模型质量**（语言模型能用这些令牌做的事）。

## 构建步骤

### 第1步：使用 EnCodec 编码

```python
from encodec import EncodecModel
import torch

model = EncodecModel.encodec_model_24khz()
model.set_target_bandwidth(6.0)  # kbps

wav = torch.randn(1, 1, 24000)
with torch.no_grad():
    encoded = model.encode(wav)
codes, scale = encoded[0]
# codes: (1, n_codebooks, n_frames), dtype=int64
```

6 kbps 码率下，`n_codebooks=8`。每个码为 0-1023 （10 位）。

### 第2步：解码并测量重构

```python
with torch.no_grad():
    wav_recon = model.decode([(codes, scale)])

from torchaudio.functional import compute_deltas
import torch.nn.functional as F

mse = F.mse_loss(wav_recon[:, :, :wav.shape[-1]], wav).item()
```

### 第3步：语义-声学拆分（Mimi 风格）

```python
from moshi.models import loaders
mimi = loaders.get_mimi()

with torch.no_grad():
    codes = mimi.encode(wav)  # 形状 (1, 8, frames@12.5Hz)

semantic = codes[:, 0]
acoustic = codes[:, 1:]
```

语义码本 0 与 WavLM 对齐。你可以训练一个文本到语义的 Transformer —— 词汇量远小于直接到音频。然后一个单独的声学到波形解码器基于说话人参考进行条件生成。

### 第4步：为什么自回归语言模型处理编解码器令牌有效

一段10秒讲话，Mimi码率12.5 Hz × 8码本：

```text
N_tokens = 10 * 12.5 * 8 = 1000 个令牌
```

1000 个令牌是 Transformer 可轻松处理的上下文。一个256M参数规模的 Transformer 在现代 GPU 上可以毫秒级生成10秒语音。

## 应用场景

任务 → 推荐编解码器：

| 任务 | 编解码器 |
|------|----------|
| 通用音乐生成 | EnCodec-24k |
| 最高保真重构 | DAC-44.1k |
| 语音上的自回归语言模型（TTS） | SNAC 或 Mimi |
| 流式全双工语音 | Mimi（12.5 Hz） |
| 含文本的音效库 | EnCodec + T5 条件 |
| 细粒度音频编辑 | DAC + 修补 |

经验法则：**如果构建生成模型，优先用 Mimi 或 SNAC；构建压缩管道，则用 Opus。**

## 陷阱

- **码本过多。** 添加码本线性提升保真度，但同时线性增加模型序列长度。通常不超过8-12个。  
- **帧率不匹配。** 在12.5 Hz的 Mimi 上训练语言模型后，再在50 Hz的 EnCodec 上微调会默默失败。  
- **假设所有码本等同。** Mimi 中码本0承载内容，丢失它会导致语音不可懂；丢失码本7几乎无感。  
- **仅用重构质量作为指标。** 优秀的重构质量并不保证可用的语言模型生成效果，语义结构差会使生成毫无价值。

## 发布

保存为 `outputs/skill-codec-picker.md`。为所需生成或压缩任务选择合适编解码器。

## 练习

1. **简单。** 运行 `code/main.py`。实现简单的标量+残差量化器，增加码本数量时测量重构误差。  
2. **中等。** 安装 `encodec`，比较1、4、8、32码本对一段留出测试语音的影响。绘制 PESQ 或 MSE 与码率关系曲线。  
3. **困难。** 加载 Mimi，编码一段语音。用随机整数替换码本0，解码；再同样替换码本7，解码。比较两种破坏：码本0破坏应导致语音不可懂，码本7破坏几乎无影响。

## 关键词

| 术语 | 通俗说法 | 实际含义 |
|-------|-----------|----------|
| RVQ | 残差量化 | 一串小码本级联，每个码本量化前一个残差。 |
| 帧率 | 编解码器速度 | 每秒的令牌帧数。帧率低 = 语言模型更快。 |
| 语义码本 | Mimi 的码本 0 | 从自监督学习（SSL）特征蒸馏而来；编码内容。 |
| 声学码本 | 其余码本 | 编码音色、韵律、噪声、细节。 |
| PESQ / ViSQOL | 感知质量 | 与主观评分（MOS）高度相关的客观指标。 |
| EnCodec | Meta 编解码器 | RVQ的基线；被 MusicGen 使用。 |
| Mimi | Kyutai 编解码器 | 12.5 Hz帧率；语义-声学拆分；支撑 Moshi。 |

## 推荐阅读

- [Défossez 等（2023）。EnCodec](https://arxiv.org/abs/2210.13438) — RVQ基线。  
- [Kumar 等（2023）。Descript 音频编解码器（DAC）](https://arxiv.org/abs/2306.06546) — 最高保真开源。  
- [Siuzdak（2024）。SNAC](https://arxiv.org/abs/2410.14411) — 多尺度 RVQ。  
- [Kyutai（2024）。Mimi 编解码器](https://kyutai.org/codec-explainer) — 语义-声学拆分，WavLM 蒸馏。  
- [Borsos 等（2023）。AudioLM](https://arxiv.org/abs/2209.03143) — 两阶段语义/声学范式。  
- [Zeghidour 等（2021）。SoundStream](https://arxiv.org/abs/2107.03312) — 最初的可流式 RVQ 编解码器。
