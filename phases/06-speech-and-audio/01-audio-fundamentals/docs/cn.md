# 音频基础 — 波形、采样、傅里叶变换

> 波形是原始信号。声谱图是信号的表示。Mel 特征是机器学习友好的形式。每个现代的自动语音识别（ASR）和文本转语音（TTS）流水线都遵循这条阶梯，而第一步是理解采样和傅里叶变换。

**类型：** 学习  
**语言：** Python  
**先决条件：** 第1阶段 · 06（向量和矩阵）、第1阶段 · 14（概率分布）  
**时长：** ~45分钟  

## 问题

麦克风产生一个压力-时间信号。你的神经网络消耗张量。两者之间有一套约定，当被违反时会造成隐蔽的错误：模型训练正常，但字错误率（WER）翻倍；TTS 发出嘶嘶声；或者语音克隆系统记住了麦克风而非说话者。

语音系统中的每个错误都可以归结为以下三个问题之一：

1. 数据录制的采样率是多少，模型期望的是哪个采样率？
2. 信号是否发生混叠（aliasing）？
3. 你处理的是原始样本还是频率表示？

答对这几个问题，其余的第6阶段内容就能顺利理解。答错，连 Whisper-Large-v4 也会产生垃圾结果。

## 概念

![波形、采样、离散傅里叶变换及频率箱示意图](../assets/audio-fundamentals.svg)

**波形（Waveform）。** 一个一维浮点数组，范围在 `[-1.0, 1.0]`。用样本编号索引。要转换为秒数，除以采样率：`t = n / sr`。一个 16 kHz 的10秒音频片段有16万浮点数。

**采样率（Sampling rate, sr）。** 每秒采样的数量。2026年常见采样率：

| 采样率 | 用途 |
|------|-----|
| 8 kHz | 电话系统，遗留的 VOIP。奈奎斯特频率4 kHz会丢失辅音。不建议用于自动语音识别。 |
| 16 kHz | ASR标准。Whisper、Parakeet、SeamlessM4T v2都使用16 kHz采样。 |
| 22.05 kHz | 用于较老TTS模型的声码器训练。 |
| 24 kHz | 现代TTS（Kokoro、F5-TTS、xTTS v2）。 |
| 44.1 kHz | CD音频，音乐。 |
| 48 kHz | 电影、专业音频、高保真TTS（VALL-E 2、NaturalSpeech 3）。 |

**奈奎斯特-香农采样定理（Nyquist-Shannon）。** 采样率为 `sr` 可无歧义地表示最高频率为 `sr/2`。`sr/2` 是**奈奎斯特频率（Nyquist frequency）**。超过奈奎斯特频率的能量会被混叠（alias）——折叠到更低频率范围——并破坏信号；所以下采样前务必低通滤波。

**位深度（Bit depth）。** 16位 PCM（signed int16，取值范围±32,767）是通用交换格式。24位用于音乐，32位浮点用于内部数字信号处理（DSP）。比如 `soundfile` 库读取 int16，但输出的数组是标定在 `[-1, 1]` 的 float32。

**傅里叶变换（Fourier Transform）。** 任意有限信号都可以看作不同频率的正弦波叠加。离散傅里叶变换（Discrete Fourier Transform, DFT）对 `N` 个样本计算出 `N` 个复数系数——每个频率箱一个系数。第 `k` 个箱对应频率为 `k · sr / N` Hz。幅度表示该频率的振幅，角度表示相位。

**快速傅里叶变换（FFT）。** 快速傅里叶变换是一种在 `N` 为2的幂时，时间复杂度为 `O(N log N)` 的 DFT 算法。所有音频库底层都使用FFT。16 kHz 采样率下，1024点FFT产生512个有效频率箱，覆盖0–8 kHz，频率分辨率为15.6 Hz。

**帧分割和窗函数（Framing + window）。** 我们不会对整个音频片段做FFT，而是将音频切成有重叠的帧（一般25 ms，跳步10 ms），对每个帧乘以窗函数（Hann窗、Hamming窗）消除边缘不连续，然后对每帧做FFT。这就是短时傅里叶变换（Short-Time Fourier Transform, STFT）。第02课将继续讲解。

## 实现步骤

### 第1步：读取音频并绘制波形

`code/main.py` 仅使用标准库 `wave` 模块，保持演示无依赖。实际生产中会用 `soundfile` 或 `torchaudio.load`（两者都返回 `(waveform, sr)` 元组）：

```python
import soundfile as sf
waveform, sr = sf.read("clip.wav", dtype="float32")  # 形状 (T,), sr 是整数
```

### 第2步：用基础原理合成正弦波

```python
import math

def sine(freq_hz, sr, seconds, amp=0.5):
    n = int(sr * seconds)
    return [amp * math.sin(2 * math.pi * freq_hz * i / sr) for i in range(n)]
```

440 Hz的正弦波（音乐中的标准 A）在16 kHz采样率下，1秒长有16,000个浮点数。用 `wave.open(..., "wb")` 保存时使用16-bit PCM编码。

### 第3步：手工计算离散傅里叶变换

```python
def dft(x):
    N = len(x)
    out = []
    for k in range(N):
        re = sum(x[n] * math.cos(-2 * math.pi * k * n / N) for n in range(N))
        im = sum(x[n] * math.sin(-2 * math.pi * k * n / N) for n in range(N))
        out.append((re, im))
    return out
```

时间复杂度为 `O(N²)` —— 对于 `N=256` 程序可以跑用来验证正确性，但对真实音频没法用。实际代码调用 `numpy.fft.rfft` 或 `torch.fft.rfft`。

### 第4步：找到主频率

幅度峰值所在的索引 `k_star` 映射到频率 `k_star * sr / N`。对440 Hz正弦波运行应在 `440 * N / sr` 处出现峰值。

### 第5步：展示混叠现象

用10 kHz采样率采样7 kHz的正弦波（奈奎斯特频率为5 kHz）。7 kHz 超过奈奎斯特频率，混叠到 `10 − 7 = 3 kHz` 处。FFT峰值出现在3 kHz。这是经典的混叠演示，也是所有数模/模数转换器（DAC/ADC）都必须配备带阻低通滤波器的原因。

## 实用工具栈

2026年实际使用的栈：

| 任务 | 库 | 原因 |
|------|----|------|
| 读写 WAV/FLAC/OGG | `soundfile`（libsndfile封装） | 速度最快，稳定，返回 float32。 |
| 重采样 | `torchaudio.transforms.Resample` 或 `librosa.resample` | 内置正确的抗混叠。 |
| STFT / Mel 特征 | `torchaudio` 或 `librosa` | 适合GPU，加深PyTorch生态集成。 |
| 实时流式处理 | `sounddevice` 或 `pyaudio` | 跨平台的 PortAudio 绑定。 |
| 文件信息查看 | `ffprobe` 或 `soxi` | 命令行工具，快速，报告采样率/声道/编码。 |

决策规则：**先匹配采样率，比匹配其他任何事情都重要**。Whisper期望16 kHz单声道float32，给它44.1 kHz立体声，结果就是混乱，像模型出错一样。

## 部署

保存为 `outputs/skill-audio-loader.md`。该技能帮助你检查音频输入是否符合下游模型的预期，并在不匹配时正确重采样。

## 练习

1. **简单。** 合成1秒的 220 Hz + 440 Hz + 880 Hz 混合波形，采样率16 kHz。运行DFT，确认在预期频率箱中出现三个峰。
2. **中等。** 录制一个3秒的48 kHz人声WAV文件。用 `torchaudio.transforms.Resample`（带抗混叠）下采样为16 kHz，然后用简单取样（每3个取1个）方式下采样为16 kHz。对两者做FFT。混叠现象在哪儿出现？
3. **困难。** 用纯 `math` 和第3步的 DFT 从零实现STFT。帧长400，跳步160，Hann窗。用 `matplotlib.pyplot.imshow` 绘制幅度谱图。这就是第02课的声谱图。

## 关键词

| 词汇 | 通俗理解 | 实际含义 |
|------|---------|----------|
| 采样率（Sample rate） | 一秒内采样数量 | ADC 采样信号的频率，单位Hz。 |
| 奈奎斯特频率（Nyquist） | 能表示的最高频率 | `sr/2`；超过此频率的能量会混叠。 |
| 位深度（Bit depth） | 每个样本的分辨率 | `int16` = 65,536 级；`float32` 在 `[-1,1]` 内约24位精度。 |
| 离散傅里叶变换（DFT） | 序列的傅里叶变换 | `N` 个样本 → `N` 个复数频率系数。 |
| 快速傅里叶变换（FFT） | 快速DFT算法 | `O(N log N)` 算法，需要 `N` 是2的幂。 |
| 频率箱（Bin） | 频率列 | `k · sr / N` Hz；分辨率 = `sr / N`。 |
| 短时傅里叶变换（STFT） | 声谱图的底层 | 分帧+加窗后随时间的FFT。 |
| 混叠（Aliasing） | 奇怪的频率“鬼影” | 超过奈奎斯特频率的能量被“折叠”回低频箱。 |

## 延伸阅读

- [Shannon (1949). Communication in the Presence of Noise](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf) — 采样定理的经典论文。
- [Smith — The Scientist and Engineer's Guide to Digital Signal Processing](https://www.dspguide.com/ch8.htm) — 免费且权威的数字信号处理教材。
- [librosa 文档 — 音频基础](https://librosa.org/doc/latest/tutorial.html) — 代码实践讲解。
- [Heinrich Kuttruff — Room Acoustics (第6版)](https://www.routledge.com/Room-Acoustics/Kuttruff/p/book/9781482260434) — 关于现实音频为何非完美正弦波的权威参考。
- [Steve Eddins — FFT 解释笔记](https://blogs.mathworks.com/steve/2020/03/30/fft-spectrum-and-spectral-densities/) — 10分钟搞定频率箱的直观理解。
