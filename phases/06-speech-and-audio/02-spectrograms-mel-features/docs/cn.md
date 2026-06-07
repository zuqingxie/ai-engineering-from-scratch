# 频谱图、梅尔尺度与音频特征

> 神经网络无法直接有效处理原始波形（raw waveforms）。它们处理频谱图（spectrograms），处理梅尔频谱图（mel spectrograms）则更佳。每一个2026年的自动语音识别（ASR）、文本转语音（TTS）和音频分类器的成败，都取决于这一单一的预处理选择。

**类型：** 构建  
**语言：** Python  
**先决条件：** 阶段 6 · 01（音频基础）  
**时间：** 约 45 分钟

## 问题

取一个10秒、16 kHz的音频片段。这是16万浮点数，全在`[-1, 1]`范围内，几乎完全与标签“狗叫”或“单词猫”无关。原始波形包含信息，但模型很难直接提取。相同的两个音素若相隔100毫秒，原始采样就完全不同。

频谱图解决了这一问题。它将人类感知忽略的时间细节（如微秒级抖动）折叠掉，同时保留人类感知关注的结构（在约10–25毫秒时间窗内，各频率的能量）。

梅尔频谱图更进一步。人类对音高是对数感知的：100 Hz与200 Hz听起来与1000 Hz与2000 Hz“距离相同”。梅尔刻度对频率轴做了变换以匹配这一感知规则。梅尔频谱图是2010年至2026年语音机器学习中最重要的特征。

## 概念

![从波形到STFT再到梅尔频谱图和MFCC阶梯图](../assets/mel-features.svg)

**STFT（短时傅里叶变换）。** 将波形切成重叠帧（典型窗长为25毫秒，跳跃为10毫秒，即16 kHz下400采样点/160采样点）。每帧乘以一个窗函数（默认汉宁窗 Hann，海明窗 Hamming 有稍微不同的权衡）。对每帧做FFT。将幅度谱堆叠成 `(n_frames, n_freq_bins)` 矩阵，即频谱图。

**对数幅度。** 原始幅度跨越5~6个数量级。取 `log(|X| + 1e-6)` 或 `20 * log10(|X|)` 来压缩动态范围。每个生产流水线都使用对数幅度，而非原始幅度。

**梅尔尺度（Mel scale）。** 频率 `f`（Hz）映射到梅尔刻度 `m` 公式为：`m = 2595 * log10(1 + f / 700)`。低于1 kHz部分大致线性，高于1 kHz部分近似对数。80个梅尔滤波器覆盖0–8 kHz是ASR的标准输入。

**梅尔滤波器组。** 一组在梅尔刻度上均匀分布的三角形滤波器。每个滤波器是相邻FFT频点的加权和。将STFT幅度乘以滤波器组矩阵，得到梅尔频谱图，等价一次矩阵乘法。

**对数梅尔频谱图。** `log(mel_spec + 1e-10)`。Whisper、Parakeet、SeamlessM4T的输入。2026通用的音频前端。

**MFCC（Mel-frequency cepstral coefficients）。** 对对数梅尔频谱图做DCT（类型II），保留前13个系数。该操作消除特征相关性并进一步压缩。直到约2015年CNN/Transformer在原始对数梅尔上的表现赶超它。仍用于说话人识别（如x-vectors、ECAPA）。

**分辨率权衡。** 更大FFT带来更好的频率分辨率，但时间分辨率更差。25 ms / 10 ms为音频机器学习默认；50 ms / 12.5 ms用于音乐；5 ms / 2 ms用于瞬态检测（如鼓点、破裂音）。

## 构建它

### 步骤1：分帧波形

```python
def frame(signal, frame_len, hop):
    n = 1 + (len(signal) - frame_len) // hop
    return [signal[i * hop : i * hop + frame_len] for i in range(n)]
```

一个10秒、16 kHz的波形，`frame_len=400, hop=160`时，会得到998帧。

### 步骤2：汉宁窗

```python
import math

def hann(N):
    return [0.5 * (1 - math.cos(2 * math.pi * n / (N - 1))) for n in range(N)]
```

在FFT前做元素逐点乘。消除因截断非零端点产生的频谱泄漏。

### 步骤3：STFT幅度

```python
def stft_magnitude(signal, frame_len=400, hop=160):
    win = hann(frame_len)
    frames = frame(signal, frame_len, hop)
    return [magnitudes(dft([w * s for w, s in zip(win, f)])) for f in frames]
```

真正的生产通常调用 `torch.stft` 或 `librosa.stft`（基于FFT，矢量化实现）。这里用循环是教学目的，适合运行于`code/main.py`中短片段。

### 步骤4：梅尔滤波器组

```python
def hz_to_mel(f):
    return 2595.0 * math.log10(1.0 + f / 700.0)

def mel_to_hz(m):
    return 700.0 * (10 ** (m / 2595.0) - 1)

def mel_filterbank(n_mels, n_fft, sr, fmin=0, fmax=None):
    fmax = fmax or sr / 2
    mels = [hz_to_mel(fmin) + (hz_to_mel(fmax) - hz_to_mel(fmin)) * i / (n_mels + 1)
            for i in range(n_mels + 2)]
    hzs = [mel_to_hz(m) for m in mels]
    bins = [int(h * n_fft / sr) for h in hzs]
    fb = [[0.0] * (n_fft // 2 + 1) for _ in range(n_mels)]
    for m in range(n_mels):
        for k in range(bins[m], bins[m + 1]):
            fb[m][k] = (k - bins[m]) / max(1, bins[m + 1] - bins[m])
        for k in range(bins[m + 1], bins[m + 2]):
            fb[m][k] = (bins[m + 2] - k) / max(1, bins[m + 2] - bins[m + 1])
    return fb
```

使用`n_fft=400`时，80个梅尔滤波器覆盖0–8 kHz得到一个 `(80, 201)` 矩阵。将维度为 `(n_frames, 201)` 的STFT幅度矩阵乘以该矩阵转置，得到 `(n_frames, 80)` 的梅尔频谱图。

### 步骤5：对数梅尔

```python
def log_mel(mel_spec, eps=1e-10):
    return [[math.log(max(v, eps)) for v in frame] for frame in mel_spec]
```

常用替代方案：`librosa.power_to_db`（基于参考归一化的分贝）或 `10 * log10(power + eps)`。Whisper使用了更复杂的截断+归一化过程（参见Whisper的`log_mel_spectrogram`）。

### 步骤6：MFCC

```python
def dct_ii(x, n_coeffs):
    N = len(x)
    return [
        sum(x[n] * math.cos(math.pi * k * (2 * n + 1) / (2 * N)) for n in range(N))
        for k in range(n_coeffs)
    ]
```

对每帧对数梅尔频谱做DCT，保留前13个系数。这是MFCC矩阵。常常丢弃第一个系数（代表整体能量）。

## 使用它

2026年技术栈：

| 任务 | 特征 |
|------|------|
| 自动语音识别（Whisper, Parakeet, SeamlessM4T） | 80个对数梅尔，10 ms跳跃，25 ms窗长 |
| TTS声学模型（VITS, F5-TTS, Kokoro） | 80个梅尔，5–12 ms跳跃实现细粒度时间控制 |
| 音频分类（AST, PANNs, BEATs） | 128个对数梅尔，10 ms跳跃 |
| 说话人嵌入（ECAPA-TDNN, WavLM） | 80个对数梅尔或原始波形自监督学习（SSL） |
| 音乐（MusicGen, Stable Audio 2） | EnCodec离散编码（非梅尔） |
| 关键词检测 | 40个MFCC，适合微型设备 |

经验法则：**除非做音乐，否则从80个对数梅尔开始。**任何偏离都需要有足够的理由。

## 2026年仍然存在的坑

- **梅尔数量不匹配。** 训练时使用80个梅尔，推理时使用128个。悄无声息的错误。两端都记录特征尺寸。
- **上游采样率不匹配。** 22.05 kHz生成的梅尔与16 kHz不同。采样率问题必须在特征提取之前修正。
- **dB与log的混淆。** Whisper期望对数梅尔，而非dB梅尔。一些HF（Hugging Face）流水线自动检测，但自定义代码不会。
- **归一化漂移。** 训练时按语句归一化，推理时用全局归一化。生产Bug导致词错误率（WER）翻倍。
- **填充泄漏。** 结尾零填充会在末尾帧生成平坦谱。应对称填充或重复边界值。

## 交付

保存为 `outputs/skill-feature-extractor.md`。该技能根据模型目标选择特征类型、梅尔数量、帧长/跳跃以及归一化方法。

## 练习

1. **简单。** 运行 `code/main.py`。它合成一个频率从200 Hz扫到4000 Hz的啁啾信号，打印每帧最大梅尔bin索引。绘图（选做）并确认匹配扫频。
2. **中等。** 修改 `n_mels` 在 `{40, 80, 128}` 和 `frame_len` 在 `{200, 400, 800}` 中组合，测量沿时间轴的带宽锐度。哪个组合对啁啾信号分辨率最好？
3. **困难。** 实现`power_to_db`，并比较一个小型CNN分类器在AudioMNIST上的ASR准确率，使用（a）原始对数梅尔，（b）归一化dB梅尔（`ref=max`），以及（c）13维MFCC + 一阶差分 + 二阶差分。报告top-1准确率。

## 关键词

| 术语 | 常见说法 | 实际含义 |
|------|----------|---------|
| Frame（帧） | 一个切片 | 25毫秒的波形块，送入一次FFT。 |
| Hop（跳跃） | 步幅 | 连续帧之间采样点数；10毫秒是ASR默认。 |
| Window（窗） | 汉宁／海明窗 | 对帧边缘做点乘，渐零化，减少泄漏。 |
| STFT | 频谱图生成器 | 分帧加窗后FFT，得到时频矩阵。 |
| Mel | 频率变换 | 对数听感尺度，`m = 2595·log10(1 + f/700)`。 |
| Filterbank（滤波器组） | 矩阵 | 三角滤波器将STFT映射到梅尔bin。 |
| Log-mel（对数梅尔） | Whisper输入 | `log(mel_spec + eps)`；2026年标准化特征。 |
| MFCC | 传统特征 | 对数梅尔的DCT，保留13系数，去相关。 |

## 延伸阅读

- [Davis, Mermelstein (1980). Comparison of parametric representations for monosyllabic word recognition](https://ieeexplore.ieee.org/document/1163420) — MFCC论文。  
- [Stevens, Volkmann, Newman (1937). A Scale for the Measurement of the Psychological Magnitude Pitch](https://pubs.aip.org/asa/jasa/article-abstract/8/3/185/735757/) — 最初的梅尔刻度。  
- [OpenAI — Whisper源码，log_mel_spectrogram](https://github.com/openai/whisper/blob/main/whisper/audio.py) — 参考实现。  
- [librosa特征提取文档](https://librosa.org/doc/main/feature.html) — `mfcc`，`melspectrogram`，跳跃和窗函数参考。  
- [NVIDIA NeMo — 音频预处理](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/asr_all.html#featurizers) — Parakeet和Canary模型的生产级流水线。
