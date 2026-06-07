# 音频分类 — 从基于 MFCC 的 k-NN 到 AST 和 BEATs

> 从“狗叫声与警报声”到“这是什么语言”，都是音频分类。特征是 mel 频率。架构每十年更新一次。评估指标保持 AUC、F1 和每类召回率。

**类型：** 构建  
**语言：** Python  
**先决条件：** 第6阶段 · 02（频谱图与 Mel）、第3阶段 · 06（卷积神经网络 CNN）、第5阶段 · 08（文本的 CNN 与循环神经网络 RNN）  
**时间：** ~75分钟

## 问题描述

你得到一个 10 秒音频片段。你想知道：“这是什么？” 城市场景声音（警报器、钻机、狗），语音命令（是/否/停止），语言识别（英语/西班牙语/阿拉伯语），说话者情绪（愤怒/中性），或环境声音（室内/室外、嘈杂声音）。这些都属于*音频分类*，到 2026 年，基线架构已成熟：log-mel → CNN 或 Transformer → softmax。

核心难点不是网络，而是数据。音频数据集类别极度不平衡，存在强烈的域偏移（干净音频与嘈杂音频）以及标签噪声（谁决定“城市场景嘈杂”和“餐厅噪声”的？）。80% 的问题在于数据整理、增强和评估，而不是换用 CNN 还是 Transformer。

## 概念

![音频分类阶梯：基于 MFCC 的 k-NN 到 AST 再到 BEATs](../assets/audio-classification.svg)

**基于 MFCC 的 k-NN（1990年代基线）。** 将每段音频的 MFCC 展平成向量，计算与带标签样本库的余弦相似度，返回前 K 个的多数投票。对干净、规模较小的数据集（如 Speech Commands 和 ESC-50）表现出乎意料地好。无需 GPU。

**基于 log-mels 的二维 CNN（2015-2019）。** 将 `(T, n_mels)` 的 log-mel 视作图像。应用 ResNet-18 或 VGG 类型网络。对时间维度使用全局均值池化。对类别使用 softmax。依然是多数 2026 年 Kaggle 比赛的基线。

**音频频谱图 Transformer，AST（2021-2024）。** 将 log-mel 切分为块（例如 16×16 的图像块），添加位置编码，输入给 ViT。监督学习下在 AudioSet（mAP 0.485）达到最先进水平。

**BEATs 和 WavLM-base（2024-2026）。** 基于数百万小时数据的自监督预训练。以 1-10% 的监督数据微调特定任务。2026 年这成为非语音音频的默认起点。BEATs-iter3 以四分之一的计算量超越 AST，在 AudioSet 上提升 1-2 mAP。

**Whisper 编码器作为冻结骨干（2024）。** 使用 Whisper 的编码器，去掉解码器，附加线性分类器。在语言 ID 和简单事件分类上接近 SOTA，无需音频增强，“免费午餐”基线。

### 类别不平衡才是真正挑战

ESC-50：50 个类别，每类 40 个样本——平衡且简单。  
UrbanSound8K：10 个类别，类别不平衡比例约为 10:1。  
AudioSet：632 个类别，长尾约 100,000:1。  
有效方法包括：

- 训练时均衡采样（评估时不采用）。
- Mixup：线性混合两段音频（和它们的标签），作为增强。
- SpecAugment：随机屏蔽时间和频率轴上的部分区域。简单且关键。

### 评估

- 多类独占（Speech Commands）：top-1 准确率，top-5 准确率。  
- 多类多标签（AudioSet、UrbanSound）：均值平均精度（mAP）。  
- 严重不平衡：每类召回率 + 宏 F1。

2026 年你应知晓的成绩：

| 基准测试 | 基线 | 2026 年最优 | 来源 |
|-----------|----------|-----------|--------|
| ESC-50 | 82%（AST） | 97.0%（BEATs-iter3） | BEATs 论文（2024） |
| AudioSet mAP | 0.485（AST） | 0.548（BEATs-iter3） | HEAR 2026 排行榜 |
| Speech Commands v2 | 98%（CNN） | 99.0%（Audio-MAE） | HEAR v2 结果 |

## 构建步骤

### 第1步：特征提取

```python
def featurize_mfcc(signal, sr, n_mfcc=13, n_mels=40, frame_len=400, hop=160):
    mag = stft_magnitude(signal, frame_len, hop)
    fb = mel_filterbank(n_mels, frame_len, sr)
    mels = apply_filterbank(mag, fb)
    log = log_transform(mels)
    return [dct_ii(frame, n_mfcc) for frame in log]
```

### 第2步：固定长度摘要

```python
def summarize(mfcc_frames):
    n = len(mfcc_frames[0])
    mean = [sum(f[i] for f in mfcc_frames) / len(mfcc_frames) for i in range(n)]
    var = [
        sum((f[i] - mean[i]) ** 2 for f in mfcc_frames) / len(mfcc_frames) for i in range(n)
    ]
    return mean + var
```

简单且强大：对时间维度求均值和方差，13 维 MFCC 得到 26 维固定向量。即时运行。至 2017 年，在 ESC-50 上击败了最先进的神经网络基线。

### 第3步：k-NN 分类

```python
def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-12
    nb = math.sqrt(sum(x * x for x in b)) or 1e-12
    return dot / (na * nb)

def knn_classify(q, bank, labels, k=5):
    sims = sorted(range(len(bank)), key=lambda i: -cosine(q, bank[i]))[:k]
    votes = Counter(labels[i] for i in sims)
    return votes.most_common(1)[0][0]
```

### 第4步：升阶到基于 log-mels 的 CNN

使用 PyTorch：

```python
import torch.nn as nn

class AudioCNN(nn.Module):
    def __init__(self, n_mels=80, n_classes=50):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(128, n_classes)

    def forward(self, x):  # x: (B, 1, T, n_mels)
        return self.head(self.body(x).flatten(1))
```

3M 参数量。使用单块 RTX 4090，训练 ESC-50 约 10 分钟。准确率超 80%。

### 第5步：2026 年默认选择 — 微调 BEATs

```python
from transformers import ASTFeatureExtractor, ASTForAudioClassification

ext = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
model = ASTForAudioClassification.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593",
    num_labels=50,
    ignore_mismatched_sizes=True,
)

inputs = ext(audio, sampling_rate=16000, return_tensors="pt")
logits = model(**inputs).logits
```

对于 BEATs，使用 `microsoft/BEATs-base` 通过 `beats` 库；transformers API 接口相同。

## 使用指南

2026 年栈：

| 情况 | 起点选择 |
|-----------|-----------|
| 小规模数据集（<1000 条音频） | 基于 MFCC 均值的 k-NN（你的基线）+ 音频增强 |
| 中等规模数据集（1K–100K） | BEATs 或 AST 微调 |
| 大规模数据集（>100K） | 从头训练或微调 Whisper 编码器 |
| 实时或边缘设备 | 40 维 MFCC 的 CNN，量化为 int8（关键词检测样式） |
| 多标签任务（AudioSet） | BEATs-iter3 配合 BCE 损失 + mixup + SpecAugment |
| 语言识别 | MMS-LID，SpeechBrain VoxLingua107 基线 |

决策规则：**从冻结骨干开始，而非新模型训练**。微调 BEATs 头部，几小时即可达到 95% 的最先进水平，不用几周。

## 发布指南

保存为 `outputs/skill-classifier-designer.md`。根据具体音频分类任务，选择架构、增强方法、类别平衡策略和评估指标。

## 练习题

1. **简单。** 运行 `code/main.py`。在 4 类合成数据集（不同音高的纯音）上训练基于 k-NN MFCC 的基线。报告混淆矩阵。  
2. **中等。** 用 [均值、方差、偏度、峰度] 替换 `summarize`。四矩池化是否优于同一合成数据集上的均值+方差？  
3. **困难。** 使用 `torchaudio`，在 ESC-50 的 fold 1 上训练二维 CNN。报告 5 折交叉验证准确率。加入 SpecAugment（时间遮罩 = 20，频率遮罩 = 10），报告准确率变化。

## 术语表

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| AudioSet | 音频领域的 ImageNet | 谷歌的 200 万剪辑、632 类弱监督 YouTube 数据集。 |
| ESC-50 | 小型分类基准 | 50 类，每类 40 个环境声音剪辑。 |
| AST | 音频频谱图 Transformer | 在 log-mel 拼块上应用 ViT；2021 年至 2024 年 SOTA。 |
| BEATs | 自监督音频模型 | 微软模型，2026 年 iter3 版本引领 AudioSet 任务。 |
| Mixup | 双样本增强 | `x = λ·x1 + (1-λ)·x2; y = λ·y1 + (1-λ)·y2`。 |
| SpecAugment | 基于掩膜的增强 | 随机屏蔽频谱图中的时间和频率区域。 |
| mAP | 主要多标签指标 | 各类和阈值的均值平均精度。 |

## 延伸阅读

- [Gong, Chung, Glass (2021). AST: Audio Spectrogram Transformer](https://arxiv.org/abs/2104.01778) — 2021 至 2024 年的明星架构。
- [Chen et al. (2022, rev. 2024). BEATs: Audio Pre-Training with Acoustic Tokenizers](https://arxiv.org/abs/2212.09058) — 2024 及以后默认方案。
- [Park et al. (2019). SpecAugment](https://arxiv.org/abs/1904.08779) — 主流的音频增强技术。
- [Piczak (2015). ESC-50 dataset](https://github.com/karolpiczak/ESC-50) — 依然流行的 50 类基准。
- [Gemmeke et al. (2017). AudioSet](https://research.google.com/audioset/) — 含 632 类的 YouTube 标签集；持续金标准。
