# 语音识别（ASR）— CTC、RNN-T、Attention（注意力机制）

> 语音识别是在每个时间步进行音频分类，由一个了解英语和静音的序列模型连接起来的过程。CTC、RNN-T 和 Attention 是实现这一目标的三种方法。选择一种并理解其原理。

**类型：** 构建  
**编程语言：** Python  
**先决条件：** 第6阶段 · 02（谱图和梅尔频率）、第5阶段 · 08（文本的 CNN 和 RNN）、第5阶段 · 10（注意力）  
**时间：** 约45分钟

## 问题描述

你有一个10秒钟、16 kHz的音频片段。你想要一个字符串："turn on the kitchen lights"。挑战在于结构上：音频帧与字符之间并非一一对应。单词 "okay" 可能持续200毫秒或1200毫秒。话语中穿插着静音。一些音素比其他音素持续时间更长。输出标记的数量事先未知。

有三种方法解决此问题：

1. **CTC（Connectionist Temporal Classification，连接时序分类）。** 针对每帧发出包含特殊*空白(blank)*标记的概率。解码时合并重复和空白。非自回归，速度快。wav2vec 2.0、MMS采用此方法。  
2. **RNN-T（Recurrent Neural Network Transducer，循环神经网络转导器）。** 联合网络根据编码器的帧和之前的标记预测下一个标记。可流式处理。Google设备端ASR、NVIDIA Parakeet采用此方法。  
3. **Attention（注意力机制）编码器-解码器。** 编码器将音频压缩为隐状态，解码器通过交叉注意力自回归生成标记。Whisper、SeamlessM4T采用此方法。

截至2026年，LibriSpeech test-clean数据集上的最新字错误率（WER）为1.4%（Parakeet-TDT-1.1B，NVIDIA）和1.58%（Whisper-Large-v3-turbo）。差别极小，但部署方式差异巨大。

## 概念介绍

![三种ASR方案：CTC、RNN-T、Attention编码器-解码器](../assets/asr-formulations.svg)

**CTC直觉。** 让编码器输出 `T` 帧的、涵盖 `V+1` 个标记的帧级分布（V个字符 + 空白(blank)）。对于长度为 `U < T` 的目标字符串 `y`，所有可折叠到 `y` 的帧对齐方式都计算在内。CTC损失对所有此类对齐方式求和。推理时：每帧取最大概率，合并重复，去除空白。

优点：非自回归，支持流式，无需向前看。缺点：*条件独立假设*——每帧预测相互独立，没有内部语言模型。可以通过束搜索或浅层融合结合外部LM解决。

**RNN-T直觉。** 添加一个*预测器*网络嵌入标记历史，一个*连接器*将预测器状态与编码器帧结合成联合分布，覆盖 `V+1` 个标记（其中 `+1` 表示无输出）。显式建模CTC忽略的条件依赖。流式可用，因为每步仅条件于过去的帧和标记。

优点：支持流式且内置语言模型。缺点：训练复杂且内存需求高（3D损失格）；RNN-T损失核是一整类专业库。

**Attention编码器-解码器。** 编码器采用6-32层Transformer处理对数梅尔谱图。解码器通过交叉注意力解码器输出，自回归生成标记。无对齐限制——注意力可覆盖任意音频位置。非流式，除非限制注意力范围（如2024年chunked Whisper-Streaming）。

优点：离线ASR质量最高，容易用标准seq2seq工具训练。缺点：自回归延迟与输出长度成正比；不做工程处理无法流式。

### WER：唯一的数字

**字错误率（Word Error Rate，WER）** = `(S + D + I) / N`，其中 S=替换数，D=删除数，I=插入数，N=参考词数。等同于词级Levenshtein编辑距离。数值越低越好。WER超过20%通常不可用；低于5%则为阅读语音的人类水平。2026年标准基准上的成绩：

| 模型 | LibriSpeech test-clean | LibriSpeech test-other | 模型大小 |
|-------|------------------------|------------------------|------|
| Parakeet-TDT-1.1B | 1.40% | 2.78% | 11亿参数 |
| Whisper-Large-v3-turbo | 1.58% | 3.03% | 8.09亿 |
| Canary-1B Flash | 1.48% | 2.87% | 10亿 |
| Seamless M4T v2 | 1.7% | 3.5% | 23亿 |

以上均为编码器-解码器或RNN-T架构。纯CTC系统（wav2vec 2.0）在test-clean表现约为1.8–2.1%。

## 构建流程

### 第1步：贪心CTC解码

```python
def ctc_greedy(frame_logits, blank=0, vocab=None):
    # frame_logits: 每帧概率向量列表
    preds = [max(range(len(p)), key=lambda i: p[i]) for p in frame_logits]
    out = []
    prev = -1
    for p in preds:
        if p != prev and p != blank:
            out.append(p)
        prev = p
    return "".join(vocab[i] for i in out) if vocab else out
```

两条规则：合并连续重复，丢弃空白。示例：`a a _ _ a b b _ c` → `a a b c`。

### 第2步：CTC束搜索

```python
def ctc_beam(frame_logits, beam=8, blank=0):
    import math
    beams = [([], 0.0)]  # (tokens, log_prob)
    for p in frame_logits:
        log_p = [math.log(max(pi, 1e-10)) for pi in p]
        candidates = []
        for seq, lp in beams:
            for t, lpt in enumerate(log_p):
                new = seq[:] if t == blank else (seq + [t] if not seq or seq[-1] != t else seq)
                candidates.append((new, lp + lpt))
        candidates.sort(key=lambda x: -x[1])
        beams = candidates[:beam]
    return beams[0][0]
```

实际生产中使用带LM融合的前缀树束搜索；此处为概念骨架。

### 第3步：计算WER

```python
def wer(ref, hyp):
    r, h = ref.split(), hyp.split()
    dp = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        dp[i][0] = i
    for j in range(len(h) + 1):
        dp[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[len(r)][len(h)] / max(1, len(r))
```

### 第4步：针对 Whisper 进行推理

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe("clip.wav")
print(result["text"])
```

2026年最强通用ASR的一行代码示例。能在24 GB GPU上以约20倍实时速率运行。

### 第5步：使用 Parakeet 或 wav2vec 2.0 流式识别

```python
from transformers import pipeline
asr = pipeline("automatic-speech-recognition", model="nvidia/parakeet-tdt-1.1b")
for chunk in streaming_audio():
    print(asr(chunk, return_timestamps=True))
```

流式识别需求编码器注意力分块和状态传递；使用支持的库（Parakeet的NeMo，或带有`chunk_length_s`参数的`transformers`管线）。

## 使用建议

2026年技术栈：

| 场景 | 推荐选择 |
|-----------|------|
| 英语，离线，最高质量 | Whisper-large-v3-turbo |
| 多语言，鲁棒性 | SeamlessM4T v2 |
| 流式，低延迟 | Parakeet-TDT-1.1B 或 Riva |
| 边缘，移动端，<500 ms延迟 | Whisper-Tiny量化版或Moonshine（2024年） |
| 长文档 | Whisper结合基于VAD的分块（WhisperX） |
| 特定领域（医疗、法律） | 微调wav2vec 2.0 + 领域LM融合 |

## 2026年仍常见的坑

- **无VAD（Voice Activity Detector，语音活动检测）。** 对静音段运行 Whisper 会出现幻听（如“Thanks for watching!”）。务必搭配VAD使用。  
- **字符、单词或子词WER的差异。** 报告时应针对*词级WER*且先进行文本规范化（小写、去除标点）。  
- **语言识别漂移。** Whisper的自动语言识别会将嘈杂音频误判为日语或威尔士语；已知语言时请强制`language="en"`。  
- **长音频无分块。** Whisper窗口为30秒。处理更长音频时使用`chunk_length_s=30, stride=5`。  

## 发布建议

保存为 `outputs/skill-asr-picker.md`。根据部署目标选择模型、解码策略、分块以及LM融合方式。

## 练习

1. **简单。** 运行 `code/main.py`。它对手工制作的CTC输出进行贪心解码并与参考计算WER。  
2. **中等。** 正式实现第2步的前缀树束搜索（考虑空白合并规则）。在10个合成示例数据集上与贪心解码比较。  
3. **困难。** 在[LibriSpeech test-clean](https://www.openslr.org/12)上运行`whisper-large-v3-turbo`，计算前100句的WER。与已发布结果对比。

## 关键词汇

| 术语 | 通俗说法 | 实际含义 |
|------|----------|----------|
| CTC | 空白标记损失 | 对所有帧到标记对齐的边际概率求和；非自回归。 |
| RNN-T | 流式损失 | CTC+下一标记预测器；处理词序关系。 |
| Attention enc-dec | Whisper式 | 编码器 + 交叉注意力解码器；最佳离线质量。 |
| WER | 报告数字 | 词级 `(S+D+I)/N`。 |
| Blank | 空白 | CTC中特殊标记，表示该帧无输出。 |
| LM fusion | 外部语言模型 | 束搜索中加权叠加语言模型对数概率。 |
| VAD | 静音门控 | 语音活动检测器，剪除非语音段。 |

## 延伸阅读

- [Graves et al. (2006). Connectionist Temporal Classification](https://www.cs.toronto.edu/~graves/icml_2006.pdf) — CTC论文。  
- [Graves (2012). Sequence Transduction with RNNs](https://arxiv.org/abs/1211.3711) — RNN-T论文。  
- [Radford et al. / OpenAI (2022). Whisper: Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) — 2022年权威论文；2024年v3-turbo扩展。  
- [NVIDIA NeMo — Parakeet-TDT卡](https://huggingface.co/nvidia/parakeet-tdt-1.1b) — 2026年开源ASR榜首。  
- [Hugging Face — Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) — 涵盖25+模型的实时排行榜。
