# 语音克隆（Voice Cloning）与语音转换（Voice Conversion）

> 语音克隆是用别人的声音朗读你的文本。语音转换是将你的声音转换成别人的声音，同时保持你说的内容。两者都基于相同的分解思想：将说话者身份与内容分离。

**类型：** 构建
**语言：** Python
**先决条件：** 第6阶段 · 06（说话者识别），第6阶段 · 07（文本转语音 TTS）
**时间：** 约75分钟

## 问题背景

在2026年，一段5秒的音频足以用消费级GPU生成任何人的高质量声音克隆。ElevenLabs、F5-TTS、OpenVoice v2、VoiceBox等均支持零样本（zero-shot）或少样本（few-shot）克隆。该技术既是福音（无障碍TTS、配音、辅助语音），也是武器（诈骗电话、政治深度伪造、知识产权盗窃）。

两个密切相关的任务：

- **语音克隆（文本侧 TTS）**：文本 + 5秒参考声音 → 生成该声音的语音。
- **语音转换（语音侧）**：源音频（A说X）+ 目标说话者B的参考声音 → 生成B说X的音频。

两者都将波形分解为（内容（content）、说话者（speaker）、韵律（prosody）），并将某个源的内容与另一个说话者的声音重新组合。

2026年你必须遵守的主要限制：**欧盟（AI法案，2026年8月生效）和加州（AB 2905，2025年生效）法律要求水印（watermarking）和同意门控（consent gates）**。你的流水线必须输出不可察觉的水印，并拒绝未经同意的克隆。

## 概念

![语音克隆与语音转换：分解、交换说话者、重组](../assets/voice-cloning.svg)

**零样本克隆。** 输入5秒音频片段给已训练数千说话者的模型。说话者编码器将该片段映射为说话人嵌入（speaker embedding）；TTS解码器以该嵌入和文本为条件生成语音。

应用：F5-TTS（2024）、YourTTS（2022）、XTTS v2（2024）、OpenVoice v2（2024）。

**少样本微调。** 录制5至30分钟目标声音。用LoRA微调基础模型1小时。质量从“还行”跃升至“无法区分”。Coqui和ElevenLabs支持此方案，社区用F5-TTS实现。

**语音转换（VC）。** 两类方法：

- **识别-合成（Recognition-synthesis）。** 运行类似自动语音识别（ASR）的模型提取内容表示（如软音素后验概率、PPG），再用目标说话者嵌入重合成。对语言和口音鲁棒。代表作：KNN-VC（2023）、Diff-HierVC（2023）。
- **解耦（Disentanglement）。** 训练自编码器，在瓶颈层将内容、说话者和韵律分离。推理时交换说话者嵌入。质量较低但速度快。代表作：AutoVC（2019）、VITS-VC变体。

**基于神经编解码器的克隆（2024+）。** VALL-E、VALL-E 2、NaturalSpeech 3、VoiceBox——将音频视为SoundStream/EnCodec的离散token，训练大规模自回归或流匹配模型。短文本质量可比ElevenLabs。

### 伦理部分，非附加功能

**水印（Watermarking）。** PerTh（Perth）和SilentCipher（2024）将约16-32位ID无感知地嵌入音频。能抵抗重编码、流传输及常规编辑。开源且可生产应用。

**同意门控（Consent gates）。** 每个克隆输出必须配对可验证的同意记录。比如 “我，Rohit，2026-04-22，授权此声音用于X用途。” 存储在防篡改日志中。

**检测（Detection）。** AASIST、RawNet2和Wav2Vec2-AASIST作为检测器发布。ASVspoof 2025挑战赛公布了针对ElevenLabs、VALL-E 2和Bark合成音的最先进检测器的错误接受率（EER）在0.8%至2.3%之间。

### 数据指标（2026）

| 模型 | 零样本？ | SECS（目标相似度） | WER（智能识别率） | 参数量 |
|-------|-----------|--------------------|--------------|--------|
| F5-TTS | 有 | 0.72 | 2.1% | 335M |
| XTTS v2 | 有 | 0.65 | 3.5% | 470M |
| OpenVoice v2 | 有 | 0.70 | 2.8% | 220M |
| VALL-E 2 | 有 | 0.77 | 2.4% | 370M |
| VoiceBox | 有 | 0.78 | 2.1% | 330M |

SECS > 0.70对于大多数听众而言通常无法与目标音区分。

## 构建它

### 步骤1：用识别-合成分解（仅代码演示，见 main.py）

```python
def clone_pipeline(ref_audio, text, target_embedder, tts_model):
    speaker_emb = target_embedder.encode(ref_audio)
    mel = tts_model(text, speaker=speaker_emb)
    return vocoder(mel)
```

概念简单；实现重点在于`tts_model`和说话者编码器。

### 步骤2：用F5-TTS实现零样本克隆

```python
from f5_tts.api import F5TTS
tts = F5TTS()
wav = tts.infer(
    ref_file="rohit_5s.wav",
    ref_text="The quick brown fox jumps over the lazy dog.",
    gen_text="Please add milk and bread to my list.",
)
```

参考文本必须与音频完全匹配；不匹配会导致对齐失败。

### 步骤3：用KNN-VC实现语音转换

```python
import torch
from knnvc import KNNVC  # 2023模型， https://github.com/bshall/knn-vc
vc = KNNVC.load("wavlm-base-plus")
out_wav = vc.convert(source="my_voice.wav", target_pool=["alice_1.wav", "alice_2.wav"])
```

KNN-VC运行WavLM提取源音频和目标池每帧嵌入，然后用目标池中最近邻帧替换源帧。非参数方法，可用一分钟目标语音。

### 步骤4：嵌入水印

```python
from silentcipher import SilentCipher
sc = SilentCipher(model="2024-06-01")
payload = b"consent_id:abc123;ts:1745353200"
watermarked = sc.embed(wav, sr=24000, message=payload)
detected = sc.detect(watermarked, sr=24000)   # 返回payload字节
```

约32位负载，能在MP3重编码和轻微噪声后检测。

### 步骤5：同意门控

```python
def cloned_inference(text, ref_audio, consent_record):
    assert verify_signature(consent_record), "需要签署同意"
    assert consent_record["speaker_id"] == hash_speaker(ref_audio)
    wav = tts.infer(ref_file=ref_audio, gen_text=text)
    wav = watermark(wav, payload=consent_record["id"])
    return wav
```

## 使用它

2026年方案：

| 场景 | 选择 |
|-----------|------|
| 5秒零样本克隆，开源 | F5-TTS 或 OpenVoice v2 |
| 商业生产克隆 | ElevenLabs Instant Voice Clone v2.5 |
| 语音转换（重写） | KNN-VC 或 Diff-HierVC |
| 多说话者微调 | StyleTTS 2 + 说话者适配器 |
| 跨语言克隆 | XTTS v2 或 VALL-E X |
| 深度伪造检测 | Wav2Vec2-AASIST |

## 注意事项

- **参考转录文本不对齐。** F5-TTS等要求参考文本与参考音频完全匹配，含标点符号。
- **混响参考音。** 回声破坏克隆效果。录制时应干净，靠近麦克风。
- **情感不匹配。** 训练参考带“高兴”情绪会导致所有克隆音都带该情绪。情绪应与目标应用匹配。
- **语言泄露。** 克隆了英语说话者后强制让模型说法语，仍可能带英语口音；要使用跨语言模型（XTTS、VALL-E X）。
- **无水印。** 2026年8月起欧盟禁止无水印产品合法发货。

## 交付它

保存为`outputs/skill-voice-cloner.md`。设计一个包含同意门控、水印和质量目标的克隆或转换流水线。

## 练习

1. **简单。** 运行`code/main.py`。展示说话者嵌入交换，并计算交换前后的余弦相似度。
2. **中等。** 用OpenVoice v2克隆你自己的声音。测量参考与克隆的SECS。用Whisper测量字符错误率（CER）。
3. **困难。** 将SilentCipher水印应用于20个克隆音频，经过128 kbps MP3编码解码后检测负载。报告比特准确率。

## 关键词

| 术语 | 常说法 | 实际含义 |
|------|--------|----------|
| Zero-shot clone | 5秒足够 | 预训练模型 + 说话者嵌入；无训练。 |
| PPG | 音素后验概率图 | 每帧ASR后验，用作语言无关的内容表示。 |
| KNN-VC | 最近邻转换 | 用目标池中最近的帧替换每个源帧。 |
| Neural codec TTS | VALL-E风格 | 在EnCodec/SoundStream token上训练自回归模型。 |
| Watermark | 无声签名 | 嵌入音频的比特，能通过重编码存活。 |
| SECS | 克隆相似度 | 目标与克隆说话者嵌入之间的余弦相似度。 |
| AASIST | 深度伪造检测器 | 反伪造模型；检测合成语音。 |

## 深入阅读

- [Chen et al. (2024). F5-TTS](https://arxiv.org/abs/2410.06885) — 开源最先进零样本克隆。
- [Baevski et al. / Microsoft (2023). VALL-E](https://arxiv.org/abs/2301.02111) 和 [VALL-E 2 (2024)](https://arxiv.org/abs/2406.05370) — 神经编解码器 TTS。
- [Qian et al. (2019). AutoVC](https://arxiv.org/abs/1905.05879) — 基于解耦的语音转换。
- [Baas, Waubert de Puiseau, Kamper (2023). KNN-VC](https://arxiv.org/abs/2305.18975) — 基于检索的语音转换。
- [SilentCipher (2024) — 音频水印](https://github.com/sony/silentcipher) — 生产级32位音频水印。
- [ASVspoof 2025结果](https://www.asvspoof.org/) — 检测器与合成器军备竞赛，2026年更新。
