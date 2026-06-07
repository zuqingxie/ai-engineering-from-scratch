# 音频评估 — WER（词错误率），MOS，UTMOS，MMAU，FAD 以及开放排行榜

> 你无法交付你无法衡量的东西。本课介绍 2026 年各类音频任务的核心指标：ASR（WER，CER，RTFx），TTS（MOS，UTMOS，SECS，ASR 往返WER），音频语言（MMAU，LongAudioBench），音乐（FAD，CLAP），以及说话人（EER）。还包括比对的排行榜。

**类型：** 学习  
**语言：** Python  
**先修条件：** 第6阶段 · 04, 06, 07, 09, 10；第2阶段 · 09（模型评估）  
**时长：** ~60分钟

## 问题

每个音频任务有多个指标，各自度量不同的维度。使用错误的指标会导致模型在仪表盘上表现良好，但在生产环境中效果糟糕。2026 年的规范清单：

| 任务 | 主要指标 | 次要指标 |
|------|---------|-----------|
| ASR | WER（词错误率） | CER（字符错误率）· RTFx（逆实时因子）· 首个标记延迟 |
| TTS | MOS（平均意见分） / UTMOS | SECS（说话人编码器余弦相似度）· ASR 往返WER · CER · TTFA（首次音频时延） |
| 语音克隆 | SECS（ECAPA 余弦相似度） | MOS · CER |
| 说话人验证 | EER（等错误率） | minDCF（最小检测成本）· FAR / FRR 于操作点 |
| 分离说话人 | DER（分离错误率） | JER （Jaccard 错误率）· 说话人混淆率 |
| 音频分类 | top-1 · mAP（平均准确率） | 宏观 F1 · 各类召回率 |
| 音乐生成 | FAD（Fréchet 音频距离） | CLAP · 听众评分 MOS |
| 音频语言模型 | MMAU-Pro | LongAudioBench · AudioCaps FENSE |
| 流式语音到语音 | 延迟 P50/P95 | WER · MOS |

## 概念

![音频评估矩阵 — 指标 vs 任务 vs 2026 排行榜](../assets/eval-landscape.svg)

### ASR 指标

**WER（词错误率）**。计算公式 `(S + D + I) / N`。评分前需转换小写，去除标点符号，数字归一化。推荐使用 `jiwer` 或 OpenAI 的 `whisper_normalizer`。低于 5% 接近人工阅读语音性能。

**CER（字符错误率）**。同公式，字符级别。适用于语调语言（如普通话、粤语）中，词边界划分模糊时使用。

**RTFx（逆实时因子）**。单位时间内处理的音频秒数。值越高越好。Parakeet-TDT 达到 3380×，Whisper-large-v3 约 30×。

**首个标记延迟**。从音频输入到第一个转录标记的真实时钟耗时。流式处理关键指标。Deepgram Nova-3 约 150 ms。

### TTS 指标

**MOS（平均意见分）**。1-5 的人工打分。黄金标准但速度慢。每个样本需 20+ 评审，模型需 100+ 样本。

**UTMOS（2022-2026）**。学习得到的 MOS 预测器。在标准基准上与人工 MOS 相关度约 0.9。F5-TTS 为 3.95，真实声音约 4.08。

**SECS（说话人编码器余弦相似度）**。语音克隆中使用。ECAPA 嵌入的参考声源与克隆输出间余弦相似度。> 0.75 表示可识别克隆。

**ASR 往返WER**。用 Whisper 对 TTS 输出音频进行转录，再与输入文本比对 WER。检测可懂度退化。2026 年 SOTA：CER < 2%。

**TTFA（首次音频时延）**。真实时钟延迟。Kokoro-82M 约 100 ms；F5-TTS 约 1 秒。

### 语音克隆专用

**SECS + MOS + CER** 三合一指标。高 SECS 但低 MOS 表示音色相符但不自然；反之则自然但说话人错误。

### 说话人验证

**EER（等错误率）**。假接受率（FAR）等于假拒绝率（FRR）的阈值。ECAPA 在 VoxCeleb1-O 上表现 0.87%。

**minDCF（最小检测成本）**。在某一操作点（常见 FAR=0.01）加权的成本。比 EER 更贴近生产环境。

### 说话人分离

**DER（分离错误率）**。计算 `(误报 + 漏报 + 混淆) / 总说话时间`。AMI 会议录音的真实值约 10-20%。pyannote 3.1 + Precision-2 商用版在高清录音中低于 10%。

**JER（Jaccard 错误率）**。DER 的替代指标，对短片段偏差稳健。

### 音频分类

多标签：所有类别的 **mAP（平均准确率）**。AudioSet 中 BEATs-iter3 达 0.548。

多分类独占：**top-1, top-5 准确率**。Speech Commands v2：99.0% top-1（Audio-MAE）。

类别不平衡时：**宏观 F1** 和 **每类召回率**。报告每类指标，整体准确率可能掩盖部分类别表现不佳。

### 音乐生成

**FAD（Fréchet 音频距离）**。真实与生成音频的 VGGish 嵌入分布距离。MusicGen-small 在 MusicCaps 上为 4.5，MusicLM 为 4.0，越低越好。

**CLAP 分数**。利用 CLAP 嵌入的文本与音频对齐分数。> 0.3 表示合理匹配。

**听众评分 MOS**。消费者级音乐质量最终标准。Suno v5 在 TTS Arena 的 ELO 1293 分（基于配对人工偏好）。

### 音频语言基准

**MMAU（Massive Multi-Audio Understanding）**。含 1 万个音频问答对。

**MMAU-Pro。** 1800 个难题，4 类别：语音 / 声音 / 音乐 / 多音频。4 选 1 随机准确率为 25%。Gemini 2.5 Pro 总体约 60%；多音频项约 22%。

**LongAudioBench。** 多分钟级长音频与语义查询。Audio Flamingo Next 优于 Gemini 2.5 Pro。

**AudioCaps / Clotho。** 标注基准。指标包括 SPICE，CIDEr，FENSE。

### 流式语音到语音

**延迟 P50 / P95 / P99。** 从用户话语结束到助手首个可听响应的真实时钟时延。Moshi：200 ms；GPT-4o Realtime：300 ms。

**输出 WER / MOS**。

**插话响应性。** 用户中断到助手静音的时间。目标 < 150 ms。

### 2026 排行榜

| 排行榜 | 赛道 | 链接 |
|------------|--------|-----|
| Open ASR Leaderboard（HF） | 英文 + 多语种 + 长文本 | `huggingface.co/spaces/hf-audio/open_asr_leaderboard` |
| TTS Arena（HF） | 英文 TTS | `huggingface.co/spaces/TTS-AGI/TTS-Arena` |
| Artificial Analysis Speech | TTS + STT，配对投票计算ELO | `artificialanalysis.ai/speech` |
| MMAU-Pro | LALM 推理 | `mmaubenchmark.github.io` |
| SpeakerBench / VoxSRC | 说话人识别 | `voxsrc.github.io` |
| MMAU 音乐子集 | 音乐 LALM | （MMAU 内部） |
| HEAR benchmark | 自监督音频 | `hearbenchmark.com` |

## 实现步骤

### 第1步：含归一化的 WER

```python
from jiwer import wer, Compose, ToLowerCase, RemovePunctuation, Strip

transform = Compose([ToLowerCase(), RemovePunctuation(), Strip()])
score = wer(
    truth="Please turn on the lights.",
    hypothesis="please turn on the light",
    truth_transform=transform,
    hypothesis_transform=transform,
)
# ~0.17
```

### 第2步：TTS 往返 WER

```python
def ttr_wer(tts_model, asr_model, texts):
    errors = []
    for txt in texts:
        audio = tts_model.synthesize(txt)
        recog = asr_model.transcribe(audio)
        errors.append(wer(truth=txt, hypothesis=recog))
    return sum(errors) / len(errors)
```

### 第3步：语音克隆的 SECS

```python
from speechbrain.inference.speaker import EncoderClassifier
sv = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb")

emb_ref = sv.encode_batch(load_wav("reference.wav"))
emb_clone = sv.encode_batch(load_wav("cloned.wav"))
secs = torch.nn.functional.cosine_similarity(emb_ref, emb_clone, dim=-1).item()
```

### 第4步：音乐生成的 FAD

```python
from frechet_audio_distance import FrechetAudioDistance
fad = FrechetAudioDistance()
score = fad.get_fad_score("generated_folder/", "reference_folder/")
```

### 第5步：说话人验证 EER（同第6课代码）

```python
def eer(same_scores, diff_scores):
    thresholds = sorted(set(same_scores + diff_scores))
    best = (1.0, 0.0)
    for t in thresholds:
        far = sum(1 for s in diff_scores if s >= t) / len(diff_scores)
        frr = sum(1 for s in same_scores if s < t) / len(same_scores)
        if abs(far - frr) < best[0]:
            best = (abs(far - frr), (far + frr) / 2)
    return best[1]
```

## 使用方法

每次部署都要配对一个固定的评估机制，覆盖所有模型更新。三条基本规则：

1. **评分前先归一化。** 转小写，去标点，数字展开。明确报告归一化规则。
2. **报告分布而非平均值。** 延迟用 P50/P95/P99 表示。分类用每类召回率。MMAU 按类别报告。
3. **至少运行一个规范的公开基准。** 即使生产数据不同，在 Open ASR / TTS Arena / MMAU 上报告，方便评审横向比较。

## 陷阱

- **UTMOS 外推风险。** 训练于 VCTK 类似的清晰语音，噪音／克隆／情绪音效表现不好。
- **MOS 听众面板偏差。** 20 名 Amazon Mechanical Turk 评测者 ≠ 20 名目标用户。重要场景需付费定制领域面板。
- **FAD 依赖参考集。** 不同模型对比时必须用同一参考集。
- **总体 WER 聚合误导。** 5% 总体 WER 可能掩盖 30% 带口音语音错误率。分人口统计报告更合理。
- **公开基准饱和。** 前沿模型多数已接近公开基准顶点。要搭建符合自身流量的内部保留数据集。

## 交付标准

保存为 `outputs/skill-audio-evaluator.md`。为任意音频模型发布挑选指标、基准和报告格式。

## 练习

1. **简单。** 运行 `code/main.py`。计算玩具输入上的 WER / CER / EER / SECS / FAD-ish / MMAU-ish。
2. **中等。** 构建 TTS 往返 WER 测评模块。用 Whisper 转录 Kokoro 或 F5-TTS 输出。对 50 条提示计算 WER。标记出 WER > 10% 的提示。
3. **困难。** 在 MMAU-Pro 的语音 + 多音频子集中（各 50 项）测试你第10课的 LALM 选择。报告按类别准确率，和发表成绩比较。

## 关键词汇

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|-----------------------|
| WER | ASR 得分 | 归一化后，词层面 `(S+D+I)/N`。 |
| CER | 字符级 WER | 用于语调语言或字符级系统。 |
| MOS | 人工评分 | 1-5 分；20+ 人 × 100 样本。 |
| UTMOS | 机器学习的 MOS 预测 | 学习模型；与人工 MOS 相关度约 0.9。 |
| SECS | 语音克隆相似度 | 参考与克隆之间 ECAPA 嵌入余弦相似度。 |
| EER | 说话人验证得分 | FAR 和 FRR 交叉点阈值。 |
| DER | 说话人分离得分 | (误报 + 漏报 + 混淆) / 总时长。 |
| FAD | 音乐生成质量 | VGGish 嵌入上的 Fréchet 距离。 |
| RTFx | 吞吐率 | 每秒钟处理的音频秒数。 |

## 延伸阅读

- [jiwer](https://github.com/jitsi/jiwer) — 带归一化工具的 WER/CER 库。  
- [UTMOS (Saeki 等，2022)](https://arxiv.org/abs/2204.02152) — 学习型 MOS 预测器。  
- [Fréchet Audio Distance (Kilgour 等，2019)](https://arxiv.org/abs/1812.08466) — 音乐生成标准。  
- [Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) — 2026 实时排名。  
- [TTS Arena](https://huggingface.co/spaces/TTS-AGI/TTS-Arena) — 人工投票 TTS 排行榜。  
- [MMAU-Pro benchmark](https://mmaubenchmark.github.io/) — LALM 推理排行榜。  
- [HEAR benchmark](https://hearbenchmark.com/) — 音频自监督学习基准。
