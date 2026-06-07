# 说话人识别与验证

> 语音识别（ASR）问“他们说了什么？”说话人识别问“这是谁说的？”数学看上去一样——嵌入加余弦相似度——但每个生产决策都依赖于一个关键的EER数值。

**类型：** 构建  
**语言：** Python  
**先决条件：** 第6阶段 · 02（频谱图与梅尔频率），第5阶段 · 22（嵌入模型）  
**时间：** ~45分钟

## 问题描述

用户说出一串口令。你想知道：这是不是他们声称的人（*验证*，1:1），还是你的注册库里第一个人（*鉴定*，1:N）？或者都不是——这是一个未知说话者（*开放集*）？

2018年前：GMM-UBM + i-vectors。EER合理但对信道变化（手机与笔记本）和情绪脆弱。2018–2022：x-vectors（带有角度边缘的TDNN骨干训练）。2022年以后：ECAPA-TDNN和WavLM-large嵌入。到2026年，该领域由三种模型和一个度量指标主导。

度量指标是**EER**——等错误率。设置决策阈值，使误接受率（False Accept Rate）=误拒率（False Reject Rate）。交叉点即为EER。每篇论文、每个排行榜、每次采购都使用该指标。

## 概念

![入驻 + 验证流程采用嵌入 + 余弦 + EER](../assets/speaker-verification.svg)

**流程。** 入驻（Enrollment）：录制目标说话人的5–30秒音频；计算固定维度嵌入（ECAPA-TDNN为192维，WavLM-large为256维）。验证（Verification）：获取测试语音嵌入；计算余弦相似度；与阈值比较。

**ECAPA-TDNN（2020年，2026年仍然主导）。** 强调通道关注、传播与聚合的时间延迟神经网络。采用带压缩激励（squeeze-excitation）的一维卷积块，多头注意力池化，之后通过线性层降为192维。在VoxCeleb 1+2数据集（2700说话人，110万语句）上，用带加性角度边缘（AAM-softmax）的损失训练。

**WavLM-SV（2022+）。** 精调预训练的WavLM-large自监督学习骨干，采用AAM损失。质量更高但更慢——300+ MB vs 15 MB。

**x-vector（基线）。** TDNN + 统计池化。经典，仍然适用于CPU/边缘设备。

**AAM-softmax。** 标准softmax加上角度空间的边缘`m`：正确类别采用`cos(θ + m)`。强制类别间的角度分隔。典型为`m=0.2`，缩放`s=30`。

### 评分方法

- **余弦相似度（Cosine）**：在入驻和测试嵌入之间计算。基于阈值做决策。
- **PLDA（概率线性判别分析）。** 将嵌入投影到潜在空间，计算同说话人和不同说话人的封闭式似然比。叠加在余弦上可减少10–20% EER。2020年前标准方法；现在只用于闭集情况。
- **分数归一化。** `S-norm` 或 `AS-norm`：用一组伪装者的均值和标准差归一化分数。跨领域评估时必需。

### 你应该知道的数据（2026）

| 模型 | VoxCeleb1-O EER | 参数量 | 吞吐量（A100） |
|------|-----------------|--------|----------------|
| x-vector（经典） | 3.10% | 5 M | 400× 实时 |
| ECAPA-TDNN | 0.87% | 15 M | 200× 实时 |
| WavLM-SV large | 0.42% | 316 M | 20× 实时 |
| Pyannote 3.1 分割 + 嵌入 | 0.65% | 6 M | 100× 实时 |
| ReDimNet（2024） | 0.39% | 24 M | 100× 实时 |

### 说话人分离（Diarization）

多说话人录音中“谁在何时说话”。流程：语音活动检测（VAD） → 分段 → 为每段生成嵌入 → 聚类（聚合或谱聚类） → 边界平滑。2026年领先方案为`pyannote.audio` 3.1，一条调用完成说话人分割 + 嵌入 + 聚类。AMI数据集上的最新SOTA DER约为15%（2022年为23%）。

## 构建过程

### 步骤1：从MFCC统计量构造玩具嵌入

```python
def embed_mfcc_stats(signal, sr):
    frames = featurize_mfcc(signal, sr, n_mfcc=13)
    mean = [sum(f[i] for f in frames) / len(frames) for i in range(13)]
    std = [
        math.sqrt(sum((f[i] - mean[i]) ** 2 for f in frames) / len(frames))
        for i in range(13)
    ]
    return mean + std  # 26维
```

远非SOTA，仅作教学用途。`code/main.py`使用此方法对合成说话人数据做概念验证。

### 步骤2：余弦相似度 + 阈值

```python
def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0

def verify(enroll, test, threshold=0.75):
    return cosine(enroll, test) >= threshold
```

### 步骤3：从相似度对计算EER

```python
def eer(same_scores, diff_scores):
    thresholds = sorted(set(same_scores + diff_scores))
    best = (1.0, 1.0, 0.0)  # (错误接受率fa, 错误拒绝率fr, 阈值)
    for t in thresholds:
        fr = sum(1 for s in same_scores if s < t) / len(same_scores)
        fa = sum(1 for s in diff_scores if s >= t) / len(diff_scores)
        if abs(fa - fr) < abs(best[0] - best[1]):
            best = (fa, fr, t)
    return (best[0] + best[1]) / 2, best[2]
```

返回（eer, 阈值）。请同时报告。

### 步骤4：用SpeechBrain生产环境

```python
from speechbrain.pretrained import EncoderClassifier

clf = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

# 入驻：对3-5个干净样本的嵌入取平均
enroll = torch.stack([clf.encode_batch(load(x)) for x in enrollment_clips]).mean(0)
# 验证
score = clf.similarity(enroll, clf.encode_batch(load("test.wav"))).item()
verdict = score > 0.25   # ECAPA典型阈值；请根据你的数据调参
```

### 步骤5：用pyannote做说话人分离

```python
from pyannote.audio import Pipeline

pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
diarization = pipe("meeting.wav", num_speakers=None)
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"{turn.start:.1f}–{turn.end:.1f}  {speaker}")
```

## 使用建议

2026年推荐栈：

| 场景 | 选择 |
|------|------|
| 闭集1:1验证，边缘设备 | ECAPA-TDNN + 余弦阈值 |
| 开放集验证，云端 | WavLM-SV + AS归一化 |
| 说话人分离（会议、播客） | `pyannote/speaker-diarization-3.1` |
| 反欺骗（重放 / 深度伪造检测） | AASIST或RawNet2 |
| 小型嵌入式设备（关键词检测 + 入驻） | Titanet-Small（NeMo） |

## 陷阱

- **信道不匹配。** 模型在VoxCeleb（网页视频）训练，不等于电话语音。务必在目标信道评估。
- **短语音。** EER在测试音频短于3秒时急剧恶化。
- **有噪声的入驻。** 一条嘈杂的入驻录音会污染锚点。至少用3条干净样本并取平均。
- **跨条件固定阈值。** 始终在目标域的保留开发集上调参阈值。
- **非归一化嵌入上计算余弦。** 先做L2归一，否则幅值主导结果。

## 交付方案

保存为`outputs/skill-speaker-verifier.md`。选择模型、入驻协议、阈值调节方案及反欺骗措施。

## 练习

1. **简单。** 运行`code/main.py`。构建合成“说话人”（不同音色特征），入驻，计算100对测试列表的EER。
2. **中等。** 用SpeechBrain的ECAPA测试VoxCeleb1中30个语句（5个说话人，每人6条）。计算基于余弦和PLDA的EER。
3. **困难。** 构建完整的入驻 → 分离 → 验证流水线，使用`pyannote.audio`。在AMI开发集上评估DER。

## 关键词

| 术语 | 俗称 | 实际含义 |
|------|------|----------|
| EER | 核心指标 | 假接受率 = 假拒率时对应的阈值。 |
| Verification（验证） | 1:1 | “这是Alice吗？” |
| Identification（鉴定） | 1:N | “这是谁在讲话？” |
| Open-set（开放集） | 可能未知 | 测试集含未登记说话人。 |
| Enrollment（入驻） | 注册 | 计算说话人的参考嵌入。 |
| AAM-softmax | 损失函数 | 带加性角度边缘的softmax，强制聚类分离。 |
| PLDA | 经典评分 | 概率线性判别分析；对嵌入进行似然比评分。 |
| DER | 分离指标 | 分离错误率——漏检 + 误警 + 混淆。 |

## 拓展阅读

- [Snyder et al. (2018). X-Vectors: Robust DNN Embeddings for Speaker Recognition](https://www.danielpovey.com/files/2018_icassp_xvectors.pdf) — 经典深度嵌入论文。
- [Desplanques et al. (2020). ECAPA-TDNN](https://arxiv.org/abs/2005.07143) — 2020–2026年主导架构。
- [Chen et al. (2022). WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing](https://arxiv.org/abs/2110.13900) — 适用于说话人验证和分离的SSL骨干。
- [Bredin et al. (2023). pyannote.audio 3.1](https://github.com/pyannote/pyannote-audio) — 生产级分离 + 嵌入堆栈。
- [VoxCeleb排行榜（2026更新）](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/) — 各模型最新EER排名。
