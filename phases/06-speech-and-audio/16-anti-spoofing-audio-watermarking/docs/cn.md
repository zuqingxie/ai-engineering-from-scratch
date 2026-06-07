# 语音反伪造（Voice Anti-Spoofing）与音频水印（Audio Watermarking） — ASVspoof 5、AudioSeal、WaveVerify

> 语音克隆的推出速度快过了防御措施。2026 年的生产语音系统需要两样东西：一个检测器（AASIST、RawNet2）用于区分真实语音与伪造语音；一个能够经受压缩和编辑的水印（AudioSeal）。要么同时交付，要么不交付语音克隆。

**类型：** 构建  
**语言：** Python  
**先决条件：** 第 6 阶段 · 06（说话人识别）、第 6 阶段 · 08（语音克隆）  
**时长：** 约 75 分钟

## 问题描述

三个相关防御：

1. **反伪造 / 深度伪造检测。** 给定一段音频，是合成的还是真实的？ASVspoof 基准（ASVspoof 2019 → 2021 → 5）是黄金标准。  
2. **音频水印。** 在生成的音频中嵌入不可察觉的信号，检测器随后能提取出来。AudioSeal（Meta）和 WavMark 是开放选项。  
3. **身份来源认证。** 对音频文件及元数据进行密码签名。C2PA / 内容真实性倡议（Content Authenticity Initiative）。  

检测针对不合作的对手。水印用于合规——AI 生成的音频应被识别为此类。2026 年两者都必须具备。

## 概念

![反伪造 vs 水印 vs 来源认证 — 三层防御](../assets/spoofing-watermark.svg)

### ASVspoof 5 — 2024-2025 基准

相比之前版本的最大变化：

- **众包数据**（非录音棚纯净环境）— 更真实条件。  
- **约 2000 名说话人**（之前约 100）。  
- **32 种攻击算法。** 包括 TTS（文本转语音）、语音转换、对抗扰动。  
- **两条赛道。** 反措施（CM）独立检测；针对生物识别系统的抗伪造 ASV（SASV）。  

ASVspoof 5 的最新技术水平：约 7.23% EER。旧的 ASVspoof 2019 LA 数据集：0.42% EER。真实部署中，预计野外环境语音 EER 在 5-10% 之间。

### AASIST 与 RawNet2 — 检测模型族

**AASIST**（2021，持续更新至 2026）。基于谱特征的图注意力模型。当前 ASVspoof 5 反措施任务的最先进模型（SOTA）。

**RawNet2。** 对原始波形使用卷积前端 + TDNN 主干。更简单的基线；调优后仍有竞争力。

**NeXt-TDNN + SSL 特征。** 2025 版本：ECAPA 风格 + WavLM 特征 + focal 损失。在 ASVspoof 2019 LA 达到 0.42% EER。

### AudioSeal — 2024 年默认水印方案

Meta 的 **AudioSeal**（2024 年 1 月发布，v0.2 2024 年 12 月）。关键设计：

- **定位性。** 以 16kHz 采样率帧级（1/16000 秒）检测水印。  
- **生成器 + 检测器联合训练。** 生成器学习嵌入不可闻信号；检测器学习通过增强找到信号。  
- **鲁棒性强。** 可承受 MP3 / AAC 压缩、均衡调整、±10% 速度变换、+10 dB 信噪比噪声混合处理。  
- **速度快。** 检测速度为实时的 485 倍；比 WavMark 快 1000 倍。  
- **容量充裕。** 16 位负载（可编码模型 ID、生成时间戳、用户 ID），每段语音都可嵌入。  

### WavMark

AudioSeal 之前的开源基线。可逆神经网络，32bits/秒。缺点：

- 同步需要暴力破解，速度慢。  
- 容易被高斯噪声或 MP3 压缩破坏。  
- 不适合实时应用。  

### WaveVerify（2025 年 7 月）

针对 AudioSeal 的弱点，特别是时间操纵（倒放、速度调整）。采用基于 FiLM 的生成器 + 专家混合模型（MoE）检测器。对标准攻击与 AudioSeal 相当，且能应对时间编辑。

### 对手利用的漏洞

AudioMarkBench 指出：“在音调平移攻击下，所有水印的比特恢复准确率低于 0.6，说明几乎完全被去除。”**音调变化是通用攻击手段。** 目前任何 2026 年的水印都无法完全抵抗剧烈音调变动。这就是为什么需要检测器（AASIST）搭配水印。

### C2PA / 内容真实性倡议

非机器学习技术 —— 一种清单（manifest）格式。音频文件携带密码签名的创建工具、作者、时间元数据。Audobox / Seamless 使用它。对来源认证有用；但若恶意者重新编码并剥离元数据则失效。

## 构建

### 第 1 步：简单谱特征检测器（玩具示例）

```python
def spectral_rolloff(spec, percentile=0.85):
    cum = 0
    total = sum(spec)
    if total == 0:
        return 0
    threshold = total * percentile
    for k, v in enumerate(spec):
        cum += v
        if cum >= threshold:
            return k
    return len(spec) - 1

def is_suspicious(audio):
    spec = magnitude_spectrum(audio)
    rolloff = spectral_rolloff(spec)
    return rolloff / len(spec) > 0.92
```

合成语音通常在高频部分能量较为平坦。生产级检测使用 AASIST，而非此示例。但直观思路相符。

### 第 2 步：AudioSeal 嵌入与检测

```python
from audioseal import AudioSeal
import torch

generator = AudioSeal.load_generator("audioseal_wm_16bits")
detector = AudioSeal.load_detector("audioseal_detector_16bits")

audio = load_wav("generated.wav", sr=16000)[None, None, :]
payload = torch.tensor([[1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0]])
watermark = generator.get_watermark(audio, sample_rate=16000, message=payload)
watermarked = audio + watermark

result, decoded_payload = detector.detect_watermark(watermarked, sample_rate=16000)
# result: 在 [0, 1] 之间的浮点数 —— 水印存在的概率
# decoded_payload: 16 位；与嵌入的负载进行匹配
```

### 第 3 步：指标评估 — EER（等错误率）

```python
def eer(real_scores, fake_scores):
    thresholds = sorted(set(real_scores + fake_scores))
    best = (1.0, 0.0)
    for t in thresholds:
        far = sum(1 for s in fake_scores if s >= t) / len(fake_scores)
        frr = sum(1 for s in real_scores if s < t) / len(real_scores)
        if abs(far - frr) < best[0]:
            best = (abs(far - frr), (far + frr) / 2)
    return best[1]
```

### 第 4 步：生产环境集成示例

```python
def safe_tts(text, voice, clone_reference=None):
    if clone_reference is not None:
        verify_consent(user_id, clone_reference)
    audio = tts_model.synthesize(text, voice)
    audio_with_wm = audioseal_embed(audio, payload=build_payload(user_id, model_id))
    manifest = c2pa_sign(audio_with_wm, user_id, timestamp=now())
    return audio_with_wm, manifest
```

每次生成都打包交付：(1) 水印，(2) 签名清单，(3) 符合保留政策的审计日志。

## 使用

| 用例                | 防御措施                          |
|---------------------|---------------------------------|
| 生产 TTS / 语音克隆 | 每个输出嵌入 AudioSeal（不可妥协） |
| 生物识别语音解锁    | AASIST + ECAPA 集成；活体检测挑战 |
| 呼叫中心欺诈检测    | 对 20% 来电采样使用 AASIST     |
| 播客真实性          | 上传时签名 C2PA，AI 生成加 AudioSeal |
| 研究 / 训练检测器   | 使用 ASVspoof 5 训练/开发/评测集 |

## 注意事项

- **水印嵌入但不运行检测器。** 无意义。检测器必须在持续集成（CI）中运行。  
- **检测无校准。** AASIST 在 ASVspoof LA 上过拟合，实际准确率会下降。需在目标领域校准。  
- **音调变动漏洞。** 剧烈音调平移能破坏大部分水印。须配合检测器作为备选。  
- **元数据剥离再上传。** C2PA 容易通过重新编码绕过。始终结合密码签名和感知水印防御。  
- **活体检测非万能。** 让用户说随机短语可防重放攻击，但不能阻止实时克隆。  

## 交付

保存至 `outputs/skill-spoof-defender.md`。选定检测模型、水印、来源清单，以及语音生成上线的操作方案。

## 练习

1. **简单。** 运行 `code/main.py`。玩具检测器 + 玩具水印嵌入/检测测试合成音频。  
2. **中等。** 安装 `audioseal`，对 TTS 输出嵌入 16 位负载，再解码。用噪声破坏音频并测量比特恢复准确率。  
3. **困难。** 在 ASVspoof 2019 LA 上微调 RawNet2 或 AASIST。测量 EER。测试 F5-TTS 生成的保留测试集，观察域外检测性能下降情况。  

## 关键术语

| 术语         | 普通说法         | 实际含义                                 |
|--------------|------------------|----------------------------------------|
| ASVspoof     | 基准             | 两年一度的挑战；2024 版本为 ASVspoof 5。   |
| CM (countermeasure) | 检测器      | 分类器：判断真实语音与合成/转换语音。         |
| SASV         | 说话人验证 + CM   | 集成生物识别 + 伪造检测。                      |
| AudioSeal    | Meta 水印        | 定位水印，16 位负载，检测速度比 WavMark 快 485 倍。  |
| Bit Recovery Accuracy | 水印存活率   | 攻击后恢复的负载比特比例。                      |
| C2PA         | 来源清单         | 关于创建/作者的密码签名元数据。                 |
| AASIST       | 检测器系列       | 基于图注意力的反伪造最先进技术。                 |

## 参考阅读

- [Todisco 等 (2024)。ASVspoof 5](https://dl.acm.org/doi/10.1016/j.csl.2025.101825) — 当前基准。  
- [Defossez 等 (2024)。AudioSeal](https://arxiv.org/abs/2401.17264) — 默认水印方案。  
- [Chen 等 (2025)。WaveVerify](https://arxiv.org/abs/2507.21150) — 用于时间攻击的 MoE 检测器。  
- [Jung 等 (2022)。AASIST](https://arxiv.org/abs/2110.01200) — 先进检测主干。  
- [AudioMarkBench (2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/5d9b7775296a641a1913ab6b4425d5e8-Paper-Datasets_and_Benchmarks_Track.pdf) — 鲁棒性评测。  
- [C2PA 规范](https://c2pa.org/specifications/specifications/) — 来源认证清单格式。
