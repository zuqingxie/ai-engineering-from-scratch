# 语音活动检测（Voice Activity Detection）与轮次切换 — Silero、Cobra 及 Flush 技巧

> 每个语音助手的成败取决于两个决定：用户现在是否在说话，是否已经说完？VAD 解决第一个问题。轮次检测（VAD + 静音保留 + 语义端点模型）解决第二个问题。两者有误，助手要么打断用户，要么一直不停。

**类型：** 构建  
**语言：** Python  
**先决条件：** Phase 6 · 11（实时音频）、Phase 6 · 12（语音助手）  
**时间：** 约45分钟  

## 问题背景

语音助手在每个20毫秒音频片段上做出的三个不同决定：

1. **这一帧是语音吗？** — VAD，帧级二元分类。  
2. **用户是否开始了新的发声？** — 开始检测（onset detection）。  
3. **用户是否说完了？** — 结束点检测（turn-end）。  

简单的方案（能量阈值）在任何噪音环境下都无法可靠工作——路噪、键盘声、人群嘈杂等。2026年的方案是：Silero VAD（开源深度学习模型） + 轮次检测模型（语义端点）+ VAD 校准的静音保留时间。

## 概念

![VAD 级联：能量 → Silero → 轮次检测器 → flush 技巧](../assets/vad-turn-taking.svg)

### 三层 VAD 级联

**第一层：能量门。** 最简单，最便宜。RMS阈值设为 -40 dBFS。可以过滤明显的静音，但对阈值以上的任何噪声都触发。

**第二层：Silero VAD**（2020-2026，MIT许可）。1百万参数，训练于6000多种语言。单线程CPU上处理30毫秒音频仅需约1毫秒。5%假阳性率下，真阳性率为87.7%。是开源默认选择。

**第三层：语义轮次检测器。** LiveKit 的轮次检测模型（2024-2026）或自定义小型分类器。区别“句中停顿”与“说话结束”。基于语言上下文（语调 + 最近词语），不仅依赖静音。

### 关键参数及默认值

- **阈值（Threshold）。** Silero 输出概率，默认大于0.5判语音，敏感模式可设为大于0.3。阈值低，首词截断少，但误报多。  
- **最短语音长度。** 丢弃低于250毫秒的语音，通常是咳嗽或椅子声。  
- **静音保留时间（end-pointing）。** VAD 返回0后等待500-800毫秒再认定轮次结束。太短会打断用户，太长会感觉延迟。  
- **预缓冲区（Pre-roll buffer）。** 在VAD触发前保留300-500毫秒音频，防止“嗨”被剪断。  

### Flush 技巧（Kyutai 2025）

流式语音转文本（STT）模型存在前瞻延迟（Kyutai STT-1B约500毫秒，STT-2.6B约2.5秒）。通常需等这么长时间才能拿到完整转录。Flush 技巧是在 VAD 认定说话结束时，**向 STT 发送 flush 信号强制立即输出**。STT 的处理速度约为实时的4倍，500ms缓存转录仅需约125ms。

端到端实现：125ms VAD 处理 + flush STT = 可对话的低延迟。

### 2026 年 VAD 对比

| VAD | 5% 假阳性率下真阳性率（TPR） | 延迟 | 许可证 |
|-----|-------------------------------|------|--------|
| WebRTC VAD（Google，2013） | 50.0% | 30毫秒 | BSD |
| Silero VAD（2020-2026） | 87.7% | ~1毫秒 | MIT |
| Cobra VAD（Picovoice） | 98.9% | ~1毫秒 | 商业 |
| pyannote segmentation | 95% | ~10毫秒 | 类 MIT |

Silero 是默认的最佳选择。Cobra 用于合规或追求更高准确度。纯能量阈值 VAD 在2026年生产环境已无立足之地。

## 实现步骤

### 步骤1：能量门

```python
def energy_vad(chunk, threshold_dbfs=-40.0):
    rms = (sum(x * x for x in chunk) / len(chunk)) ** 0.5
    dbfs = 20.0 * math.log10(max(rms, 1e-10))
    return dbfs > threshold_dbfs
```

### 步骤2：Python中使用 Silero VAD

```python
from silero_vad import load_silero_vad, get_speech_timestamps

vad = load_silero_vad()
audio = torch.tensor(waveform_16k, dtype=torch.float32)
segments = get_speech_timestamps(
    audio, vad, sampling_rate=16000,
    threshold=0.5,
    min_speech_duration_ms=250,
    min_silence_duration_ms=500,
    speech_pad_ms=300,
)
for s in segments:
    print(f"{s['start']/16000:.2f}s - {s['end']/16000:.2f}s")
```

### 步骤3：轮次结束状态机

```python
class TurnDetector:
    def __init__(self, silence_hangover_ms=500, min_speech_ms=250):
        self.state = "idle"
        self.speech_ms = 0
        self.silence_ms = 0
        self.silence_hangover_ms = silence_hangover_ms
        self.min_speech_ms = min_speech_ms

    def update(self, is_speech, chunk_ms=20):
        if is_speech:
            self.speech_ms += chunk_ms
            self.silence_ms = 0
            if self.state == "idle" and self.speech_ms >= self.min_speech_ms:
                self.state = "speaking"
                return "START"
        else:
            self.silence_ms += chunk_ms
            if self.state == "speaking" and self.silence_ms >= self.silence_hangover_ms:
                self.state = "idle"
                self.speech_ms = 0
                return "END"
        return None
```

### 步骤4：flush 技巧骨架

```python
def flush_on_end(stt_client, audio_buffer):
    stt_client.send_audio(audio_buffer)
    stt_client.send_flush()
    return stt_client.recv_transcript(timeout_ms=150)
```

STT（Kyutai、Deepgram、AssemblyAI）必须支持 flush 才能生效。Whisper 流式不支持——Whisper是基于块处理，总是等待固定块。

## 使用场景

| 场景 | VAD 选择 |
|------|---------|
| 开源、快速、通用 | Silero VAD |
| 商业呼叫中心 | Cobra VAD |
| 设备端（手机） | Silero VAD ONNX |
| 研究 / 说话者分离 | pyannote segmentation |
| 零依赖备用方案 | WebRTC VAD（传统） |
| 需要高质量轮次结束 | Silero + LiveKit 轮次检测层叠 |

经验法则：除非万不得已，绝不应只用能量检测做 VAD。

## 常见陷阱

- **固定阈值。** 静音环境可行，噪声环境失效。应设备端校准或切换到 Silero。  
- **静音保留时间过短。** 语音中间被打断。500-800毫秒是对话语音的最佳区间。  
- **静音保留时间过长。** 感觉反应迟钝。通过A/B测试调整。  
- **没有预缓冲区。** 用户语音的前200-300毫秒丢失，总是保留滚动预缓冲区。  
- **忽略语义端点检测。** “嗯，让我想想...” 长暂停，用户讨厌被打断思路。应使用 LiveKit 轮次检测器或类似方案。  

## 上线部署

保存为 `outputs/skill-vad-tuner.md`。根据任务选择 VAD 模型、阈值、保留时间、预缓冲和轮次检测策略。

## 练习

1. **简单。** 运行 `code/main.py`。模拟语音 + 静音 + 语音 + 咳嗽序列，测试三层 VAD。  
2. **中等。** 安装 `silero-vad`，处理一段5分钟录音，调节阈值最小化首词截断及误触发，报告准确率/召回率。  
3. **困难。** 构建小型轮次检测器：Silero VAD + 基于最近10词嵌入的三层MLP（用 sentence-transformers），在人工标注的轮次结束数据集上训练，F1提升10%以上。  

## 关键术语

| 术语 | 常用说法 | 实际含义 |
|------|-----------|----------|
| VAD | 语音检测器 | 帧级二元分类：这帧是语音吗？ |
| 轮次检测 | 结束点检测 | VAD + 静音保留 + 语义端点模型 |
| 静音保留 | 说话后等待 | 认定轮次结束前等待的时间；通常500-800毫秒 |
| 预缓冲 | 说话前缓存 | 在VAD触发前保留300-500毫秒音频 |
| Flush 技巧 | Kyutai 黑科技 | VAD触发→flush STT→延迟由500毫秒降至125毫秒 |
| 语义端点 | “他们是不是想停？” | 基于词语信息的机器学习分类器，而非仅静音 |
| 5%假阳性率下真阳性率（TPR） | ROC评分点 | VAD 性能标准；Silero约87.7%，WebRTC约50% |

## 拓展阅读

- [Silero VAD](https://github.com/snakers4/silero-vad) — 参考开源 VAD。  
- [Picovoice Cobra VAD](https://picovoice.ai/products/cobra/) — 商业准确率领先者。  
- [Kyutai — Unmute + flush trick](https://kyutai.org/stt) — 次200毫秒工程黑科技。  
- [LiveKit — 轮次检测](https://docs.livekit.io/agents/logic/turns/) — 生产环境语义端点检测。  
- [WebRTC VAD](https://webrtc.googlesource.com/src/) — 传统基线方案。  
- [pyannote segmentation](https://github.com/pyannote/pyannote-audio) — 说话者分离级别分割。
