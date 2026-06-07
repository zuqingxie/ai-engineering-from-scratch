# 实时音频处理

> 批处理流水线处理文件。实时流水线在下一段20毫秒音频到达前处理当前段。每个会话式 AI、广播室和电话机器人都靠这条延迟预算存活。

**类型：** 构建  
**语言：** Python  
**先决条件：** 第6阶段 · 02（频谱图Spectrograms）、第6阶段 · 04（自动语音识别ASR）、第6阶段 · 07（文本转语音TTS）  
**时间：** ~75 分钟

## 问题

你想要一款听起来“有生命”的语音助手。人类会话中轮到说话的延迟大约是230毫秒（静音到回应）。超过500毫秒会感觉机械化；超过1500毫秒则感觉断裂。2026年完整的 **听 → 理解 → 响应 → 说** 流程预算是：

| 阶段 | 预算 |
|-------|--------|
| 麦克风 → 缓冲区 | 20 ms |
| 语音活动检测VAD | 10 ms |
| 自动语音识别ASR（流式） | 150 ms |
| 大语言模型LLM（首个token） | 100 ms |
| 文本转语音TTS（首个音块） | 100 ms |
| 渲染 → 扬声器 | 20 ms |
| **总计** | **~400 ms** |

Moshi（Kyutai, 2024）实现了200毫秒全双工。GPT-4o-realtime（2024）约320毫秒。2022年级联流水线达到2500毫秒。10倍的提升得益于三大技术：（1）全面流式处理，（2）异步流水线与部分结果处理，（3）可中断生成。

## 概念

![Streaming audio pipeline with ring buffer, VAD gate, interruption](../assets/real-time.svg)

**帧 / 音块 / 窗口。** 实时音频以固定大小块流动。常见选择是20毫秒（16 kHz下320个采样点）。下游所有模块必须跟上这个节奏。

**环形缓冲区（Ring buffer）。** 固定大小的循环缓冲区。生产者线程写入新帧，消费者线程读取。防止热路径中的内存分配。大小≈最大延迟 × 采样率；2秒、16 kHz的环形缓冲大小=32,000采样点。

**VAD（语音活动检测Voice Activity Detection）。** 当无人讲话时关闭下游工作。Silero VAD 4.0（2024）在CPU上处理30毫秒音频帧耗时<1毫秒。`webrtcvad`是较老的替代。

**流式 ASR。** 音频输入时输出部分转录文本的模型。Parakeet-CTC-0.6B流式模式（NeMo，2024）在320毫秒延迟下实现2-5%词错误率。Whisper-Streaming（Macháček 等，2023）将 Whisper分块实现近流式，延迟约2秒。

**中断。** 当用户在助手说话时插话，你必须：(a) 侦测插话，(b) 停止TTS，(c) 丢弃剩余LLM输出。都需在100毫秒内完成，否则用户会觉得助手“聋了”。

**WebRTC Opus传输。** 20毫秒音频帧，48 kHz，8–128 kbps自适应比特率。浏览器和手机的标准。LiveKit、Daily.co、Pion是2026年构建语音应用的主流框架。

**抖动缓冲区（Jitter buffer）。** 网络包无序或延迟到达。抖动缓冲重新排序平滑处理；太小会有明显断点，太大会增加延迟。典型值60–80毫秒。

### 常见误区

- **线程争用。** Python的全局解释器锁GIL加上重模型可能导致音频线程得不到足够资源。要用C回调音频库（sounddevice、PortAudio），让Python不在热路径中。
- **采样率转换延迟。** 流水线中实时重采样会增加5–20毫秒延迟。要么事先重采样，要么用零延迟重采样器（PolyPhase、`soxr_hq`）。
- **TTS预热。** 即使快速TTS如 Kokoro，首次请求仍需100–200毫秒预热。缓存模型，并用空跑预热后才用正式对话。
- **回声消除。** 无AEC时，TTS音频会被麦克风录入并触发自己的ASR。WebRTC AEC3是开源默认方案。

## 构建步骤

### 第1步：环形缓冲区

```python
import collections

class RingBuffer:
    def __init__(self, capacity):
        self.buf = collections.deque(maxlen=capacity)
    def write(self, frame):
        self.buf.extend(frame)
    def read(self, n):
        return [self.buf.popleft() for _ in range(min(n, len(self.buf)))]
    def level(self):
        return len(self.buf)
```

容量决定最大缓冲延迟。16 kHz时32000采样点即2秒。

### 第2步：VAD门控

```python
def simple_energy_vad(frame, threshold=0.01):
    return sum(x * x for x in frame) / len(frame) > threshold ** 2
```

生产环境用 Silero VAD 替换：

```python
import torch
vad, _ = torch.hub.load("snakers4/silero-vad", "silero_vad")
is_speech = vad(torch.tensor(frame), 16000).item() > 0.5
```

### 第3步：流式自动语音识别ASR

```python
# Parakeet-CTC-0.6B 流式，基于 NeMo
from nemo.collections.asr.models import EncDecCTCModelBPE
asr = EncDecCTCModelBPE.from_pretrained("nvidia/parakeet-ctc-0.6b")
# chunk_ms=320 ms，look_ahead_ms=80 ms
for chunk in audio_stream():
    partial_text = asr.transcribe_streaming(chunk)
    print(partial_text, end="\r")
```

### 第4步：中断处理器

```python
class Dialog:
    def __init__(self):
        self.tts_task = None

    def on_user_speech(self, frame):
        if self.tts_task and not self.tts_task.done():
            self.tts_task.cancel()   # 插话打断
        # 然后传给流式ASR

    def on_final_user_utterance(self, text):
        self.tts_task = asyncio.create_task(self.reply(text))

    async def reply(self, text):
        async for tts_chunk in llm_then_tts(text):
            speaker.write(tts_chunk)
```

依赖异步I/O和可取消的TTS流。WebRTC中调用peerconnection.stop()关闭音轨是规范做法。

## 使用说明

2026年方案：

| 层级 | 选项 |
|-------|------|
| 传输 | LiveKit（WebRTC）或 Pion（Go） |
| VAD | Silero VAD 4.0 |
| 流式 ASR | Parakeet-CTC-0.6B 或 Whisper-Streaming |
| LLM首token | Groq、Cerebras、vLLM-streaming |
| 流式 TTS | Kokoro 或 ElevenLabs Turbo v2.5 |
| 回声消除 | WebRTC AEC3 |
| 端到端原生 | OpenAI Realtime API 或 Moshi |

## 陷阱

- **缓冲500毫秒防守。** 缓冲区是你的延迟下限。要缩小它。
- **线程不绑定。** 音频回调运行在比UI优先级低的线程会导致卡顿。
- **TTS音块太小。** 少于200毫秒的块声音合成器会产生明显假声。320毫秒是最佳块大小。
- **没有抖动缓冲。** 真实网络环境会有抖动，没缓冲会有爆音。
- **单次错误处理。** 音频流水线必须严防奔溃。一次异常会杀死整个会话。

## 发布

保存为 `outputs/skill-realtime-designer.md`。设计一个实时音频流水线，明确每个阶段的延迟预算。

## 练习

1. **简单。** 运行 `code/main.py`。模拟环形缓冲+能量VAD；打印虚拟10秒流的分阶段延迟。
2. **中等。** 利用 `sounddevice`，构建一个20毫秒音频帧的回环流程，打印每帧的VAD状态。
3. **困难。** 用 `aiortc` 构建全双工回声测试：浏览器 → WebRTC → Python → WebRTC → 浏览器。用1 kHz脉冲测量端到端延迟。

## 关键术语

| 术语 | 常用说法 | 实际含义 |
|------|----------|----------|
| 环形缓冲区Ring buffer | 循环队列 | 用于音频帧的固定大小、无锁（或单生产单消费者锁）FIFO队列。 |
| VAD | 静音门 | 模型或启发式方法判断讲话与非讲话段。 |
| 流式 ASR | 实时语音转文本 | 音频到达时即输出部分文字，带有限 lookahead。 |
| 抖动缓冲Jitter buffer | 网络平滑器 | 重排序乱序包的队列，典型60–80毫秒。 |
| AEC | 回声消除 | 减去话筒与扬声器间反馈路径。 |
| 插话Barge-in | 用户打断 | 系统检测用户在TTS中途说话，必须取消播放。 |
| 全双工Full duplex | 双向同时通话 | 用户和机器人可同时讲话；Moshi是全双工实现。 |

## 进一步阅读

- [Macháček 等（2023）。Whisper-Streaming](https://arxiv.org/abs/2307.14743) — 分块近流式 Whisper。  
- [Kyutai（2024）。Moshi](https://kyutai.org/Moshi.pdf) — 全双工200毫秒延迟。  
- [LiveKit Agents 框架（2024）](https://docs.livekit.io/agents/) — 生产级音频代理编排。  
- [Silero VAD 仓库](https://github.com/snakers4/silero-vad) — 亚1毫秒VAD，Apache 2.0。  
- [WebRTC AEC3 论文](https://webrtc.googlesource.com/src/+/main/modules/audio_processing/aec3/) — 开源回声消除。
