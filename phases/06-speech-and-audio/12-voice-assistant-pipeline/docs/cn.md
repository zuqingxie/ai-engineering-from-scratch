# 构建语音助手流程 — 第6阶段毕业项目

> 将第01-11课内容串联起来。构建一个能够听、推理并回复的语音助手。到2026年，这将是一个工程已解决的问题，而非研究问题——但集成细节决定它能否顺利发布。

**类型：** 构建  
**语言：** Python  
**前置课程：** 第6阶段 · 04, 05, 06, 07, 11；第11阶段 · 09（Function Calling（函数调用））；第14阶段 · 01（Agent Loop（代理循环））  
**时间：** 约120分钟

## 问题描述

构建一个端到端的助手：

1. 捕获麦克风输入（16 kHz 单声道）。
2. 识别用户讲话的开始和结束。
3. 流式转录。
4. 将转录文本传递给支持调用工具（定时器、天气、日历）的LLM（大语言模型）。
5. 将LLM文本流式传输至TTS（文本转语音）。
6. 播放音频给用户。
7. 若用户在生成回复中途打断，则停止。

延迟目标：用户发言结束后800 ms内在笔记本CPU上收到首个TTS音频字节。质量目标：无漏词，无静音时生成虚假字幕，无声音克隆泄露，无成功的提示注入攻击。

## 概念

![语音助手流程：mic → VAD → STT → LLM+工具 → TTS → 扬声器](../assets/voice-assistant.svg)

### 七个组件

1. **音频采集。** 麦克风 → 16 kHz单声道 → 20 ms数据块。通常在Python中用 `sounddevice`，生产环境用原生AudioUnit/ALSA/WASAPI。
2. **VAD（第11课）。** Silero VAD，阈值0.5，最短语音250 ms，静音续延500 ms。发出“开始”和“结束”信号。
3. **流式STT（第4-5课）。** Whisper-streaming、Parakeet-TDT或Deepgram Nova-3（API）。输出部分与最终转录。
4. **带工具调用的LLM。** GPT-4o / Claude 3.5 / Gemini 2.5 Flash。使用JSON schema定义工具。流式输出token。
5. **流式TTS（第7课）。** Kokoro-82M（最快开源）或Cartesia Sonic（商业）。在生成20个LLM token后启动TTS。
6. **播放。** 输出到扬声器；低带宽网络时用opus编码。
7. **中断处理。** 如果VAD在TTS播放时检测到声音，停止播放，取消LLM，重新启动STT。

### 你可能遇到的三种失败模式

1. **首词被截断。** VAD识别开始晚了一拍，用户的“hey”被漏掉。阈值设置为0.3而不是0.5。
2. **中途中断混乱。** 用户打断时LLM仍在生成回复，助手出现“讲话盖过用户”的情况。把VAD信号连线到取消LLM。
3. **静音幻觉。** Whisper在静音预热帧中输出“Thanks for watching”。一定要用VAD闸门。

### 2026年生产参考栈

| 栈                      | 延迟      | 许可        | 备注               |
|-------------------------|-----------|-------------|--------------------|
| LiveKit + Deepgram + GPT-4o + Cartesia | 350-500 ms | 商业API     | 2026年行业默认       |
| Pipecat + Whisper-streaming + GPT-4o + Kokoro | 500-800 ms | 大部分开源   | 适合DIY             |
| Moshi（全双工）           | 200-300 ms | CC-BY 4.0   | 单模型；不同架构，第15课 |
| Vapi / Retell（托管）      | 300-500 ms | 商业        | 上线最快；定制有限       |
| Whisper.cpp + llama.cpp + Kokoro-ONNX | 离线       | 开源        | 隐私/边缘设备         |

## 构建步骤

### 步骤1：带分块的麦克风采集（伪代码）

```python
import sounddevice as sd

def mic_stream(chunk_ms=20, sr=16000):
    q = queue.Queue()
    def cb(indata, frames, time, status):
        q.put(indata.copy().flatten())
    with sd.InputStream(channels=1, samplerate=sr, blocksize=int(sr * chunk_ms/1000), callback=cb):
        while True:
            yield q.get()
```

### 步骤2：VAD闸门的语音段捕获

```python
def capture_turn(stream, vad, pre_roll_ms=300, silence_ms=500):
    buf, pre, triggered = [], collections.deque(maxlen=pre_roll_ms // 20), False
    silent = 0
    for chunk in stream:
        pre.append(chunk)
        if vad(chunk):
            if not triggered:
                buf = list(pre)
                triggered = True
            buf.append(chunk)
            silent = 0
        elif triggered:
            silent += 20
            buf.append(chunk)
            if silent >= silence_ms:
                return b"".join(buf)
```

### 步骤3：流式STT → LLM → TTS

```python
async def turn(audio_bytes):
    transcript = await stt.transcribe(audio_bytes)
    async for token in llm.stream(transcript):
        async for audio in tts.stream(token):
            await speaker.play(audio)
```

### 步骤4：LLM循环内的工具调用

```python
tools = [
    {"name": "get_weather", "parameters": {"location": "string"}},
    {"name": "set_timer", "parameters": {"seconds": "int"}},
]

async for chunk in llm.stream(user_text, tools=tools):
    if chunk.type == "tool_call":
        result = dispatch(chunk.name, chunk.args)
        continue_streaming(result)
    if chunk.type == "text":
        await tts.stream(chunk.text)
```

### 步骤5：中断处理

```python
tts_task = asyncio.create_task(tts_loop())
while True:
    chunk = await mic.get()
    if vad(chunk):
        tts_task.cancel()
        await speaker.stop()
        await new_turn()
        break
```

## 使用方法

查看 `code/main.py`，这是一个可运行的模拟程序，用stub模型实现七个组件的连接，方便你即使没有硬件也能观察流程结构。实际实现时，将stub替换为：

- `silero-vad`（`pip install silero-vad`）
- `deepgram-sdk` 或 `openai-whisper`
- `openai`（GPT-4o） 或 `anthropic`
- `kokoro` 或 `cartesia`
- 用于音频I/O的 `sounddevice`

## 注意事项

- **日志中长期存储个人信息（PII）。** 大多数地区全程语音都是个人信息。建议30天保留，并在静态存储时加密。
- **不支持插话。** 用户会打断，你的助手必须停止说话。
- **阻塞型TTS。** 同步TTS会阻塞事件循环。应使用异步或独立线程。
- **无工具调用错误处理。** 工具可能失败。LLM必须获得错误信息并重试一次，之后优雅降级。
- **过度过滤幻觉。** 过滤过度时，助手重复说“我无法帮你”；过滤不足时，助手胡说八道。需要在保留集上调参。
- **无唤醒词选项。** 始终监听存在隐私风险。建议加唤醒词闸门（Porcupine或openWakeWord）。

## 发布方案

保存为 `outputs/skill-voice-assistant-architect.md`。结合预算、规模、语言和合规约束，制定完整栈规范。

## 练习

1. **简单。** 运行 `code/main.py`。它用stub模块模拟完整一次对话并打印各阶段延迟。
2. **中等。** 用真实Whisper模型替换STT stub，对预录 `.wav` 文件测试。测量词错误率（WER）和端到端延迟。
3. **困难。** 增加工具调用：实现 `get_weather`（任意API）和 `set_timer`。让LLM通过这些工具，并验证“设置一个5分钟定时器”能触发相应函数且语音回复确认。

## 关键词

| 术语       | 常用说法               | 实际含义                                  |
|------------|------------------------|-----------------------------------------|
| Turn（轮次）    | 用户+助手一次交互          | 一个VAD界定的用户语音 + 一个LLM-TTS回复         |
| Barge-in（插话） | 中断                    | 用户在助手讲话时插话；助手停止说话              |
| Wake word（唤醒词） | “嘿助手”               | 短关键词检测器；Porcupine、Snowboy、openWakeWord等 |
| End-pointing（端点检测） | 一轮结束                | VAD加最小静默时间判定用户说话结束                 |
| Pre-roll（预录）   | 讲话前的缓冲音频            | 在VAD触发前保留200-400 ms音频以避免首词截断        |
| Tool call（工具调用） | 函数调用                | LLM输出JSON；运行时分发调用；结果回传到循环中       |

## 深入阅读

- [LiveKit — 语音代理快速入门](https://docs.livekit.io/agents/) — 生产级参考。
- [Pipecat — 语音代理示例](https://github.com/pipecat-ai/pipecat) — 适合DIY的框架。
- [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime) — 托管的语音原生路径。
- [Kyutai Moshi](https://github.com/kyutai-labs/moshi) — 全双工参考（第15课）。
- [Porcupine唤醒词](https://picovoice.ai/products/porcupine/) — 唤醒词闸门。
- [Anthropic — 工具使用指南](https://docs.anthropic.com/en/docs/build-with-claude/tool-use) — LLM函数调用。
