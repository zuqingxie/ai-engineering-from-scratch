# Capstone 03 — 实时语音助理（ASR 到 LLM 到 TTS）

> 一个感觉顺畅的语音代理端到端延迟低于 800ms，能够识别用户何时停止说话，支持插话（barge-in），且能在不中断的情况下调用工具。Retell、Vapi、LiveKit Agents 和 Pipecat 都将在 2026 年达到这一标准。它们采用相同架构：流式 ASR（自动语音识别）、回合检测器、流式 LLM（大语言模型）、流式 TTS（文本转语音），整个过程通过 WebRTC 连接，每个环节都严格控制延迟预算。构建一个，测量 WER（词错误率）、MOS（主观听感评分）和误切断率，并在丢包情况下运行。

**类型：** Capstone  
**语言：** Python（代理 + 流水线）、TypeScript（网页客户端）  
**前置知识：** 阶段 6（语音与音频）、阶段 7（transformers）、阶段 11（LLM 工程）、阶段 13（工具）、阶段 14（代理）、阶段 17（基础设施）  
**涉及阶段：** P6 · P7 · P11 · P13 · P14 · P17  
**时间：** 30 小时

## 问题

语音是 2025-2026 年发展最快的 AI 用户体验（UX）类别。技术天花板每季度都在下降。OpenAI Realtime API、Gemini 2.5 Live、Cartesia Sonic-2、ElevenLabs Flash v3、LiveKit Agents 1.0 和 Pipecat 0.0.70 都实现了低于 800ms 的首次音频输出。标准不仅在于延迟，更在于交互感受：不能打断用户，也不能被用户打断，能从半句中断恢复，交谈中能调用工具且不中断音频，能承受移动网络抖动。

单靠拼接三个 REST 调用无法实现这一点。架构必须是端到端的流式流水线。搭建过程中，缺陷会显现：为电话音频调优的 VAD（语音活动检测）误触背景电视，基于标点的回合检测器等待永远不会出现的标点符号，TTS 在播报前缓冲 400ms。挑战是在负载下逐一修正这些问题，发布延迟与质量报告。

## 概念

流水线包含五个流式阶段：**音频输入**（浏览器或 PSTN 的 WebRTC）、**ASR**（Deepgram Nova-3 或 faster-whisper 逐步输出部分转录文本）、**回合检测**（结合 VAD 以及读取部分转录文本寻找结束信号的小型回合检测模型）、**LLM**（回合判定完成后快速流式输出 token）、**TTS**（在第一个 LLM token 后约 200ms 流式输出音频）。

三个跨阶段关注点。**插话（Barge-in）**：用户在代理播报时开始讲话，立即取消 TTS 并重新启用 ASR。**工具调用**：对话中途调用工具（天气、日历）必须通过侧通道运行且不打断音频；若延迟超过 300ms，代理预先输出确认词（“稍等...”）。**背压（Backpressure）**：丢包时，保留部分转录内容，VAD 提高激活阈值，代理避免在未确认消息上方讲话。

测量标准是定量的。WER 在 15 dB 信噪比 Hamming VAD 基准上低于 8%。100 个通话的 p50 首次音频输出低于 800ms。误切断率低于 3%。TTS 的 MOS 高于 4.2。单台 g5.xlarge 支持 50 路并发。这些数值即交付标准。

## 架构

```text
浏览器 / Twilio PSTN
        |
        v
   WebRTC / SIP 边缘
        |
        v
  LiveKit Agents 1.0（或 Pipecat 0.0.70）
        |
   +----+--------------+--------------+-----------------+
   |                   |              |                 |
   v                   v              v                 v
  ASR              VAD v5         回合检测器        侧通道工具
(Deepgram         (Silero)          (LiveKit)        （天气、
 Nova-3 /         语音门控          完成度评分       日历）
 Whisper-v3)      每 20ms          基于转录文本
   |                   |              |
   +--------+----------+--------------+
            v
        LLM（流式）
     GPT-4o-realtime / Gemini 2.5 Flash /
     级联 Claude Haiku 4.5
            |
            v
        TTS 流式
     Cartesia Sonic-2 / ElevenLabs Flash v3
            |
            v
     音频回传给呼叫方
            |
            v
   OpenTelemetry 语音跟踪 -> Langfuse
```

## 技术栈

- 传输：LiveKit Agents 1.0（WebRTC）加 Twilio PSTN 网关；备用框架为 Pipecat 0.0.70  
- ASR：Deepgram Nova-3（流式，首次部分转录延迟 <300ms）或 GPU 自托管的 faster-whisper Whisper-v3-turbo  
- VAD：Silero VAD v5 加 LiveKit 回合检测器（读取部分转录的小型 transformer）  
- LLM：OpenAI GPT-4o-realtime（紧耦合）、Gemini 2.5 Flash Live 或 级联 Claude Haiku 4.5（流式补全，分离音频路径）  
- TTS：Cartesia Sonic-2（最低首字节延迟）、ElevenLabs Flash v3 或开源 Orpheus（自托管）  
- 工具：FastMCP 侧通道处理天气/日历/预订；工具超过 300ms 代理先发出填充词  
- 观测：OpenTelemetry 语音跨度跟踪，Langfuse 包含音频回放的语音跟踪  
- 部署：单台 g5.xlarge（24GB VRAM）自托管 Whisper + Orpheus，或托管 API 以获取最低延迟  

## 构建步骤

1. **WebRTC 会话。** 搭建 LiveKit 房间和网页客户端，流式传输麦克风音频。服务器端挂载代理工作线程加入房间。

2. **ASR 流式。** 将 20ms PCM 音频帧送入 Deepgram Nova-3（或 GPU 上的 faster-whisper）。订阅部分和最终转录。记录每个部分转录的延迟。

3. **VAD 和回合检测。** 对帧流运行 Silero VAD v5。检测到语音结束事件时，触发 LiveKit 回合检测器对最新部分转录打分。仅当 VAD 静音 500ms 且回合检测评分高于 0.6 时，认定回合完成。

4. **LLM 流。** 回合完成启动 LLM 调用，携带当前会话和最终转录，流式输出 token。第一个 token 出现时移交给 TTS。

5. **TTS 流。** Cartesia Sonic-2 流式输出音频块。首块音频需在第一个 LLM token 后 200ms 内离开服务器。向 LiveKit 房间发出音频，客户端通过 WebRTC 抖动缓冲播放。

6. **插话（Barge-in）。** 当 VAD 识别到用户新语音且 TTS 正在播报时，立即取消 TTS 流，丢弃剩余的 LLM 输出，重新启用 ASR。发布一个 `tts_canceled` 追踪跨度。

7. **工具侧通道。** 注册天气和日历作为函数调用工具。调用时并行发起请求；若 300ms 内无返回，LLM 先输出“稍等，让我看一下”作为填充；工具返回后恢复。

8. **评估工具。** 录制 100 次通话。计算 WER（与保留转录比对）、误切断率（用户中断时 TTS 被取消的次数）、首次音频输出 p50、TTS MOS（人工或 NISQA 评分）、抖动丢包测试（丢弃 3% 包）。

9. **负载测试。** 在单台 g5.xlarge 上模拟 50 路并发呼叫。测量稳定状态的首次音频输出 p95。

## 使用示例

```text
caller: "what is the weather in tokyo tomorrow"
[asr  ] partial @280ms: "what is the"
[asr  ] partial @540ms: "what is the weather"
[turn ] completion score 0.82 at @820ms; commit
[llm  ] first token @960ms
[tool ] weather.tokyo tomorrow -> 68/52 partly cloudy @1140ms
[tts  ] first audio-out @1040ms: "Tokyo tomorrow will be partly cloudy..."
turn latency: 1040ms user-stop -> audio-out
```

## 交付物

`outputs/skill-voice-agent.md` 是最终成果。给定领域（客户支持、调度或自助终端），启动一个 LiveKit 代理，搭建符合测量标准的 ASR/VAD/LLM/TTS 流水线。评分标准：

| 权重 | 评判标准 | 测量方式 |
|:-:|---|---|
| 25 | 端到端延迟 | 100 次录音通话中 p50 首次音频输出低于 800ms |
| 20 | 轮次切换质量 | Hamming VAD 基准误切断率低于 3% |
| 20 | 工具使用正确性 | 对话中调用工具并返回正确数据且不阻塞音频 |
| 20 | 丢包下的可靠性 | 注入 3% 丢包后 WER 和轮次切换稳定性 |
| 15 | 评估工具完整性 | 可复现测量结果和公开配置 |
| **100** | | |

## 练习

1. 将 Deepgram Nova-3 替换为 g5.xlarge 上的 faster-whisper v3 turbo。测量延迟和 WER 差距。识别 CPU 与 GPU 方案的影响。

2. 添加中断仲裁策略：用户在工具调用过程中插话时，代理如何处理？比较三种策略（硬取消，完成工具后停止，排队下一轮）。

3. 运行对抗性回合检测测试：用户在句中长时间停顿。调整 VAD 静音阈值和回合检测评分阈值以实现最低误切断率且不超过 900ms。

4. 通过 Twilio 部署相同代理到 PSTN。比较 PSTN 与 WebRTC 的首次音频输出。解释抖动缓冲和编码器差异。

5. 增加对非英语语言（如日语、西班牙语）的语音活动检测。测试 Silero VAD v5 的误触发率与语言特定微调效果。

## 关键词

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| 回合检测 | “话语结束” | 基于 VAD 静音和部分转录判断用户是否停止说话的分类器 |
| 插话 | “中断处理” | VAD 检测到新用户语音时取消正在播放的 TTS |
| 首次音频输出 | “延迟” | 用户停止说话到服务器发送第一音频包的时间 |
| VAD | “语音门控” | 判断音频帧是语音还是静音的模型；Silero VAD v5 是 2026 默认选项 |
| 抖动缓冲 | “音频平滑” | 客户端短暂缓存包以吸收网络抖动 |
| 填充词 | “确认词” | 代理输出的短语，用于工具调用时避免静音 |
| MOS | “平均意见分” | 感知语音质量评分；NISQA 为自动代理评估接口 |

## 拓展阅读

- [LiveKit Agents 1.0](https://github.com/livekit/agents) — WebRTC 代理框架参考  
- [Pipecat](https://github.com/pipecat-ai/pipecat) — 备用 Python 主导的流式代理框架  
- [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime) — 集成语音模型参考  
- [Deepgram Nova-3 文档](https://developers.deepgram.com/docs) — 流式 ASR 参考  
- [Silero VAD v5](https://github.com/snakers4/silero-vad) — VAD 参考模型  
- [Cartesia Sonic-2](https://docs.cartesia.ai) — 低延迟 TTS 参考  
- [Retell AI 架构](https://docs.retellai.com) — 生产级语音代理架构  
- [Vapi.ai 生产栈](https://docs.vapi.ai) — 备用生产参考
