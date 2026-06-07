# 语音代理：Pipecat 和 LiveKit

> 语音代理在2026年成为一流的生产类别。Pipecat提供一个Python帧级管道框架（VAD → STT → LLM → TTS → 传输）。LiveKit Agents则通过WebRTC将AI模型桥接到用户。生产延迟目标为高端栈的端到端450–600ms。

**类型：** 学习  
**语言：** Python（标准库）  
**先决条件：** 第14阶段 · 01（Agent 循环）、第14阶段 · 12（工作流模式）  
**时间：** 约60分钟

## 学习目标

- 描述Pipecat的帧级管道：DOWNSTREAM（源→汇）和UPSTREAM（控制）。
- 说出典型语音管道阶段及Pipecat支持的传输方式。
- 解释LiveKit Agents的两种语音代理类（MultimodalAgent，多模态代理，VoicePipelineAgent，语音管道代理）及其适用场景。
- 概述2026年生产延迟预期及其如何推动架构选择。

## 问题

语音代理不是一个附加了TTS的文本循环。延迟预算非常严苛（约600ms），部分音频是默认，回合检测是一个模型，传输范围从电话SIP到WebRTC。要么构建基于帧的管道（Pipecat），要么依赖平台（LiveKit）。

## 概念

### Pipecat（pipecat-ai/pipecat）

- Python帧级管道框架。
- `Frame` → `FrameProcessor` 链。
- 两个流向：
  - **DOWNSTREAM** — 源头 → 汇（输入音频，输出TTS）。
  - **UPSTREAM** — 反馈与控制（取消、指标、打断）。
- `PipelineTask`管理生命周期，有事件（`on_pipeline_started`，`on_pipeline_finished`，`on_idle_timeout`）和观察者用于指标/追踪/RTVI。

典型管道：

```text
VAD (Silero) → STT → LLM（上下文在用户/助手间交替）→ TTS → 传输
```

传输：Daily，LiveKit，SmallWebRTCTransport，FastAPI WebSocket，WhatsApp。

Pipecat Flows添加结构化对话（状态机）。Pipecat Cloud为托管运行时。

### LiveKit Agents（livekit/agents）

- 通过WebRTC将AI模型桥接给用户。
- 关键概念：`Agent`，`AgentSession`，`entrypoint`，`AgentServer`。
- 两种语音代理类：
  - **MultimodalAgent** — 通过OpenAI Realtime或同类实现直接音频传输。
  - **VoicePipelineAgent** — STT → LLM → TTS 级联；提供文本级控制。
- 通过Transformer（Transformer 架构）模型进行语义回合检测。
- 原生MCP集成。
- 通过SIP实现电话功能。
- 50+模型无需API密钥通过LiveKit推理访问；200+模型通过插件。

### 商用平台

Vapi（在优化的高端栈上约450–600ms）和Retell（在180个测试电话中端到端约600ms）基于这些构建。当你想要一个托管的语音栈且没有WebRTC团队时，选择平台。

### 该模式的不足

- **无打断处理。** 用户打断，代理继续说话。需要Pipecat的UPSTREAM取消帧或LiveKit的等价机制。
- **忽视STT置信度。** 低置信度转录结果被当作真理喂给LLM。应基于置信度做门控或请求确认。
- **TTS中断。** 管道中途取消时，TTS需要感知或剪断音频。
- **忽视延迟预算。** 每个组件增加50–200ms。发布前需累计链路延迟。

### 典型2026年延迟

- VAD：20–60ms  
- STT部分响应：100–250ms  
- LLM首个token：150–400ms  
- TTS首个音频：100–200ms  
- 传输往返时延：30–80ms  

端到端450–600ms为高端水平。800–1200ms较常见。超过1500ms体验极差。

## 构建它

`code/main.py` 是基于帧的玩具管道，包含：

- `Frame`类型（音频、转录、文本、tts_audio、控制）。
- 实现`Processor`接口的`process(frame)`方法。
- 五阶段管道（VAD → STT → LLM → TTS → 传输）作为脚本处理器。
- 展示打断的UPSTREAM取消帧。

运行：

```text
python3 code/main.py
```

追踪显示正常流程和中途打断取消，停止TTS播报。

## 使用它

- **Pipecat** 用于完全控制——自定义处理器、Python优先、可插拔供应商。
- **LiveKit Agents** 用于WebRTC优先部署和电话功能。
- **Vapi / Retell** 用于无WebRTC团队的托管语音代理。
- **OpenAI Realtime / Gemini Live** 用于直接音频输入输出（MultimodalAgent）。

## 上线它

`outputs/skill-voice-pipeline.md` 脚手架了一个Pipecat形态的语音管道，包含VAD + STT + LLM + TTS + 传输以及打断处理。

## 练习

1. 给玩具管道添加指标观察者：统计每阶段每秒帧数。延迟集中在哪？
2. 实现基于置信度门控的STT：低于阈值时请求“可以重复一下吗？”。
3. 添加语义回合检测：简单规则——转录以“？”结尾即回合结束。
4. 阅读Pipecat传输文档。将标准库传输替换为SmallWebRTCTransport配置（桩代码）。
5. 测量同一查询下OpenAI Realtime与STT+LLM+TTS级联的延迟差异。文本级控制带来了什么延迟成本？

## 关键术语

| 术语 | 俗称 | 实际含义 |
|------|------|----------|
| Frame | “事件” | 管道中的类型化数据单元（音频、转录、文本、控制） |
| Processor | “管道阶段” | 实现process(frame)的处理器 |
| DOWNSTREAM | “前向流” | 源到汇：输入音频，输出语音 |
| UPSTREAM | “反馈流” | 控制流：取消、指标、打断 |
| VAD | “语音活动检测” | 检测用户讲话状态 |
| 语义回合检测 | “智能结束回合” | 基于模型判断用户说完 |
| MultimodalAgent | “直接音频代理” | 音频输入输出，中间无文本 |
| VoicePipelineAgent | “级联代理” | STT + LLM + TTS；支持文本级控制 |

## 深入阅读

- [Pipecat docs](https://docs.pipecat.ai/getting-started/introduction) — 基于帧的管道、处理器、传输
- [LiveKit Agents docs](https://docs.livekit.io/agents/) — WebRTC + 语音原语
- [Vapi](https://vapi.ai/) — 托管语音平台
- [Retell AI](https://www.retellai.com/) — 托管语音，延迟基准测试
