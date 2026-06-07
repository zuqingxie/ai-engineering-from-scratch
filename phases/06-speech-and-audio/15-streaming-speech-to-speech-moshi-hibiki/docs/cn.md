# 流式语音对语音 — Moshi、Hibiki 和全双工对话

> 2024-2026 年重新定义了语音 AI。Moshi 提供一个单模型，可在 200 毫秒延迟下实现同时听和说。Hibiki 实现了逐块流式语音对语音翻译。两者都摒弃了 ASR（自动语音识别）→ LLM（大语言模型）→ TTS（文本转语音）的流水线，采用基于 Mimi 编解码器标记的统一全双工架构。这是新的参考设计。

**类型：** 学习  
**语言：** Python  
**先决条件：** 第6阶段·13（神经音频编解码器）、第6阶段·11（实时音频）、第7阶段·05（全 Transformer）  
**耗时：** 约75分钟

## 问题所在

基于第11、12课构建的每个语音代理都有约300-500毫秒的基本延迟下限：VAD（语音活动检测）触发，STT（语音转文本）处理，LLM推理，TTS生成。每个阶段都有自身的最小延迟。你可以调优和并行化，但流水线结构限制了性能。

Moshi（Kyutai，2024-2026）提出了一个不同的问题：如果没有流水线呢？如果一个模型直接连续地输入音频并输出音频，将文本作为中间“内心独白”，而不是必经阶段呢？

答案是**全双工语音对语音**。理论延迟 160 毫秒（80 毫秒 Mimi 帧 + 80 毫秒声学延迟）。实际延迟为单个 L4 GPU 上的 200 毫秒。只有最佳流水线语音代理延迟的一半。

## 概念

![Moshi 架构：两个并行 Mimi 流 + 内心独白文本](../assets/moshi-hibiki.svg)

### Moshi 架构

**输入。** 两个 Mimi 编解码器流，均为 12.5 Hz × 8 码本：

- 流 1：用户音频（Mimi 编码，持续到达）
- 流 2：Moshi 自身音频（Moshi 生成）

**Transformer。** 一个7亿参数的时间序列 Transformer 处理两个流和文本“内心独白”流。每 80 毫秒步进时，它：

1. 消费最新的用户 Mimi 标记（8 个码本）。
2. 消费最近的 Moshi Mimi 标记（8 个码本，生成的）。
3. 生成下一个 Moshi 文本标记（内心独白）。
4. 生成下一个 Moshi Mimi 标记（通过小型深度 Transformer 预测的 8 个码本）。

这三条流——用户音频、Moshi 音频、Moshi 文本——并行运行。Moshi 能听到用户且同时讲话；当用户打断时可自我打断；可进行回馈式沟通（“嗯嗯”）且不中断主话语。

**深度 Transformer。** 在一个帧内，8 个码本不是并行预测的——它们存在码本间依赖。一个小型2层“深度 Transformer”在 80 毫秒内顺序预测它们。这是自回归（AR）编解码语言模型的标准分解（VALL-E、VibeVoice 也使用此法）。

### 为什么内心独白文本有帮助

没有显式文本，模型必须在声学流中隐式建模语言。Moshi 的洞见：强制它同时输出文本标记和音频。文本流本质上是 Moshi 说话内容的转录。这提升了语义连贯性，更容易替换语言模型头，并且免费获得转录文本。

### Hibiki：流式语音对语音翻译

相同架构，训练于翻译对。输入源语言音频，连续输出目标语言音频。Hibiki-Zero（2026年2月）消除了对词级对齐训练数据的需求——使用句级数据+GRPO强化学习进行延迟优化。

初始支持四种语言对；可用约1000小时数据适配新语言。

### 更广泛的 Kyutai 技术栈（2026）

- **Moshi** — 全双工对话（首推法语，英语支持良好）  
- **Hibiki / Hibiki-Zero** — 同声传译实时翻译  
- **Kyutai STT** — 流式自动语音识别（500 毫秒或 2.5 秒前瞻）  
- **Kyutai Pocket TTS** — 1亿参数 CPU 上运行的 TTS（2026年1月）  
- **Unmute** — 在公共服务器上整合这套完整流水线

L40S GPU 上吞吐量：64 会话并发，实时速率3倍。

### Sesame CSM — 亲缘方案

Sesame CSM（2025）采用类似思路——Llama-3 骨干与 Mimi 编解码头。但 CSM 是单向的（接收上下文+文本，生成语音），非全双工。它是市场上的最佳“语音临场感” TTS；与 Moshi 的全双工能力不同。

### 2026年性能数据

| 模型           | 延迟         | 用例                    | 许可证           |
|----------------|--------------|-------------------------|------------------|
| Moshi          | 200 毫秒（L4） | 全双工英法对话            | CC-BY 4.0        |
| Hibiki         | 12.5 Hz 帧率   | 法语 ↔ 英语流式翻译        | CC-BY 4.0        |
| Hibiki-Zero    | 同上          | 5 种语言对，无对齐数据       | CC-BY 4.0        |
| Sesame CSM-1B  | 200 毫秒 TTFA | 上下文条件TTS             | Apache-2.0       |
| GPT-4o Realtime| 约300 毫秒    | 封闭，OpenAI API          | 商业授权          |
| Gemini 2.5 Live| 约350 毫秒    | 封闭，Google API          | 商业授权          |

## 构建它

### 第1步：接口

Moshi 暴露一个 WebSocket 服务器，接受 80 毫秒的 Mimi 编码音频块，返回 80 毫秒 Mimi 编码音频块。双向。持续进行。

```python
import asyncio
import websockets
from moshi.client_utils import encode_audio_mimi, decode_audio_mimi

async def moshi_chat():
    async with websockets.connect("ws://localhost:8998/api/chat") as ws:
        mic_task = asyncio.create_task(stream_mic_to(ws))
        spk_task = asyncio.create_task(stream_from_to_speaker(ws))
        await asyncio.gather(mic_task, spk_task)
```

### 第2步：全双工循环

```python
async def stream_mic_to(ws):
    async for chunk_80ms in mic_stream_at_12_5_hz():
        mimi_tokens = encode_audio_mimi(chunk_80ms)
        await ws.send(serialize(mimi_tokens))

async def stream_from_to_speaker(ws):
    async for msg in ws:
        mimi_tokens, text_token = deserialize(msg)
        audio = decode_audio_mimi(mimi_tokens)
        await play(audio)
```

两个方向同时运行。Python 的 asyncio 或 Rust 的 futures 是标准通信方案。

### 第3步：训练目标（概念）

每个 80 毫秒帧 `t`：

- 输入：`user_mimi[0..t]`、`moshi_mimi[0..t-1]`、`moshi_text[0..t-1]`  
- 预测：先预测 `moshi_text[t]`，然后预测 `moshi_mimi[t, codebook_0..7]`

文本在音频之前预测（内心独白）；音频在深度 Transformer 内按码本顺序预测。

### 第4步：Moshi 的优势与劣势

Moshi 优势：

- 低于250毫秒端到端延迟，且使用廉价硬件。
- 自然的回馈与打断能力。
- 无流水线粘合代码。

Moshi 劣势：

- 不能调用工具（未经训练，需要单独 LLM 路径）。
- 长时间推理能力较弱（Moshi 是约8亿参数的对话模型，不是 Claude/GPT-4）。
- 专业领域的事实准确度有限。
- 多数企业级生产用途（2026 年仍依赖流水线）。

## 使用它

| 场景                     | 选择          |
|--------------------------|---------------|
| 最低延迟语音助手         | Moshi         |
| 实时翻译电话             | Hibiki        |
| 语音演示/科研            | Moshi, CSM    |
| 具备工具调用的企业代理   | 流水线（第12课），非 Moshi |
| 上下文定制语音 TTS       | Sesame CSM    |
| 任何语言的语音对语音     | GPT-4o Realtime 或 Gemini 2.5 Live（商业） |

## 陷阱

- **有限的工具调用能力。** Moshi 是对话模型，不是代理框架。需要与流水线结合使用工具。
- **特定声音条件化。** Moshi 使用单一训练的人设；克隆需单独训练。
- **语言覆盖。** 法语+英语表现优异，其他有限。Hibiki-Zero 可辅助，但仍需训练数据。
- **资源成本。** 一个完整 Moshi 会话占用一个 GPU 插槽，不适合廉价共享部署。

## 发布它

保存为 `outputs/skill-duplex-pipeline.md`。根据需求选择流水线还是全双工架构实现语音代理，并说明理由。

## 练习

1. **简单。** 运行 `code/main.py`。符号化模拟双流+内心独白架构。  
2. **中等。** 从 HuggingFace 拉取 Moshi，运行服务器，测试一段对话。测量从用户语音结束到 Moshi 开始响应的实际延迟。  
3. **困难。** 拿第12课的流水线代理，和 Moshi 在20个匹配测试语句上对比 P50 延迟。写出流水线在何时架构上仍优势。

## 关键术语

| 术语          | 通俗说法           | 实际含义                                     |
|---------------|--------------------|----------------------------------------------|
| Full-duplex   | 一边听一边说       | 同一模型上同时活动的两个音频流。               |
| Inner monologue | 模型的文本流       | Moshi 除音频输出外同时输出文本标记。             |
| Depth transformer | 码本间预测器      | 小型 Transformer，顺序预测单帧内 8 码本。         |
| Mimi          | Kyutai 编解码器    | 12.5 Hz × 8 码本；包含语义+声学；驱动 Moshi。    |
| Streaming S2S | 音频到音频实时转化 | 逐块语音翻译/对话，无流水线阶段。                 |
| Back-channeling | 回馈式反应         | Moshi 可发出小确认语音如“嗯嗯”，不中断主语句。    |

## 进一步阅读

- [Défossez 等人（2024）。Moshi — 语音-文本基础模型](https://arxiv.org/html/2410.00037v2) — 论文。  
- [Kyutai Labs（2026）。Hibiki-Zero](https://arxiv.org/abs/2602.12345) — 无需对齐数据的流式翻译。  
- [Sesame（2025）。跨越语音的“诡异谷”](https://www.sesame.com/research/crossing_the_uncanny_valley_of_voice) — CSM 规范。  
- [Kyutai — Moshi 代码仓库](https://github.com/kyutai-labs/moshi) — 安装 + 服务器。  
- [OpenAI — Realtime API](https://platform.openai.com/docs/guides/realtime) — 商业封闭同行。  
- [Kyutai — 延迟流建模](https://github.com/kyutai-labs/delayed-streams-modeling) — 底层 STT/TTS 框架。
