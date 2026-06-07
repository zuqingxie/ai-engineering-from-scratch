# 音频-语言模型 — Qwen2.5-Omni、Audio Flamingo、GPT-4o Audio

> 2026 年的音频-语言模型能够对语音 + 环境音 + 音乐进行推理。Qwen2.5-Omni-7B 在 MMAU-Pro 上匹配 GPT-4o Audio。Audio Flamingo Next 在 LongAudioBench 上击败 Gemini 2.5 Pro。开源与闭源模型之间的差距基本上已消除——除多音频任务外，所有模型的表现都接近随机水平。

**类型:** 学习  
**语言:** Python  
**前置知识:** 第6阶段 · 04 (自动语音识别 ASR)、第12阶段 · 03 (视觉-语言模型)、第7阶段 · 10 (音频 Transformer（Transformer 架构）)  
**时间:** 约 45 分钟

## 问题

你有 5 秒音频：狗叫，有人喊“停！”，然后是安静。实用问题跨越多个维度：

- **转录。** “说了什么？”——自动语音识别领域。  
- **语义推理。** “此人是否有危险？”——需要联合理解狗叫 + 呼喊 + 安静。  
- **音乐推理。** “是什么乐器演奏旋律？”  
- **长音频检索。** “在这场 90 分钟讲座里，讲师在哪讲解了梯度下降（gradient descent）？”

用一个提示就能回答这些问题的单一模型称为**音频-语言模型（LALM / ALM）**。与纯 ASR 不同：LALM 产生自由形式的自然语言答案，而非仅仅是转录文本。

## 概念

![音频-语言模型：音频编码器 + 投影器 + 大语言模型解码器](../assets/alm-architecture.svg)

### 三组件模板

每个 2026 年的 LALM 具有相同骨架：

1. **音频编码器。** Whisper 编码器 · BEATs · CLAP · WavLM · 或每个模型的定制编码器。  
2. **投影器。** 线性或 MLP，将音频编码器特征桥接到大语言模型的词元嵌入空间。  
3. **大语言模型（LLM）。** 基于 Llama / Qwen / Gemma 的解码器。接受交错文本 + 音频词元，生成文本。

训练流程：

- **第一阶段。** 冻结编码器 + LLM；只训练投影器，使用 ASR / 字幕数据。  
- **第二阶段。** 在指令跟随音频任务（问答、推理、音乐理解）上全量或 LoRA 微调。  
- **第三阶段（可选）。** 语音输入 / 语音输出增加语音解码器。Qwen2.5-Omni 和 AF3-Chat 实现了这一点。

### 2026 年模型图谱

| 模型 | 主干架构 | 音频编码器 | 输出模式 | 访问方式 |
|-------|----------|---------------|-----------------|--------|
| Qwen2.5-Omni-7B | Qwen2.5-7B | 定制 + Whisper | 文本 + 语音 | Apache-2.0 许可证 |
| Qwen3-Omni | Qwen3 | 定制 | 文本 + 语音 | Apache-2.0 许可证 |
| Audio Flamingo 3 | Qwen2 | AF-CLAP | 文本 | NVIDIA 非商业许可 |
| Audio Flamingo Next | Qwen2 | AF-CLAP v2 | 文本 | NVIDIA 非商业许可 |
| SALMONN | Vicuna | Whisper + BEATs | 文本 | Apache-2.0 许可证 |
| LTU / LTU-AS | Llama | CAV-MAE | 文本 | Apache-2.0 许可证 |
| GAMA | Llama | AST + Q-Former | 文本 | Apache-2.0 许可证 |
| Gemini 2.5 Flash/Pro（闭源） | Gemini | 专有 | 文本 + 语音 | API |
| GPT-4o Audio（闭源） | GPT-4o | 专有 | 文本 + 语音 | API |

### 基准真实性检验（2026）

**MMAU-Pro。** 1800 道涵盖语音/声音/音乐/混合的问答，含多音频子集。

| 模型 | 总体 | 语音 | 声音 | 音乐 | 多音频 |
|-------|---------|--------|-------|-------|-------------|
| Gemini 2.5 Pro | 约 60% | 73.4% | 51.9% | 64.9% | 约 22% |
| Gemini 2.5 Flash | 约 57% | 73.4% | 50.5% | 64.9% | 21.2% |
| GPT-4o Audio | 52.5% | — | — | — | 26.5% |
| Qwen2.5-Omni-7B | 52.2% | 57.4% | 47.6% | 61.5% | 约 20% |
| Audio Flamingo 3 | 约 54% | — | — | — | — |
| Audio Flamingo Next | LongAudioBench 最佳 | — | — | — | — |

**“多音频”栏目对所有模型都极其不利。** 四选一随机猜测概率为 25%；大部分模型得分就在周边。LALM 依然难以比较两个音频片段。

### 2026 年 LALM 有用场景

- **呼叫中心录音合规审计。** “代理是否提到必需的告知内容？”  
- **无障碍辅助。** 向听力障碍用户描述声音事件（不仅仅是转录）。  
- **内容审核。** 检测暴力语言 + 威胁语气 + 背景环境。  
- **播客 / 会议章节划分。** 语义摘要，而非仅仅基于说话人切换。  
- **音乐目录分析。** “找出所有含 B 段调性变化的曲目。”

### 仍不适用场景

- 精细音乐理论（低于和弦层级）。  
- 长对话中的说话人归因推理（超过 10 分钟性能下降明显）。  
- 多音频比较（22-26% 仅略高于随机）。  
- 实时流推理（大多数为离线批处理推理）。

## 实战演练

### 第一步：调用 Qwen2.5-Omni

```python
from transformers import AutoModelForCausalLM, AutoProcessor

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Omni-7B", torch_dtype="auto")

audio, sr = load_wav("clip.wav", sr=16000)
messages = [{
    "role": "user",
    "content": [
        {"type": "audio", "audio": audio},
        {"type": "text", "text": "你听到了哪些声音，发生了什么事？"},
    ],
}]
inputs = processor.apply_chat_template(messages, tokenize=True, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=200)
print(processor.decode(output[0], skip_special_tokens=True))
```

### 第二步：投影器模式

```python
import torch.nn as nn

class AudioProjector(nn.Module):
    def __init__(self, audio_dim=1280, llm_dim=4096):
        super().__init__()
        self.down = nn.Linear(audio_dim, llm_dim)
        self.act = nn.GELU()
        self.up = nn.Linear(llm_dim, llm_dim)

    def forward(self, audio_features):
        return self.up(self.act(self.down(audio_features)))
```

就是这样。投影器通常为 1 到 3 层线性层。在 ASR 对（音频 → 转录）上训练它是第一阶段的预训练任务。

### 第三步：MMAU / LongAudioBench 基准测试

```python
from datasets import load_dataset
mmau = load_dataset("MMAU/MMAU-Pro")

correct = 0
for item in mmau["test"]:
    answer = call_model(item["audio"], item["question"], item["choices"])
    if answer == item["correct_choice"]:
        correct += 1
print(f"准确率: {correct / len(mmau['test']):.3f}")
```

分别报告各类别（语音 / 声音 / 音乐 / 多音频）的准确率。总体数据会掩盖模型失败的具体领域。

## 使用建议

| 任务 | 2026 年优选模型 |
|------|-----------|
| 自由形式音频问答（开源） | Qwen2.5-Omni-7B |
| 长音频表现最佳（开源） | Audio Flamingo Next |
| 最佳闭源模型 | Gemini 2.5 Pro |
| 语音输入 / 语音输出代理 | Qwen2.5-Omni 或 GPT-4o Audio |
| 音乐推理 | Audio Flamingo 3 或 2（音乐专用 AF-CLAP） |
| 呼叫中心审计 | 通过 API 调用 Gemini 2.5 Pro，结合 RAG 在策略文档上检索 |

## 陷阱

- **对多音频过度信任。** 如果任务是“哪个片段有 X 特征”，那么随机水平表现是真实存在的。  
- **长音频性能下降。** 超过 10 分钟后，大多数模型的说话人归属表现衰减。应先说话人分离（第6课），再进行总结。  
- **沉默中的幻觉生成。** LALM 继承了基于 Whisper 编码器的同样问题。需使用 VAD 门控。  
- **基准结果挑选。** 厂商博客往往突出其最佳表现类别。请自行运行 MMAU-Pro 的多音频子集。

## 部署方案

保存为 `outputs/skill-alm-picker.md`。针对特定音频理解任务选择 LALM + 基准子集 + 输出模式（文本 vs 语音）。

## 练习

1. **简单。** 运行 `code/main.py`，观察一个玩具投影器模式 + 伪 LALM 路由（音频嵌入，文本词元）→ 输出词元。  
2. **中等。** 在 100 条 MMAU-Pro 语音题目上评估 Qwen2.5-Omni-7B，与你查看的论文数字对比。  
3. **困难。** 构建最简音频描述基线：BEATs 编码器 + 两层投影器 + 冻结 Llama-3.2-1B，只在 AudioCaps 上微调投影器。和 SALMONN 在 Clotho-AQA 上比较。

## 关键词

| 术语 | 常用说法 | 实际含义 |
|------|---------|---------|
| LALM | 音频版 ChatGPT | 音频编码器 + 投影器 + 大语言模型解码器。 |
| 投影器 | 适配器 | 小型 MLP，将音频特征映射到大语言模型嵌入空间。 |
| MMAU | 基准数据集 | 包含语音、声音、音乐的 1 万条音频问答对。 |
| MMAU-Pro | 难度更大 | 1800 条多音频 / 重推理问答。 |
| LongAudioBench | 长音频评测 | 含多分钟音频的语义查询。 |
| 语音输入 / 语音输出 | 原生语音处理 | 模型直接接收语音并输出语音，无需文本转接。 |

## 拓展阅读

- [Chu et al. (2024). Qwen2-Audio](https://arxiv.org/abs/2407.10759) — 参考架构。  
- [Alibaba (2025). Qwen2.5-Omni](https://huggingface.co/Qwen/Qwen2.5-Omni-7B) — 语音输入语音输出。  
- [NVIDIA (2025). Audio Flamingo 3](https://arxiv.org/abs/2507.08128) — 开源长音频领导者。  
- [NVIDIA (2026). Audio Flamingo Next](https://arxiv.org/abs/2604.10905) — LongAudioBench 状态最佳。  
- [Tang et al. (2023). SALMONN](https://arxiv.org/abs/2310.13289) — 双编码器先驱。  
- [MMAU-Pro 排行榜](https://mmaubenchmark.github.io/) — 2026 年实时排名。
