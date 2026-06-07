# Whisper — 架构与微调

> Whisper 是一个 30 秒窗口的 transformer encoder-decoder（编码器-解码器），在 68 万小时的多语言弱监督音频文本对上训练而成。一个架构，多任务，支持 99 种语言的鲁棒表现。2026 年的参考 ASR。

**类型：** 构建  
**语言：** Python  
**先决条件：** 第6阶段 · 04（ASR），第5阶段 · 10（Attention），第7阶段 · 05（完整 Transformer）  
**时间：** 约 75 分钟

## 问题

Whisper 由 OpenAI 于 2022 年 9 月发布，是首个作为商品推出的 ASR 模型：粘贴音频，生成文本，支持 99 种语言，抗噪声，能在笔记本上运行。到 2024 年，OpenAI 已发布 Large-v3 和 Turbo 变体；到 2026 年，Whisper 成为从播客转录到语音助手再到 YouTube 字幕的默认基线。

但 Whisper 不是一个你可以永远当作黑盒的流水线。域偏移会让它失效——技术术语、说话人口音、专有名词、短片段、静音。你需要了解：

1. 它内部真正的结构。
2. 如何正确给它提供分块、流式或长格式音频。
3. 何时微调以及如何微调。

## 概念

![Whisper 编码器-解码器、任务、分块推理、微调](../assets/whisper.svg)

**架构。** 标准的 transformer 编码器-解码器。

- 输入：30 秒的对数梅尔谱 (log-mel spectrogram)，80 个梅尔频率，10 毫秒跳跃 → 3000 帧。短片段零填充，长片段分块。
- 编码器：卷积下采样（步幅 2） + `N` 个 transformer 块。Large-v3：32 层，1280 维，20 个头。
- 解码器：`N` 个 transformer 块，带有因果自注意力（causal self-attn）和对编码器输出的交叉注意力。大小与编码器相同。
- 输出：覆盖 51865 词汇的 BPE（Byte Pair Encoding）token（符号）。

Large-v3 有 15.5 亿参数。Turbo 采用 4 层解码器（原 32 层），延迟减少 8 倍，WER（词错误率）下降不到 1%。

**提示词格式。** Whisper 是一个由解码器提示中的特殊 token 控制的多任务模型：

```text
<|startoftranscript|><|en|><|transcribe|><|notimestamps|> Hello world.<|endoftext|>
```

- `<|en|>` — 语言标签，控制翻译与转录行为。
- `<|transcribe|>` 或 `<|translate|>` — 将任何语言输入转录为英文，或逐字翻译。
- `<|notimestamps|>` — 跳过词级时间戳（更快）。

这个提示使一个模型完成多种任务。将 `<|en|>` 改为 `<|fr|>` 即可转录法语。

**30 秒窗口。** 所有处理均固定为 30 秒。更长音频需分块；短音频填充。窗口本身不支持原生流式处理——这就是 WhisperX、Whisper-Streaming 和 faster-whisper 存在的原因。

**对数梅尔规范化。** `(log_mel - mean) / std`，统计数据来自 Whisper 自身训练语料。*必须*使用 Whisper 的预处理方法（`whisper.audio.log_mel_spectrogram`），而非 `librosa.feature.melspectrogram`。

### 2026 年变体

| 变体 | 参数量 | 延迟 (A100) | WER（LibriSpeech-clean） |
|---------|--------|----------------|------------------------|
| Tiny | 3900 万 | 实时 1 倍 | 5.4% |
| Base | 7400 万 | 1× | 4.1% |
| Small | 2.44 亿 | 1× | 3.0% |
| Medium | 7.69 亿 | 1× | 2.7% |
| Large-v3 | 15.5 亿 | 2× | 1.8% |
| Large-v3-turbo | 8.09 亿 | 1/8 延迟 | 1.58% |
| Whisper-Streaming (2024) | 15.5 亿 | 流式 | 2.0% |

### 微调

2026 年经典工作流程：

1. 收集 10–100 小时目标领域音频及其对应转录文本。
2. 使用 `transformers.Seq2SeqTrainer` 运行，有 `generate_with_loss` 回调。
3. 参数高效微调：对注意力层的 `q_proj`、`k_proj`、`v_proj` 应用 LoRA，GPU 显存降低 4 倍，WER 费用<0.3。
4. 如果数据少于 10 小时，则冻结编码器，仅调解码器。
5. 使用 Whisper 自带的分词器和提示格式，切勿更换分词器。

社区成果：用 20 小时医疗口述微调 Medium，医疗词汇 WER 从 12% 降至 4.5%。用 4 小时冰岛语微调 Turbo，WER 从 18% 降到 6%。

## 构建

### 第一步：开箱即用运行 Whisper

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe(
    "clip.wav",
    language="en",
    task="transcribe",
    temperature=0.0,
    condition_on_previous_text=False,  # 防止无限重复
)
print(result["text"])
for seg in result["segments"]:
    print(f"[{seg['start']:.2f}–{seg['end']:.2f}] {seg['text']}")
```

需要始终覆盖的关键默认值：`temperature=0.0`（采样默认链是 0.0 → 0.2 → 0.4 …），`condition_on_previous_text=False`（防止级联幻觉），`no_speech_threshold=0.6`（静音检测）。

### 第二步：分块处理长格式音频

```python
# whisperx 是 2026 年长格式带词级时间戳的参考方案
import whisperx
model = whisperx.load_model("large-v3-turbo", device="cuda", compute_type="float16")
segments = model.transcribe("1hour.mp3", batch_size=16, chunk_size=30)
```

WhisperX 集成 (1) Silero VAD 门控，(2) 通过 wav2vec 2.0 的词级对齐，(3) 通过 `pyannote.audio` 实现说话人分离。2026 年生产转录的主力。

### 第三步：用 LoRA 微调

```python
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import LoraConfig, get_peft_model

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3-turbo")
lora = LoraConfig(
    r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1, bias="none", task_type="SEQ_2_SEQ_LM",
)
model = get_peft_model(model, lora)
# model.print_trainable_parameters()  -> 约 300 万可训练 / 8.09 亿 总参数
```

然后执行标准 Trainer 循环。每 1000 步保存检查点。用持出集计算 WER。

### 第四步：检查每层学到的内容

```python
# 在解码过程中抓取交叉注意力权重，观察解码器关注的编码器内容。
with torch.inference_mode():
    out = model.generate(
        input_features=features,
        return_dict_in_generate=True,
        output_attentions=True,
    )
# out.cross_attentions: 层 × 头 × 步骤 × 源序列长度
```

用热力图可视化——你会看到解码步骤扫描编码器帧时的对角线对齐。那条对角线是 Whisper 的词时间戳概念。

## 使用指南

2026 年使用栈：

| 场景 | 选择 |
|-----------|------|
| 通用英语，离线 | Large-v3-turbo，通过 `whisperx` |
| 移动/边缘设备 | Whisper-Tiny 量化版（int8）或 Moonshine |
| 多语言长格式 | Large-v3 通过 `whisperx` + 说话人分离 |
| 低资源语言 | 用 LoRA 微调 Medium 或 Turbo |
| 流式（2 秒延迟） | Whisper-Streaming 或 Parakeet-TDT |
| 词级时间戳 | WhisperX（通过 wav2vec 2.0 强制对齐） |

`faster-whisper`（基于 CTranslate2）是 2026 年最快的 CPU+GPU 推理运行时——比原版快 4 倍，输出一致。

## 2026 年仍存在的陷阱

- **静音时幻觉文本。** Whisper 在字幕训练中包含“感谢观看！”、“订阅！”、歌曲歌词等。调用前一定要进行 VAD 门控。
- **`condition_on_previous_text` 级联。** 一次幻觉会污染后续窗口。除非需要跨块流畅，否则设为 `False`。
- **短片段填充。** 2 秒音频填充到 30 秒可能在尾部静音产生幻觉。用 `pad=False` 或 VAD 门控。
- **错误的梅尔统计。** 使用 librosa 的梅尔谱会几乎随机输出。一定用 `whisper.audio.log_mel_spectrogram`。

## 发布

保存为 `outputs/skill-whisper-tuner.md`。为特定领域设计一个 Whisper 微调或推理流水线。

## 练习

1. **简单。** 运行 `code/main.py`。它会分词一个 Whisper 风格的提示词，计算解码形状预算，并打印 10 分钟音频的分块时间表。
2. **中等。** 安装 `faster-whisper`，转录 10 分钟播客，比较与人工转录的 WER。尝试 `language="auto"` 与强制 `language="en"`。
3. **困难。** 使用 HF `datasets`，选一个 Whisper 效果差的语言（例如乌尔都语），用 LoRA 微调 Medium，2 轮，2 小时数据，报告 WER 变化。

## 关键术语

| 术语 | 用户说法 | 实际含义 |
|------|-----------------|-----------------------|
| 30秒窗口 | Whisper 限制 | 硬输入上限；分块长音频。 |
| SOT | transcript开始标记 | `<|startoftranscript|>` 触发解码器提示。 |
| 时间戳 token | 时序对齐 | 每 0.02 秒偏移是 51865 词汇表的一个特殊 token。 |
| Turbo | 极速版 | 4 层解码器，速度快 8 倍，WER 下降 <1%。 |
| WhisperX | 长格式包装器 | VAD + Whisper + wav2vec 对齐 + 说话人分离。 |
| LoRA 微调 | 高效调优 | 向注意力添加低秩适配器；训练约 0.3% 参数。 |
| 幻觉 | 沉默失败 | Whisper 从噪音/静音生成人类流畅英文。 |

## 进一步阅读

- [Radford 等人（2022）。Whisper 论文](https://arxiv.org/abs/2212.04356) — 原始架构与训练方案。  
- [OpenAI（2024）。Whisper Large-v3-turbo 发布](https://github.com/openai/whisper/discussions/2363) — 4 层解码器，8 倍加速。  
- [Bain 等（2023）。WhisperX](https://arxiv.org/abs/2303.00747) — 长格式，词对齐，说话人分离。  
- [Systran — faster-whisper 仓库](https://github.com/SYSTRAN/faster-whisper) — 基于 CTranslate2，快 4 倍。  
- [HuggingFace — Whisper 微调教程](https://huggingface.co/blog/fine-tune-whisper) — 经典 LoRA / 全量微调流程。
