# 音乐生成 — MusicGen、Stable Audio、Suno，以及许可领域的地震

> 2026年音乐生成：Suno v5和Udio v4主导商业领域；MusicGen、Stable Audio Open和ACE-Step领跑开源。技术问题大致已解决。法律问题（华纳音乐5亿美元和解、环球音乐和解）在2025-2026年重塑了这一领域。

**类型:** 构建  
**语言:** Python  
**先决条件:** 第6阶段 · 02（频谱图）、第4阶段 · 10（扩散模型）  
**时间:** 约75分钟

## 问题描述

文本 → 30秒至4分钟的音乐片段，包括歌词、人声及结构。三个子问题：

1. **伴奏生成（Instrumental generation）**。文本如“lo-fi嘻哈鼓配温暖的琴声” → 音频。MusicGen、Stable Audio、AudioLDM。
2. **带人声歌词的歌曲生成（Song generation with vocals + lyrics）**。“关于多雨的德州夜晚的乡村歌” → 完整歌曲。Suno、Udio、YuE、ACE-Step。
3. **条件/可控生成（Conditional / controllable）**。扩展现有片段，重新生成桥段，交换风格，stem分离，或修补（inpaint）。Udio的修补+stem分离是2026年必备功能。

## 概念介绍

![音乐生成：token-LM与扩散模型，2026年模型图](../assets/music-generation.svg)

### 基于神经编解码符号（neural-codec tokens）的Token LM

Meta的**MusicGen**（2023年，MIT许可）及众多衍生版本：以文本/旋律嵌入为条件，自回归预测EnCodec符号（32 kHz，4个码本），再解码为音频。参数规模300M至3.3B。基准强大；超过30秒的生成表现不理想。

**ACE-Step**（开源，2026年4月发布4B XL版）扩展了该方法，实现完整歌曲带歌词条件的生成。开源社区最接近Suno的方案。

### 基于梅尔频谱或潜变量的扩散模型

**Stable Audio（2023）**和**Stable Audio Open（2024）**：在压缩音频潜变量上实现潜变量扩散。擅长循环、声音设计和环境质感，但结构化完整歌曲表现不足。

**AudioLDM / AudioLDM2**：基于文本到音频的T2I风格潜变量扩散，支持音乐、音效和语音。

### 混合（产品级）— Suno、Udio、Lyria

闭源权重。可能是自回归编解码器语言模型加基于扩散的声码器，带专用的人声/鼓声/旋律头。Suno v5（2026）是ELO 1293质量领导者。Udio v4增加了修补和stem分离（贝斯、鼓、人声单独下载）。

### 评估方式

- **FAD（Fréchet音频距离）**。使用VGGish或PANN提取特征计算生成音频与真实音频嵌入分布距离，越低越好。MusicGen small：MusicCaps数据集上4.5 FAD；SOTA约3.0。
- **音乐性（主观评估）**。人工偏好。Suno v5 ELO 1293领先。
- **文本-音频匹配度**。利用CLAP评分评估提示和输出。
- **音乐性缺陷**。节拍错误转换，人声音调漂移，30秒以上结构丢失。

## 2026年模型图谱

| 模型 | 参数量 | 生成长度 | 是否有人声 | 许可证类型 |
|------|--------|----------|------------|------------|
| MusicGen-large | 3.3B | 30秒 | 否 | MIT |
| Stable Audio Open | 1.2B | 47秒 | 否 | Stability非商业 |
| ACE-Step XL (2026年4月) | 4B | 超过2分钟 | 是 | Apache-2.0 |
| YuE | 7B | 超过2分钟 | 是，多语言 | Apache-2.0 |
| Suno v5（闭源） | ? | 4分钟 | 是，ELO 1293 | 商业 |
| Udio v4（闭源） | ? | 4分钟 | 是+stem | 商业 |
| Google Lyria 3（闭源） | ? | 实时 | 是 | 商业 |
| MiniMax Music 2.5 | ? | 4分钟 | 是 | 商业 API |

## 法律环境（2025-2026）

- **华纳音乐与Suno和解。** 5亿美元。华纳音乐集团现监管Suno上AI语音相似度、音乐版权和用户生成曲目。类似的环球音乐和解协议也适用于Udio。
- **欧盟AI法案（EU AI Act）+ 加州SB 942法案**：AI生成音乐必须披露。
- **Riffusion / MusicGen采用MIT许可**，无合规负担，但无商业人声。

合规上架方案：

1. 仅生成伴奏（MusicGen、Stable Audio Open、MIT/CC0输出）。
2. 使用带按次许可的商业API（Suno、Udio、ElevenLabs Music）。
3. 在自有或授权曲库上训练（大多数企业采用）。
4. 给生成内容打标水印+元数据。

## 开始构建

### 第一步：使用MusicGen生成音频

```python
from audiocraft.models import MusicGen
import torchaudio

model = MusicGen.get_pretrained("facebook/musicgen-small")
model.set_generation_params(duration=10)
wav = model.generate(["upbeat synthwave with driving drums, 128 BPM"])
torchaudio.save("out.wav", wav[0].cpu(), 32000)
```

三种尺寸：`small`（300M，快速）、`medium`（1.5B）、`large`（3.3B）。small已足够验证“想法是否实现”。

### 第二步：旋律条件生成

```python
melody, sr = torchaudio.load("humming.wav")
wav = model.generate_with_chroma(
    ["jazz piano cover"],
    melody.squeeze(),
    sr,
)
```

MusicGen-melody输入一个旋律色度图，保持曲调，替换音色。适合“用弦乐四重奏演绎这段旋律”。

### 第三步：FAD评估

```python
from frechet_audio_distance import FrechetAudioDistance
fad = FrechetAudioDistance()

fad.get_fad_score("generated_folder/", "reference_folder/")
```

计算VGGish嵌入层的距离。适器类或风格层面的回归测试，非替代人为听感。

### 第四步：集成到LLM-音乐工作流

结合第7至8课思路：

```python
prompt = "Write a 30-second jazz loop. Describe the drums, bass, and piano voicing."
description = llm.complete(prompt)
music = musicgen.generate([description], duration=30)
```

## 用法示例

| 目标 | 技术栈 |
|------|---------|
| 伴奏音效设计 | Stable Audio Open |
| 游戏/自适应音乐 | Google Lyria RealTime（闭源） |
| 带人声完整歌曲（商业） | Suno v5或Udio v4，需明确授权 |
| 带人声完整歌曲（开源） | ACE-Step XL或YuE |
| 短广告歌曲 | 基于哼唱旋律的MusicGen条件生成 |
| 背景音乐视频 | MusicGen + Stable Video Diffusion |

## 2026年仍存在的问题

- **版权规避式提示。** “泰勒·斯威夫特风格的歌曲” — 商业Suno/Udio已过滤，开源模型未过滤。请自行添加过滤列表。
- **30秒后重复及漂移。** 自回归模型易循环。多次生成交叉淡入或用ACE-Step维持结构连贯。
- **节奏漂移。** 模型节拍容易偏移。提示中加入BPM标签，并用librosa的`beat_track`后处理。
- **人声清晰度。** Suno表现优异；开源模型往往词不清晰。歌词重要时，推荐商业API或微调。
- **单声道输出。** 开源模型常生成单声道或伪立体声。可用ezst、Cartesia等立体声重建方案升级。

## 上线准备

保存为`outputs/skill-music-designer.md`。选择模型、许可策略、长度/结构规划及披露元数据，部署音乐生成服务。

## 练习

1. **简单。** 运行`code/main.py`。它生成一段“生成式”和弦进程+鼓点的ASCII符号——音乐生成卡通。可用任何MIDI渲染器播放。
2. **中等。** 安装`audiocraft`，用MusicGen-small针对4个风格提示生成10秒片段，测量FAD与参考风格集对比。
3. **困难。** 用ACE-Step（或MusicGen-melody）生成同一旋律的三种不同音色变体。计算CLAP相似度以验证匹配。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|-------|-------------|------------|
| FAD | 音频FID | 真实与生成音频嵌入分布的Fréchet距离。 |
| Chromagram | 旋律为音高表示 | 每帧12维向量；旋律条件输入。 |
| Stems | 独立轨道 | 分离的贝斯/鼓/人声/旋律WAV文件。 |
| Inpainting | 重生成片段 | 遮罩时间窗口，模型仅重生成该段。 |
| CLAP | 文本-音频CLIP | 对比音频-文本嵌入；评估文字与音频匹配。 |
| EnCodec | 音乐编解码器 | Meta的神经编解码器，MusicGen使用；32 kHz，4码本。 |

## 深度阅读

- [Copet 等 (2023). MusicGen](https://arxiv.org/abs/2306.05284) — 开源自回归基准。  
- [Evans 等 (2024). Stable Audio Open](https://arxiv.org/abs/2407.14358) — 声音设计默认方案。  
- [ACE-Step](https://github.com/ace-step/ACE-Step) — 开源4B完整歌曲生成，2026年4月。  
- [Suno v5平台文档](https://suno.com) — 商业质量领先者。  
- [AudioLDM2](https://arxiv.org/abs/2308.05734) — 音乐+音效潜变量扩散。  
- [WMG-Suno和解报道](https://www.musicbusinessworldwide.com/suno-warner-music-settlement/) — 2025年11月先例。
