# 音频生成

> 音频是16-48 kHz的1维信号。一个五秒的片段包含80-240k个采样点。没有任何Transformer（Transformer 架构）能够直接处理这样的序列。2026年每个生产级音频模型的解决方案都是一样的：一个神经编解码器（neural codec，Encodec、SoundStream、DAC）将音频压缩成50-75 Hz的离散tokens，然后由Transformer或扩散模型生成tokens。

**类型：** 构建  
**语言：** Python  
**先决条件：** 第6阶段 · 02（音频特征）、第6阶段 · 04（语音识别ASR）、第8阶段 · 06（扩散模型DDPM）  
**时间：** 约45分钟

## 问题

三种音频生成任务：

1. **文本转语音（Text-to-speech）。** 给定文本，生成语音。干净的语音属于窄带，具有强烈的语音结构——此类任务通过基于tokens的Transformer解决得很好。代表作有VALL-E（微软）、NaturalSpeech 3、ElevenLabs、OpenAI TTS。
2. **音乐生成。** 给定提示（文本、旋律、和弦进程、风格），生成音乐。分布更广。例子有MusicGen（Meta）、Stable Audio 2.5、Suno v4、Udio、Riffusion。
3. **音频效果 / 声音设计。** 给定提示，生成环境声或拟音（Foley）。示例有AudioGen、AudioLDM 2、Stable Audio Open。

这三者都运行在同一基底：神经音频编解码器加上token自回归（token-AR）或扩散式生成器。

## 概念

![音频生成：编解码器tokens + Transformer或扩散模型](../assets/audio-generation.svg)

### 神经音频编解码器

Encodec（Meta，2022）、SoundStream（谷歌，2021）、Descript音频编解码器（DAC，2023）。卷积编码器将波形压缩为每个时间步的向量；残差向量量化（Residual Vector Quantization，RVQ）将每个向量转换为K级码本索引的级联。解码器逆向操作。用8个RVQ码本在75 Hz下编码24 kHz音频可实现2 kbps压缩率，相当于600 tokens/秒。

```text
波形 (16000 采样/秒)
    └─ 编码器卷积 ─┐
                     ├─ RVQ 层 1 → 75 Hz 采样的索引
                     ├─ RVQ 层 2 → 75 Hz 采样的索引
                     ├─ ...
                     └─ RVQ 层 8
```

### 其上层的两种生成范式

**Token自回归。** 将RVQ tokens展平为序列，运行纯解码器Transformer。MusicGen采用“延迟并行”方案同时发射K个码本流，流间有偏移。VALL-E从文本提示加3秒语音样本中生成语音tokens。

**潜在扩散。** 将编解码器tokens作为连续潜变量，或用分类扩散模型建模。Stable Audio 2.5对连续音频潜变量使用流匹配（flow matching）。AudioLDM 2采用文本到梅尔频谱再到音频的扩散。

2024-2026年趋势：流匹配在音乐生成中领先（推理更快，样本更干净），而token-AR依然主导语音生成，因为它天生因果、适合流式处理。

## 生产态势

| 系统                 | 任务              | 主干架构                        | 延迟               |
|----------------------|-------------------|--------------------------------|--------------------|
| ElevenLabs V3        | 文本转语音(TTS)    | Token-AR + 神经声码器            | 约300毫秒首token延迟  |
| OpenAI GPT-4o 音频   | 全双工语音         | 端到端多模态AR                  | 约200毫秒           |
| NaturalSpeech 3      | 文本转语音(TTS)    | 潜变量流匹配                    | 非流式               |
| Stable Audio 2.5     | 音乐 / 音效        | DiT + 音频潜变量流匹配           | 1分钟音频约十秒      |
| Suno v4              | 全曲              | 未公开；疑似token-AR            | 每首歌约30秒        |
| Udio v1.5            | 全曲              | 未公开                         | 每首歌约30秒        |
| MusicGen 3.3B        | 音乐              | 基于Encodec 32kHz的Token-AR     | 实时                |
| AudioCraft 2         | 音乐 + 音效        | 流匹配                         | 5秒生成5秒片段      |
| Riffusion v2         | 音乐              | 频谱扩散                       | 约10秒               |

## 构建它

`code/main.py`模拟核心思想：训练一个小型的下一token预测Transformer，基于两种不同“风格”（风格A为交替高低tokens，风格B为单调斜坡）合成生成的“音频token”序列。对风格进行条件化，然后采样。

### 第1步：合成音频tokens

```python
def make_tokens(style, length, vocab_size, rng):
    if style == 0:  # “类语音”：交替
        return [i % vocab_size for i in range(length)]
    # “类音乐”：斜坡
    return [(i * 3) % vocab_size for i in range(length)]
```

### 第2步：训练小型token预测器

基于二元组（bigram）风格的预测器，带风格条件输入。核心在于流程：编解码器token → 交叉熵训练 → 自回归采样。

### 第3步：条件采样

给定风格token和起始token，从预测分布中采样下一个token，继续采样20-40个token。

## 注意事项

- **编码器质量限制输出质量。** 编解码器无法忠实表示的声音，任何生成器质量都无济于事。DAC目前是开放领域的最佳。
- **RVQ误差累积。** 每层RVQ建模上一层残差，第一层出现错误会传播。对更高层使用温度为0采样可缓解。
- **音乐结构。** 30秒tokens在75Hz下超过两万个token，Transformer难以直接处理。MusicGen采用滑动窗口和提示延续；Stable Audio用更短片段加交叉淡出。
- **边界处的伪影。** 生成片段切换时交叉淡出需要细致处理。
- **干净数据需求大。** 音乐生成器需要数万小时授权音乐。2024年Suno / Udio因RIAA诉讼事件引发关注。
- **声音克隆伦理。** 仅凭3秒样本和文本提示，VALL-E、XTTS、ElevenLabs可克隆声音。所有生产模型需具备滥用检测和拒绝列表。

## 使用它

| 任务                 | 2026年技术栈                               |
|----------------------|--------------------------------------------|
| 商业文本转语音       | ElevenLabs、OpenAI TTS或Azure Neural      |
| 语音克隆（已验证同意）| XTTS v2（开源）或ElevenLabs Pro             |
| 背景音乐、快速生成    | Stable Audio 2.5 API、Suno、Udio           |
| 带歌词的音乐         | Suno v4或Udio v1.5                          |
| 音效 / 拟音          | AudioCraft 2、ElevenLabs音效、Stable Audio Open |
| 实时语音代理         | GPT-4o实时版或Gemini Live                   |
| 开源音乐研究         | MusicGen 3.3B、Stable Audio Open 1.0、AudioLDM 2 |
| 配音 / 翻译          | HeyGen、ElevenLabs配音                       |

## 发布它

保存 `outputs/skill-audio-brief.md`。该skill接受音频简报（任务、时长、风格、声音、授权），输出模型及托管方案，提示格式（风格标签、描述、结构标记），编码器+生成器+声码器链，随机种子协议及评估计划（MOS、CLAP打分、TTS字符识别率CER、用户A/B测试）。

## 练习

1. **简单。** 运行`code/main.py`，明确指定风格。验证生成序列符合风格模式。
2. **中等。** 添加延迟并行解码：模拟2条保持1步偏移的token流。训练联合预测器。
3. **困难。** 使用HuggingFace Transformers本地运行MusicGen-small。生成10秒片段，使用3个不同提示；对比风格遵循情况。

## 关键词

| 术语            | 常见说法             | 实际含义                                      |
|-----------------|----------------------|-----------------------------------------------|
| Codec           | “神经压缩”          | 音频的编码器/解码器；典型输出为50-75 Hz的tokens。   |
| RVQ             | “残差矢量量化”       | K级量化器级联；每层量化上层残差。                   |
| Token           | “一个codec符号”       | 码本的离散索引；典型大小1024或2048。                 |
| Delayed parallel| “偏移式码本”         | 同时发射K条token流，令偏移错开以缩短序列长度。       |
| Flow matching   | “2024年音频赢家”     | 扩散替代方案，路径更直，采样更快。                     |
| Voice prompt    | “3秒采样”            | 说话人嵌入或token前缀，用以驱动克隆声音。             |
| Mel spectrogram | “可视频谱图”         | 对数幅度感知频谱图，很多TTS系统使用。                 |
| Vocoder        | “梅尔到波形”          | 将梅尔频谱神经网络转换回波形。                        |

## 生产提醒：音频是流式问题

音频是用户期望“边生成边接收”的唯一输出模态，而非一次性获得。生产环境中关注TPOT（Time Per Output Token，单位输出token时间），因为用户的“听速”就是系统目标吞吐量，而非阅读速率。对于16kHz音频，以约75 tokens/秒的编码速率（Encodec），服务器必须为每个用户生成≥75 tokens/秒以确保无缝播放。

两条架构推论：

- **流匹配音频模型无法简单流式。** Stable Audio 2.5和AudioCraft 2一次渲染固定时长音频。若流式播放，需要分块并在边界重叠——类似滑动窗口扩散——带来100-300ms的额外延迟，相较于编解码器AR模型。
- 若产品是“实时语音聊天”或“实时音乐续写”，选编解码器AR方案；若是“提交后渲染30秒片段”，流匹配方案在质量和总延迟上占优。

## 进一步阅读

- [Défossez et al. (2022). Encodec: High Fidelity Neural Audio Compression](https://arxiv.org/abs/2210.13438) — 编解码器标准。
- [Zeghidour et al. (2021). SoundStream](https://arxiv.org/abs/2107.03312) — 第一款广泛应用的神经音频编解码器。
- [Kumar et al. (2023). High-Fidelity Audio Compression with Improved RVQGAN (DAC)](https://arxiv.org/abs/2306.06546) — DAC。
- [Wang et al. (2023). Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers (VALL-E)](https://arxiv.org/abs/2301.02111) — VALL-E。
- [Copet et al. (2023). Simple and Controllable Music Generation (MusicGen)](https://arxiv.org/abs/2306.05284) — MusicGen。
- [Liu et al. (2023). AudioLDM 2: Learning Holistic Audio Generation with Self-supervised Pretraining](https://arxiv.org/abs/2308.05734) — AudioLDM 2。
- [Stability AI (2024). Stable Audio 2.5](https://stability.ai/news/introducing-stable-audio-2-5) — 流匹配驱动的2025年文本生成音乐。
