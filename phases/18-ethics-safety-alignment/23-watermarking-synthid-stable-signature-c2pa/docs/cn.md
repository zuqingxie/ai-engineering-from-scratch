# 水印技术 — SynthID、Stable Signature、C2PA

> 三项技术构建2026年AI生成内容的溯源体系。SynthID（Google DeepMind）——图像水印于2023年8月推出，文本和视频于2024年5月推出（Gemini + Veo），文本部分于2024年10月通过Responsible GenAI Toolkit开源，统一多媒体检测器于2025年11月随Gemini 3 Pro推出。文本水印通过细微调整下一个token的采样概率实现不可察觉；图像/视频水印能在压缩、裁剪、滤镜、帧率变化后依然存活。Stable Signature（Fernandez等，ICCV 2023，arXiv:2303.15435）——微调潜在扩散解码器，使每个输出包含固定信息；裁剪（内容10%）的生成图像检测准确率超过90%，误报率小于1e-6。后续研究“Stable Signature is Unstable”（arXiv:2405.07145，2024年5月）显示微调可去除水印且保持质量。C2PA——加密签名、篡改证据的元数据标准（C2PA 2.2 Explainer 2025）。水印和C2PA互补：元数据可被剥离但承载更丰富溯源；水印能穿越转码但信息量较少。

**类型：** 构建  
**语言：** Python（标准库，token水印嵌入+检测）  
**先决条件：** 阶段10·04（采样），阶段01·09（信息论）  
**时间：** 约75分钟  

## 学习目标

- 描述token级别水印（SynthID-text风格）及其可检测机制。
- 介绍Stable Signature及2024年击破该水印的去除攻击。
- 说明C2PA的作用及为何与水印技术互补。
- 描述主要限制：模型特定信号、释义重写鲁棒性，及保持语义攻击（arXiv:2508.20228）。

## 问题背景

2023-2024年，deepfake和AI生成内容大规模进入政治及消费者场景。水印技术被提出作为技术溯源信号：在生成时标记内容，后期检测。2025年研究数据显示：没有水印是绝对鲁棒的，但结合C2PA元数据后，提供了可用的溯源方案。

## 概念介绍

### 文本水印（SynthID-text风格）

Kirchenbauer等2023年提出的机制，由Google产品化：

1. 在每一步解码时，哈希前K个token，伪随机将词汇表划分为“绿色”与“红色”集合。
2. 通过对绿色集合的logits加上偏置δ，偏向于采样绿色集合的token。
3. 生成内容中绿色token比随机机会更多。

检测方法：重新哈希每个前缀，统计生成文本中绿色token数量，计算z-score。水印文本z-score >0，人类文本约为0。

属性：
- 对阅读者不可察觉（δ足够小，质量损失微乎其微）。
- 需要访问词表划分函数才能检测。
- 不耐释义重写，重写文本破坏信号。

SynthID-text于2024年10月通过Google Responsible GenAI Toolkit开源。

### Stable Signature（图像）

Fernandez等，ICCV 2023。微调潜在扩散解码器，使每张生成图像在潜在表示中嵌入固定二进制信息。检测通过神经网络从潜在表示解码。裁剪至10%内容图像检测准确率超过90%，误报率小于1e-6。

2024年5月“Stable Signature is Unstable”（arXiv:2405.07145）：微调解码器即可移除水印且保持图像质量。对抗性后期微调成本低，水印的对抗鲁棒有限。

### SynthID统一检测器（2025年11月）

随Gemini 3 Pro推出：一个多媒体检测器，通过单一API读取文本、图像、音频、视频中的SynthID信号。统一Google溯源体系。

### C2PA

内容溯源与真实性联盟。加密签名、防篡改的元数据标准。参考C2PA 2.2 Explainer（2025）。C2PA清单记录溯源声明（谁创建、何时创建、经历何种变换），由创建者密钥签名。

与水印互补：
- 元数据可被剥离；水印难以剥离。
- 元数据丰富，包含完整溯源链；水印仅携带少量信息。
- C2PA依赖平台采纳；水印可自动嵌入。

Google在搜索、广告和“关于这张图”中集成两者。

### 限制

- **模型特定。** SynthID水印只存在于具备SynthID的模型生成；无SynthID信号不代表真实性。
- **释义问题。** 文本水印不耐含义不变的释义重写。
- **变换攻击。** arXiv:2508.20228（2025）展示了破坏文本和众多图像水印的保持语义攻击。
- **微调移除。** “Stable Signature is Unstable”显示后期微调可移除嵌入水印。

### 欧盟AI法案第50条

AI生成内容标识透明守则（首稿2025年12月，二稿2026年3月，预计2026年6月定稿，详见[欧盟委员会状态页](https://digital-strategy.ec.europa.eu/en/policies/code-practice-ai-generated-content)）。截至2026年4月守则仍为草案，时间线存在变动。规制层是技术层的支持。deepfake必须标识。

### 本内容在阶段18中的位置

课程22-23讲解模型输出（私有数据、溯源信号）。课程27涵盖训练数据治理。课程24讲解技术规范要求的监管框架。

## 练习使用

`code/main.py`构建了一个玩具版文本水印。token为0到N-1的整数；采样时偏向哈希定义的绿色集合。检测器计算绿色token的z-score。你可以观察1000token生成文本的检测，目睹释义破坏信号，并测量人类文本的误报率。

## 交付成果

本课生成`outputs/skill-provenance-audit.md`。给定带溯源声明的内容部署，审计包括：水印机制（如有）、C2PA签名链（如有）、各自的对抗鲁棒性、以及各模态覆盖情况。

## 练习题

1. 运行`code/main.py`。报告1000token水印生成文本与人类文本的z-score。确认95%置信阈值下的误报率。

2. 实现释义攻击，替换30%token为同义词。重新测量z-score。

3. 阅读Kirchenbauer等2023年第6节关于鲁棒性的论述。为什么文本水印在释义失败而图像水印能在裁剪后存活？

4. 设计一个部署方案，结合SynthID-text和C2PA元数据。描述消费者看到的溯源链。指出各组件的一个失败模式。

5. 2024年“Stable Signature is Unstable”结果表明微调能移除图像水印。设计一个部署管控以限制此攻击——如要求微调的检查点必须签名发布。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| SynthID | “Google的水印” | 跨模态溯源信号；文本、图像、音频、视频 |
| Token水印 | “Kirchenbauer风格” | 通过绿色token z-score检测的偏采样文本水印 |
| Stable Signature | “图像水印” | 微调解码器水印；ICCV 2023 |
| C2PA | “元数据标准” | 加密签名防篡改的溯源元数据 |
| 释义鲁棒性 | “改写会破坏吗” | 文本水印属性；目前有限 |
| 微调移除 | “对抗性去水印” | 通过解码器微调移除图像水印的攻击 |
| 跨模态检测器 | “统一SynthID” | 2025年11月跨模态统一API |

## 延伸阅读

- [Kirchenbauer等 — 大语言模型的水印（ICML 2023，arXiv:2301.10226）](https://arxiv.org/abs/2301.10226) — token水印机制  
- [Fernandez等 — Stable Signature（ICCV 2023，arXiv:2303.15435）](https://arxiv.org/abs/2303.15435) — 图像水印论文  
- [“Stable Signature is Unstable”（arXiv:2405.07145）](https://arxiv.org/abs/2405.07145) — 移除攻击论文  
- [Google DeepMind — SynthID](https://deepmind.google/models/synthid/) — 跨模态水印  
- [C2PA 2.2 Explainer（2025）](https://c2pa.org/specifications/specifications/2.2/explainer/Explainer.html) — 元数据标准
