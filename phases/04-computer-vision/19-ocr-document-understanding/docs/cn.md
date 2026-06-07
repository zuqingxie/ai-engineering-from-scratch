# OCR（Optical Character Recognition）与文档理解（Document Understanding）

> OCR（光学字符识别，Optical Character Recognition）是一个三阶段的流水线：检测文本框（text detection），识别字符（recognition），然后进行排版（layout）。每个现代 OCR 系统都会重新排序或合并这些阶段。

**类型：** 学习 + 使用  
**语言：** Python  
**先决条件：** 第4阶段第06课（检测），第7阶段第02课（自注意力）  
**时间：** 约45分钟

## 学习目标

- 理解经典 OCR 流水线（检测 -> 识别 -> 排版）及现代端到端替代方案（Donut、Qwen-VL-OCR）  
- 实现序列到序列 OCR 训练用的 CTC（Connectionist Temporal Classification，连接时序分类）损失函数  
- 使用 PaddleOCR 或 EasyOCR 进行无训练的生产环境文档解析  
- 区分 OCR、布局解析和文档理解，并根据任务选择合适工具  

## 双语术语速查

| 中文术语 | English term |
|----------|--------------|
| 光学字符识别 | Optical Character Recognition, OCR |
| 文档理解 | Document understanding |
| 文本检测 | Text detection |
| 文本识别 | Text recognition |
| 版面分析 | Layout analysis |
| 布局解析 | Layout parsing |
| 阅读顺序 | Reading order |
| 连接时序分类 | Connectionist Temporal Classification, CTC |
| CRNN | Convolutional Recurrent Neural Network, CRNN |
| 双向 LSTM | Bidirectional LSTM, BiLSTM |
| 字符错误率 | Character error rate, CER |
| 词错误率 | Word error rate, WER |
| Levenshtein 距离 | Levenshtein distance |
| 端到端 OCR | End-to-end OCR |
| Donut | Document Understanding Transformer, Donut |
| 视觉语言 OCR | VLM-OCR |


## 问题描述

到处都是充满文本的图像：收据、发票、身份证、扫描书籍、表格、白板、标识、截图。从中提取结构化数据——不仅是字符，还有“这是总金额”这类信息——是计算机视觉领域非常有价值的应用问题之一。

这个领域分为三个技能层次：

1. **OCR本身**：将像素转换为文本。  
2. **布局解析**：将 OCR 输出分组为区域（标题、正文、表格、页眉）。  
3. **文档理解**：从布局中提取结构化字段（例如“invoice_total = $42.50”）。  

每个层次都有经典和现代方法，而“我想从图像中得到文本”与“我需要这张收据的总金额”之间的差距，比大多数团队想象的还要大。

## 概念解析

### 关键公式（Key equations）

序列式 OCR 常用 CTC 损失，把所有能折叠成目标文本 $y$ 的对齐路径概率相加：

$$
P(y \mid x) = \sum_{\pi \in \mathcal{B}^{-1}(y)} \prod_{t=1}^{T} P(\pi_t \mid x)
$$

$$
\mathcal{L}_{\mathrm{CTC}} = -\log P(y \mid x)
$$

### 经典流水线

```mermaid
flowchart LR
    IMG["图像"] --> DET["文本检测<br/>(DB, EAST, CRAFT)"]
    DET --> BOX["单词/行<br/>边界框"]
    BOX --> CROP["裁剪每个区域"]
    CROP --> REC["识别<br/>(CRNN + CTC)"]
    REC --> TXT["文本字符串"]
    TXT --> LAY["布局<br/>排序"]
    LAY --> OUT["阅读顺序文本"]

    style DET fill:#dbeafe,stroke:#2563eb
    style REC fill:#fef3c7,stroke:#d97706
    style OUT fill:#dcfce7,stroke:#16a34a
```

- **文本检测**生成每行或每词的四边形区域。  
- **识别**将每个区域裁剪至固定高度，运行 CNN + BiLSTM + CTC 生成字符序列。  
- **布局**根据语言（拉丁文为自上而下、自左向右，阿拉伯文、日文等则不同）重建阅读顺序。  

### CTC 一段话解释

OCR 识别从固定长度的特征图生成可变长序列。CTC（Graves 等，2006年）允许无需字符级对齐进行训练。模型在每个时间步输出一个（词汇 + 空白）分布；CTC 损失对所有合并重复字符并移除空白后变为目标文本的排列进行边缘化。

```text
原始输出: "h h h _ _ e e l l _ l l o _ _"
合并重复并移除空白后: "hello"
```

CTC 是 2015 年 CRNN 成功训练及至 2026 年多数生产 OCR 模型仍用的关键。

### 现代端到端模型

- **Donut**（Kim 等，2022）—ViT 编码器 + 文本解码器；从图像直接输出 JSON，无需文本检测或布局模块。  
- **TrOCR** — ViT + Transformer 解码器，行级 OCR。  
- **Qwen-VL-OCR / InternVL** — 视觉语言大模型微调后的 OCR 任务，2026 年在复杂文档准确率最佳。  
- **PaddleOCR** — 传统 DB + CRNN 管线的成熟生产方案，仍是开源主力。  

端到端模型需求更多数据和计算，但避免了多阶段流水线中的误差累积。

### 布局解析

对于结构化文档，运行布局检测器（如 LayoutLMv3、DocLayNet）对区域标注：标题、段落、图形、表格、脚注等。阅读顺序即“依布局顺序迭代区域并串联”。

对于表单，使用**键值提取**模型（Donut 用于视觉丰富文档，LayoutLMv3 用于纯扫描件），输入图像 + 检测文本 + 位置，预测结构化键值对。

### 评估指标

- **字符错误率 (CER)** — Levenshtein 距离 / 参考文本长度，越低越好。生产目标：清晰扫描件 < 2%。  
- **词错误率 (WER)** — 同理，但以词为单位。  
- **结构化字段的 F1 分数** — 用于键值任务，衡量如 `{invoice_total: 42.50}` 是否正确出现。  
- **JSON 编辑距离** — 用于端到端文档解析，Donut 论文提出了归一化树编辑距离。  

## 实现步骤

### 第一步：CTC 损失 + 贪心解码器

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=0):
    """
    log_probs:      (T, N, C) 对词汇（包含空白，索引0）取对数的 softmax
    targets:        (N, S) 整数目标（无空白）
    input_lengths:  (N,) 每个样本时间步数
    target_lengths: (N,) 每个样本目标长度
    """
    return F.ctc_loss(log_probs, targets, input_lengths, target_lengths,
                      blank=blank, reduction="mean", zero_infinity=True)


def greedy_ctc_decode(log_probs, blank=0):
    """
    log_probs: (T, N, C) 对数 softmax
    返回: 去除空白和重复的索引序列列表
    """
    preds = log_probs.argmax(dim=-1).transpose(0, 1).cpu().tolist()
    out = []
    for seq in preds:
        decoded = []
        prev = None
        for idx in seq:
            if idx != prev and idx != blank:
                decoded.append(idx)
            prev = idx
        out.append(decoded)
    return out
```

`F.ctc_loss` 会在有条件时调用高效的 CuDNN 实现。贪心解码器比束搜索简单，通常 CER 误差在1%以内。

### 第二步：微型 CRNN 识别器

极简 CNN + BiLSTM 用于行 OCR。

```python
class TinyCRNN(nn.Module):
    def __init__(self, vocab_size=40, hidden=128, feat=32):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, feat, 3, 1, 1), nn.BatchNorm2d(feat), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(feat, feat * 2, 3, 1, 1), nn.BatchNorm2d(feat * 2), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(feat * 2, feat * 4, 3, 1, 1), nn.BatchNorm2d(feat * 4), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(feat * 4, feat * 4, 3, 1, 1), nn.BatchNorm2d(feat * 4), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
        )
        self.rnn = nn.LSTM(feat * 4, hidden, bidirectional=True, batch_first=True)
        self.head = nn.Linear(hidden * 2, vocab_size)

    def forward(self, x):
        # x: (N, 1, H, W)
        f = self.cnn(x)                # (N, C, H', W')
        f = f.mean(dim=2).transpose(1, 2)  # (N, W', C)
        h, _ = self.rnn(f)
        return F.log_softmax(self.head(h).transpose(0, 1), dim=-1)  # (W', N, vocab)
```

输入为固定高度（CNN通过最大池化使高度变为1），宽度作为 CTC 的时间维度。

### 第三步：合成 OCR 数据

生成黑底白字的数字字符串，用于端到端烟雾测试。

```python
import numpy as np

def synthetic_line(text, height=32, char_width=16):
    W = char_width * len(text)
    img = np.ones((height, W), dtype=np.float32)
    for i, c in enumerate(text):
        x = i * char_width
        shade = 0.0 if c.isalnum() else 0.5
        img[6:height - 6, x + 2:x + char_width - 2] = shade
    return img


def build_batch(strings, vocab):
    H = 32
    W = 16 * max(len(s) for s in strings)
    imgs = np.ones((len(strings), 1, H, W), dtype=np.float32)
    target_lengths = []
    targets = []
    for i, s in enumerate(strings):
        imgs[i, 0, :, :16 * len(s)] = synthetic_line(s)
        ids = [vocab.index(c) for c in s]
        targets.extend(ids)
        target_lengths.append(len(ids))
    return torch.from_numpy(imgs), torch.tensor(targets), torch.tensor(target_lengths)


vocab = ["_"] + list("0123456789abcdefghijklmnopqrstuvwxyz")
imgs, targets, lengths = build_batch(["hello", "world"], vocab)
print(f"图片: {imgs.shape}   目标: {targets.shape}   长度: {lengths.tolist()}")
```

真实 OCR 数据集会增加字体、噪声、旋转、模糊和颜色。流程基本相同。

### 第四步：训练示范

```python
model = TinyCRNN(vocab_size=len(vocab))
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

for step in range(200):
    strings = ["abc" + str(step % 10)] * 4 + ["xyz" + str((step + 1) % 10)] * 4
    imgs, targets, target_lens = build_batch(strings, vocab)
    log_probs = model(imgs)  # (W', 8, vocab)
    input_lens = torch.full((8,), log_probs.size(0), dtype=torch.long)
    loss = ctc_loss(log_probs, targets, input_lens, target_lens, blank=0)
    opt.zero_grad(); loss.backward(); opt.step()
```

在这个简单的合成数据上，损失应从约 3 降至约 0.2，进行 200 步。

## 使用指南

三种生产环境路径：

- **PaddleOCR** — 成熟、快速、多语言。单行使用：`paddleocr.PaddleOCR(lang="en").ocr(image_path)`。  
- **EasyOCR** — Python 原生，多语言，基于 PyTorch。  
- **Tesseract** — 经典 OCR，当模型难以处理老旧扫描文档时仍有用。  

端到端文档解析使用 Donut 或视觉语言模型：

```python
from transformers import DonutProcessor, VisionEncoderDecoderModel

processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
```

针对收据、发票和结构重复的表单，微调 Donut。对于任意文档或需要推理的 OCR，像 Qwen-VL-OCR 这类 VLM 是当前默认选择。

## 交付成果

本课产出：

- `outputs/prompt-ocr-stack-picker.md` — 根据文档类型、语言和结构选择 Tesseract / PaddleOCR / Donut / VLM-OCR 的提示词。  
- `outputs/skill-ctc-decoder.md` — 从零实现贪心和束搜索 CTC 解码器的技能，包括长度归一化。  

## 练习

1. **（简单）** 在5位随机数字字符串上训练 TinyCRNN 500步。报告在留出集上的 CER。  
2. **（中等）** 将贪心解码替换为束搜索（beam_width=5）。报告 CER 变化。束搜索在哪些输入上效果更好？  
3. **（困难）** 使用 PaddleOCR 处理 20 张收据，提取行项目，计算 `{item_name, price}` 对的人工标注真值的 F1。  

## 关键词

| 术语 | 一般说法 | 实际含义 |
|------|----------------|----------------------|
| OCR | “从像素得到文本” | 将图像区域转为字符序列 |
| CTC | “无对齐损失” | 在无逐步标签情况下训练序列模型的损失；对所有对齐方式边缘化 |
| CRNN | “经典 OCR 模型” | 卷积特征提取 + BiLSTM + CTC；2015年基线，仍用于生产 |
| Donut | “端到端 OCR” | ViT 编码器 + 文本解码器；直接从图像输出 JSON |
| Layout parsing（布局解析） | “找到区域” | 检测并标注文档中标题/表格/图形/段落区域 |
| Reading order（阅读顺序） | “文本序列” | 识别区域排序成句子；拉丁文简单，混合布局复杂 |
| CER / WER | “错误率” | 字符或词级的 Levenshtein 距离 / 参考长度 |
| VLM-OCR | “能读文本的 LLM” | 已训练或提示用于 OCR 的视觉语言模型；复杂文档当前 SOTA |

## 深入阅读

- [CRNN（Shi 等，2015）](https://arxiv.org/abs/1507.05717) — 原始的 CNN+RNN+CTC 架构
- [CTC（Graves 等，2006）](https://www.cs.toronto.edu/~graves/icml_2006.pdf) — 原始的 CTC 论文；算法思想密集呈现
- [Donut（Kim 等，2022）](https://arxiv.org/abs/2111.15664) — 无 OCR 的文档理解 Transformer（Transformer 架构）
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) — 开源生产级 OCR 技术栈
