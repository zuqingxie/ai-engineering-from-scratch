# Capstone 04 — 多模态文档问答（以视觉优先的 PDF、表格、图表）

> 2026 年的文档问答前沿已从 OCR-先文本（OCR-then-text）转向视觉优先的后期交互（vision-first late interaction）。ColPali、ColQwen2.5 和 ColQwen3-omni 将每个 PDF 页视为图像，以多向量后期交互方式嵌入，让查询直接关注图像块。针对财务 10-K 报告、科学论文和手写笔记，这种模式远超 OCR-先文本的效果。在 1 万页文档上端到端构建管道，并发布与 OCR-先文本的对比结果。

**类型：** Capstone（顶点项目）  
**语言：** Python（管道），TypeScript（查看器 UI）  
**先决条件：** 第 4 阶段（计算机视觉）、第 5 阶段（NLP）、第 7 阶段（transformers（Transformer 架构））、第 11 阶段（大语言模型工程）、第 12 阶段（多模态）、第 17 阶段（基础设施）  
**涉及阶段：** P4 · P5 · P7 · P11 · P12 · P17  
**时间：** 30 小时

## 问题

企业持有的 PDF 通常被 OCR 流程破坏：扫描的 10-K 报告带旋转表格、科学论文密布公式、图表仅能以图像形式理解、手写批注。以文本优先处理意味着丢失一半信号。2026 年的答案是对原始页图像进行后期交互多向量检索。ColPali（Illuin Tech）首创此法；ColQwen2.5-v0.2 和 ColQwen3-omni 提升了准确率。在 ViDoRe v3 数据集上，视觉优先检索显著领先 OCR-先文本，在图表、表格和手写笔记的差距更大。

这带来存储和延迟的权衡。ColQwen 嵌入是每页约 2048 个 PATCH 向量，而非单一 1024 维向量。原始存储膨胀。DocPruner（2026 年）实现 50% 剪枝，准确率几乎无损。你会索引 1 万页，测量 ViDoRe v3 nDCG@5，保证回答延迟低于 2 秒，并与 OCR-先文本基线直接对比。

## 概念

后期交互（late interaction）意味着每个查询 token 与每个图像块 token 评分，每个查询 token 取最大分数后求和。无需单一池化向量即可获得细粒度匹配。多向量索引（Vespa、Qdrant 多向量或 AstraDB）存储每个图像块的嵌入，检索时运行 MaxSim。

回答器是视觉语言模型（VLM），输入为查询加检索出的 top-k 页图像，输出带证据区域（边界框或页码引用）的答案。2026 年前沿模型选项包括 Qwen3-VL-30B、Gemini 2.5 Pro 和 InternVL3。对于公式和科学符号，加入 OCR 回退（Nougat、dots.ocr）作为可选文本通道。

评估是二维矩阵。一轴是内容类型（纯文本段落、密集表格、柱状/折线图、手写笔记、公式）；另一轴是检索方式（视觉优先后期交互、OCR-先文本、混合） 。每个单元报告 nDCG@5 和答案准确率。评测报告作为交付物。

## 架构

```text
PDFs -> 页面渲染器（PyMuPDF，180 DPI）
           |
           v
  ColQwen2.5-v0.2 嵌入（每页多向量，约 2048 个图像块）
           |
           +------> DocPruner 50% 压缩
           |
           v
   多向量索引（Vespa 或 Qdrant 多向量）
           |
查询 ----+----> 检索 top-k 页（MaxSim）
           |
           v
  VLM 回答器：Qwen3-VL-30B | Gemini 2.5 Pro | InternVL3
    输入：查询 + top-k 页图像 + 可选 OCR 文本
           |
           v
  带引用页码和证据区域的答案
           |
           v
  Streamlit / Next.js 查看器：原页突出显示边界框
```

## 技术栈

- 页面渲染：PyMuPDF（fitz），180 DPI，纵向归一化  
- 后期交互模型：ColQwen2.5-v0.2 或 ColQwen3-omni（ViDoRe 团队 Hugging Face）  
- 索引：带多向量字段的 Vespa，或 Qdrant 多向量，或带 MaxSim 的 AstraDB  
- 剪枝：DocPruner 2026 策略（保留高方差图像块，50% 压缩准确率损失 < 0.5%）  
- OCR 回退（公式/密集表格）：dots.ocr 或 Nougat  
- VLM 回答器：自托管 Qwen3-VL-30B 或托管 Gemini 2.5 Pro；InternVL3 备选  
- 评测：ViDoRe v3 基准，M3DocVQA 多页推理  
- 查看器 UI：Next.js 15 搭配画布覆盖证据区域  

## 构建步骤

1. **摄取。** 遍历包含 10-K 报告、科学论文、扫描文档的 1 万页 PDF 语料库。每页渲染为 1536x2048 PNG。持久化 `{doc_id, page_num, image_path}`。

2. **嵌入。** 对每页图像运行 ColQwen2.5-v0.2。输出形状约为 2048 个 128 维图像块向量。应用 DocPruner 保留信号最高的半数。写入 Vespa 多向量字段或 Qdrant 多向量。

3. **查询。** 对每个传入查询，使用查询塔进行 token 级别嵌入。用 MaxSim 与索引检索：对每个查询 token，取与页面图像块向量的最大点积，求和。返回 top-k 页面。

4. **合成。** 调用 Qwen3-VL-30B，输入查询与 top-5 页图像。提示：“仅使用提供的页面回答。每个断言引用（doc_id，页码）并命名区域（图表、表格、段落）。”

5. **证据区域。** 后处理答案提取被引用区域。如果 VLM 输出边界框（如 Qwen3-VL），则在查看器中渲染叠加。

6. **OCR 回退。** 对于被判定为公式密集的页面（基于图像方差启发），运行 Nougat 或 dots.ocr，作为附加文本通道与图像并行输入。

7. **评测。** 运行 ViDoRe v3（检索 nDCG@5）和 M3DocVQA（多页问答准确率）。对相同语料与相同合成器同时运行 OCR-先文本管线。生成内容类型 × 检索方案矩阵。

8. **UI。** 先做 Streamlit 原型，再开发 Next.js 15 生产查看器，支持逐页证据区域叠加。

## 使用示例

```text
$ doc-qa ask "what was the 2024 operating margin change for segment EMEA?"
[retrieve]   top-5 页，耗时 320ms（ColQwen2.5，MaxSim，Vespa）
[synth]      qwen3-vl-30b，1.4s，带引用（form-10k-2024，第 88 页）+（...，第 92 页）
答案：
  EMEA 经营利润率从 18.2% 下降到 16.8%，降幅 140 个基点。
  引用：10-K-2024.pdf 第 88 页（表 4，分部经营利润率）
         10-K-2024.pdf 第 92 页（MD&A，经营表现）
[viewer]     打开查看器，88 页表 4 上叠加高亮边界框
```

## 交付物

`outputs/skill-doc-qa.md` 描述交付成果：针对特定语料调优的视觉优先多模态文档问答系统，并在 ViDoRe v3 上相较 OCR-先文本基线进行评测。

| 权重 | 评判标准 | 测量方式 |
|:-:|---|---|
| 25 | ViDoRe v3 / M3DocVQA 准确率 | 与 OCR-文本基线和已发布排行榜的基准数值 |
| 20 | 证据区域定位 | 被引用区域实际包含答案跨度的比例 |
| 20 | 存储和延迟工程 | DocPruner 压缩率，索引 p95 延迟，回答 p95 延迟 |
| 20 | 多页推理 | 一套 100 题手标注多页准确率 |
| 15 | 来源检查用户体验 | 查看器清晰度、叠加精度、并排比较工具 |
| **100** |  |  |

## 练习

1. 比较 ColQwen2.5-v0.2 与 ColQwen3-omni 在同一语料库上的表现。哪些页面一者正确另一者错？为索引添加“内容类别”标签按类型路由。

2. 激进剪枝嵌入（75%、90%）。找出压缩断崖点：ViDoRe nDCG@5 低于 OCR 基线的临界点。

3. 搭建混合系统：OCR-先文本与 ColQwen 并行，使用 RRF 融合，交叉编码器重排。混合方案是否优于任一单独方案？在哪些情况最有效？

4. 使用更小的 VLM 替换 Qwen3-VL-30B（如 Qwen2.5-VL-7B）。测量每美元准确率曲线。

5. 增加手写笔记支持。渲染手写语料，采用 ColQwen 嵌入，测量检索准确率。与手写 OCR 管线比较。

## 关键词

| 术语 | 常见说法 | 实际含义 |
|------|---------|----------|
| Late interaction | “ColPali 样式检索” | 查询 token 独立对所有页面图像块评分；MaxSim 聚合 |
| Multi-vector | “每块嵌入” | 每文档含多个向量，不是单一池化向量 |
| MaxSim | “后期交互评分” | 每查询 token 取文档向量最大相似度；求和得分 |
| DocPruner | “图像块压缩” | 2026 年剪枝策略，保留 50% 图像块，准确率损失极小 |
| ViDoRe v3 | “文档检索基准” | 2026 年衡量视觉文档检索的标准基准 |
| Evidence region | “引用边界框” | 源页上定位答案跨度的边界框 |
| OCR fallback | “公式通道” | 针对公式或表格密集页采用的文本管线补充 |

## 深入阅读

- [ColPali（Illuin Tech）仓库](https://github.com/illuin-tech/colpali) — 参考后期交互文档检索  
- [ColPali 论文（arXiv:2407.01449）](https://arxiv.org/abs/2407.01449) — 基础方法论文  
- [ColQwen 系列 Hugging Face](https://huggingface.co/vidore) — 生产级模型检查点  
- [M3DocRAG（Adobe）](https://arxiv.org/abs/2411.04952) — 多页多模态 RAG 基线  
- [Vespa 多向量教程](https://docs.vespa.ai/en/colpali.html) — 参考服务栈  
- [Qdrant 多向量支持](https://qdrant.tech/documentation/concepts/vectors/#multivectors) — 备用索引  
- [AstraDB 多向量](https://docs.datastax.com/en/astra-db-serverless/databases/vector-search.html) — 备用托管索引  
- [Nougat OCR](https://github.com/facebookresearch/nougat) — 支持公式的 OCR 回退方案
