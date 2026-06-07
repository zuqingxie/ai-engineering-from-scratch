# ColPali 和视觉原生文档 RAG

> 传统的 RAG 会将 PDF 解析成文本，拆分成小块，嵌入小块，存储向量。每一步都会丢失信息：OCR 会丢失图表数据，分块会打断表格行，文本嵌入忽略图形。ColPali（Faysse 等，2024 年 7 月）则提出了一个更简单的问题：为什么还要提取文本？直接通过 PaliGemma 嵌入页面图像，用 ColBERT 风格的后期交互（late interaction）进行检索，保留文档携带的所有布局、图形、字体和格式信号。公开基准测试显示：在视觉丰富的文档上，端到端准确率比文本 RAG 高出 20-40%。ColQwen2、ColSmol 和 VisRAG 继承并扩展了这一模式。本课将阅读视觉原生 RAG 论文，构建一个类似 ColPali 的微型索引器。

**类型：** 构建  
**语言：** Python（标准库，多向量索引器 + MaxSim 评分器）  
**先决条件：** 第 11 阶段（LLM 工程——RAG 基础）、第 12 阶段 · 05（LLaVA）  
**时间：** ~180 分钟  

## 学习目标

- 解释双编码器检索（每文档一个向量）与后期交互检索（每文档多个向量）的区别。  
- 描述 ColBERT 的 MaxSim 操作以及 ColPali 如何将其从文本 token 推广到图像 patch。  
- 构建一个微型的类似 ColPali 的索引器：页面 → patch 嵌入 → query-token 嵌入上的 MaxSim → top-k 页面。  
- 比较 ColPali + Qwen2.5-VL 生成器 与 文本 RAG + GPT-4 在发票/财务报告用例上的表现。  

## 问题说明

文本 RAG 处理 PDF 时会丢弃大部分文档信息。财务报告的第三季度营收增长通常呈现在图表中；医学报告的诊断结果往往在带注释的图像里；法律合同的签字区是一种版面信息，而非文本信息。

文本 RAG 流程：

1. PDF → 通过 OCR / pdftotext 转为文本。  
2. 文本 → 300-500 token 的小块。  
3. 小块 → 双编码器嵌入（一向量）。  
4. 用户查询 → 嵌入 → 余弦相似度 → top-k 小块。  
5. 小块 + 查询 → LLM 生成答案。

五个有信息丢失的步骤。图表未被捕获，表格被拆分成多个块，多栏布局被折叠，图形注释消失。

ColPali 的解决方案是：跳过 OCR，直接嵌入页面图像。使用 ColBERT 风格的后期交互检索，使模型能在查询时关注细粒度的 patch。

## 核心概念

### ColBERT（2020）

ColBERT（Khattab & Zaharia，arXiv:2004.12832）是一种文本检索方法。它不为整个文档生成一个向量，而是为每个 token 生成向量。查询时：

- 查询 token 各有自己的嵌入（N_q 个向量）。  
- 文档 token 有嵌入（N_d 个向量，通常缓存）。  
- 分数 = 查询中每个 token 对文档各 token 余弦相似度的最大值之和：Σ_i max_j cos(q_i, d_j)。

这就是 MaxSim 操作。每个查询 token “选取” 最匹配的文档 token，最终得分为所有查询 token 最高匹配分数的累加。

优点：召回强，能处理词级语义。缺点：文档包含 N_d 个向量，存储成本高。

### ColPali

ColPali（Faysse 等，arXiv:2407.01449）将 ColBERT 模式应用于图像。

- 每页通过 PaliGemma（ViT + 语言模型）编码为 patch 嵌入：每页 N_p 个向量。  
- 用户查询（文本）编码为查询 token 嵌入：N_q 个向量。  
- 分数 = Σ_i max_j cos(q_i, p_j)，即查询文本 token 和页面图像 patch 之间的 MaxSim。  
- 通过总分检索 top-k 页面。

文档摄取时：用 PaliGemma 嵌入每页，存储所有 patch 嵌入。查询时：嵌入查询 token，计算与所有存储页面嵌入的 MaxSim，返回 top-k 页面。

优点：端到端性能在视觉丰富文档上比文本 RAG 高 20-40%。每个 patch 向量捕获局部布局和内容。

缺点：每页 N_p 个 patch × 4 字节浮点 × D 维向量，存储需求增长快。通过 PQ/OPQ 量化缓解。

### ColQwen2 和 ColSmol

ColQwen2（illuin-tech，2024-2025）用 Qwen2-VL 代替 PaliGemma，编码器更好，检索性能提升。

ColSmol 是更小规模版本，适合本地/边缘设备。大约 10 亿参数的 ColSmol 检索器可以在消费者 GPU 上运行。

### VisRAG

VisRAG（Yu 等，arXiv:2410.10594）是另一种变体：不是对 patch 计算 MaxSim，而是用 VLM 对每页池化为单一向量，再用双编码器检索。索引更快、存储更小，但召回较弱。

质量与成本的权衡：ColPali 追求质量，VisRAG 追求规模。

### M3DocRAG

M3DocRAG（Cho 等，arXiv:2411.04952）将多模态检索扩展到多页多文档推理。跨文档检索页面，组合多页上下文给 VLM 使用。

### ViDoRe —— 基准测试

ColPali 配套的基准测试。视觉文档检索评估。任务涵盖财务报告、科研论文、行政文件、医疗记录、手册。指标：nDCG@5。

ColPali-v1 在 ViDoRe 上获得约 80% nDCG@5，文本 RAG 在相同文档上约为 50-60%。

### 端到端 RAG 流程

视觉原生 RAG：

1. 摄取：PDF → 页面图像 → PaliGemma 编码 → 存储所有 patch 嵌入。  
2. 查询：用户文本 → 查询 token 嵌入 → 与所有索引页做 MaxSim → top-k 页。  
3. 生成：top-k 页图像 + 查询 → VLM（Qwen2.5-VL 或 Claude）→ 生成答案。

整个流程无 OCR。图形、图表、字体、版式全都进入答案。

### 存储计算

一个 50 页的财报，每页 729 个 patch，128 维嵌入：

- ColPali：50 × 729 × 128 × 4 字节 ≈ 18 MB 原始数据，PQ 量化后约 4 MB。  
- 文本 RAG：50 个小块 × 768 维 × 4 字节 ≈ 150 KB。

ColPali 存储需求约为文本 RAG 的 30 倍。大规模时 OPQ/PQ 量化降低到约 5-10 倍，通常可以接受。

### 适合文本 RAG 的场景

- 纯文本文档，无版面信息（维基文章、聊天记录），文本 RAG 更简单且存储成本更低。  
- 数百万页的归档，存储成本成为主导。  
- 需要可提取 OCR 文本以满足严格监管要求。

2026 年及以后，财报、科研论文、法律合同、医疗记录、UX 文档等领域视觉原生 RAG 更具优势。

## 使用方法

`code/main.py`：

- 玩具 patch 编码器：将“页面”（小网格特征向量）映射为 patch 嵌入数组。  
- MaxSim 评分器：计算查询 token 嵌入集与页面 patch 嵌入集之间的 ColBERT 风格分数。  
- 索引 5 个玩具页面，运行 3 次查询，返回带分数的 top-k。

## 发布成果

本课生成 `outputs/skill-vision-rag-designer.md`。在文档 RAG 项目中，选择 ColPali / ColQwen2 / VisRAG / 文本 RAG 并计算存储容量。

## 练习

1. 一个 200 页年报，每页 729 个 patch，128 维嵌入，4 字节浮点。计算原始存储和 PQ 压缩后（8 倍压缩）的存储。  

2. MaxSim 是 Σ_i max_j cos(q_i, p_j)。这个求和捕捉了简单均值相似度没有的什么信息？  

3. ColPali 按 patch 集合索引页面。如果改为按词级别索引（如 ColBERT 所做），会有哪些变化？利弊？  

4. 设计一个 100 万页语料库的端到端流程，查询延迟预算 500 毫秒。选择 ColQwen2 或 VisRAG 并说明理由。  

5. 阅读 M3DocRAG（arXiv:2411.04952）。描述多页注意力模式，并说明它与单页 ColPali 检索的区别。  

## 关键词汇

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| 后期交互（Late interaction） | “ColBERT 风格” | 使用按 token 或 patch 嵌入 + MaxSim 检索，不是单一文档向量 |
| MaxSim | “补丁最大相似度” | 每个查询 token 选取文档 token 最大相似度，查询 token 分数求和 |
| 双编码器（Bi-encoder） | “单向量” | 每文档一个向量，速度快但丢失细节粒度 |
| 多向量（Multi-vector） | “每文档多个向量” | 每文档/页面存 N_p 个向量，存储成本增大但召回提升 |
| Patch 嵌入 | “页面特征” | 来自 VLM 编码器的每个图像 patch 向量，按页缓存 |
| ViDoRe | “视觉文档基准” | ColPali 的视觉文档检索基准套件 |
| PQ 量化 | “乘积量化” | 压缩技术，保持向量相似度，同时将存储缩小约 8 倍 |

## 相关阅读

- [Faysse 等 — ColPali（arXiv:2407.01449）](https://arxiv.org/abs/2407.01449)  
- [Khattab & Zaharia — ColBERT（arXiv:2004.12832）](https://arxiv.org/abs/2004.12832)  
- [Yu 等 — VisRAG（arXiv:2410.10594）](https://arxiv.org/abs/2410.10594)  
- [Cho 等 — M3DocRAG（arXiv:2411.04952）](https://arxiv.org/abs/2411.04952)  
- [illuin-tech/colpali GitHub](https://github.com/illuin-tech/colpali)
