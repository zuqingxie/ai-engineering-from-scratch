# ASCII 艺术和视觉越狱（Jailbreak）攻击

> Jiang, Xu, Niu, Xiang, Ramasubramanian, Li, Poovendran 发表的论文《ArtPrompt: 基于 ASCII 艺术的对齐大型语言模型（Aligned LLMs）越狱攻击》（ACL 2024，arXiv:2402.11753）。方法是将有害请求中的安全相关词语掩盖，用相同字母的 ASCII 艺术表现形式替换，发送伪装后的提示。GPT-3.5、GPT-4、Gemini、Claude、Llama-2 都无法稳健识别 ASCII 艺术词元。该攻击绕过了困惑度（Perplexity，PPL）过滤器、复述防御（Paraphrase），以及重新分词（Retokenization）。相关研究：ViTC 基准测试衡量对非语义视觉提示的识别能力；StructuralSleight 将攻击推广到非常见文本编码结构（树、图、嵌套 JSON），作为一类编码攻击家族。

**类型：** 构建  
**语言：** Python（标准库，ArtPrompt 词元掩盖框架）  
**先决条件：** 第18阶段 · 12课（PAIR），第18阶段 · 13课（MSJ）  
**时间：** 约 60 分钟

## 学习目标

- 描述 ArtPrompt 攻击：单词识别步骤、ASCII 艺术替换、最终伪装提示。
- 解释为何标准防御（PPL、复述、重新分词）会在 ArtPrompt 攻击下失效。
- 定义 ViTC 并说明其测量内容。
- 描述 StructuralSleight，作为对任意非常见文本编码结构（UTES）的泛化。

## 问题

通过复述和角色扮演（第12课）以及长上下文（第13课）进行的攻击，作用于文本层面模式。ArtPrompt 攻击作用于识别层面：模型不解析被禁止的词元，而是解析以字符呈现的图像。安全过滤器见到的是无害的标点符号，模型却看到一个词。

## 概念

### ArtPrompt，两个步骤

步骤 1. 单词识别。攻击者利用大型语言模型定位安全相关单词（如“bomb”在“how to make a bomb”中）。

步骤 2. 伪装提示生成。用 ASCII 艺术渲染（7x5 或 7x7 字符块形成字母形状）替换识别出的每个单词。模型接收的是一个由标点和空格组成的网格，有足够能力的模型能识别出词；安全过滤器仅见到该网格。

结果：GPT-4、Gemini、Claude、Llama-2 和 GPT-3.5 均失败。在其基准子集中攻击成功率超过 75%。

### 标准防御失效原因

- **PPL（困惑度过滤器）。** ASCII 艺术困惑度高，但所有新颖输入困惑度也高。阻止 ArtPrompt 的阈值同样阻挡合法的结构化输入。
- **复述。** 对提示进行复述会破坏 ASCII 艺术，但实际复述模型往往保留或重构该艺术。
- **重新分词。** 不同的词元拆分不会改变模型视觉上识别字母形状的事实。

根本问题是安全过滤器工作在词元或语义层面，而 ArtPrompt 在视觉识别层面操作。

### ViTC 基准测试

识别非语义视觉提示的能力。衡量模型读取 ASCII 艺术、Wingdings 字体和其他非文本语义视觉内容的能力。ArtPrompt 效果与 ViTC 准确率相关：模型读取视觉文本越好，ArtPrompt 攻击效果越佳。这是能力与安全之间的权衡。

### StructuralSleight

是 ArtPrompt 的泛化：非常见文本编码结构（UTES）。树、图、嵌套 JSON、JSON 中的 CSV、差异样式代码块。如果结构在训练安全数据中罕见但模型能解析，便可隐藏有害内容。

防御启示：安全必须涵盖模型能解析的结构化表示，且此类表示集合庞大且不断扩展。

### 图像模态类比

视觉大型语言模型（GPT-5.2、Gemini 3 Pro、Claude Opus 4.5、Grok 4.1）扩大了攻击面。基于真实图像的 ArtPrompt 风格攻击比 ASCII 艺术对应形式更强，因为图像编码器提供更丰富信号。

### 本课程第18阶段中的位置

第12至14课介绍三个正交攻击向量：迭代精炼（PAIR）、上下文长度（MSJ）、编码（ArtPrompt/StructuralSleight）。第15课转向模型中心外的系统边界攻击（间接提示注入）。第16课描述防御工具响应。

## 使用方法

`code/main.py` 构建了一个简易的 ArtPrompt。你可以用 ASCII 艺术字形为有害查询中的特定词语伪装，验证伪装字符串能通过简单关键词过滤，并（可选）用简单识别器反解该字符串。

## 交付物

本课产出 `outputs/skill-encoding-audit.md`。在给定越狱防御报告的情况下，列举涵盖的编码攻击家族（ASCII 艺术、base64、leet 语、UTF-8 同形异义字、UTES）及对应的防御层。

## 练习

1. 运行 `code/main.py`。验证伪装字符串是否通过简单关键词过滤。报告所需的字符级修改。

2. 实现第二种编码：对同一目标词使用 base64 编码。比较其绕过过滤率与 ArtPrompt 以及恢复难度。

3. 阅读 Jiang et al. 2024 第4.3节（五模型结果）。提出 Claude 在同一基准上 ArtPrompt 抵抗力高于 Gemini 的可能原因。

4. 设计一个生成前防御，检测提示中的 ASCII 艺术形状区域。测量其在合法代码、表格和数学符号上的误报率。

5. StructuralSleight 列出10种编码结构。构思一个处理全部10种结构的泛化防御方案，并估算每条提示的计算成本。

## 关键词

| 术语 | 通常说法 | 实际含义 |
|------|----------|-----------|
| ArtPrompt | “ASCII 艺术攻击” | 两步越狱：用 ASCII 艺术渲染掩盖安全词 |
| Cloaking（伪装） | “隐藏词语” | 用模型能读懂但过滤器看不见的视觉表示替换禁止词元 |
| UTES | “非常见结构” | 非常见文本编码结构 — 如树、图、嵌套 JSON 等用于走私内容 |
| ViTC | “视觉文本能力” | 测量模型读取非语义视觉编码的基准 |
| Perplexity filter（困惑度过滤器） | “PPL 防御” | 拒绝困惑度高的提示；因合法结构输入也困惑度高而失效 |
| Retokenization（重新分词） | “分词器变换防御” | 用不同分词器预处理提示；因识别是视觉上的而失效 |
| Homoglyph（同形异义字） | “相似字符” | 看起来与拉丁字母完全相同的 Unicode 字符，绕过子串检查 |

## 延伸阅读

- [Jiang et al. — ArtPrompt（ACL 2024，arXiv:2402.11753）](https://arxiv.org/abs/2402.11753) — ASCII 艺术越狱论文
- [Li et al. — StructuralSleight（arXiv:2406.08754）](https://arxiv.org/abs/2406.08754) — UTES 泛化
- [Chao et al. — PAIR（第12课，arXiv:2310.08419）](https://arxiv.org/abs/2310.08419) — 互补迭代攻击
- [Anil et al. — Many-shot Jailbreaking（第13课）](https://www.anthropic.com/research/many-shot-jailbreaking) — 互补长度攻击
