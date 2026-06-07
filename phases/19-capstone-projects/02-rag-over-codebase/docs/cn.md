# 结业项目 02 — 基于代码库的 RAG（跨仓库语义搜索）

> 到 2026 年，每个严肃的工程组织都会运行一个理解语义而不仅仅是字符串的内部代码搜索。Sourcegraph Amp、Cursor 的代码库答案、Augment 的企业图谱、Aider 的 repomap、Pinterest 的内部 MCP —— 形态相同。摄取多个仓库，使用 tree-sitter 解析，嵌入函数和类级别的代码块，混合搜索，重排序，带引证地回答。此结业项目要求你构建一个能处理跨 10 个仓库 200 万行代码且支持每次 git push 增量重索引的系统。

**类型：** 结业项目  
**语言：** Python（摄取），TypeScript（API + UI）  
**前置要求：** 第 5 阶段（NLP 基础）、第 7 阶段（transformers）、第 11 阶段（LLM 工程）、第 13 阶段（工具）、第 17 阶段（基础设施）  
**练习阶段：** P5 · P7 · P11 · P13 · P17  
**时长：** 30 小时  

## 问题

到 2026 年，每个前沿的编码代理都配备了一个代码库检索层，因为仅靠上下文窗口无法解决跨仓库的问题。Claude 的百万 token 上下文非常有帮助；但它无法替代排名检索的需求。对原始代码块进行的朴素余弦检索会因为生成代码、monorepo 代码重复和很少导入的符号长尾问题而污染结果。生产环境中的解决方案是对具备 AST 语法树感知的代码块进行混合（dense + BM25）搜索，并辅以重排序器，同时基于符号引用图谱支撑。

你将通过索引真实舰队（而非单个示例仓库），来衡量 MRR@10、引证的准确度和增量新鲜度。失败模式为基础设施：10 万文件的 monorepo，一次 push 涉及半数文件修改，查询需跨四个仓库才能正确答复。

## 概念

AST 感知的摄取流水线使用 tree-sitter 解析每个文件，提取函数和类节点，在节点边界而非固定 token 窗口处分块。每个代码块获得三种表示：稠密向量嵌入（Voyage-code-3 或 nomic-embed-code）、稀疏 BM25 词项，以及简短的自然语言摘要。摘要作为第三种可检索模态，用户询问“X 是如何授权的”，摘要会提到“authz”，即使代码中仅有 `check_permission`。

检索为混合式。查询同时触发稠密和 BM25 检索，合并 top-k，然后交给交叉编码器重排序器（Cohere rerank-3 或 bge-reranker-v2-gemma-2b）。重排序列表传给长上下文合成器（Claude Sonnet 4.7 带提示缓存，或自托管 Llama 3.3 70B），指令要求每条结论必须有文件和行号的引用。无引用的答案会被后期过滤拒绝。

增量新鲜度是基础设施问题。Git push 触发差异检测：哪些文件变化，哪些符号变化。只重新嵌入受影响的代码块。跨文件符号边（导入、方法调用）也会被重新计算。索引保持一致，无需每次提交处理 200 万行代码。

## 架构

```text
git push --> webhook --> 摄取 worker (LlamaIndex 工作流)
                           |
                           v
             tree-sitter 解析 + AST 分块
                           |
            +--------------+----------------+
            v              v                v
          稠密向量       BM25 索引          摘要（LLM）
        (Voyage / bge)   (Tantivy)          (Haiku 4.5)
            |              |                |
            +------> Qdrant / pgvector <----+
                            |
                            v
                      符号图谱 (Neo4j / kuzu)
                            |
  查询 --> LangGraph 代理 (检索 -> 重排序 -> 合成)
                            |
                            v
                 Claude Sonnet 4.7 1M 上下文
                            |
                            v
                 答案 + 文件:行号 引用
```

## 技术栈

- 解析：tree-sitter，支持 17 种语言语法（Python、TS、Rust、Go、Java、C++ 等）
- 稠密嵌入：Voyage-code-3（托管）或 nomic-embed-code-v1.5（自托管），bge-code-v1 备用
- 稀疏索引：采用 Tantivy（Rust）和 BM25F，在符号名与代码体上加权
- 向量数据库：Qdrant 1.12 支持混合搜索，或对 < 5000 万向量团队使用 pgvector + pgvectorscale
- 代码块摘要模型：Claude Haiku 4.5 或 Gemini 2.5 Flash，带提示缓存
- 重排序器：Cohere rerank-3 或自托管 bge-reranker-v2-gemma-2b
- 编排：LlamaIndex 工作流负责摄取，LangGraph 负责查询代理
- 合成器：Claude Sonnet 4.7（百万上下文）带提示缓存
- 符号图谱：Neo4j（托管）或 kuzu（嵌入式）构建导入和调用边
- 可观察性：Langfuse，记录每步检索与合成的 span

## 构建步骤

1. **摄取遍历。** 在每次 push 的钩子中遍历 git 历史，收集变更文件。对每个文件，用 tree-sitter 解析，提取函数和类节点及其完整源码范围。生成代码块记录 `{repo, path, start_line, end_line, symbol, body}`。

2. **代码块摘要。** 将代码块批量送入带有系统预设缓存的 Haiku 4.5 模型。提示语：“用一句话总结此函数，描述其公共契约和副作用。”将摘要与代码块一起存储。

3. **嵌入池。** 两条并行队列：稠密向量（Voyage-code-3 批量 128），摘要（同模型，输入摘要文本）。向 Qdrant 写入向量及载荷 `{repo, path, start_line, end_line, symbol, kind}`。

4. **BM25 索引。** 采用字段加权的 Tantivy 索引：符号名权重 4，符号体权重 1，摘要权重 2。支持“查找名为 X 的函数”及“查找做 X 的函数”查询。

5. **符号图谱。** 对每个代码块，记录边：导入（此文件使用 Z 仓库的符号 Y）、调用（此函数调用 C 类的 M 方法）、继承。在 kuzu 存储。查询时用于跨仓库扩展检索。

6. **查询代理。** 使用 LangGraph 由三节点组成。`retrieve` 并发触发稠密和 BM25 检索，按（repo, path, symbol）去重。`rerank` 在 top-50 上运行交叉编码器，保留 top-10。`synth` 调用带重排序内容上下文的 Claude Sonnet 4.7，缓存系统提示，要求带文件:行号引用。

7. **引证强制。** 解析模型输出；任何无 `(repo/path:start-end)` 锚点的声明会被标记重新询问或丢弃。仅返回带引证的答案给用户。

8. **增量重索引。** 每次 webhook，计算符号级差异。仅重新嵌入文本变更的代码块。重新计算导入变更代码块的符号边。指标：50 个文件的 push 低于 60 秒完成重索引，适用于 200 万 LOC 队伍。

9. **评测。** 标注 100 个跨仓库问题的黄金文件:行号答案。测量 MRR@10、nDCG@10、引证准确度（声明带有可验证锚点比例）、p50/p99 延迟。

## 使用示例

```text
$ code-rag ask "how is S3 multipart abort wired into our retry budget?"
[retrieve]  12 稠密块 + 7 BM25 块，去重后 16 块
[rerank]    保留 top-5（cohere rerank-3）
[synth]     claude-sonnet-4.7，缓存命中率 68%，2.1 秒
答案：
  Multipart aborts 由 services/uploader/retry.go:122-148 中的 `AbortMultipartOnFail` 触发，
  会减少 config/budgets.yaml:34-51 中定义的每桶重试预算...
  引证：[services/uploader/retry.go:122-148, config/budgets.yaml:34-51,
         libs/s3client/multipart.ts:44-61]
```

## 交付

交付产物 `outputs/skill-codebase-rag.md`。给定仓库语料，搭建摄取流水线、混合索引和查询代理，并返回带引证的跨仓库问题答案。评分标准：

| 权重 | 标准 | 测量方式 |
|:-:|---|---|
| 25 | 检索质量 | 100 个问题的 MRR@10 和 nDCG@10 |
| 20 | 引证准确度 | 答案中带有可验证文件:行号锚点的声明比例 |
| 20 | 延迟与规模 | 10k QPS 下的 p95 查询延迟，基于索引语料规模 |
| 20 | 增量索引正确性 | 50 文件提交从 git push 到可检索时间 |
| 15 | 用户体验与答案格式 | 引用点击性、代码片段预览、后续交互能力 |
| **100** | | |

## 练习

1. 用自托管的 nomic-embed-code 替换 Voyage-code-3。测量 MRR@10 变化。报告启用重排序后差距是否缩小。

2. 向语料中注入 20% 的生成代码（LLM 产出的模板代码），重新评估。观察检索污染。为载荷添加 "generated" 标记，并降低这类命中的权重。

3. 在你的语料规模下对比 Qdrant 混合搜索与 pgvector + pgvectorscale。报告批量大小为 1 时的 p99 延迟。

4. 增加基于抽样的漂移检测：每周重新执行 100 问评测。MRR@10 下降超过 5% 发出告警。

5. 扩展跨语言符号解析：如一个 Python 函数调用通过 gRPC 的 Go 服务。利用符号图谱将它们连接起来。

## 关键词

| 术语 | 常用说法 | 实际含义 |
|------|---------|----------|
| AST-aware chunking | “函数级分块” | 在 tree-sitter 节点边界而非固定 token 窗口处分块 |
| Hybrid search | “稠密+稀疏” | 并行运行 BM25 和向量检索，合并 top-k，重排序 |
| Cross-encoder rerank | “二阶段排序” | 对每个（查询，候选）对共同打分，准确率高于余弦相似度 |
| Prompt caching | “缓存系统提示” | 2026 年 Claude / OpenAI 功能，重复前缀 token 折扣达 90% |
| Symbol graph | “代码图” | 跨文件和仓库的导入、调用、继承边 |
| Citation faithfulness | “有据答案率” | 用户可点击引用锚点并验证答复声明比例 |
| Incremental re-index | “推送到可检索时间” | 从 git push 到变更符号可被查询的实时时间 |

## 延伸阅读

- [Sourcegraph Amp](https://ampcode.com) — 生产环境跨仓库代码智能  
- [Sourcegraph Cody RAG 架构](https://sourcegraph.com/blog/how-cody-understands-your-codebase) — 本结业项目的权威深度解析  
- [Aider repo-map](https://aider.chat/docs/repomap.html) — tree-sitter 排序代码视图  
- [Augment Code 企业图谱](https://www.augmentcode.com) — 商用符号图 RAG  
- [Qdrant 混合搜索文档](https://qdrant.tech/documentation/concepts/hybrid-queries/) — 参考实现  
- [Voyage AI 代码嵌入](https://docs.voyageai.com/docs/embeddings) — Voyage-code-3 详情  
- [Cohere rerank-3](https://docs.cohere.com/reference/rerank) — 交叉编码器参考  
- [Pinterest MCP 内部搜索](https://medium.com/pinterest-engineering) — 内部平台参考
