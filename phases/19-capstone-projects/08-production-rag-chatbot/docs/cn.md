# Capstone 08 — 受监管垂直领域的生产级 RAG 聊天机器人

> Harvey、Glean、Mendable 和 LlamaCloud 都在 2026 年运行相同的生产形态。使用 docling 或 Unstructured 进行文档导入，ColPali 处理视觉内容。混合搜索。使用 bge-reranker-v2-gemma 重排序。使用 Claude Sonnet 4.7 结合提示缓存（命中率 60-80%）进行综合生成。使用 Llama Guard 4 和 NeMo Guardrails 进行安全保护。使用 Langfuse 和 Phoenix 监控。用一套 200 题金标数据集和红队测试集评判。构建一个受监管领域（法律、临床、保险）的解决方案，本项任务的目标是在通过金标集测试、红队和漂移监控仪表盘。

**类型：** Capstone  
**语言：** Python（流水线 + API）、TypeScript（聊天界面）  
**先决条件：** 第 5 阶段（NLP），第 7 阶段（transformers），第 11 阶段（LLM 工程），第 12 阶段（多模态），第 17 阶段（基础设施），第 18 阶段（安全）  
**涉及阶段：** P5 · P7 · P11 · P12 · P17 · P18  
**时间估计：** 30 小时

## 问题

受监管领域的 RAG（法律合同、临床试验方案、保险政策）是 2026 年最常见的生产形态，因为投资回报显著且风险具体可控。Harvey（Allen & Overy）为法律领域构建此方案。Mendable 为开发者文档版本。Glean 覆盖企业搜索。整体模式是：高保真文档导入、混合检索+重排序、带引用强制和提示缓存的综合生成、多层安全守护、持续漂移监控。

难点不在模型，而在于具备法律管辖区合规意识（HIPAA、GDPR、SOC2）、引用级别可审计性、成本控制（提示缓存高命中率时可节省 60-90% 成本）、通过 RAGAS 检测幻觉、以及当源文档更新但索引滞后时检测漂移。本 capstone 要求你在一套 200 题金标数据集及红队测试集上完整交付这一切。

## 概念

流水线分两部分。**导入**：docling 或 Unstructured 解析结构化文档；ColPali 处理视觉丰富文档；将文档切片，加上摘要、标签和基于角色的访问控制标签。向量存储到 pgvector + pgvectorscale（向量规模 < 5000 万）或 Qdrant Cloud；同时运行稀疏 BM25 检索。**会话**：LangGraph 负责多轮记忆；每次查询进行混合检索，结合重排序器 bge-reranker-v2-gemma-2b；用 Claude Sonnet 4.7（提示缓存）综合；输出通过 Llama Guard 4 和 NeMo Guardrails 安全过滤；给出带引用锚点的回答。

评估堆栈有四层。**金标集**（200 条带引用的问答）测准确性。**红队**（越狱、PII 提取尝试、越界问题）测安全性。**RAGAS** 自动度量每轮生成的真实性/答案相关性/上下文精确度。**漂移仪表盘**（Arize Phoenix）每周监控检索质量和幻觉得分。

提示缓存是成本杠杆。Claude 4.5+ 和 GPT-5+ 支持缓存系统提示和检索上下文。命中率在 60-80% 时，每次调用成本可减少 3-5 倍。流水线设计须稳定使用固定前缀（系统提示 + 重排序上下文优先），以获得高缓存命中率。

## 架构

```text
documents (contracts, protocols, policies)
      |
      v
docling / Unstructured parse + ColPali for visuals
      |
      v
chunks + summaries + role-labels + jurisdiction tags
      |
      v
pgvector + pgvectorscale  +  BM25 (Tantivy)
      |
query + role + jurisdiction
      |
      v
LangGraph conversational agent
   +--- retrieve (hybrid)
   +--- filter by role + jurisdiction
   +--- rerank (bge-reranker-v2-gemma-2b or Voyage rerank-2)
   +--- synthesize (Claude Sonnet 4.7, prompt cached)
   +--- guard (Llama Guard 4 + NeMo Guardrails + Presidio output PII scrub)
   +--- cite + return
      |
      v
eval:
  RAGAS faithfulness / answer_relevance / context_precision (online)
  Langfuse annotation queue (sampled)
  Arize Phoenix drift (weekly)
  red team suite (pre-release)
```

## 技术栈

- 导入：Unstructured.io 或 docling 解析结构化文档；ColPali 处理视觉丰富 PDF  
- 向量数据库：pgvector + pgvectorscale（小于 5000 万向量）；否则用 Qdrant Cloud  
- 稀疏索引：Tantivy BM25，带字段加权  
- 编排：LlamaIndex Workflows（导入） + LangGraph（会话）  
- 重排序器：bge-reranker-v2-gemma-2b 自部署，或 Voyage rerank-2 托管版  
- 大语言模型：Claude Sonnet 4.7 提示缓存；回退 Llama 3.3 70B 自部署  
- 评估：线上 RAGAS 0.2，DeepEval 辅助幻觉检测和越狱测试  
- 监控观察性：Langfuse 自部署含注释队列；Arize Phoenix 监测漂移  
- 安全护栏：Llama Guard 4 输入/输出分类器，NeMo Guardrails v0.12 策略，Presidio PII 清理  
- 合规性：切片上带基于角色访问标签和 GDPR/HIPAA 法域标签  

## 构建步骤

1. **导入。** 用 Unstructured 或 docling 解析语料库（正式构建需 1000-10000 文档）。扫描或视觉密集页用 ColPali 处理。生成切片，带摘要、角色标签、法域标签。

2. **建索引。** 使用 Voyage-3 或 Nomic-embed-v2 生成向量，存入 pgvector + pgvectorscale。BM25 通过 Tantivy 建立辅助索引。角色和法域信息作为附加负载。

3. **混合检索。** 先通过角色和法域过滤；再并行执行密集向量检索和 BM25；用相互排位融合合并结果；取 top-20 到重排序器；top-5 到综合生成。

4. **提示缓存综合。** 缓存系统提示和静态策略为缓存头；重排序上下文为缓存扩展；用户提问为未缓存后缀。目标稳态命中率 60-80%。

5. **安全护栏。** 输入端应用 Llama Guard 4；NeMo Guardrails 拦截越界或政策禁止话题；Presidio 清理输出中的意外 PII；综合结果后置引用强制。

6. **金标集。** 200 组问题/回答由领域专家提供（含答案和引用）。打分标准为准确匹配引用、答案正确性、真实性（RAGAS）。

7. **红队测试。** 50 条对抗式提示：越狱（PAIR、TAP）、PII 泄露尝试、越界问题、多法域泄漏。评分方式为通过/失败及严重度。

8. **漂移仪表盘。** Arize Phoenix 每周追踪检索质量（nDCG、引用真实性）。质量下滑 5% 发出警报。

9. **成本报告。** Langfuse 提示缓存命中率统计；每查询 token 量统计；各阶段 $/查询 细分。

## 使用示例

```text
$ chat --role=analyst --jurisdiction=GDPR
> what is the data-retention obligation for EU user profiles under our contract?
[retrieve]  hybrid top-20 filtered to GDPR + analyst-role
[rerank]    top-5 kept
[synth]     claude-sonnet-4.7, cache hit 74%, 0.8s
answer:
  The contract (Section 12.4, Master Services Agreement dated 2024-03-11)
  obligates EU user profile deletion within 30 days of termination per GDPR
  Article 17. The DPA amendment (DPA-v2.1, Section 5) extends this to 14 days
  for "restricted" category data.
  citations: [MSA-2024-03-11 s12.4, DPA-v2.1 s5]
```

## 部署交付

`outputs/skill-production-rag.md` 描述了交付物。一个带合规标签的受监管域聊天机器人，完成评分标准测试，通过红队检查，配备实时漂移监控。

| 权重 | 评分标准 | 测量方式 |
|:-:|---|---|
| 25 | RAGAS 真实性 + 答案相关性 | Online score on a 200-question gold set |
| 20 | 引用正确性 | Share of answers with verifiable source anchors |
| 20 | 安全护栏覆盖 | Llama Guard 4 pass rate plus jailbreak test results |
| 20 | 成本/延迟工程 | Prompt-cache hit rate, p95 latency, and $/query breakdown |
| 15 | 漂移监控仪表盘 | Phoenix live dashboard with weekly retrieval-quality trend |
| **100** |  |  |

## 练习

1. 构建另一法域的语料切片（例如 HIPAA 与 GDPR 并存）。演示角色+法域过滤防止跨法域泄露，使用 20 题跨域测试。

2. 监测一周生产流量中的提示缓存命中率。识别哪些查询破坏缓存前缀。重构设计。

3. 增加多轮记忆，含 10k token 摘要缓冲。检测随着对话长度增加，真实性是否降低。

4. 用 Llama 3.3 70B 自部署替换 Claude Sonnet 4.7。测量 $/查询 和真实性变化。

5. 加入“不确定”模式：当重排序最高分低于阈值，回复改为“我没有自信的引用”而非回答。测量虚假自信的减少。

## 关键词

| 术语 | 通俗说法 | 实际含义 |
|------|----------|----------|
| Prompt caching | “缓存系统提示+上下文” | Claude/OpenAI 功能，命中时缓存前缀 token 享 60-90% 折扣 |
| RAGAS | “RAG 评估器” | 自动评分真实性、答案相关性、上下文精确度 |
| Golden set | “标注评测集” | 200+ 条专家标注问答及引注；事实真相 |
| Jurisdiction tag | “合规标签” | GDPR/HIPAA/SOC2 范围附加在切片上；检索时强制执行 |
| Citation faithfulness | “依据答案率” | 受支持声明中可检索源范围占比 |
| Drift | “检索质量下降” | 每周 nDCG 或引用得分变化；5% 阈值触发告警 |
| Red team | “对抗测试” | 发布前越狱、PII 泄漏与越界探测 |

## 延伸阅读

- [Harvey AI](https://www.harvey.ai) — 法律生产级参考栈  
- [Glean enterprise search](https://www.glean.com) — 企业级 RAG 参考  
- [Mendable documentation](https://mendable.ai) — 开发者文档 RAG 参考  
- [LlamaCloud Parse + Index](https://docs.llamaindex.ai/en/stable/examples/llama_cloud/llama_parse/) — 托管导入示例  
- [Anthropic prompt caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) — 成本杠杆参考  
- [RAGAS 0.2 documentation](https://docs.ragas.io/) — 标准 RAG 评估框架  
- [Arize Phoenix](https://github.com/Arize-ai/phoenix) — 漂移监控参考  
- [Llama Guard 4](https://ai.meta.com/research/publications/llama-guard-4/) — 2026 年安全分类器  
- [NeMo Guardrails v0.12](https://docs.nvidia.com/nemo-guardrails/) — 策略护栏框架
