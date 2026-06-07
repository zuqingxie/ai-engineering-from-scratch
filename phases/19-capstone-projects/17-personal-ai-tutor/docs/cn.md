# 毕业设计 17 — 个人 AI 导师（自适应、多模态，带记忆）

> Khanmigo（可汗学院）、Duolingo Max、Google LearnLM / Gemini for Education、Quizlet Q-Chat 和 Synthesis Tutor 都在2026年推出了大规模自适应多模态辅导。其共同特征是苏格拉底式策略（绝不直接给答案），在每次互动后更新的学习者模型（贝叶斯知识追踪风格），语音+文本+拍照数学输入，课程图检索，间隔重复调度，以及针对年龄适宜内容的严格安全过滤。毕业设计目标是交付一个学科专属导师（K-12代数或入门Python），开展为期两周的10名学习者效能研究，并通过内容安全审核。

**类型：** 毕业设计  
**语言：** Python（后端，学习者模型）、TypeScript（网页应用）、SQL（通过 Postgres + Neo4j 实现课程图）  
**前置条件：** 阶段5（NLP）、阶段6（语音）、阶段11（LLM工程）、阶段12（多模态）、阶段14（智能体）、阶段17（基础设施）、阶段18（安全）  
**涉及阶段：** P5 · P6 · P11 · P12 · P14 · P17 · P18  
**时间：** 30小时  

## 问题

自适应辅导曾是教育技术研究的细分领域。到2026年已成为消费产品。Khanmigo已在美国大部分学区部署。Duolingo Max月活跃用户达数千万。谷歌的 LearnLM / Gemini for Education 支持 Google Classroom 中的辅导。Quizlet Q-Chat 与闪卡并列。Synthesis Tutor 以好奇孩子导师模式迅速走红。共同元素是：多模态输入（打字、语音、拍摄方程式），苏格拉底教学法（先问后讲解），每次交互后更新的学习者模型，以及严格的年龄适宜安全。

你将为特定人群构建此类系统。衡量标准为真实效能研究：10名学习者在两周内的前测及后测成绩。语音回路必须自然流畅（参考毕业设计03子栈）。记忆必须尊重隐私。安全过滤须通过面向K-12的 COPPA 合规红队测试。

## 概念

四个组件。**导师策略** 是一个苏格拉底循环：当学习者请求答案时，策略反问引导性问题；答对则转向下一个概念；卡壳时提供分步提示。**学习者模型** 是贝叶斯知识追踪（或简单变体），在每次互动后更新每个课程节点的掌握概率。**课程图** 是 Neo4j 实现的概念图，带有先决条件边；策略在图上遍历选择下一个概念。**记忆** 是情节式 + 语义存储（agentmemory 风格），保存过往交互、错误和偏好。

UX为多模态。文字输入用于键入答案。语音输入通过 LiveKit+Whisper（复用毕业设计03）。数学题拍照输入依赖 dots.ocr 或 PaliGemma 2。语音输出用 Cartesia Sonic-2。安全采用 Llama Guard 4 加年龄适宜过滤器（屏蔽成人内容、暴力、自残），并且记忆保留策略符合 COPPA 要求。

效能研究是交付成果。10名学习者，前测和后测，历时两周。报告学习增益差值及置信区间。对照一组非自适应基线（无导师策略，内容线性递送）。

## 架构

```text
learner device
  |
  +-- text         -> web app
  +-- voice        -> LiveKit Agents (ASR + TTS)
  +-- photo math   -> dots.ocr / PaliGemma 2
       |
       v
  tutor policy (LangGraph)
       - Socratic decision head
       - next-concept chooser (curriculum graph walk)
       - hint scaffolder
       - mastery update
       |
       v
  learner model (BKT / item-response theory)
       - per-concept mastery probability
       - spaced-repetition scheduler (SM-2 or FSRS)
       |
       v
  memory (agentmemory-style)
       - episodic: every interaction
       - semantic: learned mistakes, preferences
       - retention policy: COPPA / GDPR aware
       |
       v
  curriculum graph (Neo4j)
       - prerequisite edges
       - OER content attached
       |
       v
  safety:
    Llama Guard 4 + age-appropriate filter
    memory access guarded by learner ID scope
```

## 技术栈

- 学科选择：K-12代数或入门Python（二选一，深入开发）  
- 导师策略：基于 Claude Sonnet 4.7 的 LangGraph（带提示缓存）  
- 学习者模型：经典贝叶斯知识追踪或 FSRS 间隔重复调度  
- 课程图：Neo4j 实现概念图 + 先决条件边 + OER内容  
- 记忆：agentmemory风格持久化向量 + 情节存储 + 语义存储  
- 语音：LiveKit Agents 1.0 + Cartesia Sonic-2（复用毕业设计03子栈）  
- 拍照数学：dots.ocr 或 PaliGemma 2进行方程式识别  
- 安全：Llama Guard 4 + 自定义年龄适宜过滤器  
- 评估：基于Bloom水平的题目生成、前后测框架、效能研究工具支持  

## 构建步骤

1. **课程图。** 使用 Neo4j 构建包含50-150个概念节点（例如K-12代数“数轴”到“二次公式”），并创建先决条件边。为每个节点附加OER内容（开放教科书、OpenStax）。

2. **学习者模型。** 初始化贝叶斯知识追踪模型，设定先验参数：猜测、失误、学习率。每次交互后更新每个概念掌握度。为每个学习者持久化数据。

3. **导师策略。** 使用 LangGraph 创建节点：`read_signal`（学习者答案是否正确/部分/卡壳），`select_concept`（遍历课程图选择优先级最高的概念），`scaffold`（苏格拉底提示），`update_mastery`。

4. **记忆。** 每次交互写入情节存储。错误和偏好提炼成语义记忆。制定 COPPA 合规保留策略：一年后自动删除，家长可访问。

5. **语音路径。** LiveKit Agents 工作线程连接到导师策略。ASR 使用 Whisper-v3-turbo，TTS 使用 Cartesia Sonic-2。支持抢话（复用毕业设计03机制）。

6. **拍照数学路径。** 上传或拍摄图片；运行 dots.ocr 或 PaliGemma 2识别方程；将结构化输入发送给导师。

7. **安全。** 每次模型输出通过 Llama Guard 4 和年龄适宜过滤器（阻断自残、成人内容、暴力）。记忆访问受学习者ID范围限制；家长有删除入口。

8. **效能研究。** 10名学习者，标准化30题前测，历时两周的辅导互动（每周3次），后测。与另一组10名无自适应基线学习者比较。

9. **周进展报告。** 为每位学习者自动生成 PDF，汇总学习主题、掌握轨迹及推荐下一步。

## 使用示例

```text
learner: "我不明白为什么 3x + 6 = 12 就是 x = 2"
[signal]   卡壳
[concept]  '变量隔离'（先决条件：加法-减法-等式）
[scaffold] "两边先减去哪个数字？"
learner: "6"
[signal]   正确
[mastery]  加法-减法-等式：0.62 -> 0.77
[concept]  继续 '变量隔离'
[scaffold] "很好。那 3x 除以 3 是多少？"
```

## 交付物

`outputs/skill-ai-tutor.md` 是成果文件。具备学科专属自适应导师，多模态输入、学习者模型、记忆、安全以及效能测量。

| 权重 | 评分标准       | 测量方式                                 |
|:----:|--------------|---------------------------------------|
|  25  | 学习增益差值   | 10名学习者两周内前后测分差              |
|  20  | 苏格拉底一致性 | 脚本样本评分量表                       |
|  20  | 多模态用户体验 | 语音+拍照+文本的端到端连贯性            |
|  20  | 安全与隐私态度 | Llama Guard 4通过率 + COPPA保留规范     |
|  15  | 课程广度及图质量| 概念覆盖率 + 先决条件图一致性           |
| **100** |              |                                       |

## 练习

1. 开展效能研究，分别有无自适应学习者模型（随机概念顺序）。报告差值。预期自适应方案表现更优，关注差距大小。

2. 增加多模态探测：同一概念题以文字、语音、拍照三种形式展现。测量学习者是否更快在其偏好模态中达到收敛。

3. 构建家长仪表盘：练习主题、掌握轨迹、即将学习的概念、安全事件（任何防护触发）。符合 COPPA。

4. 增加语言切换模式：导师接受西班牙语输入，西班牙语授课。测量 X-Guard 覆盖情况。

5. 加强记忆隐私防护：验证学习者 A 无法通过语音剪辑重导入攻击访问学习者 B 的数据。记录尝试访问并报警。

## 关键词

| 术语                  | 流行说法            | 实际含义                             |
|-------------------|-----------------|---------------------------------|
| Socratic policy（苏格拉底策略） | “提问，不直接给答案”      | 导师通过引导问题而非直接给出答案            |
| Bayesian knowledge tracing（贝叶斯知识追踪） | “BKT”             | 经典的基于概念的掌握概率学习者模型方程       |
| FSRS                    | “自由间隔重复调度器”    | 2024年发布，优于 SM-2 的间隔重复调度算法     |
| Curriculum graph（课程图）       | “概念有向无环图”          | Neo4j构建的带先决条件边的概念图               |
| Episodic memory（情节记忆）       | “每次交互日志”           | 保存每次交互以备后续检索                       |
| Semantic memory（语义记忆）       | “学习模式库”             | 从情节记忆中提炼出的错误与偏好综合               |
| COPPA                    | “儿童隐私法”            | 美国法律，限制13岁以下儿童数据采集               |

## 扩展阅读

- [Khanmigo（可汗学院）](https://www.khanmigo.ai) — 参考消费级K-12导师  
- [Duolingo Max](https://blog.duolingo.com/duolingo-max/) — 参考语言学习导师  
- [Google LearnLM / Gemini for Education](https://blog.google/technology/google-deepmind/learnlm) — 托管参考模型  
- [Quizlet Q-Chat](https://quizlet.com) — 替代参考  
- [Synthesis Tutor](https://www.synthesis.com) — 创业公司参考  
- [FSRS算法](https://github.com/open-spaced-repetition/fsrs4anki) — 间隔重复调度器  
- [贝叶斯知识追踪](https://en.wikipedia.org/wiki/Bayesian_knowledge_tracing) — 经典学习者模型  
- [LiveKit Agents](https://github.com/livekit/agents) — 语音技术栈
