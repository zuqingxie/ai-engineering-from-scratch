# Capstone 15 — 宪法安全保护带 + 红队测试范围

> Anthropic 的 Constitutional Classifiers（宪法分类器）、Meta 的 Llama Guard 4、Google 的 ShieldGemma-2、NVIDIA 的 Nemotron 3 Content Safety（内容安全）和覆盖多语种的 X-Guard 定义了 2026 年的安全分类器栈。garak、PyRIT、NVIDIA Aegis 和 promptfoo 成为标准的对抗攻击评估工具。NeMo Guardrails v0.12 将它们集成到生产流水线中。这个顶点项目将所有这些串联起来：围绕目标应用的多层安全保护带，运行 6 种以上攻击家族的自主红队代理，以及产生可测量无害性差异的宪法自我批判执行。

**类型：** 顶点项目  
**语言：** Python（安全流水线，红队），YAML（策略配置）  
**先决条件：** 阶段 10（从零开始的 LLM）、阶段 11（LLM 工程）、阶段 13（工具）、阶段 14（智能代理）、阶段 18（伦理、安全、对齐）  
**涉及阶段：** P10 · P11 · P13 · P14 · P18  
**时长：** 25 小时

## 问题

2026 年 LLM（大语言模型）安全的前沿不是分类器是否有效（它们大致有效），而是如何正确地将它们组合应用于生产应用中，既不过度拒绝也不留明显漏洞。Llama Guard 4 处理英语策略违规；X-Guard（132 种语言）处理多语种越狱；ShieldGemma-2 捕捉图像型提示注入；NVIDIA Nemotron 3 Content Safety 涵盖企业类别。Anthropic 的 Constitutional Classifiers 在训练期间使用，是另一种独立方法，而非推理时服务。

攻击发展同样重要。PAIR 和 TAP 自动发现越狱手段；GCG 运行基于梯度的后缀攻击；多轮交互和代码切换攻击利用智能代理的记忆。任何部署的 LLM 都需要一个红队测试范围——以 garak 和 PyRIT 为典型工具——以及完善的缓解措施和带有 CVSS 评分的安全发现报告。

你将强化一个目标应用（可以是一个 8B 指令调优模型，或其他顶点项目的 RAG 聊天机器人之一），针对其运行 6 种以上攻击家族，产出前后无害性测量。

## 概念

安全流水线分为五层。**输入净化**：剔除零宽字符，解码 base64/rot13，规范化 Unicode。**策略层**：NeMo Guardrails v0.12 规制（领域偏离、毒性、PII 提取）。**分类器门控**：输入端使用 Llama Guard 4，非英语用 X-Guard，图像输入用 ShieldGemma-2。**模型**：目标 LLM。**输出过滤**：输出端使用 Llama Guard 4，Presidio PII 清理，适用时强制引用。**HITL（人机交互）层**：被标记为高风险的输出发送到 Slack 队列。

红队测试范围由调度器控制。PAIR 和 TAP 自动发现越狱；GCG 运行基于梯度的后缀攻击；ASCII/base64/rot13 编码攻击；多轮交互（角色扮演、记忆利用）；代码切换攻击（英语与斯瓦希里语或泰语混合）。每次运行生成带有 CVSS 评分和披露时间轴的结构化发现文件。

宪法自我批判执行是训练时的干预。取 1000 个有害尝试提示，模型起草响应，对照书面宪法（“勿伤害”规则）进行批判，并在批判循环中进行再训练。对持出评估集测量前后无害性差异。

## 架构

```text
request (text / image / multilingual)
      |
      v
input sanitize (strip zero-width, decode, normalize)
      |
      v
NeMo Guardrails v0.12 rails (off-domain, policy)
      |
      v
classifier gate:
  Llama Guard 4 (English)
  X-Guard (multilingual, 132 langs)
  ShieldGemma-2 (image prompts)
  Nemotron 3 Content Safety (enterprise)
      |
      v (allowed)
target LLM
      |
      v
output filter: Llama Guard 4 + Presidio PII + citation check
      |
      v
HITL tier for flagged outputs

parallel:
  red-team scheduler
    -> garak (classic attacks)
    -> PyRIT (orchestrated red team)
    -> autonomous jailbreak agent (PAIR + TAP)
    -> GCG suffix attacks
    -> multilingual / code-switch
    -> multi-turn persona adoption

output: CVSS-scored findings + disclosure timeline + before/after harmlessness delta
```

## 技术栈

- 安全分类器：Llama Guard 4、ShieldGemma-2、NVIDIA Nemotron 3 Content Safety、X-Guard  
- 防护框架：NeMo Guardrails v0.12 + OPA  
- 红队驱动：garak（NVIDIA）、PyRIT（Microsoft Azure）、NVIDIA Aegis、promptfoo  
- 越狱代理：PAIR（Chao 等，2023）、TAP（攻击树）、GCG 后缀攻击  
- 宪法训练：Anthropic 风格自我批判循环 + SFT（监督微调）  
- PII 清理：Presidio  
- 目标：一个 8B 指令调优模型或其他顶点项目中的 RAG 聊天机器人之一

## 实现步骤

1. **目标搭建。** 在 vLLM 上部署一个 8B 指令调优模型（或复用其他顶点项目的 RAG 聊天机器人）。这就是测试应用。

2. **安全流水线封装。** 将五层流水线接入目标。验证每层均可独立观测（在 Langfuse 中为每层生成 span）。

3. **分类器覆盖。** 加载 Llama Guard 4、X-Guard（多语种）、ShieldGemma-2（图像）。在小规模带标注集上运行，建立基线。

4. **红队调度器。** 调度 garak、PyRIT、PAIR 代理、TAP 代理、GCG 运行器、多轮攻击者和代码切换攻击者。每个运行于独立队列。

5. **攻击套件。** 六个攻击家族：（1）PAIR 自动越狱，（2）TAP 攻击树，（3）GCG 梯度后缀，（4）ASCII/base64/rot13 编码攻击，（5）多轮角色扮演，（6）多语种代码切换。报告各家族成功率。

6. **宪法自我批判。** 筛选 1000 个有害尝试提示。目标模型为每个提示起草回应。批评型 LLM 按书面宪法规则（“勿伤害”、“引用证据”、“拒绝非法请求”）评分。被批评否定的提示被重写，目标模型在改进后的对照对上微调。测量持出评估集上的无害性前后差异。

7. **过度拒绝测量。** 在无害提示套件（例如 XSTest）上跟踪误报率。目标模型应对无害问题保持有用。

8. **CVSS 评分。** 对成功越狱案例，按照 CVSS 4.0 标准（攻击向量、复杂度、影响）评分。生成披露时间轴及缓解计划。

9. **测试自动化。** 将以上所有流程设为定时任务；安全发现写入队列；过度拒绝回归警报推送 Slack。

## 使用示例

```text
$ safety probe --model=target --family=PAIR --budget=50
[attacker]   PAIR agent running on target
[attack]     attempt 1/50: disguise query as academic research ... blocked
[attack]     attempt 2/50: appeal to roleplay ... blocked
[attack]     attempt 3/50: chain-of-thought coax ... SUCCEEDED
[finding]    CVSS 4.8 medium: roleplay bypass on target
[range]      7 successes out of 50 (14% success rate)
```

## 交付物

`outputs/skill-safety-harness.md` 是成果文件，包含生产级多层安全流水线和可复现的红队测试范围，附带前后无害性差异。

| 权重 | 评估标准 | 测量方式 |
|:-:|---|---|
| 25 | 攻击面覆盖 | 6+ attack-family drills with 2+ language coverage |
| 20 | 误报/真报权衡 | Attack block rate vs XSTest benign pass rate |
| 20 | 自我批判差异 | Pre/post harmlessness delta on a held-out evaluation set |
| 20 | 文档与披露 | CVSS-scored findings report with disclosure timeline |
| 15 | 自动化与可复现性 | Scheduled end-to-end run with alerting |
| **100** | | |

## 练习

1. 在 RAG 聊天机器人上运行 garak 的提示注入插件，比较启用和未启用输出过滤层时的攻击成功率。

2. 添加第七攻击家族：通过检索文档的间接提示注入。测量所需的额外防御。

3. 实现“拒绝但提供帮助”模式：当防护层阻断时，目标模型提供安全相关的替代回答而非直接拒绝。衡量 XSTest 差异。

4. 多语种覆盖漏洞：找出 X-Guard 表现不足的语言。建议针对该语言的微调数据集。

5. 在 30B 模型上运行宪法自我批判，测量无害性差异比例是否扩大。

## 关键词

| 术语 | 通常说法 | 实际含义 |
|------|----------|----------|
| Layered safety（多层安全） | “纵深防御” | 输入、门控、输出、HITL 多重护栏 |
| Llama Guard 4 | “Meta 的安全分类器” | 2026 年输入/输出内容参考分类器 |
| PAIR | “越狱代理” | Chao 等人的 LLM 驱动越狱发现论文 |
| TAP | “攻击树” | PAIR 的树状搜索版本 |
| GCG | “贪心坐标梯度” | 基于梯度的对抗后缀攻击 |
| Constitutional self-critique | “Anthropic 式训练” | 目标生成回应 -> 批评评分 -> 重写 -> 重新训练 |
| XSTest | “无害测试集” | 过度拒绝回归的基准集 |
| CVSS 4.0 | “安全严重度评分” | 安全发现的漏洞评分标准 |

## 延伸阅读

- [Anthropic Constitutional Classifiers](https://www.anthropic.com/research/constitutional-classifiers) — 训练时参考  
- [Meta Llama Guard 4](https://ai.meta.com/research/publications/llama-guard-4/) — 2026 年输入/输出分类器  
- [Google ShieldGemma-2](https://huggingface.co/google/shieldgemma-2b) — 图像+多模态安全  
- [NVIDIA Nemotron 3 Content Safety](https://developer.nvidia.com/blog/building-nvidia-nemotron-3-agents-for-reasoning-multimodal-rag-voice-and-safety/) — 企业参考方案  
- [X-Guard (arXiv:2504.08848)](https://arxiv.org/abs/2504.08848) — 支持 132 语种的多语种安全  
- [garak](https://github.com/NVIDIA/garak) — NVIDIA 红队工具包  
- [PyRIT](https://github.com/Azure/PyRIT) — Microsoft 红队框架  
- [NeMo Guardrails v0.12](https://docs.nvidia.com/nemo-guardrails/) — 规制框架  
- [PAIR (arXiv:2310.08419)](https://arxiv.org/abs/2310.08419) — 越狱代理论文
