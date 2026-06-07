# Capstone 10 — 多代理软件工程团队

> SWE-AF 的工厂架构、MetaGPT 的基于角色的提示、AutoGen 0.4 的类型化 actor 图、Cognition 的 Devin 和 Factory 的 Droids 全部汇聚成同一个2026年的形态：架构师负责规划，N 个编码员在并行的 worktree 中工作，审核员负责把关，测试员进行验证。并行 worktree 将墙钟时间转化为吞吐量。共享状态和交接协议成为故障面。毕业设计是构建该团队，在 SWE-bench Pro 上评估，并报告哪些交接失败以及失败频率。

**类型：** 毕业设计  
**语言：** Python / TypeScript（代理），Shell（worktree 脚本）  
**先决条件：** 阶段 11（LLM 工程）、阶段 13（工具）、阶段 14（代理）、阶段 15（自主）、阶段 16（多代理）、阶段 17（基础设施）  
**涉及阶段：** P11 · P13 · P14 · P15 · P16 · P17  
**时间：** 40 小时

## 问题

单代理编码方法在大规模任务上遇到天花板。原因不是某个单独代理能力不足，而是因为 20 万 token 的上下文无法同时容纳架构规划、四个并行代码片段、审核员评论和测试输出。多代理工厂将问题拆分：架构师负责规划，编码员在并行 worktree 中负责实现，审核员把关，测试员验证。SWE-AF 的“工厂”架构、MetaGPT 的角色划分、AutoGen 的类型化 actor 图——这三者描述了同一种形态。

失败面是交接环节。架构师规划了编码员无法实现的内容。编码员产生冲突的 diff。审核员审核了虚幻的修复。测试员竞跑于仍在编码的编码员。你将构建其中一个团队，在 50 个 SWE-bench Pro 问题上运行，跟踪每一次交接，并撰写事后分析。

## 概念

角色即类型化代理。**架构师**（Claude Opus 4.7）阅读 issue，编写计划，并将其拆解为带有显式接口的子任务。**编码员**（Claude Sonnet 4.7，N 个并行实例，每个在一个 `git worktree` + Daytona 沙盒中）独立实现子任务。**审核员**（GPT-5.4）阅读合并的 diff，批准或请求具体修改。**测试员**（Gemini 2.5 Pro）在隔离环境中运行测试套件，报告通过/失败及相关工件。

通信通过共享任务看板（基于文件或 Redis）。每个角色消费它允许处理的任务。交接是类型化的 A2A 协议消息。协调关注点包括：合并冲突解决（协调者角色或自动三方合并）、共享状态同步（规划一旦开始编码即冻结；重新规划为独立事件）、审核把关（审核员不可批准自己或自己提议的更改）。

token 放大是隐性成本。每个角色边界增加摘要提示和交接上下文。40 轮单代理执行变成四个角色共 160 轮。评分标准专门权衡 token 效率与单代理基线，因为问题不是“多代理是否有效”，而是“花费每美元是否胜出”。

## 架构

```text
GitHub issue URL
      |
      v
架构师 (Opus 4.7)
   阅读 issue，产出带子任务 + 接口的计划
      |
      v
任务看板 (文件 / Redis)
      |
   +-- 子任务 1 ---+-- 子任务 2 ---+-- 子任务 3 ---+-- 子任务 4 ---+
   v                v                v                v                v
编码员 A          编码员 B          编码员 C          编码员 D          (4 个并行)
 (Sonnet)         (Sonnet)         (Sonnet)         (Sonnet)
 worktree A       worktree B       worktree C       worktree D
 Daytona          Daytona          Daytona          Daytona
      |                |                |                |
      +--------+-------+-------+--------+
               v
           合并协调者  (三方合并 + 冲突解决)
               |
               v
           审核员 (GPT-5.4)
               |
               v
           测试员  (Gemini 2.5 Pro)  -> 通过？ -> 开启 PR
                                    -> 失败？ -> 返回编码员
```

## 技术栈

- 编排：LangGraph，支持共享状态 + 每代理子图  
- 消息传递：A2A 协议（Google 2025），用于类型化代理间消息  
- 模型：Opus 4.7（架构师）、Sonnet 4.7（编码员）、GPT-5.4（审核员）、Gemini 2.5 Pro（测试员）  
- worktree 隔离：每编码员一个 `git worktree add` + Daytona 沙盒  
- 合并协调者：自定义三方合并 + LLM 辅助冲突调解  
- 评测：SWE-bench Pro（50 个问题）、SWE-AF 场景、HumanEval++ 单元测试  
- 可观测性：Langfuse，带角色标签的跨度追踪，按代理统计 token  
- 部署：K8s，每角色一个 Deployment + 基于积压自动扩缩（HPA）

## 构建步骤

1. **任务看板。** 文件支持的 JSONL，类型化消息包括：`plan_request`，`subtask`，`diff_ready`，`review_needed`，`test_needed`，`approved`，`rejected`，`replan_needed`。代理订阅对应标签。

2. **架构师。** 读取 GitHub issue，运行 Opus 4.7，使用计划模板要求显式子任务接口（修改文件、公共函数、测试影响）。产出一条包含子任务 DAG 的 `plan_request`。

3. **编码员。** N 个并行工人，各自从看板认领一个子任务。为每个子任务新建一个 `git worktree add` 分支和 Daytona 沙盒，实现子任务。提交 `diff_ready`，包含补丁和测试差异。

4. **合并协调者。** 等所有编码完毕后，将 N 条分支通过三方合并合并到临时分支。仅当文件级冲突时启用 LLM 调解。

5. **审核员。** GPT-5.4 读取合并后的 diff，不可批准自己撰写的 diff。输出 `approved`（无操作）或带具体变更请求的 `review_feedback`，并转回对应编码员。

6. **测试员。** Gemini 2.5 Pro 在干净沙盒中运行测试套件。收集工件。发出 `test_passed` 或 `test_failed`，失败时带堆栈信息。失败测试循环回归给对应编码员。

7. **交接统计。** 每条跨角色边界消息产生 Langfuse 跨度，记录负载大小和用到的模型。计算每子任务 token 放大比（编码员 token + 审核员 token + 测试员 token + 架构师占比 / 编码员 token）。

8. **评估。** 在 50 个 SWE-bench Pro 问题上运行。对比 pass@1 和 每解决问题成本美元数，相对于单代理基线（单个 Sonnet 4.7 在单一 worktree）。

9. **事后分析。** 每个失败问题，识别导致失败的交接（规划模糊、合并冲突、审核员误批、测试波动），输出交接失败直方图。

## 使用示例

```text
$ team run --issue https://github.com/acme/widget/issues/842
[architect] 规划：4 个子任务（parser，cache，api，migration）
[board]     分发给 4 个编码员并行 worktree
[coder-A]   子任务 parser  -> 42 行，测试本地通过
[coder-B]   子任务 cache   -> 88 行，测试本地通过
[coder-C]   子任务 api     -> 31 行，测试本地通过
[coder-D]   子任务 migration -> 19 行，测试本地通过
[merge]     三方合并：0 冲突
[reviewer]  对 cache（线程池大小）发表评论；转给 coder-B
[coder-B]   重复修改：92 行；提交
[reviewer]  批准
[tester]    全部 412 个测试通过
[pr]        开启 #3382   4 编码员，1 次修订，4.90$，耗时18分
```

## 交付成果

`outputs/skill-multi-agent-team.md` 是交付物。输入 issue URL 和并行度，团队产出合并就绪的 PR，并记录每个角色的 token 用量。

| 权重 | 评判标准 | 衡量方式 |
|:-:|---|---|
| 25 | SWE-bench Pro pass@1 | 匹配 50 个问题子集，pass@1 达成率 |
| 20 | 并行提速 | 墙钟时间相较单代理基线 |
| 20 | 审核质量 | 注入漏洞探测的误批率 |
| 20 | token 效率 | 每解决问题的总 token 数相较单代理 |
| 15 | 协调工程 | 合并冲突解决策略，交接失败直方图 |
| **100** | | |

## 练习

1. 在运行中途的 diff 中注入明显错误（主逻辑前多插入 `return None`），测量审核员的误批率。调试审核员提示，直到误批率低于 5%。

2. 缩减到两个编码员（架构师 + 单编码员 + 审核员 + 测试员，编码员串行执行两个子任务）。对比墙钟时间和通过率。

3. 用单写限制替换合并协调者（子任务触及文件集不重叠）。测量架构师的规划负担。

4. 将审核员从 GPT-5.4 替换为 Claude Opus 4.7。测量误批率和 token 成本的差异。

5. 添加第五个角色：文档员（Haiku 4.5）。审核后其产生变更日志条目。衡量文档质量是否足以抵消额外的 token 花费。

## 关键术语

| 术语 | 人们说 | 实际含义 |
|------|--------|----------|
| 并行 worktree | “隔离分支” | 每个编码员一个 `git worktree add` 的独立工作区 |
| 任务看板 | “共享消息总线” | 文件或 Redis 存储的类型化消息，代理订阅 |
| 交接 | “角色边界” | 任意跨角色上下文的消息 |
| token 放大 | “多代理开销” | 多角色间总 token 数 / 单代理相同任务 token 数 |
| A2A 协议 | “代理到代理” | Google 2025 年代理间消息的类型规范 |
| 合并协调者 | “集成者” | 执行三方合并并调解冲突的组件 |
| 误批 | “审核员幻觉” | 审核员批准包含已知 bug 的 diff |

## 相关阅读

- [SWE-AF 工厂架构](https://github.com/Agent-Field/SWE-AF) — 2026年多代理工厂参考  
- [MetaGPT](https://github.com/FoundationAgents/MetaGPT) — 基于角色的多代理框架  
- [AutoGen v0.4](https://github.com/microsoft/autogen) — 微软类型化 actor 框架  
- [Cognition AI (Devin)](https://cognition.ai) — 参考产品  
- [Factory Droids](https://www.factory.ai) — 另一个参考产品  
- [谷歌 A2A 协议](https://developers.google.com/agent-to-agent) — 代理间消息规范  
- [git worktree 文档](https://git-scm.com/docs/git-worktree) — 隔离基础设施  
- [SWE-bench Pro](https://www.swebench.com) — 评测目标
