# Capstone 09 — 代码迁移代理（仓库级语言/运行时升级）

> 亚马逊的 MigrationBench（Java 8 到 17）和谷歌的 App Engine Py2 到 Py3 迁移器设定了 2026 年的标杆。Moderne 的 OpenRewrite 在大规模上进行确定性 AST 改写。Grit 针对同一问题，采用 codemod 风格的 DSL。生产模式结合了两者：一个用于安全重写的确定性基座，以及一个处理模糊情况的代理层，每个分支构建的沙箱，和一个在 PR 开启前变绿的测试环境。毕业设计目标是迁移 50 个真实仓库，并发布通过率及失败分类。

**类型：** 毕业设计  
**语言：** Python（代理），Java / Python（目标），TypeScript（仪表盘）  
**先修：** 第5阶段（NLP）、第7阶段（Transformer（Transformer 架构））、第11阶段（大语言模型（LLM）工程）、第13阶段（工具）、第14阶段（代理）、第15阶段（自主）、第17阶段（基础设施）  
**涉及阶段：** P5 · P7 · P11 · P13 · P14 · P15 · P17  
**时间：** 30 小时

## 问题

大规模代码迁移是 2026 年编码代理最干净的生产应用之一。地面真实标准显而易见（迁移后测试套件是否通过？），回报真实（Java 8 舰队迁移是大规模人员项目），基准公开（MigrationBench 50 个仓库子集）。Moderne 的 OpenRewrite 处理确定性部分。代理层则解决 OpenRewrite 规则无法覆盖的问题：模糊的重写、构建系统漂移、边缘语法、传递依赖破坏。

你将构建一个代理，接收 Java 8 仓库（或 Python 2 仓库），输出绿色 CI 迁移分支。你将测量通过率、测试覆盖率保持情况、每个仓库成本，并建立失败分类。与仅使用确定性基座的对照实验，帮助你了解代理价值所在。

## 概念

管道含两层。**确定性基座**（Java 使用 OpenRewrite，Python 使用 libcst）安全执行大部分机械化重写：导入、方法签名、空安全编辑、try-with-resources、弃用 API 替换。其速度快且产生可审计的代码差异。**代理层**（基于 OpenAI Agents SDK 或 LangGraph，使用 Claude Opus 4.7 和 GPT-5.4-Codex）处理规则不能覆盖的情况：构建文件升级（Maven/Gradle/pyproject）、传递依赖冲突、测试不稳定、自定义注解。

每个仓库对应一个预装目标运行时的 Daytona 沙箱。代理循环：执行构建、分类失败、应用修复、重新运行。硬限制：每仓库 30 分钟，8 美元，20 次代理回合。如果全部测试通过且覆盖率没有下降，分支开启 PR。若未通过，则仓库贴标签归类失败，并附证据。

失败分类是最终产物。50 个仓库中，什么出了问题？传递依赖？自定义注解？构建工具版本？与迁移无关的测试不稳定？每个类别统计数量并附示例差异。未来规则作者可聚焦头部三类问题。

## 架构

```text
target repo
      |
      v
OpenRewrite / libcst 确定性规则
   （安全、快速、可审计，涵盖约70-80%修复）
      |
      v
Daytona 沙箱（每分支一个）
      |
      v
代理循环（Claude Opus 4.7 / GPT-5.4-Codex）：
   - 运行构建 -> 捕获失败
   - 分类失败（构建、测试、静态检查）
   - 应用修复（补丁或重试规则）
   - 重新执行
   - 预算：30分钟，8美元，20轮
      |
      v
测试 + 覆盖率差异门控
      |
      v（通过）
开启 PR
      |
      v（失败）
归档失败类 + 附复现
```

## 技术栈

- 确定性基座：OpenRewrite（Java）或 libcst（Python）
- 代理：OpenAI Agents SDK 或 LangGraph，基于 Claude Opus 4.7 + GPT-5.4-Codex
- 沙箱：Daytona devcontainers，每分支预装目标运行时（Java 17 / Python 3.12）
- 构建系统：Maven、Gradle、uv（Python）
- 基准：亚马逊 MigrationBench 50 仓库子集（Java 8 到 17）、谷歌 App Engine Py2 到 Py3 仓库
- 测试环境：并行运行器，使用 Jacoco（Java）或 coverage.py（Python）测覆盖率
- 可观察性：Langfuse + 每个仓库每个差异块的追踪包
- 仪表盘：失败分类仪表盘，显示各类计数及示例差异

## 构建步骤

1. **运行规则。** 先运行 OpenRewrite（Java）或 libcst（Python）规则。抓取 70-80% 机械迁移。提交为“规则”提交。

2. **构建尝试。** Daytona 沙箱：安装目标运行时，执行构建。若绿灯，跳到测试。若红灯，交由代理。

3. **代理循环。** 结合工具：`run_build`、`read_file`、`edit_file`、`run_test`、`git_diff`。代理分类失败类型（依赖、语法、测试、构建工具），应用精准修复。重新执行。

4. **预算限制。** 每仓库 30 分钟实钟，8 美元成本，20 轮代理。超限停止，归类为“budget_exhausted”，附当前差异。

5. **测试与覆盖率门控。** 构建绿灯后运行测试套件。比较覆盖率与基线仓库差异。下降超过 2%，归档为“coverage_regression”。

6. **开启 PR。** 成功后推分支，打开 PR，附差异及应用规则和代理提交摘要。

7. **失败分类。** 每个失败仓库贴标签：`dep_upgrade_required`、`build_tool_drift`、`custom_annotation`、`test_flake`、`syntax_edge_case`、`budget_exhausted`。构建仪表盘。

8. **50 仓库跑测。** 在 MigrationBench 子集执行。报告各类通过率、每仓库成本、覆盖率保持，以及与仅确定性基座的对比基线。

## 使用示例

```text
$ migrate legacy-java-service --target java17
[recipe]   27 rewrites applied (JUnit 4->5, HashMap initializer, try-with-resources)
[build]    FAIL: cannot find symbol sun.misc.BASE64Encoder
[agent]    turn 1 classify: removed_jdk_api
[agent]    turn 2 apply: sun.misc.BASE64Encoder -> java.util.Base64
[build]    OK
[tests]    412/412 passing; coverage 84.1% -> 84.3%
[pr]       opened #1841  cost=$3.20  turns=4
```

## 交付成果

`outputs/skill-migration-agent.md` 是交付物。给定仓库，执行确定性规则及代理循环，产生绿色迁移分支，若失败则归入失败分类。

| 权重 | 评判标准 | 评判方法 |
|:-:|---|---|
| 25 | MigrationBench 通过率 | 50 仓库子集 pass@1 |
| 20 | 测试覆盖保持 | 平均覆盖率差异（相对基线） |
| 20 | 每迁移仓库成本 | 通过运行的 $/仓库 |
| 20 | 代理与确定性工具集成 | OpenRewrite 处理的修复比例 vs 代理原创比例 |
| 15 | 失败分析报告 | 分类完整性及示例 |
| **100** | | |

## 练习

1. 仅用 OpenRewrite 运行迁移管道（无代理），比较通过率和完整管道差异。识别代理唯一贡献的案例。

2. 实现“lint 清理”检查：迁移后运行风格检查器（Java 用 spotless，Python 用 ruff）。若新增 lint 报错，PR 失败。测量覆盖未降但风格退步的比例。

3. 添加“最小差异”优化器：代理分支通过测试后，执行第二遍修剪不必要改动。报告差异大小缩减量。

4. 扩展到第三种迁移：Node 18 到 Node 22。复用沙箱封装，替换规则层为自定义 codemod。

5. 测量首个绿色构建时间（TTFGB）作为用户体验指标。目标：p50 低于 10 分钟。

## 关键词

| 词汇 | 通俗说法 | 实际含义 |
|------|----------|---------|
| Deterministic substrate | “规则引擎” | OpenRewrite / libcst：具安全保证的声明式 AST 改写 |
| Codemod | “代码修改程序” | 机械修改源代码的改写规则 |
| Build drift | “工具版本偏差” | Maven / Gradle / uv 在不同主版本间的微妙行为差异 |
| Failure class | “分类桶” | 仓库迁移失败的标签原因：依赖、语法、测试、构建工具、预算等 |
| Coverage delta | “覆盖率保持” | 测试覆盖率百分比从基线到迁移分支的变化 |
| Agent turn | “工具调用轮” | 代理循环中一次计划-执行-观察的周期 |
| Budget exhaustion | “达到上限” | 仓库用尽 30 分钟 / 8 美元 / 20 轮限制未通过 |

## 延伸阅读

- [Amazon MigrationBench](https://aws.amazon.com/blogs/devops/amazon-introduces-two-benchmark-datasets-for-evaluating-ai-agents-ability-on-code-migration/) — 2026 正式基准  
- [Moderne.io OpenRewrite 平台](https://www.moderne.io) — 确定性基座参考  
- [OpenRewrite 文档](https://docs.openrewrite.org) — 规则编写  
- [Grit.io](https://www.grit.io) — 另一种 codemod DSL  
- [OpenAI 沙箱迁移烹饪书](https://developers.openai.com/cookbook/examples/agents_sdk/sandboxed-code-migration/sandboxed_code_migration_agent) — Agents SDK 参考  
- [Google App Engine Py2 到 Py3 迁移器](https://cloud.google.com/appengine) — 另一迁移基准  
- [libcst](https://github.com/Instagram/LibCST) — Python 确定性基座  
- [Daytona 沙箱](https://daytona.io) — 每分支沙箱参考
