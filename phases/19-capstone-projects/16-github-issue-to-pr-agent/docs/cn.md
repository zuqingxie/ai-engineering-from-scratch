# 毕业项目 16 — GitHub Issue 到 PR 的自主代理

> AWS Remote SWE Agents、Cursor Background Agents、OpenAI Codex 云端和 Google Jules 都发布了相同的 2026 产品形态：标记一个 issue，获得一个 PR。在云沙箱中运行代理，验证测试通过，并发布带有理由说明的可审查 PR。难点在于自动重现仓库的构建环境、防止凭证泄漏、执行每仓库预算、确保代理不能强制推送。本毕业项目构建自托管版本，并在成本和通过率上与托管方案进行对比。

**类型：** 毕业项目  
**语言：** Python（代理）、TypeScript（GitHub App）、YAML（Actions）  
**先决条件：** 阶段 11（LLM 工程）、阶段 13（工具）、阶段 14（代理）、阶段 15（自主）、阶段 17（基础设施）  
**涉及阶段：** P11 · P13 · P14 · P15 · P17  
**时长：** 30 小时

## 问题

异步云端编码代理是与交互式编码代理（毕业项目 01）不同的产品类别。用户体验是 GitHub 标签。你给 issue 打标签 `@agent fix this`，后台工作器在云沙箱中启动，克隆仓库，运行测试，编辑文件，验证，然后打开带有代理理由说明的 PR。无交互循环，无终端。AWS Remote SWE Agents、Cursor Background Agents、OpenAI Codex 云端、Google Jules 及 Factory Droids 都走向这一方案。

工程挑战很具体：环境复刻（代理需从头构建仓库，无缓存开发镜像），不稳定测试（必须重试或隔离），凭证范围限制（具有最小细粒度权限的 GitHub App），每仓库每日预算执行，以及禁止强制推送策略。本毕业项目测量通过率、成本和安全性，并与托管替代方案做比较。

## 概念

触发器是 GitHub webhook（issue 标签或 PR 评论）。调度器向 ECS Fargate 或 Lambda 排队任务。工作器将仓库拉取到 Daytona 或 E2B 沙箱，使用从仓库（语言、框架）推断的通用 Dockerfile。代理运行 mini-swe-agent 或 SWE-agent v2 循环，基于 Claude Opus 4.7 或 GPT-5.4-Codex。迭代过程：读取代码，提出修复，应用补丁，运行测试。

验证是关键步骤。必须在沙箱中通过完整 CI 后，才能打开 PR。计算覆盖率增减；若负值超过阈值，PR 仍打开但标记为 `needs-review`。代理将理由发布为 PR 描述及一个 `@agent` 线程供审阅者跟进。

安全性通过两个不同的 GitHub 面向实现：App 提供短期安装令牌，含 `workflows: read` 和狭义的仓库内容及 PR 范围权限；分支保护（非 App 权限）强制执行“禁止直接写入 `main`”和“禁止强制推送”——应用永远不加入绕过名单。路径范围的只读访问 `.github/workflows` 不是 GitHub App 原生权限，因此代理的文件编辑白名单须在工作器端实施。每仓库每日预算上限由调度器强制执行（例如，每仓库每天最多 5 个 PR，每个 PR 20 美元）。

## 架构

```text
GitHub issue 被标记 `@agent fix` 或 PR 评论
            |
            v
    GitHub App webhook -> AWS Lambda 调度器
            |
            v
    ECS Fargate 任务（或 GitHub Actions 自托管 runner）
       - 拉取仓库
       - 推断 Dockerfile（语言，包管理器）
       - Daytona / E2B 沙箱与目标运行时
       - 克隆 -> git worktree -> 代理分支
            |
            v
    mini-swe-agent / SWE-agent v2 循环
       使用 Claude Opus 4.7 或 GPT-5.4-Codex
       工具：ripgrep，tree-sitter，读/写文件，运行测试，git
            |
            v
    沙箱内验证 CI 通过 + 覆盖率增减检测
            |
            v（验证通过）
    git push + 通过 GitHub App 打开 PR
       PR 内容 = 理由 + 差异摘要 + 跟踪 URL
       标签：needs-review
            |
            v
    操作员审查；可 @-mention 代理跟进
```

## 技术栈

- 触发：具有细粒度令牌的 GitHub App；Lambda 或 Fly.io 接收 webhook
- 工作器：ECS Fargate 任务（或 GitHub Actions 自托管 runner）
- 沙箱：每任务 Daytona 开发容器或 E2B 沙箱
- 代理循环：基于 Claude Opus 4.7 / GPT-5.4-Codex 的 mini-swe-agent 基线或 SWE-agent v2
- 检索：tree-sitter 仓库映射 + ripgrep
- 验证：沙箱中完整 CI + 覆盖率增减关卡
- 观测：Langfuse，PR 内容中链接每个 PR 跟踪归档
- 预算：每仓库每日美元上限；每仓库每日最大 PR 数

## 构建步骤

1. **GitHub App。** 细粒度安装令牌：issues 读写，pull_requests 写入，contents 读写，workflows 读取。分支保护（唯一可实现限制的面向）执行“禁止直接推送 `main`”和“禁止强制推送”；App 不在绕过名单中。工作器在提交差异时检查“禁止修改 `.github/workflows`”作为白名单检查，因为 GitHub App 权限不是路径范围。

2. **Webhook 接收。** Lambda 函数接收 issue 标签/PR 评论 webhook。以标签 `@agent fix this` 过滤。任务加入 SQS 队列。

3. **调度器。** 从 SQS 弹出任务。执行每仓库每日预算限制。启动 ECS Fargate 任务，带入仓库 URL、issue 内容及全新的 Daytona 沙箱。

4. **环境推断。** 检测语言（Python，Node，Go，Rust）及包管理器（uv，pnpm，go mod，cargo）。若无 Dockerfile，动态生成。

5. **代理循环。** 运行 mini-swe-agent 或 SWE-agent v2（Claude Opus 4.7）。工具包括 ripgrep，tree-sitter 仓库映射，读写文件，运行测试，git。硬限制：20 美元成本，30 分钟实时时间，30 轮对话。

6. **验证。** 循环结束后，在沙箱内运行完整测试套件。利用 jacoco / coverage.py 计算覆盖率增减。若 CI 失败：停止且不打开 PR。若覆盖率下降超过 2%：打开带 `needs-review` 标签的 PR。

7. **PR 发布。** 推送代理分支。通过 GitHub API 打开 PR，包含：标题、理由说明、差异摘要、跟踪 URL、成本、轮次数。

8. **凭证卫生。** 工作器运行时使用短期 GitHub App 安装令牌。日志存档前进行敏感信息清理。

9. **评估。** 30 个多难度内部种子 issue。测量通过率、PR 质量（差异大小、样式、覆盖率）、成本、延迟。与 Cursor Background Agents 和 AWS Remote SWE Agents 在相同 issue 上对比。

## 使用示例

```text
# 在 github.com 上
  - 用户给 issue #842 打标签 `@agent fix this`
  - 14 分钟后出现 PR #1903
  - PR 内容：
    > 修复 widget.dedupe() 中由 null 比较器条目引发的 NPE。
    > 添加回归测试 widget_test.go::TestDedupeNullComparator。
    > 覆盖率变化：+0.12%
    > 对话轮数：7  成本：$1.80  跟踪：langfuse:...
    > 标签：needs-review
```

## 发布成果

`outputs/skill-issue-to-pr.md` 是成果交付物。一个 GitHub App + 异步云工作器，能将标记的 issue 转换为带预算限制和范围权限的审核就绪 PR。

| 权重 | 标准 | 评估方式 |
|:-:|---|---|
| 25 | 30 个 issue 的通过率 | 端到端成功（CI 通过 + 覆盖率合格） |
| 20 | PR 质量 | 差异大小，覆盖率变化，样式符合度 |
| 20 | 每 resolved issue 的成本和延迟 | 每 PR 的美元成本和实时时间 |
| 20 | 安全性 | 范围令牌、每仓库预算、禁止强推、凭证卫生 |
| 15 | 操作员用户体验 | 理由评论、重试能力、@-mention 跟踪 |
| **100** | | |

## 练习

1. 添加“修复不稳定测试”模式：标签 `@agent stabilize-flake TestX` 在沙箱中运行该测试 50 次，并提出最小改动以稳定测试。

2. 在三个共享 issue 上与 Cursor Background Agents 进行成本对比，报告不同工具在哪些情形下表现更优。

3. 实现预算仪表盘：每仓库每日成本，每用户成本。异常时报警。

4. 构建“演练”模式，开启草稿 PR，不运行 CI，方便审查者低成本检查计划。

5. 添加保留策略：PR 分支超过 7 天未合并自动删除。

## 关键术语

| 术语 | 业界说法 | 实际含义 |
|------|---------|---------|
| GitHub App | “有范围的机器人身份” | 拥有细粒度权限和短期安装令牌的应用 |
| 异步云代理 | “后台代理” | 非交互式的云沙箱工作器，无终端交互 |
| 环境推断 | “Dockerfile 合成” | 检测语言+包管理器，若无则生成 Dockerfile |
| 验证 | “沙箱内 CI” | 在工作器内运行完整测试套件，才打开 PR |
| 覆盖率变化 | “覆盖率保持” | 代理分支相较基线的测试覆盖率百分比变动 |
| 每仓库预算 | “每日上限” | 调度器执行的美元和 PR 数量上限 |
| 理由 | “PR 内容解释” | 代理变更的总结和理由，写入 PR 正文 |

## 延伸阅读

- [AWS Remote SWE Agents](https://github.com/aws-samples/remote-swe-agents) — 典型异步云代理参考  
- [SWE-agent](https://github.com/SWE-agent/SWE-agent) — CLI 参考  
- [Cursor Background Agents](https://docs.cursor.com/background-agent) — 商业替代方案  
- [OpenAI Codex（云端）](https://openai.com/codex) — 托管竞品  
- [Google Jules](https://jules.google) — Google 托管版本  
- [Factory Droids](https://www.factory.ai) — 另一商业参考  
- [GitHub App 文档](https://docs.github.com/en/apps) — 范围机器人身份  
- [Daytona 云沙箱](https://daytona.io) — 参考沙箱
