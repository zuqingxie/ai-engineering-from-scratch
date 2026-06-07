# Capstone 06 — Kubernetes 的 DevOps 故障排查代理

> AWS 的 DevOps 代理已正式发布，Resolve AI 发布了其 K8s playbooks，NeuBird 演示了语义监控，Metoro 将 AI SRE 绑定到每个服务的 SLO。生产形态已定：警报 webhook 触发，代理读取遥测，遍历 K8s 对象图，排序根因假设，并带审批按钮发布 Slack 简报。默认只读。所有修复动作均需人工审核。该 Capstone 即实现此代理，在 20 个合成事件上评估，并在三个共享案例上与 AWS 代理对比。

**类型：** Capstone  
**语言：** Python（代理）、TypeScript（Slack 集成）  
**先决条件：** 第 11 阶段（LLM 工程）、第 13 阶段（工具与 MCP）、第 14 阶段（代理）、第 15 阶段（自治）、第 17 阶段（基础设施）、第 18 阶段（安全）  
**涉及阶段：** P11 · P13 · P14 · P15 · P17 · P18  
**时间：** 30 小时

## 问题

2025-2026 年的 SRE 叙事变成：“AI 代理负责事件分诊，人类批准修复。”AWS DevOps Agent、Resolve AI、NeuBird、Metoro、PagerDuty AIOps 均在生产环境采用此方案。代理读取 Prometheus 指标、Loki 日志、Tempo 追踪、kube-state-metrics 与 K8s 对象知识图谱。它在五分钟内生成带遥测出处的根因假设排序。绝不执行破坏性命令，除非通过 Slack 获得明确人类批准。

工作重点在于范围界定与安全，不是推理。代理需要默认只读的 RBAC 权限面、加固的 MCP 工具服务器以及每条命令“考虑与执行”都要有审计日志。它还需知道自身认知边界并升级报警。且得保持运行成本低，避免 OOM-kill 级联导致超过 5 千美元代理账单。

## 概念

代理操作于知识图谱上。节点是 K8s 对象（Pods、Deployments、Services、Nodes、HPAs、PVCs）及遥测源（Prometheus 指标流、Loki 日志流、Tempo 追踪）。边编码了所有权（Pod -> ReplicaSet -> Deployment）、调度（Pod -> Node）、观测（Pod -> Prometheus 指标）。图谱由 kube-state-metrics 同步保持新鲜，每次警报时重新采样。

警报触发后，代理从受影响对象开始根因定位。它遍历边，拉取相关过去 15 分钟的遥测片段，拟定假设。假设根据证据排序：有多少遥测出处支持，时间多近，具体度如何。排名前三的假设发送到 Slack，附带图路径可视化和修复审批按钮。

修复操作需审批。默认允许的动作为只读。破坏性动作（缩容、回滚、删除 Pods）需 Slack 审批；ArgoCD 回滚挂钩需授权令牌，代理不持有。审计日志记录代理*“考虑过”*的每条命令——不止执行的，这样复审流程才能捕获近失误。

## 架构

```text
PagerDuty / Alertmanager webhook
           |
           v
     FastAPI 接收器
           |
           v
   LangGraph 根因代理
           |
           +---- 只读 MCP 工具 ----+
           |                      |
           v                      v
   K8s 知识图谱               遥测切片
     (Neo4j / kuzu)          Prometheus, Loki, Tempo
   所有权 + 调度               最近 15 分钟，范围限定
           |
           v
   假设排序（证据权重）
           |
           v
   Slack 简报 + 审批按钮
           |
           v（已批准）
   ArgoCD 回滚挂钩 / PagerDuty 升级
           |
           v
   审计日志：考虑 vs 执行，每条命令
```

## 技术栈

- 可观测源：Prometheus、Loki、Tempo、kube-state-metrics  
- 知识图谱：Neo4j（托管）或 kuzu（嵌入式）K8s 对象 + 遥测边  
- 代理：LangGraph，带有每个工具的白名单，默认只读  
- 工具传输：FastMCP 通过 StreamableHTTP；破坏性工具由审批网关隔离的单独服务器托管  
- 模型：Claude Sonnet 4.7 负责根因推理，Gemini 2.5 Flash 用于日志摘要  
- 修复：ArgoCD 回滚 webhook，PagerDuty 报警升级，Slack 审批卡片  
- 审计：追加式结构日志（考虑、执行、批准、结果）  
- 部署：K8s 部署，具有限定 RBAC 权限；分离命名空间  

## 构建步骤

1. **图谱摄取。** 每 30 秒同步 kube-state-metrics 至 Neo4j/kuzu。节点包括 Pod、Deployment、Node、Service、PVC、HPA。边包括 OWNED_BY、SCHEDULED_ON、EXPOSES、MOUNTS、SCALES。遥测叠加边：OBSERVED_BY（一个 Pod 被 Prometheus 指标流观测）。

2. **报警接收器。** FastAPI 端点，接收 PagerDuty 或 Alertmanager webhook。解析受影响对象及 SLO 违约。

3. **只读工具面。** 通过 FastMCP 包装 kubectl、Prometheus query、Loki logql、Tempo traceql。每个工具具有限定 RBAC 动词（“get”、“list”、“describe”）。默认服务器无 “delete”、“exec”、“scale” 权限。

4. **根因代理。** LangGraph 有三个节点：`sample` 拉取最近 15 分钟遥测片段，`walk` 查询图邻居对象，`hypothesize` 拟定并排序根因候选假设，带遥测出处。

5. **证据评分。** 每个假设得分 = 近期程度 × 具体度 × 图路径长度倒数 × 引用数。返回前 3。

6. **Slack 简报。** 发布附件，含假设内容、图路径可视化（服务器端渲染子图像）和最多一条修复动作的审批按钮。

7. **修复门控。** 破坏性工具（缩容、回滚、删除）在另一台需审批令牌的 MCP 服务器运行。代理只能在 Slack 卡批准后调用。

8. **审计日志。** 追加式 JSONL：每条候选命令，记录是否考虑、是否执行、谁批准。每日发往 S3。

9. **合成事件套件。** 构建 20 种场景：OOMKill 级联、DNS 波动、HPA 震荡、PVC 填满、噪声邻居、故障 Sidecar、错误 ConfigMap 发布、证书轮换、镜像拉取回退等。对根因准确率和假设生成时长进行评分。

## 使用示例

```text
webhook: alert.pagerduty.com -> checkout-api SLO 违约，错误率 14%
[graph]   受影响：Deployment checkout-api（3 个 Pods，节点 ip-10-2-3-4）
[walk]    邻居：ReplicaSet checkout-api-abc，Service checkout-api，
           最近发布 14 分钟前
[sample]  prometheus 错误率 14%，上升趋势；loki /api/v2/pay 出现 500 错误
[hypo]    #1 发布故障：最新镜像 checkout-api:v2.41 /healthz 失败
          引用: deploy.yaml（版本 42），prometheus 错误率，loki 500 堆栈
[slack]   [回滚到 v2.40]  [升级处理]  [忽略处理]
          （需审批；代理不单方面回滚）
```

## 交付物

`outputs/skill-devops-agent.md` 为交付成果。给定一个 K8s 集群与警报源，代理产出排名根因假设及基于 Slack 审批的修复流程。

| 权重 | 标准 | 评估方式 |
|:-:|---|---|
| 25 | 场景套件根因准确率 | 20 个合成事件根因准确率 ≥ 80% |
| 20 | 安全性 | 破坏性操作无 Slack 审批时永不触发（审计日志验证） |
| 20 | 假设生成时长 | p50 从警报到 Slack 简报不超过 5 分钟 |
| 20 | 可解释性 | 每个假设含图路径与遥测出处 |
| 15 | 集成完整性 | PagerDuty、Slack、ArgoCD、Prometheus 端到端可用 |
| **100** |  |  |

## 练习

1. 在 AWS DevOps 代理演示的同三个事件上运行你的代理。发布并排版结果。分析两者差异。

2. 增加“近失误”审计，标记代理*考虑过*但未经审批即会造成破坏的命令。统计一周内近失误率。

3. 将假设模型从 Claude Sonnet 4.7 替换为自托管的 Llama 3.3 70B。测量根因准确率变化及单次事件花费。

4. 构建因果筛选器：区分相关的遥测峰值与真正根因。基于 20 个场景标签训练小型分类器。

5. 添加回滚预演：用 ArgoCD 在预备集群执行同一 manifest 回滚。确认回滚计划正确，之后再通过 Slack 审批按钮。

## 关键词

| 术语 | 常用说法 | 实际含义 |
|------|-----------------|------------------------|
| K8s 知识图谱 | “集群图谱” | 节点 = K8s 对象 + 遥测序列；边 = 所有权、调度、观测 |
| 默认只读 | “范围限定 RBAC” | 代理服务账户仅具备 get/list/describe 权限；破坏性操作权限单独服务器并需审批 |
| 审计日志 | “考虑 vs 执行” | 对所有候选命令的追加日志，含是否执行及谁批准 |
| 假设排序 | “证据评分” | 近期 × 具体 × 图路径长度倒数 × 引用计数 |
| Slack 审批卡片 | “HITL 网关” | 含修复按钮的交互式 Slack 消息；无人工点击代理无法继续 |
| 遥测引用 | “证据指针” | 支持假设的 Prometheus 查询、Loki 选择器或 Tempo 追踪链接 |
| MTTR | “恢复时间” | 从警报触发到 SLO 恢复的实际时间 |

## 延伸阅读

- [AWS DevOps Agent GA](https://aws.amazon.com/blogs/aws/aws-devops-agent-helps-you-accelerate-incident-response-and-improve-system-reliability-preview/) — 2026 年权威参考  
- [Resolve AI K8s 故障排查](https://resolve.ai/blog/kubernetes-troubleshooting-in-resolve-ai) — 竞品参考  
- [NeuBird 语义监控](https://www.neubird.ai) — 语义图谱方法  
- [Metoro AI SRE](https://metoro.io) — SLO 为先的生产框架  
- [kube-state-metrics](https://github.com/kubernetes/kube-state-metrics) — 集群状态数据源  
- [LangGraph](https://langchain-ai.github.io/langgraph/) — 代理编排器参考  
- [FastMCP](https://github.com/jlowin/fastmcp) — Python MCP 服务器框架  
- [ArgoCD 回滚](https://argo-cd.readthedocs.io/en/stable/user-guide/commands/argocd_app_rollback/) — 带审批门控的修复目标
