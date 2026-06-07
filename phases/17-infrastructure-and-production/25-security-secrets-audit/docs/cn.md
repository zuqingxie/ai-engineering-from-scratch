# 安全 — 密钥管理、API 密钥轮换、审计日志、防护措施

> 通过集中式 vault（HashiCorp Vault、AWS Secrets Manager、Azure Key Vault）消除密钥泛滥。绝不在配置文件、版本控制中的环境文件或电子表格中存储凭证。优先使用 IAM 角色而非静态密钥；CI/CD 使用 OIDC。AI 网关模式是 2026 年解决方案：应用 → 网关 → 模型提供者，网关运行时从 vault 拉取凭证。仅需在 vault 中轮换，所有应用几分钟内自动更新 —— 无需重新部署，无需在 Slack 上询问“谁有新密钥”。轮换策略不超过 90 天；每次提交用 TruffleHog / GitGuardian / Gitleaks 扫描。零信任：多因素认证（MFA）、单点登录（SSO）、基于角色/属性的访问控制（RBAC/ABAC）、短效令牌、设备安全状态。PII 清理使用实体识别（entity recognition）在转发前遮蔽 PHI/PII；一致性标记化（Mesh 方法）将敏感值映射到稳定占位符，确保 LLM 保持代码/关系语义。网络出口：LLM 服务部署在独立 VPC/VNet 子网，只允许 `api.openai.com`、`api.anthropic.com` 等白名单访问；阻止所有其他出站流量。2026 年事件主因：Vercel 通过被攻破的 CI/CD 凭证的供应链攻击，横向窃取了数千个客户部署的环境变量。

**类型：** 学习  
**语言：** Python（标准库，简单 PII 清理器 + 审计日志写入器）  
**先修：** 阶段 17 · 19（AI 网关）、阶段 17 · 13（可观测性）  
**时长：** ~60 分钟

## 学习目标

- 列举密钥管理的四大反模式（版本控制中的配置文件、硬编码环境变量、电子表格、静态密钥）及其替代方案。
- 说明 AI 网关从 vault 拉取密钥的模式，作为 2026 年生产标准。
- 实现带一致性标记化（相同值映射为相同占位符）的 PII 清理器，保持语义完整。
- 讲述 2026 年 Vercel 供应链事件及其对 CI/CD 凭证管理的启示。

## 问题

一位实习生提交了含 API 密钥的 `.env` 文件，又迅速删除。密钥已在 git 历史中 —— GitGuardian 扫描发现，您的轮换流程是“发 Slack 消息给团队，更新 40 个配置文件，重新部署所有服务”。8 小时后，半数服务在线，另一半等待部署窗口。

此外，用户提示包含 “我的社会保障号是 123-45-6789”。提示发给 OpenAI。尽管有业务伙伴协议（BAA），公司内部政策要求转发前屏蔽 PII，但你没做到。

另外，您的 EKS 集群上的 LLM Pod 能访问所有互联网主机。有人通过 DNS 查询向攻击者控制的域名导出数据，毫无阻拦。

LLM 服务安全必须覆盖这三条攻击路径：vault 支持的凭证管理，PII 清理，网络出口过滤，及审计日志。

## 概念

### 集中式 vault + IAM 角色拉取

**Vault**：HashiCorp Vault、AWS Secrets Manager、Azure Key Vault、GCP Secret Manager。唯一可信来源。

**IAM 角色**：应用/网关以其 IAM 身份验证，而非静态密钥。vault 返回令牌生命周期内的秘密。

**AI 网关模式**：网关在请求时从 vault 拉取 `OPENAI_API_KEY`。vault 中轮换密钥，下一个请求即获取新密钥，无需重新部署。

### 轮换策略 ≤ 90 天

所有 API 密钥、vault 根令牌、CI/CD 凭证。可自动轮换时启用，人工轮换需记录和追踪。

### 密钥扫描

- **TruffleHog** — 正则+熵检测提交。
- **GitGuardian** — 商用，高准确率。
- **Gitleaks** — 开源，CI 中运行。

每次提交运行，发现新密钥阻止 PR 合入。

### 零信任防护

- 所有账户均需多因素认证（MFA）。
- 通过 SAML/OIDC 实现单点登录（SSO）。
- 基于角色（RBAC）或属性（ABAC）细粒度访问控制。
- 短有效期令牌（小时级别，而非天）。
- 设备安全状态检查——仅允许企业设备且磁盘加密。

### PII/PHI 清理

提示离开您的基础设施前：

1. 实体识别（spaCy NER、Presidio、商业产品）。
2. 遮蔽匹配实体： `"My SSN is 123-45-6789"` → `"My SSN is [SSN_TOKEN_A3F]"`。
3. 一致性标记化（Mesh 方法）：相同值映射至相同占位符，保持 LLM 关系语义。
4. 可选的反向映射，恢复真实值到 LLM 响应。

使用静态正则过滤基础模式，NER 捕获更多。两者结合。

### 输入和输出防护措施

输入：阻断已知越狱提示、禁用话题；对用户限流。

输出：基于正则清理泄露密钥（API 密钥模式、拒绝场景下的邮箱模式），使用分类器检测策略违规。

### 网络出口白名单

LLM 服务部署于专用子网：  
- 白名单：`api.openai.com`、`api.anthropic.com`、向量数据库终端、vault 终端。  
- 其余全部丢弃。  
- DNS 使用仅允许名单解析器，避免 DNS 隧道渗透。

### 审计日志

不可篡改的每次 LLM 调用日志，包括：  
- 时间戳。  
- 用户 / 租户。  
- 提示哈希（不存储原始提示保护隐私）。  
- 模型及版本。  
- 令牌计数。  
- 成本。  
- 响应哈希。  
- 任何防护触发记录。

按照合规要求保留（SOC 2 1 年，HIPAA 6 年）。

### 2026 年 Vercel 事件

供应链攻击：CI/CD 凭证被攻破后泄露了数千个客户环境变量。教训是：CI/CD 凭证等同生产权限，必须存 vault，权限最小化，积极轮换。

### 关键数字

- 轮换策略：≤ 90 天。  
- 每次提交扫描：TruffleHog / GitGuardian / Gitleaks。  
- Vercel 2026 事件：CI/CD 凭证被泄露，成千上万客户环境变量外泄。  
- 审计日志保留：SOC 2 = 1 年，HIPAA = 6 年。

## 练手项目

`code/main.py` 实现了一个简易的 PII 清理器，带一致性标记化和追加式审计日志。

## 交付物

本节产出 `outputs/skill-llm-security-plan.md`。结合合规和当前状态，规划 vault 迁移、清理器、出口、安全日志方案。

## 练习题

1. 运行 `code/main.py`，发送两个包含相同 SSN 的提示，确认它们映射至同一占位符。
2. 为在 EKS 上运行的 vLLM 设计网络出口策略，调用 OpenAI、Anthropic 和 Weaviate。
3. 发现了一个两年前的密钥，正确的处理是轮换、清理历史还是两者都做？请说明理由。
4. 审计日志每天增长 10 GB，设计分层保留策略（热数据 30 天，温数据 12 个月，冷数据 6 年）。
5. 分析反向标记化（将真实值恢复至 LLM 响应）相较于保持占位符可见是否值得复杂度。

## 关键词汇

| 术语                 | 俗称                         | 实际含义                         |
|----------------------|------------------------------|----------------------------------|
| Vault                | “秘密存储”                  | 集中凭证管理服务                 |
| IAM 角色             | “基于身份验证”              | 应用假设的角色，返回短效凭证     |
| CI/CD 的 OIDC        | “云端发行的令牌”            | CI 里无静态密钥，通过 OIDC 验证身份 |
| TruffleHog / GitGuardian / Gitleaks | “密钥扫描器”       | 提交时的密钥检测                 |
| RBAC / ABAC          | “访问控制”                  | 基于角色或基于属性的访问控制     |
| PII 清理             | “数据遮蔽”                  | 删除或标记敏感实体               |
| 一致性标记化         | “稳定占位符”                | 相同值每次映射至相同令牌         |
| Mesh 方法            | “Mesh 标记化”               | 保持语义的标记化模式             |
| 出口白名单           | “出站允许名单”              | 仅允许指定域名访问               |
| 审计日志             | “不可篡改历史”              | 用于合规的追加式记录             |

## 拓展阅读

- [Doppler — Advanced LLM Security](https://www.doppler.com/blog/advanced-llm-security)  
- [Portkey — Manage LLM API keys with secret references](https://portkey.ai/blog/secret-references-ai-api-key-management/)  
- [Datadog — LLM Guardrails Best Practices](https://www.datadoghq.com/blog/llm-guardrails-best-practices/)  
- [JumpServer — Secrets Management Best Practices 2026](https://www.jumpserver.com/blog/secret-management-best-practices-2026)  
- [Microsoft Presidio](https://github.com/microsoft/presidio) — PII 检测与匿名化。  
- [HashiCorp Vault 文档](https://developer.hashicorp.com/vault/docs)
