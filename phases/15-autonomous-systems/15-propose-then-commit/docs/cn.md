# Human-in-the-Loop: 提议-然后-提交（Propose-Then-Commit）

> 2026年对HITL（Human-in-the-Loop，人工参与环节）的共识非常具体，并非“代理请求，用户点击同意”。而是提议-然后-提交：将提议的操作持久化存储，带有幂等键；将其展示给审阅者，附带操作意图、数据溯源、权限影响范围、影响半径及回滚计划；只在正面确认后提交；执行后验证以确认副作用确实发生。LangGraph的`interrupt()`配合PostgreSQL检查点、微软Agent Framework的`RequestInfoEvent`以及Cloudflare的`waitForApproval()`均实现了相同的模式。典型的失败模式是橡皮图章式批准：“同意？”点击却没有审阅。已记录的缓解措施是带有明确核对清单的质询与响应机制。

**类型：** 学习  
**语言：** Python（标准库，带幂等性的提议-提交状态机）  
**先修课程：** 第15阶段·12（持久执行），第15阶段·14（触发器）  
**时间：** 约60分钟

## 问题

代理执行操作，用户必须决定是否批准。如果决策是即时的，通常不是审阅；如果决策是结构化的，虽然缓慢但可信。工程问题是如何让结构化审阅成为最优路径。

2023年时期的HITL模式是同步提示：“代理想向X发送邮件，内容为Y——批准吗？”用户点击批准，系统看似安全。实际中该界面被大量橡皮图章式批准：用户快速批准，批准行为难以预测，代理出错时审计轨迹显示用户记不清楚的漫长批准历史。

2026年模式——提议-然后-提交——让HITL基于持久化载体，附加结构化元数据，要求正面提交。每个受管代理SDK都有对应版本：LangGraph的`interrupt()`、微软Agent Framework的`RequestInfoEvent`、Cloudflare的`waitForApproval()`。API名不同，形态一致。

## 概念

### 提议-然后-提交状态机

1. **提议（Propose）。**代理生成一个提议操作，持久化存储（PostgreSQL、Redis、Durable Object）中包含：
   - 意图（为什么代理这样做）
   - 数据溯源（哪个源导致该提议）
   - 涉及权限（涉及哪些作用域/文件/端点）
   - 影响半径（最坏情况是什么）
   - 回滚计划（若提交了，如何撤销）
   - 幂等键（每个提议唯一；重复提交返回相同记录）
2. **展示（Surface）。**审阅者查看带元数据的提议，审阅者为人类（非代理自审）。
3. **提交（Commit）。**正面确认，执行操作。
4. **验证（Verify）。**执行后读取并确认副作用。如果验证失败，系统处于已知异常状态，启动告警。

### 幂等键

没用幂等键时，瞬时失败的重试可能导致获批的操作被执行两次。真实场景示例：用户批准“从A转账100美元到B”，网络抖动，工作流重试。用户只批准一次，转账却执行两次。幂等键绑定批准和唯一的副作用；第二次执行无效。

这是Stripe和AWS API常用的幂等模式。微软Agent Framework文档明确指出代理批准应复用。

### 持久化：为什么批准状态超过程存在

等待批准的状态由代理不拥有，工作流暂停（第12课）。批准抵达时，工作流精确从暂停点恢复。这就是为何LangGraph将`interrupt()`与PostgreSQL检查点结合，而不是单纯内存状态——两天后的批准依然能找回工作流状态。

### 橡皮图章式批准与质询-响应缓解措施

默认HITL界面（“批准”/“拒绝”按钮）产生快速批准，无真实审阅。缓解措施是质询-响应核对清单，必须正面回答指定问题方可启用批准按钮。具体形式：

- “你理解此操作涉及的资源吗？[ ]”
- “你已验证影响半径可接受吗？[ ]”
- “如果失败你有回滚计划吗？[ ]”

非为官僚主义，而是强制机制。审阅人不能勾选所有框则请求澄清（升级）或拒绝（安全默认）。Anthropic代理安全研究明确指出核对清单驱动的HITL能缓解橡皮图章批准。

### 什么是关键操作

非所有操作都需提议-然后-提交，2026年指导如下：

- **关键操作**（始终HITL）：不可逆写入、财务交易、外发通信、生产数据库变更、破坏性文件系统操作。
- **可逆操作**（有时HITL）：本地文件编辑、预发布环境变更、带明确回滚的可逆写入。
- **读取与检查**（从不HITL）：文件读取、资源列表、只读API调用。

### 操作后验证

“提交成功”≠“副作用发生”。网络分区和竞态条件可能导致工作流误判成功，但后端未持久化。验证步骤提交后重新读取目标资源确认。这类似数据库事务中的`RETURNING`语句或AWS的`GetObject`确认`PutObject`。

### 欧盟AI法案第14条

第14条规定高风险AI系统须有效人工监督，“有效”非装饰。法规明确排除橡皮图章式模式。提议-然后-提交加质询-响应是微软Agent治理工具合规文档中通过第14条审查的形态。

## 使用方法

`code/main.py`用Python标准库实现了提议-提交状态机，持久存储为JSON文件，幂等键为(thread_id, action_signature)的哈希。驱动模拟了：清洁批准流程、瞬时失败后重试（不二次执行）、橡皮图章默认与质询-响应流程。

## 交付

`outputs/skill-hitl-design.md`审查提议-提交HITL工作流，标记缺失的元数据、幂等性、验证或质询响应层。

## 练习题

1. 运行`code/main.py`。确认获批提议的重试使用持久记录且不重复执行。修改幂等键包含时间戳，证明重试造成双重执行。

2. 在提议记录增加`rollback`字段。模拟验证失败执行，展示自动触发回滚。

3. 阅读微软Agent Framework的`RequestInfoEvent`文档，找出玩具引擎缺少的元数据字段。添加它并解释该字段防护何种风险。

4. 为具体操作（如“发推公开账号”）设计质询-响应核对清单。审阅者需回答的3个问题是什么？为何是这三个？

5. 选一个场景，说明同步“批准？”提示即可（无持久存储），解释原因及接受的风险类别。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|---|---|---|
| Propose-then-commit | “两阶段批准” | 持久化提议 + 正面提交 + 验证 |
| 幂等键（Idempotency key） | “重试安全令牌” | 每个提议唯一；第二次执行无操作 |
| 数据溯源（Data lineage） | “来源” | 导致该提议的具体源内容 |
| 影响半径（Blast radius） | “最坏情况” | 操作错误导致的影响范围 |
| 橡皮图章（Rubber-stamp） | “快速批准” | 未真实审阅即点击批准 |
| 质询-响应（Challenge-and-response） | “强制核对清单” | 审阅者必须对特定问题给出肯定答复 |
| RequestInfoEvent | “微软Agent Framework原语” | 带结构化元数据的持久HITL请求 |
| `interrupt()` / `waitForApproval()` | “框架原语” | LangGraph 和 Cloudflare 同类型实现 |

## 延伸阅读

- [Microsoft Agent Framework — 人工参与环节](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) — `RequestInfoEvent`，持久批准。
- [Cloudflare Agents — 人工参与环节](https://developers.cloudflare.com/agents/concepts/human-in-the-loop/) — `waitForApproval()` 和 Durable Objects。
- [Anthropic — 实践中衡量代理自主性](https://www.anthropic.com/research/measuring-agent-autonomy) — HITL作为长时风险的缓解措施。
- [欧盟AI法案 — 第14条：人工监督](https://artificialintelligenceact.eu/article/14/) — 高风险系统的监管基线。
- [Anthropic — Claude的宪章（2026年1月）](https://www.anthropic.com/news/claudes-constitution) — 监督的宪政框架。
