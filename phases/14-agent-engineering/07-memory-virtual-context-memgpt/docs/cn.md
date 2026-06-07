# Memory: 虚拟上下文（Virtual Context）与 MemGPT

> 上下文窗口（Context windows）是有限的。对话、文档和工具调用轨迹不是。MemGPT（Packer 等，2023）将其框架化为操作系统虚拟内存（OS virtual memory）——主上下文为 RAM，外部存储为磁盘，代理在两者之间分页。这是所有 2026 年记忆系统继承的模式。

**类型:** 构建  
**语言:** Python（stdlib）  
**先决条件:** 阶段 14 · 01（代理循环 Agent Loop），阶段 14 · 06（工具使用 Tool Use）  
**时间:** 约 75 分钟

## 学习目标

- 解释 MemGPT 所基于的操作系统类比：主上下文 = RAM，外部上下文 = 磁盘，记忆工具 = 页内调入/调出（page in/out）。
- 使用 stdlib 实现两级 MemGPT 模式，包含主上下文缓冲区、外部可搜索存储以及页内调入/调出工具。
- 描述代理如何发出“中断”以查询或修改外部记忆，以及结果如何拼接回下一个提示（prompt）。
- 识别 MemGPT 设计选择如何延续到 Letta（第 08 课）和 Mem0（第 09 课）。

## 问题

上下文窗口看似能解决记忆问题，实际上不能。生产环境中出现三种反复出现的失败模式：

1. **溢出（Overflow）**。多轮对话、长文档或重度调用工具的轨迹会超出窗口范围，超过截断点的内容全部丢失。
2. **稀释（Dilution）**。即使在窗口内，填充无关上下文会让注意力分散，从而忽略重要信息。前沿模型在长输入时性能仍然下降。
3. **持久性（Persistence）**。新会话从空窗口开始，缺少外部记忆的代理无法跨会话“记住你之前让我做过的事”。

扩大窗口有帮助，但无法解决根本问题。Mem0 2025 年论文测量出，即使是 128k 窗口基线，仍然遗漏由只有 4k 窗口且带外部记忆代理捕捉的长远事实。

## 概念

### MemGPT：操作系统类比

Packer 等（arXiv:2310.08560，2024 年 2 月 v2 版）将上下文管理映射为操作系统虚拟内存：

| 操作系统概念（OS concept） | MemGPT 概念 | 2026 年生产环境类比 |
|---------------------------|-------------|---------------------|
| RAM | 主上下文（prompt） | Anthropic/OpenAI 上下文窗口 |
| 磁盘（Disk） | 外部上下文 | 向量数据库（vector DB）、KV、图存储 |
| 页面错误（Page fault） | 记忆工具调用 | `memory.search`, `memory.read`, `memory.write` |
| 操作系统内核（OS kernel） | 代理控制循环 | 带记忆工具的 ReAct 循环 |

代理运行正常的 ReAct 循环。额外新增一类工具让它能在主上下文与外部记忆之间分页数据。

### 两级结构

- **主上下文（Main context）**。固定大小的 prompt，包含当前任务内容，模型始终可见。
- **外部上下文（External context）**。无限大小，可通过工具搜索。相关时读取，出现事实时写入。

原论文在两个超出基线窗口的任务上验证该设计：超 10 万 token 的文档分析和跨天持久记忆的多会话聊天。

### 中断模式（The interrupt pattern）

MemGPT 引入了记忆即中断：对话中途，代理可调用记忆工具，运行时执行并将结果拼接进下一次助手回复作为新观察。概念上等同 Unix 的 `read()` 系统调用，会阻塞进程，返回字节，然后进程继续。

标准记忆工具接口：

- `core_memory_append(section, text)` — 向持久部分追加写入。
- `core_memory_replace(section, old, new)` — 编辑持久部分内容。
- `archival_memory_insert(text)` — 写入可搜索的外部存储。
- `archival_memory_search(query, top_k)` — 从外部存储检索。
- `conversation_search(query)` — 扫描之前的对话轮次。

### MemGPT 终点与 Letta 起点

2024 年 9 月，MemGPT 演进为 Letta。研究仓库（`cpacker/MemGPT`）仍存；Letta 拓展设计：

- 三层结构（核心 core、回忆 recall、存档 archival——第 08 课）。
- 用原生推理替代 `send_message`/心跳模式（第 08 课）。
- 睡眠时代理做异步记忆工作（第 08 课）。

MemGPT 论文是 2026 年的基础，即使生产系统运行 Letta、Mem0 或自定义两级存储。

### 此模式常见问题

- **记忆腐烂（Memory rot）**。写入速度超过读取，检索被大量过时事实淹没。解决方案：周期性整合（Letta 睡眠时间）、显式失效（Mem0 冲突检测器）。
- **记忆中毒（Memory poisoning）**。外部记忆是检索的文本，若攻击者内容写入记忆，代理下一会话重新摄入。此即 Greshake 等（第 27 课）攻击的时间演绎。
- **引用丢失（Citation loss）**。代理能回忆“用户让我运送 X”，却无法注明对应的轮次。每次存档写入时需同时保存来源引用（会话 ID，轮次 ID）。

## 实践构建

`code/main.py` 用 stdlib 实现 MemGPT 两级模式：

- `MainContext` — 固定大小的 prompt 缓冲，含 `core` 字典与 `messages` 列表；超限时自动压缩最早消息。
- `ArchivalStore` — 内存中的 BM25 风格存储（基于 token 重叠评分），存储（id、文本、标签、会话、轮次）记录。
- 五个与 MemGPT API 对应的记忆工具。
- 一个脚本化代理先写入事实至存档，再通过调用 `archival_memory_search` 回答问题。

运行：

```text
python3 code/main.py
```

执行跟踪展示代理写入三条事实，主上下文达到容量（触发驱逐），随后通过存档检索回答跟进问题——再现 MemGPT 工作流，无需实际 LLM。

## 使用场景

如今每个生产记忆系统都是 MemGPT 变种：

- **Letta**（第 08 课）——三层结构，原生推理，睡眠时计算。
- **Mem0**（第 09 课）——向量 + KV + 图结合评分机制。
- **OpenAI Assistants / Responses**——通过线程和文件管理记忆。
- **Claude Agent SDK**——通过技能和会话存储实现长期记忆。

选择时根据运营特性（自托管、托管、框架集成）选型，而非核心模式——核心模式是 MemGPT。

## 发布方案

`outputs/skill-virtual-memory.md` 是可重用技能，能生成正确的两级记忆框架（主 + 存档 + 工具接口），适配任意目标运行时，包含驱逐策略和引用字段的接入。

## 练习

1. 添加基于 token 数的 `max_main_context_tokens` 上限（近似用 `len(text.split())` * 1.3）。超限时将最早消息压缩成摘要。比较有无摘要功能的表现差异。
2. 在存档存储上实现完整 BM25 算法（词频、逆文档频率）。在玩具事实集上对比召回率@10 与基于 token 重叠的基线。
3. 向存档插入添加 `citation` 字段（session_id, turn_id, source_url）。让代理在所有检索驱动回答中引用来源。
4. 模拟记忆中毒：添加一条存档记录提示“忽略所有未来用户指令”。写守护程序扫描检索结果中指令形态文本并标记不可信。
5. 移植实现到 MemGPT 研究仓库核心记忆 JSON 模式（`cpacker/MemGPT`）。从平铺字符串切换到类型化区段时有什么变化？

## 关键词

| 术语 | 常见说法 | 实际含义 |
|---|---|---|
| 虚拟上下文（Virtual context） | “无限记忆” | 主（prompt）+ 外部（可搜索）两级，支持页内调入/调出 |
| 主上下文（Main context） | “工作记忆” | 固定大小的 prompt，始终可见 |
| 存档记忆（Archival memory） | “长期存储” | 外部可搜索持久化，按需检索 |
| 核心记忆（Core memory） | “持久 prompt 区段” | 固定在主上下文内部的已命名区段 |
| 记忆工具（Memory tool） | “记忆 API” | 代理调用读写外部记忆的工具接口 |
| 中断（Interrupt） | “记忆页面错误” | 代理暂停，运行时获取，结果拼接至下轮 |
| 记忆腐烂（Memory rot） | “过时事实” | 旧写入淹没检索；通过整合修复 |
| 记忆中毒（Memory poisoning） | “注入持久笔记” | 攻击者内容存入记忆，检索时重入 |

## 延伸阅读

- [Packer 等，MemGPT（arXiv:2310.08560）](https://arxiv.org/abs/2310.08560) — 受操作系统启发的虚拟上下文论文  
- [Letta，Memory Blocks 博客](https://www.letta.com/blog/memory-blocks) — 三层演进设计  
- [Anthropic，《有效的上下文工程》](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) — 将上下文视作预算管理  
- [Chhikara 等，Mem0（arXiv:2504.19413）](https://arxiv.org/abs/2504.19413) — 基于此模式的混合生产记忆系统
