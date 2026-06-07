# 基准测试：WebArena 和 OSWorld

> WebArena 测试跨四个自托管应用的 web-agent（网页代理）能力。OSWorld 测试跨 Ubuntu、Windows、macOS 的 desktop-agent（桌面代理）能力。发布时（2023–2024）两者都显示出顶尖代理与人类之间存在巨大差距。差距正在缩小；失败模式未变。

**类型：** 学习  
**语言：** Python（stdlib 标准库）  
**先决条件：** 第14·19阶段（SWE-bench，GAIA）  
**时间：** ~60 分钟

## 学习目标

- 描述 WebArena 的四个自托管应用及为何基于执行的评估重要。  
- 解释为何 OSWorld 使用真实操作系统截图而非辅助功能 API。  
- 说出 OSWorld 的两大主要失败模式：GUI grounding（图形界面定位）和 operational knowledge（操作知识）。  
- 总结 OSWorld-G 和 OSWorld-Human 相较基础基准测试的附加内容。  

## 问题

通用代理可以调用工具。它们能否驱动浏览器完成 20 次点击的购物结算？能否仅用键盘和鼠标配置 Linux 盒子？这些是 WebArena 和 OSWorld 回答的问题。

## 概念

### WebArena（Zhou et al., ICLR 2024）

- 跨四个自托管的 Web 应用 812 个长时程任务：购物网站、论坛、类似 GitLab 的开发工具、企业内容管理系统（CMS）。  
- 附加工具：地图、计算器、便签本。  
- 评估基于执行，通过 gym API 实现——订单是否完成，问题是否关闭，CMS 页面是否更新？  
- 发布时：最佳 GPT-4 代理成功率为 14.41%，人类为 78.24%。  

自托管的框架至关重要——基准测试不受波动影响，因为目标应用版本固定且可复现。

### 扩展

- **VisualWebArena** — 视觉定位任务，成功依赖图像解读（截图作为一级观测）。  
- **TheAgentCompany**（2024年12月） — 增加终端和编码功能，更像真实远程工作环境。  

### OSWorld（Xie et al., NeurIPS 2024）

- 跨 Ubuntu、Windows、macOS 平台 369 个真实计算机任务。  
- 自由形式键盘和鼠标操作真实应用。  
- 1920×1080 分辨率截图作为观测。  
- 发布时：最佳模型成功率 12.24%，人类为 72.36%。  

### 主要失败模式

1. **GUI grounding（图形界面定位）。** 像素到界面元素的映射。模型难以在 1920×1080 分辨率中稳定定位 UI 元素。  
2. **Operational knowledge（操作知识）。** 哪个菜单有设置，哪个快捷键，哪个偏好面板。是人类多年积累的知识尾巴。  

### 后续工作

- **OSWorld-G** — 包含 564 个定位样本与 Jedi 训练集。将定位与规划分解，可分别衡量。  
- **OSWorld-Human** — 手工策划的黄金操作轨迹。显示顶尖代理步骤数比必要多 1.4-2.7 倍（轨迹效率差距）。  

### 重要性

Claude 计算机使用、OpenAI CUA、Gemini 2.5 Computer Use（第21课）均基于 WebArena 和 OSWorld 生成的工作负载进行训练。基准是目标，生产模型是交付的成果。

### 基准测试常见错误

- **仅用截图评估。** OSWorld 基于截图；用 DOM 或辅助功能 API 评估则忽视了定位挑战。  
- **忽略轨迹长度。** 仅按成功率打分忽略了 OSWorld-Human 显示的 1.4-2.7 倍额外步骤的低效率。  
- **自托管应用版本过旧。** WebArena 锁定特定版本，更新时如果不重新策划会破坏可比性。  

## 构建示例

`code/main.py` 实现了一个玩具级网页代理执行环境：

- 一个简易“购物应用”状态机：list_items（列出商品）、add_to_cart（加入购物车）、checkout（结账）。  
- 3 个任务的黄金轨迹。  
- 一个尝试完成每个任务的脚本化代理。  
- 基于执行的评估器（状态检查）和轨迹效率指标（步骤数 vs 黄金轨迹）。  

运行：

```bash
python3 code/main.py
```

输出：每个任务的成功率和轨迹效率，复刻 OSWorld-Human 的方法。

## 使用方法

- **WebArena Verified** 自托管于内部集群，持续评测。  
- **OSWorld** 运行于虚拟机群，针对桌面代理。  
- **计算机使用代理**（第21课）——Claude、OpenAI CUA、Gemini——均在类似工作负载上训练。  
- **你自己的产品流程**——为前20个任务采集黄金轨迹；每周运行代理测试。  

## 交付

`outputs/skill-web-desktop-harness.md` 构建了包含基于执行的评估和轨迹效率度量的网页/桌面代理执行环境。

## 练习

1. 在玩具环境中添加第二个应用（论坛），编写 3 个任务和黄金轨迹。  
2. 为每个任务添加轨迹效率报告。在你的玩具代理中，是 1 倍、2 倍还是 3 倍于黄金轨迹？  
3. 实现一个“干扰”工具——黄金轨迹从不使用的工具。脚本代理会不会尝试用？  
4. 读 OSWorld-G。你如何在自己的评测中区分定位失败和规划失败？  
5. 阅读 WebArena 应用的 README。升级其中一个被锁定的应用版本会出现什么问题？  

## 关键词

| 术语 | 俗称 | 实际含义 |
|------|------|----------|
| WebArena | “网页代理基准” | 跨 4 个自托管应用的 812 个任务；gym 风格评估 |
| VisualWebArena | “视觉 WebArena” | 视觉定位的 WebArena；截图作为观测 |
| OSWorld | “桌面代理基准” | Ubuntu/Windows/macOS 上的 369 个任务 |
| GUI grounding | “像素到元素映射” | 模型在 1920x1080 分辨率定位 UI 元素 |
| Operational knowledge | “操作系统知识” | 哪个菜单、快捷键、偏好面板 |
| OSWorld-G | “定位套件” | 564 个仅定位样本及训练集 |
| OSWorld-Human | “黄金轨迹” | 专家手工操作序列测量效率 |
| Trajectory efficiency | “步骤数相对黄金” | 代理步骤数除以人类最小步骤数 |

## 参考阅读

- [Zhou 等, WebArena (arXiv:2307.13854)](https://arxiv.org/abs/2307.13854) — 四应用网页基准  
- [Xie 等, OSWorld (arXiv:2404.07972)](https://arxiv.org/abs/2404.07972) — 跨操作系统桌面基准  
- [Anthropic, Introducing computer use](https://www.anthropic.com/news/3-5-models-and-computer-use) — Claude 基准塑造的能力  
- [OpenAI, Computer-Using Agent](https://openai.com/index/computer-using-agent/) — OSWorld 和 WebArena 数据
