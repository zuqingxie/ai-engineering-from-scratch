# 毕业设计 12 — 视频理解流水线（场景，问答，搜索）

> Twelve Labs 将 Marengo + Pegasus 商业化。VideoDB 发布了视频的 CRUD API。AI2 的 Molmo 2 公开了 VLM 检查点。Gemini 长上下文本地支持数小时视频。TimeLens-100K 实现了大规模的时间定位。2026 年的视频流水线已经确定：场景切分、每场景字幕与嵌入、字幕对齐、多向量索引，查询返回带（起始，结束）时间戳及帧预览的答案。毕业设计目标是处理 100 小时视频，达到公开基准，并测量计数与动作问题的幻觉率。

**类型:** 毕业设计  
**语言:** Python（流水线），TypeScript（UI）  
**先决条件:** 第4阶段（计算机视觉），第6阶段（语音），第7阶段（Transformer 架构），第11阶段（大语言模型工程），第12阶段（多模态），第17阶段（基础设施）  
**涉及阶段:** P4 · P6 · P7 · P11 · P12 · P17  
**时间:** 30 小时

## 问题

长视频问答是 2026 规模下最带宽密集的多模态问题。Gemini 2.5 Pro 可本地原生读取 2 小时视频，但将 100 小时视频导入可查询语料库仍需场景级索引。生产形态结合了场景切分（TransNetV2 或 PySceneDetect）、每场景通过 VLM（Gemini 2.5、Qwen3-VL-Max 或 Molmo 2）生成字幕、字幕对齐（Whisper-v3-turbo 带单词时间戳）及存储字幕、关键帧和转录文本的多向量索引。查询流水线返回带时间戳和帧预览的答案。

基准测试包括公开的 ActivityNet-QA、NeXT-GQA 及自定义 100 查询集合。计数和动作类问题的幻觉是已知难点，本项目明确对其进行测量。

## 概念

摄取时三条流水线并行运行。**场景切分**将视频划分为多个场景。**VLM 字幕生成**为每个场景生成字幕和关键帧的帧嵌入。**自动语音识别（ASR）对齐**输出词级时间戳。三条流以（scene_id，时间范围）连接。每个场景在多向量索引（Qdrant）中存储三类向量：字幕嵌入、关键帧嵌入、转录嵌入。

查询时，自然语言问题对三类向量均执行密集检索；结果通过逆序排名融合（RRF）合并；时间定位适配器（TimeLens 风格）在最高相关场景内精细调整（开始，结束）时间窗。VLM 合成器（Gemini 2.5 Pro 或 Qwen3-VL-Max）基于查询+顶级场景+裁剪帧生成带引用时间戳和帧预览的答案。

幻觉率测量尤其重要。计数问题（“有多少人进入房间？”）和动作问题（“厨师是在搅拌之前倒料吗？”）是公认的不可靠类别。准确率分开报告。

## 架构

```text
视频文件 / URL
      |
      v
PySceneDetect / TransNetV2  （场景切分）
      |
      +--- 每场景关键帧 --- VLM 字幕 + 帧嵌入
      |                            （Gemini 2.5 Pro / Qwen3-VL-Max / Molmo 2）
      |
      +--- 音频通道 --- Whisper-v3-turbo ASR + 词时间戳
      |
      v
多向量索引 Qdrant：{caption_emb, keyframe_emb, transcript_emb}
      |
查询：
  对三向量执行密集查询 -> 逆序排名融合（RRF）-> top-k 场景
      |
      v
TimeLens / VideoITG 时间定位（在场景内精细调整起止时间）
      |
      v
VLM 合成：查询 + 顶级场景 + 帧预览
      |
      v
答案 + （起始，结束）时间戳 + 帧缩略图 + 引用
```

## 技术栈

- 场景切分：TransNetV2（2024-26 先进技术）或 PySceneDetect
- ASR：使用 faster-whisper 的 Whisper-v3-turbo，带词时间戳
- VLM 字幕生成与回答：Gemini 2.5 Pro、Qwen3-VL-Max 或 Molmo 2
- 时间定位：TimeLens-100K 训练的适配器或 VideoITG
- 索引：支持多向量的 Qdrant（字幕 / 帧 / 转录）
- UI：Next.js 15 + HTML5 视频播放器和场景缩略图条
- 评测：ActivityNet-QA、NeXT-GQA，自定义 100 题手工标注集
- 幻觉基准：计数和动作类问题的手工标注子集

## 构建步骤

1. **摄取管理器。** 支持 YouTube URL 或本地 MP4。必要时降采样到 720p。持久化 `{video_id, file_path}`。

2. **场景切分。** 使用 TransNetV2 或 PySceneDetect 生成 `[{scene_id, start_ms, end_ms, keyframe_path}]`。目标 100 小时约 6k-8k 场景。

3. **ASR 处理。** 使用 Whisper-v3-turbo 处理音频，导出词级时间戳，划分为场景转录片段。

4. **VLM 字幕生成。** 对每个场景，调用 Gemini 2.5 Pro（或 Qwen3-VL-Max）以关键帧和简短模板生成字幕及帧嵌入。

5. **多向量索引。** 创建 Qdrant 集合，含三类命名向量。负载包含 `{video_id, scene_id, start_ms, end_ms, keyframe_url}`。

6. **查询。** 自然语言查询针对三种向量执行密集检索；用逆序排名融合合并；top-k=5 场景。

7. **时间定位。** 针对 top 场景使用 TimeLens 风格适配器精细调整场景内部的（起始，结束）时间窗。

8. **VLM 合成。** 调用 Gemini 2.5 Pro 输入查询+前三场景剪辑（图像或短片）+转录文本。输出需包含 `(video_id, start_ms, end_ms)` 引用。

9. **评测。** 运行 ActivityNet-QA 和 NeXT-GQA，构建 100 查询自定义集。报告整体准确率及分分类统计（计数、动作、描述类）。

## 使用示例

```text
$ video-qa ask --url=https://youtube.com/watch?v=X "how many cars pass the intersection in the first minute?"
[scene]    识别出 23 个场景
[asr]      转录完成，共 4分12秒
[index]    写入 69 个向量（23 场景 x 3）
[query]    最高场景：场景 3 [01:32-01:54], 置信度 0.84
[ground]   精细时间窗：[00:12-00:58]
[synth]    gemini 2.5 pro，用时 1.4 秒
答案:      在 00:12 到 00:58 之间有 5 辆车通过路口。
引用:      [场景 3: 00:12-00:58]
           [帧预览时间：00:14, 00:27, 00:44, 00:51, 00:57]
```

## 交付内容

`outputs/skill-video-qa.md` 是交付文件。给定 YouTube URL 或上传视频，流水线进行场景索引，并回答带时间戳引用的问题。

| 权重 | 评估标准 | 计量方法 |
|:-:|---|---|
| 25 | 时间定位 IoU | Intersection-over-union on a held-out temporal localization set |
| 20 | 问答准确率 | NeXT-GQA plus a custom 100-question set |
| 20 | 摄取吞吐量 | Processed video hours per dollar |
| 20 | UI 和引用体验 | Timestamp links, thumbnail strip, and jump-to-frame behavior |
| 15 | 幻觉率 | Separate count accuracy and action-class accuracy statistics |
| **100** | | |

## 练习任务

1. 用 Qwen3-VL-Max 替换 Gemini 2.5 Pro 进行字幕生成；在 50 个场景人工打分样本上报告字幕质量差异。

2. 将每场景帧嵌入降至单一池化向量，测量检索性能回退。

3. 构建“计数严格”模式：合成器提取每个计数实例及时间戳，要求用户点击验证。评估用户验证是否降低幻觉。

4. 评测摄取成本：基于三款 VLM 测量每美元处理的视频小时数，选出最佳平衡点。

5. 增加说话人分离转录：对音频运行 pyannote 说话人分离，嵌入每个说话人文本，实现“Alice 关于 X 说了什么？”类查询。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------|----------|
| Scene segmentation | “镜头检测” | 按镜头边界切分视频为场景 |
| Multi-vector index | “字幕 + 帧 + 转录” | Qdrant 集合，按表示类型命名向量 |
| Temporal grounding | “事件精确时间” | 精细定位查询答案的（开始，结束）时间窗 |
| Frame embedding | “视觉表示” | 关键帧的向量嵌入，用于场景视觉相似度 |
| RRF fusion | “逆序排名融合” | 经典的多排名列表合并策略 |
| Counting hallucination | “计数错误” | VLM 在“多少个X”问题上的已知失败模式 |
| ActivityNet-QA | “视频问答基准” | 长视频问答准确率评测基准 |

## 延伸阅读

- [AI2 Molmo 2](https://allenai.org/blog/molmo2) — 公开的 VLM 检查点  
- [TimeLens (CVPR 2026)](https://github.com/TencentARC/TimeLens) — 大规模时间定位  
- [Gemini Video long-context](https://deepmind.google/technologies/gemini) — 官方参考实现  
- [VideoDB](https://videodb.io) — 视频 CRUD API 参考  
- [Twelve Labs Marengo + Pegasus](https://www.twelvelabs.io) — 商用参考  
- [TransNetV2](https://github.com/soCzech/TransNetV2) — 场景切分模型  
- [PySceneDetect](https://github.com/Breakthrough/PySceneDetect) — 经典开源替代方案  
- [ActivityNet-QA](https://arxiv.org/abs/1906.02467) — 评测基准参考
