# 3D 生成

> 3D 是 2D 到 3D 利用率最高的模态。2023 年的突破是 3D Gaussian Splatting（3D 高斯散点）。2024-2026 年生成推进是在多视图扩散 + 3D 重建基础上，结合单一提示词或照片生成物体和场景。

**类型：** 学习  
**语言：** Python  
**先决条件：** 第 4 阶段（视觉），第 8 · 07 课（潜空间扩散）  
**时间：** 约 45 分钟

## 问题

3D 内容构建很痛苦：

- **表示。** 网格（mesh）、点云（point clouds）、体素网格（voxel grids）、带符号距离场（SDF），神经辐射场（NeRF）、3D 高斯，每种都有权衡。
- **数据稀缺。** ImageNet 有 1400 万张图。最大规模的干净 3D 数据集（Objaverse-XL，2023）大约有 1000 万个对象，但质量多数较低。
- **内存。** 512³ 体素网格有 1.28 亿体素；一个有用的场景 NeRF 需要每条射线 100 万采样点。生成比重建更难。
- **监督。** 2D 图像有像素值；3D 通常只有几个 2D 视角，需要升维到 3D。

2026 技术栈将两者拆开。第一步，用扩散模型生成*2D 多视图图像*。第二步，拟合一个*3D 表示*（通常是 Gaussian splatting）到这些图像上。

## 概念

![3D 生成：多视图扩散 + 3D 重建](../assets/3d-generation.svg)

### 表示：3D Gaussian Splatting（Kerbl 等，2023）

用大约 100 万个 3D 高斯体（Gaussian）表示场景。每个高斯体有 59 个参数：位置（3 个）、协方差（6 个，或四元数 4 个 + 尺度 3 个）、不透明度（1 个）、球谐颜色（3 次球谐有 48 个，0 次有 3 个）。

渲染 = 投影 + alpha 组合。速度快（4090 显卡 1080p 可达 100 帧/秒）。可微分。通过梯度下降拟合真实照片。一个场景用普通消费 GPU 5-30 分钟完成拟合。

2023-2024 两大创新：  
- **生成式高斯点云。** 如 LGM、LRM、InstantMesh 直接从一张或几张图像预测高斯云。  
- **4D 高斯散点。** 针对动态图场景，每帧带偏移的高斯。

### 多视图扩散

在预训练图像扩散模型基础上微调，生成同一物体的多个视角，提示来自文字或单图。代表工作：Zero123（Liu 等，2023）、MVDream（Shi 等，2023）、SV3D（Stability，2024）、CAT3D（Google，2024）。通常输出物体周围 4-16 视角，用高斯散点或 NeRF 升维到 3D。

### 文本到 3D 流水线

| 模型 | 输入 | 输出 | 时间 |
|-------|-------|-------|-------|
| DreamFusion (2022) | 文字 | 通过 SDS 获得 NeRF | 约 1 小时每资产 |
| Magic3D | 文字 | 网格 + 纹理 | 约 40 分钟 |
| Shap-E（OpenAI，2023） | 文字 | 隐式 3D | 约 1 分钟 |
| SJC / ProlificDreamer | 文字 | NeRF / 网格 | 约 30 分钟 |
| LRM（Meta，2023） | 图像 | 三平面 | 约 5 秒 |
| InstantMesh（2024） | 图像 | 网格 | 约 10 秒 |
| SV3D（Stability，2024） | 图像 | 新视角 | 约 2 分钟 |
| CAT3D（Google，2024） | 1-64 张图 | 3D NeRF | 约 1 分钟 |
| TripoSR（2024） | 图像 | 网格 | 约 1 秒 |
| Meshy 4（2025） | 文字 + 图像 | PBR 网格 | 约 30 秒 |
| Rodin Gen-1.5（2025） | 文字 + 图像 | PBR 网格 | 约 60 秒 |
| 腾讯混元3D 2.0（2025） | 图像 | 网格 | 约 30 秒 |

2025-2026 方向：直接生成带 PBR 材质的文本到网格模型，适用于游戏引擎。多视图扩散作为中间步骤依旧是普遍物体表现最好的方案。

### NeRF（背景介绍）

神经辐射场（Mildenhall 等，2020）。一个小型 MLP 输入 `(x, y, z, 视角方向)`，输出 `(颜色, 密度)`。通过积分沿射线渲染。画质优于基于网格的新视角合成，但渲染慢 100-1000 倍。已被高斯散点取代主流实时应用，但研究仍占主导。

## 实现它

`code/main.py` 实现了一个玩具的 2D “高斯散点”拟合：用多个 2D 高斯点积表示合成目标图像（平滑渐变）。通过梯度下降优化位置、颜色和协方差来匹配目标。你会看到两个核心操作：前向渲染（散点 + alpha 合成）和梯度下降拟合。

### 第一步：2D 高斯点

```python
def gaussian_at(x, y, gaussian):
    px, py = gaussian["pos"]
    sigma = gaussian["sigma"]
    d2 = (x - px) ** 2 + (y - py) ** 2
    return math.exp(-d2 / (2 * sigma * sigma))  # 高斯计算
```

### 第二步：通过累加散点渲染

```python
def render(image_size, gaussians):
    img = [[0.0] * image_size for _ in range(image_size)]
    for g in gaussians:
        for y in range(image_size):
            for x in range(image_size):
                img[y][x] += g["color"] * gaussian_at(x, y, g)
    return img
```

真正的 3D 高斯散点会按深度排序并按顺序 alpha 合成。我们的 2D 玩具仅简单累加。

### 第三步：用梯度下降拟合

```python
for step in range(steps):
    pred = render(size, gaussians)
    loss = mse(pred, target)
    gradients = compute_grads(pred, target, gaussians)
    update(gaussians, gradients, lr)
```

## 陷阱

- **视角不一致。** 如果单独生成 4 个视角且结构冲突，3D 拟合会模糊。解决方案：多视图扩散共享关注机制。
- **背面幻觉。** 单图生成 3D 必须“猜测”背面，质量不稳定。
- **高斯点爆炸。** 无限制训练高斯点数量爆炸至 1000 万，造成过拟合。密集化 + 剪枝启发式（3DGS 原创）必不可少。
- **拓扑问题。** 隐式场（SDF）生成的网格常有洞或自交。出货前用重网格工具（如 Blender 体素重网格）修复。
- **训练数据授权。** Objaverse 有混合许可，商业使用依模型不同而异。

## 使用它

| 任务 | 2026 推荐 |
|------|-----------|
| 照片场景重建 | Gaussian splatting (3DGS, Gsplat, Scaniverse) |
| 游戏用文本到 3D 物体 | Meshy 4 或 Rodin Gen-1.5（PBR 输出） |
| 图像到 3D | Hunyuan3D 2.0、TripoSR、InstantMesh |
| 少量图像新视角合成 | CAT3D、SV3D |
| 动态场景重建 | 4D Gaussian Splatting |
| 虚拟形象 / 穿衣人像 | Gaussian Avatar、HUGS |
| 研究 / 前沿 | 最近发布的最新模型 |

在游戏或电商管线中生产 3D 对象：Meshy 4 或 Rodin Gen-1.5 输出的 PBR 网格可直接导入 Unity / Unreal。

## 交付它

保存为 `outputs/skill-3d-pipeline.md`。技能接收一个 3D 需求简报（输入：文字/单图/少量图；输出：网格/散点/NeRF；用途：渲染/游戏/VR），输出：流水线（多视图扩散 + 拟合或直接网格模型）、基础模型、迭代预算、拓扑后处理、所需材质通道。

## 练习

1. **简单。** 用 4、16、64 个高斯运行 `code/main.py`。报告最终与目标的 MSE。
2. **中等。** 扩展到彩色高斯（RGB）。确认重建色彩模式正确。
3. **困难。** 用 gsplat 或 Nerfstudio，从 50 张照片重建真实物体。报告拟合时间和保留视角的 SSIM。

## 关键词

| 词汇 | 俗称 | 实际含义 |
|------|--------|-----------|
| 3D Gaussian Splatting | “3DGS” | 用 3D 高斯云表示场景；可微分 alpha 合成渲染。 |
| NeRF | “神经辐射场” | MLP 在 3D 点输出颜色和密度；通过射线积分渲染。 |
| Triplane | “三幅二维平面” | 把 3D 分解成三个 2D 轴对齐特征网格；比体积更节省。 |
| SDS | “Score Distillation Sampling” | 用 2D 扩散模型梯度作为伪梯度训练 3D 模型。 |
| 多视图扩散 | “多视角同时生成” | 输出多摄像机视角一致性的扩散模型。 |
| PBR | “物理基渲染” | 材质包含反照率、粗糙度、金属度、法线通道。 |
| 密集化 | “增生散点” | 3DGS 训练启发式：在高梯度区分裂/克隆高斯点。 |

## 生产说明：2026 年 3D 尚无统一底层

与图像（潜空间扩散 + DiT）和视频（时空 DiT）不同，2026 年 3D 没有单一主流运行时。生产决策树按表示分叉：

- **NeRF / triplane。** 推理是每个采样的射线行进 + MLP 前向。512² 渲染需百万级 MLP 前向。对射线采样批量优化，适用 SDPA/xformers。
- **多视图扩散 + LRM 重建。** 两阶段流水线。第一阶段（多视图 DiT）是扩散服务，与第 07 课类似。第二阶段（LRM Transformer）一次前向所有视角。整体延迟是“扩散 + 一次性”，服务基础选型需对应。
- **SDS / DreamFusion。** 每资产优化，不是推理。构建作业，不是请求处理。

大部分 2026 产品正确答案：“按需运行多视图扩散模型，异步重建至 3DGS，实时服务 3DGS 查看”，GPU 推理服务器（快）和离线优化器（慢）分开。

## 拓展阅读

- [Mildenhall et al. (2020). NeRF: Representing Scenes as Neural Radiance Fields](https://arxiv.org/abs/2003.08934) — NeRF。
- [Kerbl et al. (2023). 3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://arxiv.org/abs/2308.04079) — 3DGS。
- [Poole et al. (2022). DreamFusion: Text-to-3D using 2D Diffusion](https://arxiv.org/abs/2209.14988) — SDS。
- [Liu et al. (2023). Zero-1-to-3: Zero-shot One Image to 3D Object](https://arxiv.org/abs/2303.11328) — Zero123。
- [Shi et al. (2023). MVDream](https://arxiv.org/abs/2308.16512) — 多视图扩散。
- [Hong et al. (2023). LRM: Large Reconstruction Model for Single Image to 3D](https://arxiv.org/abs/2311.04400) — LRM。
- [Gao et al. (2024). CAT3D: Create Anything in 3D with Multi-View Diffusion Models](https://arxiv.org/abs/2405.10314) — CAT3D。
- [Stability AI (2024). Stable Video 3D (SV3D)](https://stability.ai/research/sv3d) — SV3D。
