# 3D 高斯点渲染（3D Gaussian Splatting）——从零实现

> 一个场景是由数百万个 3D 高斯点云（3D Gaussians）组成。每个高斯点都有位置、方向、尺度、不透明度，以及基于观察方向变化的颜色。对它们进行光栅化（rasterization），对光栅化过程进行反向传播，完成渲染。

**类型：** 构建（Build）  
**语言：** Python  
**先修课程：** 第4阶段第13课（3D视觉与NeRF），第1阶段第12课（张量操作），第4阶段第10课（扩散基础，可选）  
**时间：** 约90分钟

## 学习目标

- 解释为什么3D高斯点渲染在2026年取代NeRF成为真实感3D重建的生产默认方案
- 陈述每个高斯点的六个参数（位置、旋转四元数、尺度、不透明度、球谐颜色、可选特征）及其各自所占浮点数数量
- 从头实现一个2D高斯点光栅化器，使用`alpha`合成（alpha compositing），并展示3D情形如何投影到同一循环
- 使用`nerfstudio`、`gsplat`或`SuperSplat`从20-50张照片重建场景，并导出为`KHR_gaussian_splatting` glTF扩展或OpenUSD 26.03的`UsdVolParticleField3DGaussianSplat`模式

## 双语术语速查

| 中文术语 | English term |
|----------|--------------|
| 3D 高斯点渲染 | 3D Gaussian Splatting, 3DGS |
| 高斯点 | Gaussian splat |
| 显式场景表示 | Explicit scene representation |
| 协方差 | Covariance |
| 各向异性尺度 | Anisotropic scale |
| 四元数旋转 | Quaternion rotation |
| 不透明度 | Opacity |
| Alpha 合成 | Alpha compositing |
| 光栅化 | Rasterization |
| 稠密化 | Densification |
| 剪枝 | Pruning |
| 球谐函数 | Spherical harmonics |
| 视角相关颜色 | View-dependent color |
| glTF 扩展 | glTF extension |
| KHR_gaussian_splatting | KHR_gaussian_splatting |
| OpenUSD | Open Universal Scene Description, OpenUSD |


## 问题背景

NeRF将场景存储为一个多层感知机（MLP）的权重。每个渲染像素需沿射线进行数百次MLP查询。训练耗时数小时，渲染耗时数秒，且权重不可编辑——如果想移动场景中的椅子，必须重新训练。

3D高斯点渲染（Kerbl, Kopanas, Leimkühler, Drettakis, SIGGRAPH 2023）彻底改变了这一点。场景以显式的3D高斯点集合表示。渲染通过GPU光栅化实现，每秒超过100帧。训练时间仅数分钟。编辑也更直接：平移部分高斯点即可移动椅子。到2026年，Khronos集团正式批准了glTF高斯点扩展，OpenUSD 26.03版本提供了高斯点模式，实现该技术的Zillow和Apartments.com利用其进行房产渲染，而大多数新兴的3D重建论文都基于这一核心3DGS概念进行变体研究。

思维模型简单，数学细节丰富，通常介绍会从光栅化开始，跳过投影和球谐函数部分。本课将从头构建整个过程——先实现2D版本，再扩展到3D。

## 核心概念

### 关键公式（Key equations）

每个高斯点用均值和协方差表示空间密度，投影到屏幕后再进行 alpha 合成：

$$
G(\mathbf{x}) = \exp\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^\top \Sigma^{-1}(\mathbf{x}-\boldsymbol{\mu})\right)
$$

$$
C = \sum_{i=1}^{N} T_i \alpha_i c_i, \qquad T_i = \prod_{j<i}(1-\alpha_j)
$$

### 一个高斯点携带的参数

一个3D高斯点是空间中具有以下属性的参数化模糊体：

```text
position         mu         (3,)    世界坐标系中心点
rotation         q          (4,)    单位四元数，表示方向
scale            s          (3,)    各轴的对数尺度（渲染时指数化）
opacity          alpha      (1,)    后sigmoid不透明度值 [0, 1]
SH coefficients  c_lm       (3 * (L+1)^2,)   视角依赖的颜色
```

旋转+尺度构建了一个3×3协方差矩阵：`Sigma = R S S^T R^T`，定义了高斯点在3D空间的形状。球谐函数（Spherical harmonics，SH）使颜色随观察方向变化——可以捕捉镜面高光、微妙光泽和方向依赖发光——无需为每个观察方向存储纹理。SH次数设为3时，每个颜色通道有16个系数，每个高斯点颜色就用48个浮点数表示。

一个场景通常包含1至5百万高斯点。每个存储约60个浮点数（3 + 4 + 3 + 1 + 48 + 杂项）。五百万高斯点场景约240MB，比带点纹理的点云小得多，比用高分辨率重渲染的NeRF MLP权重小一个数量级。

### 光栅化，而非射线行进

```mermaid
flowchart LR
    SCENE["数百万个3D高斯点<br/>(位置, 旋转, 尺度,<br/>不透明度, SH颜色)"] --> PROJ["投影到2D<br/>(相机外参+内参)"]
    PROJ --> TILES["分配到瓦片<br/>(屏幕空间16x16)"]
    TILES --> SORT["每瓦片内按深度排序"]
    SORT --> ALPHA["前向后Alpha合成"]
    ALPHA --> PIX["像素颜色"]

    style SCENE fill:#dbeafe,stroke:#2563eb
    style ALPHA fill:#fef3c7,stroke:#d97706
    style PIX fill:#dcfce7,stroke:#16a34a
```

流程共五步，均适合GPU并行。不需要每像素查询MLP。一张RTX 3080 Ti能以147帧每秒渲染600万个高斯点。

### 投影步骤

世界坐标下位置为`mu`、协方差矩阵为`Sigma`的3D高斯点，投影到屏幕空间后变为位置`mu'`和2D协方差`Sigma'`：

```text
mu' = project(mu)
Sigma' = J W Sigma W^T J^T          (2 x 2)

W = 视点变换（相机旋转+平移）
J = mu'处透视投影的雅可比矩阵（Jacobian）
```

二维高斯点的投影轨迹是个椭圆，其主轴为`Sigma'`的特征向量。椭圆内的每个像素都根据权重`exp(-0.5 * (p - mu')^T Sigma'^-1 (p - mu'))`接收该高斯点对像素值的影响。

### Alpha合成规则

对于单个像素，覆盖该像素的高斯点按远近顺序（从后到前排序）进行合成，也可以前到后结合反向公式。颜色值与所有半透明光栅化器自1980年代以来采用相同公式：

```text
C_pixel = sum_i alpha_i * T_i * c_i

T_i = prod_{j < i} (1 - alpha_j)       i之前的透射率
alpha_i = opacity_i * exp(-0.5 * d^T Sigma'^-1 d)   局部贡献
c_i = eval_SH(SH_i, view_direction)    视角依赖颜色
```

这实际上**等同于NeRF的体积渲染公式**，唯一不同是求和离散于显式稀疏的高斯点集，而非射线上密集采样。该等价性使得渲染质量和NeRF匹配——两者积分的都是相同的辐射场方程。

### 可微分性原因

每一步——投影、瓦片分配、alpha合成、球谐函数计算——对高斯点参数是可微分的。给定真实图像，计算渲染像素损失，通过光栅化器反向传播，用梯度下降更新`(mu, q, s, alpha, c_lm)`参数。约3万次迭代后，高斯点会收敛到正确的位置、尺度与颜色。

### 密集化和剪枝

固定数目高斯点无法覆盖复杂场景，训练包括两种自适应机制：

- **克隆（Clone）**：当梯度幅值较大且尺度较小时，在当前点位置复制一个高斯点——重建细节需要加强。
- **拆分（Split）**：当大尺度高斯点梯度高时，将其拆分成两个小高斯点——大点过于平滑无法拟合区域。
- **剪枝（Prune）**：不透明度低于阈值的高斯点被剔除——它们对渲染贡献微弱。

密集化每隔若干次迭代执行。场景通常从约10万初始高斯点（源自结构光模型SfM点）增长到1-5百万个。

### 一段话说清球谐函数

视角依赖颜色是单位球面上的函数`c(direction)`。球谐函数是球面的傅里叶基。截断到次数`L`，每个颜色通道有`(L+1)^2`个基函数。在新视角下计算颜色即为高斯点学得的SH系数与基函数值的点积。零阶是恒定颜色，三阶有16个系数，可捕捉朗伯光照（Lambertian shading）、镜面反射和轻微反光。Stable Diffusion Gaussian Splatting论文默认采用三阶。

### 2026年生产技术栈

```text
1. 采集           智能手机 / DJI无人机 / 手持扫描仪
2. SfM / MVS      COLMAP或GLOMAP计算相机位姿+稀疏点云
3. 训练3DGS       nerfstudio / gsplat / inria官方 / PostShot（RTX 4090约10-30分钟）
4. 编辑           SuperSplat / SplatForge（清理漂浮体，分割）
5. 导出           .ply -> glTF KHR_gaussian_splatting或.usd（OpenUSD 26.03）
6. 浏览           Cesium / Unreal / Babylon.js / Three.js / Vision Pro
```

### 4D与生成式变体

- **4D高斯点渲染**——高斯点随时间变化，用于体积视频（如Superman 2026，A$AP Rocky的"Helicopter"）
- **生成式高斯点**——文本生成至高斯点模型（World Labs出品的Marble），可生成完整场景
- **3D高斯无迹变换（Unscented Transform）**——NVIDIA NuRec自动驾驶仿真的变体

## 动手实现

### 第1步：二维高斯点

首先构建2D光栅化器，3D投影后归约为2D高斯点光栅化。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def eval_2d_gaussian(means, covs, points):
    """
    means:  (G, 2)      中心坐标
    covs:   (G, 2, 2)   协方差矩阵
    points: (H, W, 2)   像素坐标
    返回:   (G, H, W)  每个高斯在每个像素的密度
    """
    G = means.size(0)
    H, W, _ = points.shape
    flat = points.view(-1, 2)
    inv = torch.linalg.inv(covs)
    diff = flat[None, :, :] - means[:, None, :]
    d = torch.einsum("gpi,gij,gpj->gp", diff, inv, diff)  # 计算diff^T Sigma^-1 diff
    density = torch.exp(-0.5 * d)
    return density.view(G, H, W)
```

`einsum`实现了对每个（高斯点，像素）对的二次型计算。

### 第2步：二维高斯点光栅化器

采用前向后alpha合成。二维场景深度没有意义，用一个可训练的高斯点标量来决定顺序。

```python
def rasterise_2d(means, covs, colours, opacities, depths, image_size):
    """
    means:     (G, 2)
    covs:      (G, 2, 2)
    colours:   (G, 3)
    opacities: (G,)     范围[0, 1]
    depths:    (G,)     用于排序的标量
    image_size: (H, W)
    返回:      (H, W, 3) 渲染图像
    """
    H, W = image_size
    yy, xx = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=means.device),
        torch.arange(W, dtype=torch.float32, device=means.device),
        indexing="ij",
    )
    points = torch.stack([xx, yy], dim=-1)

    densities = eval_2d_gaussian(means, covs, points)
    alphas = opacities[:, None, None] * densities
    alphas = alphas.clamp(0.0, 0.99)

    order = torch.argsort(depths)
    alphas = alphas[order]
    colours_sorted = colours[order]

    T = torch.ones(H, W, device=means.device)
    out = torch.zeros(H, W, 3, device=means.device)
    for i in range(means.size(0)):
        a = alphas[i]
        out += (T * a)[..., None] * colours_sorted[i][None, None, :]
        T = T * (1.0 - a)
    return out
```

速度不快——实际实现会用基于瓦片的CUDA内核——但数学公式完全正确且可微。

### 第3步：可训练的二维高斯点场景

```python
class Splats2D(nn.Module):
    def __init__(self, num_splats=128, image_size=64, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        H, W = image_size, image_size
        self.means = nn.Parameter(torch.rand(num_splats, 2, generator=g) * torch.tensor([W, H]))
        self.log_scale = nn.Parameter(torch.ones(num_splats, 2) * math.log(2.0))
        self.rot = nn.Parameter(torch.zeros(num_splats))  # 2D单角度旋转
        self.colour_logits = nn.Parameter(torch.randn(num_splats, 3, generator=g) * 0.5)
        self.opacity_logit = nn.Parameter(torch.zeros(num_splats))
        self.depth = nn.Parameter(torch.rand(num_splats, generator=g))

    def covs(self):
        s = torch.exp(self.log_scale)
        c, si = torch.cos(self.rot), torch.sin(self.rot)
        R = torch.stack([
            torch.stack([c, -si], dim=-1),
            torch.stack([si, c], dim=-1),
        ], dim=-2)
        S = torch.diag_embed(s ** 2)
        return R @ S @ R.transpose(-1, -2)

    def forward(self, image_size):
        covs = self.covs()
        colours = torch.sigmoid(self.colour_logits)
        opacities = torch.sigmoid(self.opacity_logit)
        return rasterise_2d(self.means, covs, colours, opacities, self.depth, image_size)
```

`log_scale`、`opacity_logit` 和 `colour_logits` 都是不受约束的参数，在渲染时通过正确的激活函数映射。这是每个 3DGS（3D Gaussian Splatting，高斯点光源投影）实现的标准模式。

### 第4步：拟合二维高斯分布到目标图像

```python
import math
import numpy as np

def make_target(size=64):
    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    img = np.zeros((size, size, 3), dtype=np.float32)
    # 红色圆形
    mask = (xx - 20) ** 2 + (yy - 20) ** 2 < 10 ** 2
    img[mask] = [1.0, 0.2, 0.2]
    # 蓝色正方形
    mask = (np.abs(xx - 45) < 8) & (np.abs(yy - 40) < 8)
    img[mask] = [0.2, 0.3, 1.0]
    return torch.from_numpy(img)


target = make_target(64)
model = Splats2D(num_splats=64, image_size=64)
opt = torch.optim.Adam(model.parameters(), lr=0.05)

for step in range(200):
    pred = model((64, 64))
    loss = F.mse_loss(pred, target)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 40 == 0:
        print(f"step {step:3d}  mse {loss.item():.4f}")
```

经过200步，64个高斯分布稳定地拟合出两种形状。这就是整个思路——对显式几何基元做梯度下降。

### 第5步：从二维到三维

三维扩展保持相同的循环。新增内容：

1. 每个高斯的旋转用四元数替代单一角度。
2. 协方差 `Covariance` 表达为 `R S S^T R^T`，其中 `R` 从四元数构造，`S = diag(exp(log_scale))`。
3. 投影 `(mu, Sigma) -> (mu', Sigma')` 利用摄像机外参和在 `mu` 处的透视投影雅可比矩阵。
4. 颜色用球谐函数（spherical harmonics，SH）展开，在视线方向上评估。
5. 深度排序`Depth-sort`用摄像机空间的 z 值，而非学习得到的标量。

每个生产级实现（如 `gsplat`、`inria/gaussian-splatting`、`nerfstudio`）都在 GPU 上使用基于瓦片的 CUDA 内核来做精确计算。

### 第6步：球谐函数评估

最高到3阶的球谐函数基有每通道16项。计算如下：

```python
def eval_sh_degree_3(sh_coeffs, dirs):
    """
    sh_coeffs: (..., 16, 3)   最后一维是RGB通道
    dirs:      (..., 3)       单位向量
    返回:      (..., 3)
    """
    C0 = 0.282094791773878
    C1 = 0.488602511902920
    C2 = [1.092548430592079, 1.092548430592079,
          0.315391565252520, 1.092548430592079,
          0.546274215296039]
    x, y, z = dirs[..., 0], dirs[..., 1], dirs[..., 2]
    x2, y2, z2 = x * x, y * y, z * z
    xy, yz, xz = x * y, y * z, x * z

    result = C0 * sh_coeffs[..., 0, :]
    result = result - C1 * y[..., None] * sh_coeffs[..., 1, :]
    result = result + C1 * z[..., None] * sh_coeffs[..., 2, :]
    result = result - C1 * x[..., None] * sh_coeffs[..., 3, :]

    result = result + C2[0] * xy[..., None] * sh_coeffs[..., 4, :]
    result = result + C2[1] * yz[..., None] * sh_coeffs[..., 5, :]
    result = result + C2[2] * (2.0 * z2 - x2 - y2)[..., None] * sh_coeffs[..., 6, :]
    result = result + C2[3] * xz[..., None] * sh_coeffs[..., 7, :]
    result = result + C2[4] * (x2 - y2)[..., None] * sh_coeffs[..., 8, :]

    # 此处省略3阶项，完整16系数实现见代码文件
    return result
```

学习到的 `sh_coeffs` 储存了该高斯在各方向的“颜色”。渲染时对当前视角方向求值，得到3维RGB颜色向量。

## 使用说明

针对真实3DGS工作，使用 `gsplat`（Meta）或 `nerfstudio`：

```bash
pip install nerfstudio gsplat
ns-download-data example
ns-train splatfacto --data path/to/data
```

`splatfacto` 是 nerfstudio 的3DGS训练器。典型场景在 RTX 4090 上训练时间为10-30分钟。

2026年重要导出选项：

- `.ply` — 原始高斯点云（可移植，文件最大）。
- `.splat` — PlayCanvas / SuperSplat 量化格式。
- glTF `KHR_gaussian_splatting` — Khronos标准，可跨查看器通用（2026年2月RC版）。
- OpenUSD `UsdVolParticleField3DGaussianSplat` — USD原生格式，用于NVIDIA Omniverse及Vision Pro管线。

针对4D / 动态场景，`4DGS` 和 `Deformable-3DGS` 利用时间变化的均值和不透明度扩展相同机制。

## 部署应用

本课成果包括：

- `outputs/prompt-3dgs-capture-planner.md` — 用于规划采集流程（照片数量、相机路径、光照）针对不同场景类型的提示词。
- `outputs/skill-3dgs-export-router.md` — 根据下游查看器或引擎自动选取合适的导出格式（`.ply` / `.splat` / glTF / USD）的技能。

## 练习

1. **（简单）** 在不同的合成图像上运行上面二维splat训练器。令 `num_splats` 在 `[16, 64, 256]` 之间变化，绘制每个的MSE与步骤关系。找出收益递减点。
2. **（中等）** 扩展二维光栅器，使每个高斯的RGB颜色随标量“视角”变量通过二阶谐波变化。在一对目标图像上训练，验证模型能重建两张图。
3. **（困难）** 克隆 `nerfstudio` 并用任意场景（办公桌、植物、面部、房间）的20张照片训练 `splatfacto`。导出成glTF的 `KHR_gaussian_splatting` 并用Three.js的 `GaussianSplats3D`、SuperSplat或Babylon.js V9查看器打开。报告训练时长、高斯数量与渲染帧率。

## 关键词

| 术语 | 俗称 | 含义 |
|------|------|------|
| 3DGS | “Gaussian splats” | 显式场景表示为百万个3D高斯，每个高斯包含位置、旋转、尺度、不透明度、球谐颜色 |
| Covariance（协方差） | “Shape of the Gaussian” | `Sigma = R S S^T R^T`，单个高斯的方向和各向异性尺度 |
| Alpha compositing（Alpha合成） | “Back-to-front blend” | 与NeRF体积渲染方程相同，但作用于显式稀疏集合 |
| Densification（稠密化） | “Clone and split” | 自适应地在重建不足区域新增高斯 |
| Pruning（剪枝） | “Delete low-opacity” | 在训练中移除已收敛为近零不透明度的高斯 |
| Spherical harmonics（球谐函数） | “View-dependent colour” | 球面上的傅里叶基函数；存储颜色随视角变化的函数 |
| Splatfacto | “nerfstudio的3DGS” | 2026年最易用的3DGS训练方案 |
| `KHR_gaussian_splatting` | “glTF标准” | Khronos 2026年扩展，使3DGS格式跨查看器和引擎通用 |

## 延伸阅读

- [3D Gaussian Splatting 用于实时光度场渲染（Kerbl 等，SIGGRAPH 2023）](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) — 原始论文
- [gsplat (Meta/nerfstudio)](https://github.com/nerfstudio-project/gsplat) — 生产级CUDA光栅器
- [nerfstudio Splatfacto](https://docs.nerf.studio/nerfology/methods/splat.html) — 参考训练配方
- [Khronos KHR_gaussian_splatting 扩展](https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_gaussian_splatting/README.md) — 2026年可移植格式
- [OpenUSD 26.03 发布说明](https://openusd.org/release/) — `UsdVolParticleField3DGaussianSplat` 结构说明
- [THE FUTURE 3D 2026年高斯点光源投影状态](https://www.thefuture3d.com/blog-0/2026/4/4/state-of-gaussian-splatting-2026) — 行业概览
