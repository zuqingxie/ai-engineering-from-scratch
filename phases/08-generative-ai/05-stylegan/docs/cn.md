# StyleGAN

> 大多数生成器会同时将 `z` 注入到每一层。StyleGAN 则拆分开来：首先将 `z` 映射到一个中间变量 `w`，然后通过 AdaIN（自适应实例归一化）在每个分辨率级别*注入* `w`。这一改变解开了潜在空间的纠缠，使得七年来生成逼真脸部图像成为已解决的问题。

**类型：** 构建  
**语言：** Python  
**先决条件：** Phase 8 · 03（GANs），Phase 4 · 08（归一化），Phase 3 · 07（CNNs）  
**时长：** ~45分钟

## 问题所在

DCGAN 通过一堆转置卷积将 `z` 映射到图像。问题是：`z` 控制着所有内容——姿态、光照、身份、背景——这些因素纠缠在一起。沿着 `z` 的一个维度变化，以上四者全变。你无法让模型实现“同一人，不同姿态”的要求，因为其表示无法因式分解出这些内容。

Karras 等人（2019，NVIDIA）提出：停止将 `z` 直接输入卷积层。用一个常量的 `4×4×512` 张量作为网络输入。训练一个 8 层 MLP，将 `z ∈ Z → w ∈ W`。通过*自适应实例归一化*（AdaIN）在每个分辨率注入 `w`：先标准化每个卷积特征图，然后用 `w` 的仿射投影对其进行缩放和平移。每层加噪声以控制随机细节（如皮肤毛孔、发丝）。

结果是：`W` 空间中的轴大致正交，区分了“高级样式”（姿态、身份）和“细节样式”（光照、颜色）。你可以将两幅图的样式互换，使用图像 A 的 `w` 进行低分辨率级别，图像 B 的 `w` 用于高分辨率级别。这样解锁了图像编辑、跨域风格迁移和整个 StyleGAN-反演 研究路线。

## 核心概念

![StyleGAN: mapping network + AdaIN + per-layer noise](../assets/stylegan.svg)

**映射网络（Mapping network）。** `f: Z → W`，8 层 MLP。`Z = N(0, I)^512`，`W` 不被限制为高斯分布，而是学习到数据适应的形状。

**合成网络（Synthesis network）。** 从一个学习到的常量 `4×4×512` 开始。每个分辨率块依次为：`上采样 → 卷积 → AdaIN(w_i) → 噪声 → 卷积 → AdaIN(w_i) → 噪声`。分辨率按顺序翻倍：4、8、16、32、64、128、256、512、1024。

**AdaIN 公式。**

```text
AdaIN(x, y) = y_scale · (x - mean(x)) / std(x) + y_bias
```

其中 `y_scale` 和 `y_bias` 来自于 `w` 的仿射投影。先对每个特征图归一化，再进行样式转换。这里的“样式”是指特征图的一阶和二阶统计量。

**每层噪声（Per-layer noise）。** 给每个特征图加单通道高斯噪声，并且通过一个学习到的每通道因子进行缩放。控制随机细节，不影响全局结构。

**截断技巧（Truncation trick）。** 生成阶段，先采样 `z`，计算 `w = mapping(z)`，然后用公式 `w' = ŵ + ψ·(w - ŵ)` 对 `w` 进行截断，其中 `ŵ` 是大量样本的平均 `w`。`ψ < 1` 时平衡多样性与质量。几乎所有 StyleGAN 演示中都用 `ψ ≈ 0.7`。

## StyleGAN 1 → 2 → 3

| 版本       | 年份 | 创新点                                      |
|------------|------|---------------------------------------------|
| StyleGAN   | 2019 | 映射网络 + AdaIN + 噪声 + 进阶增长。         |
| StyleGAN2  | 2020 | 权重调制替代 AdaIN（解决水滴伪影）；跳跃/残差结构；路径长度正则化。 |
| StyleGAN3  | 2021 | 无别名卷积 + 等变核；消除纹理固定在像素网格的问题。         |
| StyleGAN-XL| 2022 | 条件类别生成，1024²，ImageNet。                |
| R3GAN      | 2024 | 更强正则化；用更少参数缩小与扩散模型在 FFHQ-1024 上的差距。    |

到 2026 年，StyleGAN3 依旧是默认选择，用于 (a) 窄域真实感图像高帧率生成，(b) 少样本域适应（用 100 张新图训练，冻结映射网络），(c) 基于反演的编辑（寻找能重建真实照片的 `w` 并编辑）。若是开放领域文本生成，扩散模型是首选。

## 构建它

`code/main.py` 实现了一个一维的“style-GAN lite”示例：包含映射 MLP，合成函数从一个学习到的常量向量开始，并用 `w` 派生的缩放/偏置调制，同时每层加噪声。展示了用仿射调制注入 `w` 能匹配或优于将 `z` 直接拼接到生成器输入。

### 第 1 步：映射网络

```python
def mapping(z, M):
    h = z
    for i in range(num_layers):
        h = leaky_relu(add(matmul(M[f"W{i}"], h), M[f"b{i}"]))
    return h
```

### 第 2 步：自适应实例归一化

```python
def adain(x, w_scale, w_bias):
    mu = mean(x)
    sd = std(x)
    x_norm = [(xi - mu) / (sd + 1e-8) for xi in x]
    return [w_scale * xi + w_bias for xi in x_norm]
```

每个特征图的缩放和平移由 `w` 经过线性投影获得。

### 第 3 步：每层噪声

```python
def add_noise(x, sigma, rng):
    return [xi + sigma * rng.gauss(0, 1) for xi in x]
```

每通道的 sigma 是可学习的。

## 常见陷阱

- **水滴伪影。** StyleGAN1 在特征图中产生模糊的水滴状伪影，因为 AdaIN 会将均值归零。StyleGAN2 用权重调制代替修正了这个问题，通过缩放卷积权重规避伪影。
- **纹理粘贴。** StyleGAN1 和 2 的纹理跟随像素坐标而非对象坐标（在插值时可见）。StyleGAN3 用无别名卷积和窗函数 sinc 滤波器解决此问题。
- **模式覆盖。** 截断 `ψ < 0.7` 虽然样本看起来干净，但取样来自狭窄的锥形区间；若需要多样性请用 `ψ = 1.0`。
- **反演有损。** 将真实照片反演到 `W` 空间通常通过优化或编码器实现（e4e、ReStyle、HyperStyle）。多次迭代会导致结果漂移。

## 使用场景

| 用例               | 方案                                            |
|--------------------|-------------------------------------------------|
| 真实感人脸（动漫、产品、狭域） | StyleGAN3 FFHQ / 自定义微调                              |
| 从照片进行脸部编辑          | e4e 反演 + StyleSpace / InterFaceGAN 方向               |
| 人脸替换 / 驯服             | StyleGAN + 编码器 + 混合                                |
| 头像流水线               | StyleGAN3 搭配 ADA 进行低数据微调                        |
| 少量图像域适应            | 冻结映射网络，微调合成网络                                |
| 多模态或文本条件生成       | 不建议，推荐使用扩散模型                                  |

对于产品级的“照片级人脸”演示，StyleGAN 在推理成本（单次前向推理，RTX 4090 下 <10ms）和同等质量下的锐度方面均优于扩散模型。

## 部署

保存文件 `outputs/skill-stylegan-inversion.md`。该技能接受真实照片输出：反演方法（e4e / ReStyle / HyperStyle）、预期潜变量损失、编辑预算（在 `W` 中可移动距离前的伪影出现阈值）及一系列已知的良好编辑方向（年龄、表情、姿态）。

## 练习

1. **简单。** 运行 `code/main.py`，分别设置 `adain_on=True` 和 `adain_on=False`。比较固定潜变量和随机扰动潜变量时输出的分布。
2. **中等。** 实现混合正则化：在训练批次中，计算 `w_a`、`w_b`，合成过程前半段使用 `w_a`，后半段使用 `w_b`。解码器是否学会了解耦样式？
3. **困难。** 使用预训练 StyleGAN3 FFHQ 模型（ffhq-1024.pkl）。训练 SVM 来找到操控“微笑”方向的 `w`，报告推送此方向多久前身份开始漂移。

## 关键词

| 术语               | 常见表述                | 实际意义                                    |
|--------------------|-------------------------|---------------------------------------------|
| 映射网络（Mapping network）  | “MLP”                  | `f: Z → W`，8 层，解耦潜变量几何与数据统计。       |
| W 空间（W space）         | “风格空间”               | 映射网络输出，粗略解耦。                           |
| AdaIN               | “自适应实例归一化”          | 归一化特征图，再用 `w` 投影缩放和平移。               |
| 截断技巧（Truncation trick） | “Psi”                  | `w = mean + ψ·(w - mean)`，ψ<1权衡多样性与质量。    |
| 路径长度正则化（Path-length regularization） | “PL正则化”               | 惩罚 `w` 单位变动导致的图像大变化；使 `W` 空间更平滑。   |
| 权重调制（Weight demodulation） | “StyleGAN2 修复”         | 归一化卷积权重而非激活；消除水滴伪影。                 |
| 无别名（Alias-free）      | “StyleGAN3 技巧”           | 窗函数 sinc 滤波器；消除纹理粘附像素网格问题。            |
| 反演（Inversion）          | “为真实图像找 w”           | 优化或编码 `x → w` 使得 `G(w) ≈ x`。                  |

## 生产说明：为何 StyleGAN 至 2026 年仍具部署价值

StyleGAN3 在 RTX 4090 上生成 1024² 分辨率 FFHQ 脸部低于 10 ms — `num_steps = 1`，无 VAE 解码，无交叉注意力步骤。从生产角度看，这是任何图像生成模型的最低延迟。相比之下，同分辨率下 50 步 SDXL + VAE 解码流水线需时约 3 秒。这是 **300 倍差距**，在狭域产品（头像服务、身份证流水线、库存人脸生成）上具有成本优势。

两个实际影响：

- **不用调度器，不用批处理。** 静态批量在目标占用率下最优。连续批处理（LLMs 和扩散必需）无益，因为每个请求 FLOPs 相同。
- **截断参数 `ψ` 是安全阀。** `ψ < 0.7` 时采样范围成狭窄锥体；这是服务层控制样本方差的唯一杠杆。峰值负载时降低 `ψ`，高端用户提高 `ψ`。

## 进一步阅读

- [Karras et al. (2019). A Style-Based Generator Architecture for GANs](https://arxiv.org/abs/1812.04948) — StyleGAN。
- [Karras et al. (2020). Analyzing and Improving the Image Quality of StyleGAN](https://arxiv.org/abs/1912.04958) — StyleGAN2。
- [Karras et al. (2021). Alias-Free Generative Adversarial Networks](https://arxiv.org/abs/2106.12423) — StyleGAN3。
- [Tov et al. (2021). Designing an Encoder for StyleGAN Image Manipulation](https://arxiv.org/abs/2102.02766) — e4e 反演。
- [Sauer et al. (2022). StyleGAN-XL: Scaling StyleGAN to Large Diverse Datasets](https://arxiv.org/abs/2202.00273) — StyleGAN-XL。
- [Huang et al. (2024). R3GAN: The GAN is dead; long live the GAN!](https://arxiv.org/abs/2501.05441) — 现代极简 GAN 配方。
