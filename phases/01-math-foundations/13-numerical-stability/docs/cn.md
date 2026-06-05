# 数值稳定性

> 浮点数是一个有漏洞的抽象。它会在训练时咬你一口，而你却未曾察觉。

**类型:** 构建  
**语言:** Python  
**先决条件:** 阶段 1，课程 01-04  
**时间:** 约120分钟

## 学习目标

- 使用最大值减法技巧实现数值稳定的 softmax 和 log-sum-exp
- 识别浮点计算中的溢出、下溢和灾难性抵消
- 使用中心差分验证解析梯度与数值梯度
- 解释为何 bfloat16 优于 float16 用于训练，以及损失缩放如何防止梯度下溢

## 问题

你的模型训练了三个小时，之后损失变为 NaN。你加了打印语句。step 9000 时 logits 正常。step 9001 时变为 `inf`。step 9002 时每个梯度都是 `nan`，训练彻底挂了。

或者：你的模型训练跑完了，但准确率比论文低 2%。你检查所有。架构匹配。超参数匹配。数据匹配。问题是论文用的是 float32，而你用了 float16 且没有正确缩放。32 位累积的舍入误差悄无声息地吞噬了你的准确率。

或者：你从零实现了交叉熵损失。它在小的 logits 上正常工作。当 logits 超过 100 时，它返回 `inf`。softmax 溢出，因为 `exp(100)` 超出 float32 能表示的范围。每个 ML 框架都用两行代码解决了这个问题。你不知道这个技巧的存在。

数值稳定性不是理论上的问题。它是训练跑成功和悄无声息失败之间的差别。每个你最后必须调试的严重 ML bug，归根结底都是浮点问题。

## 概念

### IEEE 754：计算机如何存储实数

计算机按照 IEEE 754 标准将实数存储为浮点数。浮点数有三部分：符号位、指数和尾数（有效数字）。

```text
Float32 布局（共 32 位）:  
[1 符号位] [8 指数] [23 尾数]

数值 = (-1)^sign * 2^(exponent - 127) * 1.mantissa
```

尾数决定精度（有效数字位数），指数决定范围（数值大小上下限）。

```text
格式        位数   指数位   尾数位   十进制有效位数  大致范围
float64    64     11       52       ~15-16         +/- 1.8e308
float32    32     8        23       ~7-8           +/- 3.4e38
float16    16     5        10       ~3-4           +/- 65,504
bfloat16   16     8        7        ~2-3           +/- 3.4e38
```

float32 大约有 7 位十进制有效数字。它可以区分 1.0000001 和 1.0000002，但不能区分 1.00000001 和 1.00000002。超过 7 位后，全部都是舍入噪声。

float16 约有 3 位有效数字。它能表示的最大数是 65,504。对于常见超过该范围的 ML logits、梯度和激活值来说，这非常有限。

bfloat16 是 Google 针对 float16 范围问题的解决方案。它拥有和 float32 同样的 8 位指数（范围相同，最高可达 3.4e38），但只有 7 位尾数（精度低于 float16）。训练神经网络时，范围比精度更重要，因而 bfloat16 通常更好用。

### 为什么 0.1 + 0.2 != 0.3

数字 0.1 无法用二进制浮点数精确表示。在二进制下它是无限循环小数：

```text
0.1 的二进制表示 = 0.0001100110011001100110011...（无限循环）
```

float32 会截断为 23 位尾数。存储的值约为 0.100000001490116。类似地，0.2 约存为 0.200000002980232。它们之和约为 0.300000004470348，不是 0.3。

```text
Python 中：
>>> 0.1 + 0.2
0.30000000000000004

>>> 0.1 + 0.2 == 0.3
False
```

这在 ML 中很要紧，因为：

1. 类似 `if loss < threshold` 的损失比较可能出错
2. 累积许多小值（数千步的梯度更新）时，和会逐渐偏离真实值
3. 校验和以及可复现性测试用 `==` 比较浮点数会失败

解决方法：不要用 `==` 比较浮点数，用 `abs(a - b) < epsilon` 或 `math.isclose()`。

### 灾难性抵消（Catastrophic Cancellation）

当你相减两个非常接近的浮点数时，有效数字被抵消，剩下的都是舍入噪声升格为主要数字。

```text
a = 1.0000001    （float32 存储约为 1.00000011920929）
b = 1.0000000    （float32 存储为 1.00000000000000）

真实差值:  0.0000001
计算差值:  0.00000011920929

相对误差: 19.2%
```

单次相减带来 19% 相对误差。在 ML 中，每当你：

- 计算均值很大数据的方差：`E[x^2] - E[x]^2`
- 相减近似相等的对数概率
- 用太小的 epsilon 计算有限差分梯度

都会遇到这问题。

解决方案：重排公式，避免相减大且接近的数。方差用 Welford 算法或先对数据中心化。对数概率运算全程用对数空间。

### 溢出和下溢

溢出发生在结果太大无法表示时。下溢发生在结果太小（更接近零，小于能表示的最小正数）时。

```text
Float32 边界：
  最大值:     3.4028235e+38
  最小正数（正规数）: 1.175e-38
  最小正数（非正规数）: 1.401e-45
  溢出:      > 3.4e38 变为 inf
  下溢:      < 1.4e-45 变为 0.0
```

`exp()` 是 ML 里溢出的主因：

```text
exp(88.7)  = 3.40e+38   （刚好在 float32 范围内）
exp(89.0)  = inf        （溢出）
exp(-87.3) = 1.18e-38   （刚好高于下溢）
exp(-104)  = 0.0        （下溢变零）
```

`log()` 则走向相反方向：

```text
log(0.0)   = -inf
log(-1.0)  = nan
log(1e-45) = -103.3      （正常）
log(1e-46) = -inf        （输入下溢为 0，log(0) = -inf）
```

在 ML 中，`exp()` 出现于 softmax、sigmoid 和概率计算。`log()` 出现在交叉熵、对数似然和 KL 散度。`log(exp(x))` 的组合如无技巧就是雷区。

### Log-Sum-Exp 技巧

直接计算 `log(sum(exp(x_i)))` 数值不稳。大 x_i 导致 `exp(x_i)` 溢出。全为极小 x_i 导致所有 `exp(x_i)` 下溢为零，`log(0)` 为 `-inf`。

技巧：先减去最大值再指数。

```text
log(sum(exp(x_i))) = max(x) + log(sum(exp(x_i) - max(x)))
```

为什么有效：减去最大值后，最大的指数变为 `exp(0) = 1`，不会溢出。至少有一项是 1，和至少为 1，`log(1) = 0`，不会下溢为 `-inf`。

证明：

```text
log(sum(exp(x_i)))
= log(sum(exp(x_i - c + c)))                   （加减 c）
= log(sum(exp(x_i - c) * exp(c)))               （指数乘法）
= log(exp(c) * sum(exp(x_i - c)))               （提取公共因子）
= c + log(sum(exp(x_i - c)))                    （对数乘积为加法）
```

取 `c = max(x)`，溢出消除。

此技巧在 ML 到处可见：  
- Softmax 归一化  
- 交叉熵损失计算  
- 序列模型中对数概率求和  
- 高斯混合模型  
- 变分推断

### 为何 Softmax 需要最大值减法技巧

Softmax 将 logits 转概率：

```text
softmax(x_i) = exp(x_i) / sum(exp(x_j))
```

没用技巧时，logits [100, 101, 102] 会导致溢出：

```text
exp(100) = 2.69e43
exp(101) = 7.31e43
exp(102) = 1.99e44
sum      = 2.99e44
```

它们超出 float32 (最大约 3.4e38)？实际上：  
exp(88.7) 已接近 float32 最大值，exp(100) 在 float32 中是 inf。

用技巧，减最大值 102：

```text
exp(100 - 102) = exp(-2) = 0.135
exp(101 - 102) = exp(-1) = 0.368
exp(102 - 102) = exp(0)  = 1.000
sum = 1.503

softmax = [0.090, 0.245, 0.665]
```

概率一致，计算安全。这不是优化，是正确性的必需。

### NaN 和 Inf：检测与预防

`nan`（非数字）和 `inf`（无穷）会在计算中病毒式传播。梯度更新中出现一个 `nan`，权重变 `nan`，随后所有输出皆 `nan`。训练瞬间崩溃。

`inf` 产生原因：  
- 对大正数做 `exp()`  
- 除以零：`1.0 / 0.0`  
- float32 累积溢出  

`nan` 产生原因：  
- `0.0 / 0.0`  
- `inf - inf`  
- `inf * 0`  
- 负数求平方根 `sqrt()`  
- 负数取对数 `log()`  
- 涉及已有 `nan` 的运算

检测方法：

```python
import math

math.isnan(x)       # x 是 nan 时 True
math.isinf(x)       # x 是 +inf 或 -inf 时 True
math.isfinite(x)    # x 既非 nan 也非 inf 时 True
```

预防策略：

1. 对 `exp()` 输入限制范围：`exp(clamp(x, -80, 80))`
2. 分母添加 epsilon：`x / (y + 1e-8)`
3. `log()` 内部加 epsilon：`log(x + 1e-8)`
4. 使用稳定版本实现（log-sum-exp，稳定 softmax）
5. 梯度裁剪防止权重爆炸
6. 调试时每次前向后检测 `nan`/`inf`

### 数值梯度检查

解析梯度（反向传播得到的）可能有错。数值梯度检查通过有限差分计算梯度验证解析梯度。

中心差分公式：

```text
df/dx ≈ (f(x + h) - f(x - h)) / (2h)
```

它是 O(h²) 级别准确，比单边差分 `(f(x+h) - f(x)) / h`（O(h) 级）更精确。

选 h：太大近似错，太小灾难性抵消。通常取 `h=1e-5` 到 `1e-7`。

验算公式：

```text
relative_error = |grad_analytical - grad_numerical| / max(|grad_analytical|, |grad_numerical|, 1e-8)
```

经验规则：  
- relative_error < 1e-7：完美，梯度正确  
- relative_error < 1e-5：可接受，很可能正确  
- relative_error > 1e-3：有问题  
- relative_error > 1：梯度完全出错  

实现新层或损失函数时一定要检查梯度。PyTorch 提供 `torch.autograd.gradcheck()`。

### 混合精度训练

现代 GPU 拥有 Tensor Core 专用硬件，float16 矩阵乘法比 float32 快 2-8 倍。混合精度训练利用此：

```text
1. 保留 float32 主权重副本
2. 前向传播用 float16（快）
3. 损失计算用 float32（防溢出）
4. 反向传播用 float16（快）
5. 梯度缩放转换回 float32
6. 更新 float32 主权重
```

纯 float16 训练问题：梯度常很小（1e-8 甚至更小）。float16 会把小于 ~6e-8 的数下溢为零，模型不再学习因为所有梯度都为零。

解决方案是损失缩放：

```text
1. 将损失乘以大规模因子（如 1024）
2. 反向传播计算 (loss * 1024) 的梯度
3. 所有梯度扩大 1024 倍（浮点16下溢被抑制）
4. 更新权重前将梯度除以 1024
5. 总效果：更新一致，但无梯度下溢
```

动态损失缩放自动调整缩放因子。起始大值（65536）。若梯度溢出为 `inf`，则缩小一半。无溢出过若干步，则扩大一倍。

### bfloat16 vs float16：为什么 bfloat16 在训练中胜出

```text
float16:   [1 符号位] [5 指数]  [10 尾数]
bfloat16:  [1 符号位] [8 指数]  [7 尾数]
```

float16 拥有更高的精度（10 位尾数 vs 7 位）但范围有限（最大约 65,504）。bfloat16 精度较低，但范围与 float32 相同（最大约 3.4e38）。

用于训练神经网络时：

- 激活值和 logits 在训练高峰时常常超过 65,504。float16 会溢出；bfloat16 能处理此情况。
- float16 需要损失缩放（loss scaling），而 bfloat16 通常不需要，因为其范围覆盖了梯度幅度的全谱。
- bfloat16 是对 float32 的简单截断：丢弃尾数的低 16 位。转换过程简单且指数部分无损。

float16 更适合推理，其中数值有界且精度更重要。bfloat16 更适合训练，其中数值范围更重要。这就是为什么 TPU 和现代 NVIDIA GPU（A100、H100）原生支持 bfloat16 的原因。

### 梯度裁剪（Gradient Clipping）

梯度爆炸发生在梯度通过多层网络指数增长时（常见于 RNN、深层网络和 Transformer）。单个巨大梯度可能一步破坏所有权重。

两种裁剪方式：

**按值裁剪（Clip by value）：** 独立限制每个梯度元素。

```text
grad = clamp(grad, -max_val, max_val)
```

简单但可能改变梯度向量的方向。

**按范数裁剪（Clip by norm）：** 缩放整个梯度向量，使其范数不超过阈值。

```text
if ||grad|| > max_norm:
    grad = grad * (max_norm / ||grad||)
```

保留梯度方向。这就是 `torch.nn.utils.clip_grad_norm_()` 的实现，也是标准选择。

典型值：Transformer 用 `max_norm=1.0`，强化学习（RL）用 `max_norm=0.5`，简单网络用 `max_norm=5.0`。

梯度裁剪不是小技巧，而是一种安全机制。没有它，单个异常样本可能产生巨大梯度，毁掉数周的训练成果。

### 归一化层作为数值稳定器

批量归一化（Batch Normalization）、层归一化（Layer Normalization）和 RMS 归一化通常被认为是加速训练收敛的正则器。它们同样是数值稳定器。

没有归一化时，激活值会随着层数呈指数增长或下降：

```text
层 1：数值范围 [0, 1]
层 5：数值范围 [0, 100]
层 10：数值范围 [0, 10,000]
层 50：数值范围 [0, 无穷大]
```

归一化在每层对激活进行中心化和缩放：

```text
LayerNorm(x) = (x - mean(x)) / (std(x) + epsilon) * gamma + beta
```

`epsilon`（通常是 1e-5）防止所有激活相同时除零错误。学习参数 `gamma` 和 `beta` 让网络恢复任何所需尺度。

这让网络中所有数值保持在数值安全范围，防止前向传播溢出和反向传播梯度爆炸。

### 常见机器学习数值错误（Bugs）

**错误：训练几轮后损失变为 NaN。**  
原因：logits 变得过大，softmax 溢出，或学习率过高导致权重发散。  
修复：使用稳定 softmax（减去最大值），降低学习率，增加梯度裁剪。

**错误：损失停留在 log(类别数) 附近。**  
原因：模型输出接近均匀概率，通常表示梯度消失或模型完全不学习。  
修复：检查数据标签是否正确，验证损失函数，检查死 ReLU。

**错误：验证准确率比预期低 1-3%。**  
原因：混合精度使用不当没启用正确的损失缩放。梯度下溢导致小更新被静默清零。  
修复：启用动态损失缩放，或使用 bfloat16。

**错误：某些层梯度范数为 0.0。**  
原因：死 ReLU 神经元（输入全为负）或 float16 下溢。  
修复：使用 LeakyReLU 或 GELU，使用梯度缩放，检查权重初始化。

**错误：模型一台 GPU 上正常，另一台结果不同。**  
原因：浮点数累加顺序非确定性。GPU 并行规约在不同硬件上顺序不同，浮点加法不满足结合律。  
修复：接受小差异（1e-6），或设置 `torch.use_deterministic_algorithms(True)` 并接受速度损失。

**错误：损失计算时 `exp()` 返回 `inf`。**  
原因：未经最大值减法处理的原始 logits 传入 `exp()`。  
修复：使用 `torch.nn.functional.log_softmax()`，其内部实现了 log-sum-exp。

**错误：从 float32 改为 float16 后训练发散。**  
原因：float16 无法表示小于 6e-8 的梯度幅度或大于 65,504 的激活值。  
修复：使用带损失缩放（AMP）的混合精度，或改用 bfloat16。

## 实践

### 步骤 1：展示浮点精度限制

```python
print("=== 浮点精度示例 ===")
print(f"0.1 + 0.2 = {0.1 + 0.2}")
print(f"0.1 + 0.2 == 0.3? {0.1 + 0.2 == 0.3}")
print(f"差值: {(0.1 + 0.2) - 0.3:.2e}")
```

### 步骤 2：实现简单版和稳定版 softmax

```python
import math

def softmax_naive(logits):
    exps = [math.exp(z) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def softmax_stable(logits):
    max_logit = max(logits)
    exps = [math.exp(z - max_logit) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

safe_logits = [2.0, 1.0, 0.1]
print(f"简单版:  {softmax_naive(safe_logits)}")
print(f"稳定版: {softmax_stable(safe_logits)}")

dangerous_logits = [100.0, 101.0, 102.0]
print(f"稳定版: {softmax_stable(dangerous_logits)}")
# softmax_naive(dangerous_logits) 会返回 [nan, nan, nan]
```

### 步骤 3：实现稳定版 log-sum-exp

```python
def logsumexp_naive(values):
    return math.log(sum(math.exp(v) for v in values))

def logsumexp_stable(values):
    c = max(values)
    return c + math.log(sum(math.exp(v - c) for v in values))

safe = [1.0, 2.0, 3.0]
print(f"简单版:  {logsumexp_naive(safe):.6f}")
print(f"稳定版: {logsumexp_stable(safe):.6f}")

large = [500.0, 501.0, 502.0]
print(f"稳定版: {logsumexp_stable(large):.6f}")
# logsumexp_naive(large) 返回 inf
```

### 步骤 4：实现稳定版交叉熵

```python
def cross_entropy_naive(true_class, logits):
    probs = softmax_naive(logits)
    return -math.log(probs[true_class])

def cross_entropy_stable(true_class, logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = math.log(sum(math.exp(s) for s in shifted))
    log_prob = shifted[true_class] - log_sum_exp
    return -log_prob

logits = [2.0, 5.0, 1.0]
true_class = 1
print(f"简单版:  {cross_entropy_naive(true_class, logits):.6f}")
print(f"稳定版: {cross_entropy_stable(true_class, logits):.6f}")
```

### 步骤 5：梯度检查

```python
def numerical_gradient(f, x, h=1e-5):
    grad = []
    for i in range(len(x)):
        x_plus = x[:]
        x_minus = x[:]
        x_plus[i] += h
        x_minus[i] -= h
        grad.append((f(x_plus) - f(x_minus)) / (2 * h))
    return grad

def check_gradient(analytical, numerical, tolerance=1e-5):
    for i, (a, n) in enumerate(zip(analytical, numerical)):
        denom = max(abs(a), abs(n), 1e-8)
        rel_error = abs(a - n) / denom
        status = "OK" if rel_error < tolerance else "FAIL"
        print(f"  参数 {i}: analytical={a:.8f} numerical={n:.8f} "
              f"相对误差={rel_error:.2e} [{status}]")

def f(params):
    x, y = params
    return x**2 + 3*x*y + y**3

def f_grad(params):
    x, y = params
    return [2*x + 3*y, 3*x + 3*y**2]

point = [2.0, 1.0]
analytical = f_grad(point)
numerical = numerical_gradient(f, point)
check_gradient(analytical, numerical)
```

## 使用示例

### 混合精度模拟

```python
import struct

def float32_to_float16_round(x):
    packed = struct.pack('f', x)
    f32 = struct.unpack('f', packed)[0]
    packed16 = struct.pack('e', f32)
    return struct.unpack('e', packed16)[0]

def simulate_bfloat16(x):
    packed = struct.pack('f', x)
    as_int = int.from_bytes(packed, 'little')
    truncated = as_int & 0xFFFF0000
    repacked = truncated.to_bytes(4, 'little')
    return struct.unpack('f', repacked)[0]
```

### 梯度裁剪

```python
def clip_by_norm(gradients, max_norm):
    total_norm = math.sqrt(sum(g**2 for g in gradients))
    if total_norm > max_norm:
        scale = max_norm / total_norm
        return [g * scale for g in gradients]
    return gradients

grads = [10.0, 20.0, 30.0]
clipped = clip_by_norm(grads, max_norm=5.0)
print(f"原始范数: {math.sqrt(sum(g**2 for g in grads)):.2f}")
print(f"裁剪后范数:  {math.sqrt(sum(g**2 for g in clipped)):.2f}")
print(f"方向保持：{[c/clipped[0] for c in clipped]} == {[g/grads[0] for g in grads]}")
```

### NaN/Inf 检测

```python
def check_tensor(name, values):
    has_nan = any(math.isnan(v) for v in values)
    has_inf = any(math.isinf(v) for v in values)
    if has_nan or has_inf:
        print(f"警告 {name}: nan={has_nan} inf={has_inf}")
        return False
    return True

check_tensor("正常", [1.0, 2.0, 3.0])
check_tensor("异常",  [1.0, float('nan'), 3.0])
check_tensor("极端", [1.0, float('inf'), 3.0])
```

完整实现及所有边界情况示例详见 `code/numerical.py`。

## 交付成果

本课产出：

- `code/numerical.py`，包含稳定版 softmax、log-sum-exp、交叉熵、梯度检查和混合精度模拟  
- `outputs/prompt-numerical-debugger.md`，用于诊断训练中的 NaN/Inf 与数值问题  

这些稳定实现会在第三阶段构建训练循环时和第四阶段实现注意力机制时再次使用。

## 练习

1. **灾难性抵消（Catastrophic cancellation）。** 使用浮点数 float32 计算 [1000000.0, 1000001.0, 1000002.0] 的方差，先用简单公式 `E[x^2] - E[x]^2`，后用 Welford 在线算法。对比两者与真实方差（0.6667）的误差。

2. **精度极限。** 找出 Python 中最小正 float32 值 `x`，使得 `1.0 + x == 1.0`。即机器 epsilon。验证是否与 `numpy.finfo(numpy.float32).eps` 匹配。

3. **log-sum-exp 边界测试。** 测试稳定版 `logsumexp_stable` 函数在（a）所有值相等，（b）一个值远大于其它，（c）所有值非常小（-1000）时，验证其正确性和简单版失败的情况。

4. **梯度检查神经网络层。** 实现单线性层 `y = Wx + b` 及其解析反向函数。用 `numerical_gradient` 验证 3x2 权重矩阵的正确性。

5. **损失缩放实验。** 模拟 float16 训练：生成随机梯度范围 [1e-9, 1e-3]，转成 float16，统计为零的比例。再进行损失缩放（乘以1024），转换，缩放还原并检测零比例变化。

## 关键词汇

| 术语             | 常见说法           | 实际含义                                               |
|------------------|--------------------|--------------------------------------------------------|
| IEEE 754         | “浮点标准”         | 定义二进制浮点格式、舍入规则和特殊值（inf, nan）的国际标准。现代 CPU 和 GPU 都实现它。 |
| Machine epsilon  | “精度极限”         | 给定浮点格式下，最小的使 1.0 + e ≠ 1.0 的值。float32 大约为 1.19e-7。           |
| Catastrophic cancellation | “减法导致精度丢失” | 两个非常接近的浮点数相减导致有效数字抵消，舍入误差占主导。                             |
| Overflow         | “数字太大溢出”     | 结果超过可表示最大值变为 inf。exp(89) 会溢出 float32。                            |
| Underflow        | “数字太小下溢”     | 结果小于正的最小可表示数字变为 0.0。exp(-104) 会下溢 float32。                     |
| Log-sum-exp trick| “先减最大值”       | 计算 log(sum(exp(x))) 时先提取 max(x)，防止溢出和下溢。用于 softmax、交叉熵和对数概率计算。 |
| Stable softmax   | “稳定的 softmax”   | 先减去最大 logits 以避免指数运算溢出，结果数值等价且稳定。                           |
| Gradient checking| “验证反向传播梯度” | 用数值微分对比反向传播得到的解析梯度，捕捉实现错误。                                   |
| Mixed precision  | “前向用 float16，反向用 float32” | 重要运算用低精度浮点加速，敏感计算用高精度浮点。加速通常是 2-3 倍。                   |
| Loss scaling     | “防止梯度下溢”     | 反向传播前将损失乘以大数保证梯度在可表示范围，更新时除回去。                           |
| bfloat16         | “Brain 浮点格式”   | Google 的 16 位浮点格式，8 位指数（和 float32 范围相同），7 位尾数（比 float16 精度低），训练首选。 |
| Gradient clipping| “限制梯度范数”     | 将梯度向量缩放至不超过指定范数，防止梯度爆炸破坏权重。                                 |
| NaN              | “非数值”           | 由未定义运算产生的特殊浮点值（0/0, inf-inf, sqrt(-1)），在后续运算中传播。            |
| Inf              | “无穷大”           | 溢出或除零产生的特殊浮点值，可导致 NaN（inf - inf, inf * 0）。                        |
| Numerical gradient | “数值微分”         | 通过计算 f(x+h) 和 f(x-h) 的差除以 2h 逼近导数。速度慢但可靠用于验证。                |

## 深入阅读

- [每个计算机科学家都应了解的浮点数算术（Goldberg 1991）](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html) -- 权威参考，内容密集但完整
- [混合精度训练（Micikevicius 等，2018）](https://arxiv.org/abs/1710.03740) -- NVIDIA 论文，介绍了用于 float16 训练的损失缩放
- [AMP：自动混合精度（PyTorch 文档）](https://pytorch.org/docs/stable/amp.html) -- PyTorch 中混合精度的实用指南
- [bfloat16 格式（Google Cloud TPU 文档）](https://cloud.google.com/tpu/docs/bfloat16) -- Google 选择该格式用于 TPU 的原因
- [Kahan 求和算法（维基百科）](https://en.wikipedia.org/wiki/Kahan_summation_algorithm) -- 用于减少浮点求和中的舍入误差的算法
