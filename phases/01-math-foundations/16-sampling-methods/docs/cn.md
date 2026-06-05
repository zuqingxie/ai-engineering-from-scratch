# 采样方法

> 采样是 AI 探索可能性空间的方式。

**类型：** 构建  
**语言：** Python  
**先决条件：** 第一阶段，课程 06-07（概率，贝叶斯定理）  
**时长：** 约 120 分钟

## 学习目标

- 使用仅有的均匀随机数，从零实现逆 CDF、拒绝采样和重要性采样  
- 构建语言模型令牌生成的 temperature（温度采样）、top-k 和 top-p（nucleus，核采样）采样  
- 解释 reparameterization trick（重参数化技巧）及其为何使 VAE 中的采样支持反向传播  
- 运行 Metropolis-Hastings MCMC（马尔科夫链蒙特卡罗）从未归一化的目标分布采样

## 问题描述

一个语言模型处理完你的提示后，生成了一个长度为 50,000 的 logits 向量。每个对应词汇表中的一个词元。现在它得选一个。怎么选？

如果总是选概率最高的词元，所有回应都一样。确定性。无聊。如果均匀随机选，输出就是胡言乱语。答案介于这两个极端之间，而采样就是控制这个“中间地带”的方法。

采样不仅限于文本生成。强化学习通过采样轨迹估计策略梯度。VAE 通过从学到的分布采样并反向传播随机性，学习潜变量表示。扩散模型通过采样噪声并多次去噪生成图像。蒙特卡洛方法估计无解析解的积分。MCMC 算法在不可枚举的高维后验分布中探索。

每个生成式 AI 系统都是采样系统。采样策略决定输出的质量、多样性和可控性。本课从零构建所有主要采样方法，从均匀随机数开始，到驱动现代大语言模型和生成式模型的技术。

## 基本概念

### 为什么采样重要

采样在 AI 和机器学习中承担四个基本角色：

**生成。** 语言模型、扩散模型和 GAN 都通过采样产生输出。采样算法直接控制创造力、一致性和多样性。温度、top-k 和核采样是工程师每天调节的旋钮。

**训练。** 随机梯度下降采样小批量。Dropout 采样神经元以使其失活。数据增强采样随机变换。重要性采样在强化学习（如 PPO、TRPO）中重加权样本以减少梯度方差。

**估计。** 许多机器学习量没有解析解。数据分布上的期望损失、基于能量模型的配分函数、贝叶斯推断中的证据。蒙特卡洛估计通过对样本求平均近似这些量。

**探索。** MCMC 算法探索贝叶斯推断的后验分布。进化策略采样参数扰动。Thompson 采样在多臂赌博机问题中平衡探索与利用。

核心挑战：你只能直接从简单分布（均匀、高斯）采样。对于其他分布，你需要方法把简单采样转换到目标分布的采样。

### 均匀随机采样

所有采样方法都从这里开始。均匀随机数生成器输出 [0, 1) 内的值，每个等长子区间概率相等。

```text
U ~ Uniform(0, 1)

P(a <= U <= b) = b - a    其中 0 <= a <= b <= 1

性质：
  E[U] = 0.5
  Var(U) = 1/12
```

从 n 个离散项均匀采样，生成 U 并返回 floor(n * U)。从连续区间 [a, b] 采样，计算 a + (b - a) * U。

关键洞见：一个单独的均匀随机数蕴含着恰好足够的随机性来生成任意分布的一个样本。难点是找到合适的变换。

### 逆 CDF 方法（逆变换采样）

累积分布函数 CDF 把数值映射到概率：

```text
F(x) = P(X <= x)

性质：
  F 非减
  F(-∞) = 0
  F(+∞) = 1
  F 将实数映射到 [0, 1]
```

逆 CDF 把概率映射回数值。如果 U ~ Uniform(0, 1)，则 X = F_inverse(U) 服从目标分布。

```text
算法：
  1. 生成 u ~ Uniform(0, 1)
  2. 返回 F_inverse(u)

原理：
  P(X <= x) = P(F_inverse(U) <= x) = P(U <= F(x)) = F(x)
```

**指数分布示例：**

```text
PDF: f(x) = lambda * exp(-lambda * x)，x >= 0
CDF: F(x) = 1 - exp(-lambda * x)

解方程 F(x) = u 得到 x:
  u = 1 - exp(-lambda * x)
  exp(-lambda * x) = 1 - u
  x = -ln(1 - u) / lambda

因为 (1 - U) 和 U 分布相同：
  x = -ln(u) / lambda
```

当你能写出闭式的 F_inverse 时，此法完美适用。正态分布无闭式逆 CDF，只能用其他方法（Box-Muller 变换，或者数值逼近）。

**离散版：** 离散分布构建累积分布，生成 U，找到第一个使累积和超过 U 的索引。此法等同于第 06 课的 `sample_categorical`。

### 拒绝采样

当不能逆 CDF，但能计算目标 PDF（即使未归一化），可以用拒绝采样。

```text
目标分布：p(x)  （可计算，可能未归一化）
提议分布：q(x)  （可采样）
界限 M，保证 p(x) <= M * q(x) 对所有 x 成立

算法：
  1. 采样 x ~ q(x)
  2. 采样 u ~ Uniform(0, 1)
  3. 若 u < p(x) / (M * q(x))，接受 x
  4. 否则拒绝，重复步骤 1

接受率 = 1 / M
```

界限 M 越紧，接受率越高。低维度（1-3）拒绝采样表现良好。高维度时接受率指数级下降，绝大多数提议被拒，称为拒绝采样的维度灾难。

**示例：从截断正态分布采样。** 在截断区间上使用均匀提议。边界 M 是该范围内正态概率密度函数的最大值。

**示例：从半圆采样。** 在包围半圆的矩形中均匀采样。点在半圆内则接受。蒙特卡洛估算 π 就用此法：接受率等于面积比 π/4。

### 重要性采样

有时你不需要从目标分布 p(x) 采样，而是估计 p(x) 下的期望，却只能从另一个分布 q(x) 采样。

```text
目标：估计 E_p[f(x)] = ∫ f(x) p(x) dx

变形：
  E_p[f(x)] = ∫ f(x) * (p(x)/q(x)) * q(x) dx
            = E_q[f(x) * w(x)]

权重 w(x) = p(x) / q(x) 称为重要性权重

估计器：
  E_p[f(x)] ≈ (1/N) * Σ [f(x_i) * w(x_i)]   其中 x_i ~ q(x)
```

这在强化学习中尤为关键。在 PPO（近端策略优化）中，你用旧策略 π_old 收集轨迹，但想优化新策略 π_new。权重是 π_new(a|s) / π_old(a|s)，PPO 会剪辑此权重以防新旧策略偏离过大。

重要性采样估计的方差取决于 q 与 p 的相似度。若两者相差甚远，少数样本权重极大，主导估计。自归一化重要性采样通过除以权重和来减缓此问题：

```text
E_p[f(x)] ≈ Σ (w_i * f(x_i)) / Σ w_i
```

### 蒙特卡洛估计

蒙特卡洛估计通过样本平均逼近积分。大数定律保证收敛。

```text
目标：估计 I = ∫_D g(x) dx

方法：
  1. 从 D 均匀采样 x_1, ..., x_N
  2. I ≈ (D 的体积 / N) * Σ g(x_i)

误差：O(1/√N) 与维度无关
```

误差率不依赖维度，这就是蒙特卡洛方法在高维中优于格点积分的原因。

**估计 π：**

```text
均匀采样 (x, y) ∈ [-1, 1] x [-1, 1]
计算落入单位圆内的点的数量：x^2 + y^2 ≤ 1
π ≈ 4 * （圆内点数 / 总点数）
```

**估计期望：**

```text
E[f(X)] ≈ (1/N) * Σ f(x_i)    其中 x_i ~ p(x)

样本均值收敛到真实期望
估计器方差 = Var(f(X)) / N
```

### 马尔科夫链蒙特卡洛（MCMC）：Metropolis-Hastings 方法

MCMC 构造一个马尔科夫链，其平稳分布是目标分布 p(x)。足够长时间后链上的样本近似服从 p(x)。

```text
目标分布：p(x)  （知晓至归一化常数）
提议分布：q(x'|x)  （给定当前状态提议下一个状态）

Metropolis-Hastings 算法：
  1. 从某个 x_0 开始
  2. 对 t = 1, 2, ..., T：
     a. 提议 x' ~ q(x'|x_t)
     b. 计算接受率：
        alpha = [p(x') * q(x_t|x')] / [p(x_t) * q(x'|x_t)]
     c. 以概率 min(1, alpha) 接受：
        - 若 u < alpha (u ~ Uniform(0,1))，则 x_{t+1} = x'
        - 否则 x_{t+1} = x_t
  3. 舍弃前 B 个样本（烧入期）
  4. 返回剩余样本
```

若提议分布对称（q(x'|x) = q(x|x')），接受率简化为 p(x')/p(x)，即原始 Metropolis 算法。

**为何有效。** 接受规则保证详细平衡：从 x 到 x' 的概率等于从 x' 到 x 的概率。详细平衡保证 p(x) 是链的平稳分布。

**实用注意事项：**  
- 烧入期：舍弃链未达到平稳前的样本  
- 抽稀：每隔 k 个样本保留一个，减少自相关  
- 提议尺度：尺度太小，链移动慢（高接受率，探索慢）；尺度太大，大多数提议被拒（低接受率，陷入原地）  
- 高维中高斯提议的最优接受率约为 0.234

### Gibbs 采样

Gibbs 采样是多变量 MCMC 的特殊情况。不是同时移动所有维度，而是每次更新一个变量，从它的条件分布采样。

```text
目标分布：p(x_1, x_2, ..., x_d)

算法：
  对每次迭代 t：
    采样 x_1^{t+1} ~ p(x_1 | x_2^t, x_3^t, ..., x_d^t)
    采样 x_2^{t+1} ~ p(x_2 | x_1^{t+1}, x_3^t, ..., x_d^t)
    ...
    采样 x_d^{t+1} ~ p(x_d | x_1^{t+1}, x_2^{t+1}, ..., x_{d-1}^{t+1})
```

Gibbs 采样要求能从每个条件分布 p(x_i | x_{-i}) 采样。许多模型中此条件分布易得：

- 贝叶斯网络：条件分布来源于图结构
- 高斯混合模型：条件分布仍是高斯
- Ising 模型：每个自旋条件只依赖邻居

接受率始终为 1（每个采样自动接受），因为从精确条件采样满足详细平衡。

**限制。** 当变量高度相关时，Gibbs 混合慢，因为一次只更新一个变量，无法沿对角线方向大步移动。

### 温度采样（用于大语言模型）

语言模型输出词汇表中每个词元的 logits z_1, ..., z_V。Softmax 把它们转换为概率。温度参数先对 logits 进行缩放：

```text
p_i = exp(z_i / T) / sum(exp(z_j / T))

T = 1.0: 标准 softmax（原始分布）
T -> 0:  argmax（确定性，总是选择最高的 logit）
T -> inf: 均匀分布（所有token等概率）
T < 1.0: 锐化分布（更自信，较少多样性）
T > 1.0: 平滑分布（不那么自信，更加多样）
```

**为什么有效。** 将 logits 除以 T < 1 会放大 logits 之间的差异。如果 z_1 = 2，z_2 = 1，除以 T = 0.5 后，z_1/T = 4，z_2/T = 2，使得差距更大。经过 softmax 后，最高 logit 的 token 获得更大的概率份额。

**实际应用：**
- T = 0.0：贪婪解码，适合事实问答
- T = 0.3-0.7：略带创造性，适合代码生成
- T = 0.7-1.0：平衡，适合一般对话
- T = 1.0-1.5：创造性写作，头脑风暴
- T > 1.5：越来越随机，较少有用

温度不会改变可能产生的token，只改变分配给每个token的概率质量。

### Top-k 采样

Top-k 采样限制候选集为 k 个具有最高概率的 token，然后重新归一化并从限制集合中采样。

```text
算法：
  1. 计算所有 V 个 token 的 softmax 概率
  2. 按概率降序排序 token
  3. 仅保留前 k 个 token
  4. 重新归一化：p_i' = p_i / sum(p_j for j in top-k)
  5. 从重新归一化的分布中采样

k = 1：贪婪解码
k = V：无过滤（标准采样）
k = 40：典型设置，去除概率极低的长尾 token
```

Top-k 防止模型选择极不可能的 token（拼写错误、无意义词），这些通常存在于词汇概率分布的长尾中。问题是：k 是固定的，不考虑上下文。当模型非常自信（一个 token 有 95% 概率）时，k=40 仍允许有 39 个备选。当模型不确定（概率分布在 1000 个 token 上）时，k=40 会截断很多合理选项。

### Top-p（核采样）采样

Top-p 采样动态调整候选集合大小。它不是保留固定数量的 token，而是保留概率累计超过 p 的最小 token 集合。

```text
算法：
  1. 计算所有 V 个 token 的 softmax 概率
  2. 按概率降序排序 token
  3. 找到最小的 k 使得前 k 个 token 的概率和 >= p
  4. 只保留这 k 个 token
  5. 归一化并采样

p = 0.9：保留覆盖 90% 概率质量的 token
p = 1.0：无过滤
p = 0.1：非常严格，几乎是贪婪
```

当模型自信时，核采样保留的 token 很少（可能 2-3 个），当模型不确定时，保留很多（可能 200 个）。这种自适应行为是核采样优于 top-k 的原因。

**常用组合：**
- 温度 0.7 + top-p 0.9：通用良好设置
- 温度 0.0（贪婪）：确定性任务最佳
- 温度 1.0 + top-k 50：Fan 等人（2018）论文原始设置

Top-k 和 top-p 可以结合使用。先执行 top-k，再对剩余集合应用 top-p。

### 重参数技巧（用于变分自编码器）

变分自编码器（VAEs）通过编码输入为潜在空间中的分布，从该分布中采样，并解码样本回输入。问题是：采样操作不可微，无法反向传播。

```text
标准采样（不可微）：
  z ~ N(mu, sigma^2)

  随机性阻断了梯度流动。
  d/d_mu [从 N(mu, sigma^2) 采样] = ???
```

重参数化技巧把随机性和参数分开：

```text
重参数采样：
  epsilon ~ N(0, 1)          （固定随机噪声，无参数）
  z = mu + sigma * epsilon   （参数的确定性函数）

  现在 z 是 mu 和 sigma 的确定性、可微函数。
  d(z)/d(mu) = 1
  d(z)/d(sigma) = epsilon

  梯度可以通过 mu 和 sigma 流动。
```

这是因为 N(mu, sigma^2) 的分布等价于 mu + sigma * N(0, 1)。关键见解：将随机性移动到不依赖参数的来源（epsilon），再通过可微变换得到样本。

**VAE 训练循环中：**
1. 编码器输出每个输入的 mu 和 log(sigma^2)
2. 采样 epsilon ~ N(0, 1)
3. 计算 z = mu + sigma * epsilon
4. 解码 z 以重构输入
5. 通过步骤 4, 3, 2, 1 反向传播（可行因为步骤3可微）

没有重参数化技巧，VAEs 不能用标准反向传播训练。这一见解使 VAEs 实用。

### Gumbel-Softmax（可微的分类采样）

重参数化技巧适用于连续分布（高斯）。对于离散分类分布，需要不同方法。Gumbel-Softmax 提供一种对分类采样的可微近似。

**Gumbel-Max 技巧（不可微）：**

```text
从带对数概率 log(p_1), ..., log(p_k) 的类别分布采样：
  1. 对每个类别采样 g_i ~ Gumbel(0, 1)
     （g = -log(-log(u))，其中 u ~ Uniform(0, 1)）
  2. 返回 argmax(log(p_i) + g_i)

这得到精确的类别采样。
```

**Gumbel-Softmax（可微近似）：**

```text
用软 max 替代硬 argmax：
  y_i = exp((log(p_i) + g_i) / tau) / sum(exp((log(p_j) + g_j) / tau))

tau（温度）控制近似：
  tau -> 0: 接近 one-hot 向量（硬分类）
  tau -> inf: 接近均匀分布 (1/k, ..., 1/k)
  tau = 1.0: 软近似
```

Gumbel-Softmax 生成离散样本的连续放松。输出是概率向量（软 one-hot），而非硬 one-hot。梯度通过 softmax 流动。训练中可用“直通（straight-through）”估计器：前向用硬 argmax，反向用软 Gumbel-Softmax 的梯度。

**应用：**
- VAE 中的离散潜变量
- 神经架构搜索（选择离散操作）
- 硬注意力机制
- 离散动作的强化学习

### 分层采样

标准蒙特卡洛采样可能偶然在样本空间留下空白。分层采样通过将空间划分为若干层级，并从每层级采样，强制均匀覆盖。

```text
标准蒙特卡洛：
  从 [0, 1] 均匀采样 N 点
  可能出现区域聚集，区域空缺

分层采样：
  将 [0, 1] 均分成 N 个层级：[0, 1/N), [1/N, 2/N), ..., [(N-1)/N, 1)
  每层级内均匀采样一个点
  x_i = (i + u_i) / N   其中 u_i ~ Uniform(0, 1), i = 0, ..., N-1
```

分层采样的方差总是小于或等于标准蒙特卡洛：

```text
Var(分层) <= Var(标准蒙特卡洛)

当 f(x) 平滑时改进最大。
对分段常数函数，分层采样是精确的。
```

**应用：**
- 数值积分（准蒙特卡洛）
- 训练数据划分（保证每折中类别平衡）
- 带分层的重采样（结合两种技术）
- NeRF（神经辐射场）沿摄像机光线使用分层采样

### 扩散模型的关联

扩散模型通过采样过程生成图像。前向过程向图像添加高斯噪声，经过 T 步使其变为纯噪声。反向过程学习去噪，逐步恢复原始图像。

```text
前向过程（已知）：
  x_t = sqrt(alpha_t) * x_{t-1} + sqrt(1 - alpha_t) * epsilon
  其中 epsilon ~ N(0, I)

  T 步后: x_T ~ N(0, I) （纯噪声）

反向过程（学习得到）：
  x_{t-1} = (1/sqrt(alpha_t)) * (x_t - (1 - alpha_t)/sqrt(1 - alpha_bar_t) * epsilon_theta(x_t, t)) + sigma_t * z
  其中 z ~ N(0, I)

  每一步去噪即一次采样。
```

与本课方法的联系：
- 每一步去噪使用重参数技巧（采样噪声，应用确定性变换）
- 噪声调度 {alpha_t} 控制温度退火形式
- 训练通过蒙特卡洛估计近似 ELBO（证据下界）
- 扩散模型的祖先采样是马尔科夫链（每步仅依赖当前状态）

整个图像生成过程是迭代采样：从噪声开始，每步采样稍微去噪的版本，条件是学习到的去噪模型。

## 动手实现

### 第1步：均匀和逆CDF采样

```python
import math
import random

def sample_uniform(a, b):
    return a + (b - a) * random.random()

def sample_exponential_inverse_cdf(lam):
    u = random.random()
    return -math.log(u) / lam
```

生成 10,000 个指数分布样本，验证均值为 1/lambda。

### 第2步：拒绝采样

```python
def rejection_sample(target_pdf, proposal_sample, proposal_pdf, M):
    while True:
        x = proposal_sample()
        u = random.random()
        if u < target_pdf(x) / (M * proposal_pdf(x)):
            return x
```

使用拒绝采样从截断正态分布抽样。通过直方图验证样本形状。

### 第3步：重要性采样

```python
def importance_sampling_estimate(f, target_pdf, proposal_pdf, proposal_sample, n):
    total = 0
    for _ in range(n):
        x = proposal_sample()
        w = target_pdf(x) / proposal_pdf(x)
        total += f(x) * w
    return total / n
```

用均匀分布作为 proposal，估计正态分布下 E[X^2]。与已知结果 (mu^2 + sigma^2) 比较。

### 第4步：蒙特卡洛估计 π

```python
def monte_carlo_pi(n):
    inside = 0
    for _ in range(n):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        if x*x + y*y <= 1:
            inside += 1
    return 4 * inside / n
```

### 第5步：Metropolis-Hastings MCMC

```python
def metropolis_hastings(target_log_pdf, proposal_sample, proposal_log_pdf, x0, n_samples, burn_in):
    samples = []
    x = x0
    for i in range(n_samples + burn_in):
        x_new = proposal_sample(x)
        log_alpha = (target_log_pdf(x_new) + proposal_log_pdf(x, x_new)
                     - target_log_pdf(x) - proposal_log_pdf(x_new, x))
        if math.log(random.random()) < log_alpha:
            x = x_new
        if i >= burn_in:
            samples.append(x)
    return samples
```

从双峰分布（两个高斯混合）采样。可视化链的轨迹。

### 第6步：Gibbs采样

```python
def gibbs_sampling_2d(conditional_x_given_y, conditional_y_given_x, x0, y0, n_samples, burn_in):
    x, y = x0, y0
    samples = []
    for i in range(n_samples + burn_in):
        x = conditional_x_given_y(y)
        y = conditional_y_given_x(x)
        if i >= burn_in:
            samples.append((x, y))
    return samples
```

### 第7步：温度采样

```python
def softmax(logits):
    max_l = max(logits)
    exps = [math.exp(z - max_l) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def temperature_sample(logits, temperature):
    scaled = [z / temperature for z in logits]
    probs = softmax(scaled)
    return sample_from_probs(probs)
```

展示温度（temperature）如何改变一组 token logits 的输出分布。

### 第8步：Top-k 和 top-p 采样

```python
def top_k_sample(logits, k):
    indexed = sorted(enumerate(logits), key=lambda x: -x[1])
    top = indexed[:k]
    top_logits = [l for _, l in top]
    probs = softmax(top_logits)
    idx = sample_from_probs(probs)
    return top[idx][0]

def top_p_sample(logits, p):
    probs = softmax(logits)
    indexed = sorted(enumerate(probs), key=lambda x: -x[1])
    cumsum = 0
    selected = []
    for token_idx, prob in indexed:
        cumsum += prob
        selected.append((token_idx, prob))
        if cumsum >= p:
            break
    sel_probs = [pr for _, pr in selected]
    total = sum(sel_probs)
    sel_probs = [pr / total for pr in sel_probs]
    idx = sample_from_probs(sel_probs)
    return selected[idx][0]
```

### 第9步：重参数技巧（reparameterization trick）

```python
def reparam_sample(mu, sigma):
    epsilon = random.gauss(0, 1)
    return mu + sigma * epsilon

def reparam_gradient(mu, sigma, epsilon):
    dz_dmu = 1.0
    dz_dsigma = epsilon
    return dz_dmu, dz_dsigma
```

演示梯度如何通过重参数采样流动，而不会通过直接采样流动。

### 第10步：Gumbel-Softmax

```python
def gumbel_sample():
    u = random.random()
    return -math.log(-math.log(u))

def gumbel_softmax(logits, temperature):
    gumbels = [math.log(p) + gumbel_sample() for p in logits]
    return softmax([g / temperature for g in gumbels])
```

展示降低温度如何使输出趋近于 one-hot 向量。

完整实现及所有可视化在 `code/sampling.py` 中。

## 使用方法

使用 NumPy 和 SciPy 的生产版本：

```python
import numpy as np

rng = np.random.default_rng(42)

exponential_samples = rng.exponential(scale=2.0, size=10000)
print(f"指数分布均值：{exponential_samples.mean():.4f}（预期为 2.0）")

from scipy import stats
normal = stats.norm(loc=0, scale=1)
print(f"在 1.96 处的累计分布函数值(CDF)：{normal.cdf(1.96):.4f}")
print(f"0.975 处的逆CDF值：{normal.ppf(0.975):.4f}")

logits = np.array([2.0, 1.0, 0.5, 0.1, -1.0])
temperature = 0.7
scaled = logits / temperature
probs = np.exp(scaled - scaled.max()) / np.exp(scaled - scaled.max()).sum()
token = rng.choice(len(logits), p=probs)
print(f"采样得到的 token 索引：{token}")
```

要大规模使用 MCMC，请使用专门库：
- PyMC：带有 NUTS（自适应 HMC）的全贝叶斯建模
- emcee：集合 MCMC 采样器
- NumPyro/JAX：GPU 加速的 MCMC

这些方法你都是从零构建的。现在你知道库函数背后在做什么了。

## 练习

1. 实现柯西分布的逆CDF采样。其累积分布函数（CDF）为 F(x) = 0.5 + arctan(x)/π。生成 10,000 个样本并绘制直方图与真实概率密度函数（PDF）对比。注意其厚尾（远离中心的极端值）。

2. 使用拒绝采样（rejection sampling）从 Beta(2, 5) 分布生成样本，提议分布为 Uniform(0, 1)。绘制接受样本与真实 Beta PDF。理论接受率是多少？

3. 用蒙特卡洛（Monte Carlo）方法估计积分 ∫₀^π sin(x) dx，采样数分别为 1,000、10,000 和 100,000。比较各层次误差，验证误差按 O(1/√N) 缩放。

4. 实现 Metropolis-Hastings，从二维分布 p(x, y) ∝ exp(-(x²y² + x² + y² - 8x - 8y)/2) 采样。绘制样本和链的轨迹。尝试不同的提议标准差。

5. 构建完整的文本生成演示：给定包含 10 个单词的词汇表及 logits，生成长度为 20 个 token 的序列，分别使用 (a) 贪心，(b) temperature=0.7，(c) top-k=3，(d) top-p=0.9。比较 5 次运行输出的多样性。

## 关键词

| 术语 | 俗语 | 实际含义 |
|------|----------------|----------------------|
| Sampling（采样） | "抽取随机值" | 根据概率分布生成值。所有生成式 AI 的机制基础 |
| Uniform distribution（均匀分布） | "都一样可能" | [a,b] 区间内的每个值概率密度均为 1/(b-a)。所有采样方法的起点 |
| Inverse CDF（逆累积分布函数） | "概率变换" | F_inverse(U) 将均匀样本变为任意已知 CDF 的分布样本。准确且高效 |
| Rejection sampling（拒绝采样） | "提出并接受/拒绝" | 从简单提议分布生成样本，按目标/提议比率概率接受。准确但会浪费样本 |
| Importance sampling（重要性采样） | "重加权样本" | 用 q(x) 采样，通过 p(x)/q(x) 权重估计 p 下的期望。强化学习中 PPO 核心 |
| Monte Carlo（蒙特卡洛） | "随机样本的平均" | 将积分近似为样本均值。误差为 O(1/√N)，与维度无关 |
| MCMC（马尔可夫链蒙特卡洛） | "收敛的随机游走" | 构建马尔可夫链，其平稳分布即目标分布。Metropolis-Hastings 是基础算法 |
| Metropolis-Hastings | "接受上坡，有时接受下坡" | 提议移动，按密度比接受。详细平衡保证收敛到目标分布 |
| Gibbs sampling | "一次一个变量" | 固定其他变量，按条件分布更新每个变量。100% 接受率 |
| Temperature（温度） | "置信度调节旋钮" | 在 softmax 前除以 T。T<1 使分布更尖锐（更自信），T>1 使分布平缓（更多样） |
| Top-k sampling | "保留前 k 个" | 把除概率最高 k 个 token 外清零，重新归一化采样。候选集大小固定 |
| Nucleus sampling (top-p)（核采样） | "保留高概率集合" | 保留累计概率达到 p 的最小 token 集合。候选集大小自适应 |
| Reparameterization trick（重参数技巧） | "把随机性移出" | 写成 z = mu + sigma * epsilon，epsilon ~ N(0,1)。使采样可微分，VAE 训练必需 |
| Gumbel-Softmax | "软分类采样" | 用 Gumbel 噪声 + 温度 softmax 近似可微分类采样 |
| Stratified sampling（分层采样） | "强制覆盖" | 将采样空间分层，从每层采样。方差总比普通蒙特卡洛低 |
| Burn-in（预热期） | "热身阶段" | MCMC 初始样本丢弃，直到链达到平稳分布 |
| Detailed balance（详细平衡） | "可逆条件" | p(x) * T(x→y) = p(y) * T(y→x)，确保 p 是马尔可夫链的平稳分布的充分条件 |
| Diffusion sampling（扩散采样） | "迭代去噪" | 从噪声开始，通过学习的去噪步骤生成数据，每步为条件采样操作 |

## 深入阅读

- [Holbrook (2023)：The Metropolis-Hastings Algorithm](https://arxiv.org/abs/2304.07010) - MCMC 基础详尽教程
- [Jang, Gu, Poole (2017)：Categorical Reparameterization with Gumbel-Softmax](https://arxiv.org/abs/1611.01144) - Gumbel-Softmax 原始论文
- [Holtzman et al. (2020)：The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751) - 核采样（top-p）论文
- [Kingma & Welling (2014)：Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) - 介绍重参数技巧的 VAE 论文
- [Ho, Jain, Abbeel (2020)：Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) - DDPM 将采样与图像生成连接起来
