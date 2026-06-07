# JAX 入门

> PyTorch 对张量进行变异。TensorFlow 构建计算图。JAX 编译纯函数。最后这一点改变了你对深度学习的思考方式。

**类型：** 构建  
**语言：** Python  
**先决条件：** 第 03 阶段第 01-10 课，基础 NumPy  
**时间：** 约 90 分钟

## 学习目标

- 使用 JAX 的函数式 API（jax.numpy、jax.grad、jax.jit、jax.vmap）编写纯函数神经网络代码
- 解释 PyTorch 的即时变异（eager mutation）与 JAX 的函数式编译模型之间的关键设计差异
- 应用 jit 编译和 vmap 向量化技术，加速训练循环，相较于普通 Python 实现
- 使用 JAX 训练简单网络，并对比显式状态管理与 PyTorch 面向对象方法的区别

## 问题描述

你知道如何用 PyTorch 构建神经网络。你定义一个 `nn.Module`，调用 `.backward()`，走一次优化器步伐。这很好用，数百万用户都在用。

但 PyTorch 有个内在限制：它在 Python 中逐个即时跟踪操作。每个 `tensor + tensor` 都是一次独立的内核启动。每一训练步都会重新解释相同的 Python 代码。这在一般场景下没问题，但要在 2048 个 TPU 上训练一个 5400 亿参数的模型时，开销就成了瓶颈。

Google DeepMind 在 JAX 上训练 Gemini。Anthropic 在 JAX 上训练 Claude。这些不是小规模操作——它们是地球上最大规模的神经网络训练。它们选择 JAX，是因为 JAX 把训练循环当成可编译程序，而不是一连串 Python 调用。

JAX 是带有三项超能力的 NumPy：自动微分、JIT 编译到 XLA、自动向量化。你写一个只处理单个样本的函数，JAX 给你一个可以处理批次、计算梯度、编译成本机代码、并在多设备运行的函数。且不需修改原函数。

## 核心概念

### JAX 哲学

JAX 是函数式框架。无类，无可变状态，无 `.backward()` 方法。取而代之：

| PyTorch | JAX |
|---------|-----|
| 带状态的 `nn.Module` 类 | 纯函数：`f(params, x) -> y` |
| `loss.backward()` | `jax.grad(loss_fn)(params, x, y)` |
| 即时执行 | 通过 XLA JIT 编译 |
| `for x in batch:` 手动循环 | `jax.vmap(f)` 自动向量化 |
| `DataParallel` / `FSDP` | `jax.pmap(f)` 自动并行 |
| 可变的 `model.parameters()` | 不可变的数组 pytree |

这不是风格偏好，而是编译器要求。JIT 编译需要纯函数——相同输入必产生相同输出，无副作用。正是此约束带来 100 倍加速。

### jax.numpy：熟悉的表面

JAX 在加速器上重写了 NumPy API：

```python
import jax.numpy as jnp

a = jnp.array([1.0, 2.0, 3.0])
b = jnp.array([4.0, 5.0, 6.0])
c = jnp.dot(a, b)
```

函数名、广播规则、切片语义都相同。但数组驻留于 GPU/TPU，且每个操作均可被编译器追踪。

关键区别：JAX 数组不可变。不能写 `a[0] = 5`。改为：`a = a.at[0].set(5)`。起初感到不适应，但很快理解——不可变性使得像 `grad`、`jit`、`vmap` 这些转换可组合。

### jax.grad：函数式自动微分

PyTorch 将梯度附加到张量（`.grad`）。JAX 将梯度附加到函数。

```python
import jax

def f(x):
    return x ** 2

df = jax.grad(f)
df(3.0)
```

`jax.grad` 接受函数，返回计算梯度的新函数。无需 `.backward()`。无张量上的计算图存储。梯度本身也是函数，可调用、组合或 JIT 编译。

能无限组合：

```python
d2f = jax.grad(jax.grad(f))
d2f(3.0)
```

可求二阶导、三阶导、雅可比矩阵、黑森矩阵。全部通过组合 `grad` 实现。PyTorch 也能（`torch.autograd.functional.hessian`），但更多是附加实现。JAX 中，它是基础。

限定条件：`grad` 只对纯函数有效。追踪时不执行打印操作；不允许变异外部状态；无显式秘钥管理不可使用随机数生成。

### jit：编译到 XLA

```python
@jax.jit
def train_step(params, x, y):
    loss = loss_fn(params, x, y)
    return loss

fast_step = jax.jit(train_step)
```

首次调用时，JAX 追踪函数，记录操作但不执行。随后交给 XLA（Google 针对 TPU 和 GPU 的加速线性代数编译器）。XLA 融合操作，消除冗余内存复制，生成优化的机器码。

后续调用跳过 Python，编译代码以 C++ 速度在加速器上运行。

适用场景：
- 训练步骤（同一计算重复数千次）
- 推理（相同模型，不同输入）
- 对形状类似输入多次调用的函数

不适用：
- 控制流基于追踪数组值的函数（如 `if x > 0`）
- 一次性计算（编译开销大于执行时间）
- 调试（追踪隐藏实际执行）

控制流限制真实存在。`jax.lax.cond` 代替 `if/else`。`jax.lax.scan` 代替 `for` 循环。此乃编译代价。

### vmap：自动向量化

你写一个处理单个样本的函数：

```python
def predict(params, x):
    return jnp.dot(params['w'], x) + params['b']
```

`vmap` 会将其升阶处理批次：

```python
batch_predict = jax.vmap(predict, in_axes=(None, 0))
```

`in_axes=(None, 0)` 意为：不对 `params` 批次化（共享），对 `x` 轴 0 批次化。无需手写循环，无需变形，无需显式传递批次维度。JAX 会自动识别批次维度，完成向量化计算。

这不仅是语法糖。`vmap` 生成融合的向量化代码，比 Python 循环快 10-100 倍。且可与 `jit`、`grad` 组合：

```python
per_example_grads = jax.vmap(jax.grad(loss_fn), in_axes=(None, 0, 0))
```

逐样本梯度，一行搞定。这几乎在 PyTorch 中做不到，除非用非常规手段。

### pmap：设备间数据并行

```python
parallel_step = jax.pmap(train_step, axis_name='devices')
```

`pmap` 将函数复制到所有可用设备（GPU/TPU），拆分批次执行。函数内部，`jax.lax.pmean` 和 `jax.lax.psum` 用于设备间梯度同步。

Google 用 `pmap`（及其继任者 `shard_map`）在数千个 TPU v5e 芯片上训练 Gemini。编程模型：编写单设备版本，包裹 `pmap`，即完成。

### Pytrees：通用数据结构

JAX 操作“pytrees”——列表、元组、字典和数组嵌套组合。你模型的参数就是一个 pytree：

```python
params = {
    'layer1': {'w': jnp.zeros((784, 256)), 'b': jnp.zeros(256)},
    'layer2': {'w': jnp.zeros((256, 128)), 'b': jnp.zeros(128)},
    'layer3': {'w': jnp.zeros((128, 10)),  'b': jnp.zeros(10)},
}
```

每个 JAX 转换——`grad`、`jit`、`vmap`——都知道如何遍历 pytree。`jax.tree.map(f, tree)` 会对每个叶子节点调用 `f`。优化器就是这么一次性更新所有参数的：

```python
params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
```

无 `.parameters()` 方法。无参数注册。树结构就是模型。

### 函数式与面向对象

PyTorch 在对象内存储状态：

```python
class Model(nn.Module):
    def __init__(self):
        self.linear = nn.Linear(784, 10)

    def forward(self, x):
        return self.linear(x)
```

JAX 使用带显式状态的纯函数：

```python
def predict(params, x):
    return jnp.dot(x, params['w']) + params['b']
```

参数显式传入。无存储，无变异。这样每个函数都易于测试、组合和编译。也意味着你需要自己管理参数，或者用像 Flax、Equinox 这样的库。

### JAX 生态系统

JAX 提供原语，库提升易用性：

| 库 | 角色 | 风格 |
|----|------|------|
| **Flax**（Google） | 神经网络层 | 带显式状态的 `nn.Module` |
| **Equinox**（Patrick Kidger） | 神经网络层 | 基于 pytree，Python 风格 |
| **Optax**（DeepMind） | 优化器 + 学习率调度 | 可组合的梯度变换 |
| **Orbax**（Google） | 检查点 | 保存/恢复 pytrees |
| **CLU**（Google） | 指标 + 日志 | 训练循环工具 |

Optax 是标准优化器库。它将梯度变换（Adam、SGD、裁剪）与参数更新分离，易于组合：

```python
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate=1e-3),
)
```

### 何时用 JAX 与 PyTorch

| 因素 | JAX | PyTorch |
|-------|------|---------|
| TPU 支持 | 一流（Google 自研） | 社区维护（torch_xla） |
| GPU 支持 | 良好（通过 XLA 的 CUDA） | 顶级（原生 CUDA 支持） |
| 调试 | 较难（追踪与编译） | 简单（即时执行、逐行） |
| 生态系统 | 聚焦研究（Flax、Equinox） | 规模庞大（HuggingFace、torchvision 等） |
| 招聘 | 小众（Google/DeepMind/Anthropic） | 主流（普遍应用） |
| 大规模训练 | 优秀（XLA、pmap、mesh） | 良好（FSDP、DeepSpeed） |
| 原型开发速度 | 较慢（函数式开销） | 更快（变异即可） |
| 生产推理 | TensorFlow Serving、Vertex AI | TorchServe、Triton、ONNX |
| 使用者 | DeepMind（Gemini）、Anthropic（Claude） | Meta（Llama）、OpenAI（GPT）、Stability AI |

诚实回答：除非有特定原因，否则用 PyTorch。具体原因包括——可用 TPU、需要逐样本梯度、超大规模多设备训练，或工作于 Google/DeepMind/Anthropic。

### JAX 中的随机数

JAX 无全局随机状态。每次随机操作都需明确 PRNG 秘钥：

```python
key = jax.random.PRNGKey(42)
key1, key2 = jax.random.split(key)
w = jax.random.normal(key1, shape=(784, 256))
```

起初比较麻烦，但保证跨设备和编译的可复现性——这是 PyTorch 中 `torch.manual_seed` 无法跨多 GPU 保证的。

## 实战

### 第 1 步：环境与数据准备

我们用 JAX 和 Optax 训练一个 3 层 MLP，数据集是 MNIST。输入 784，两个隐藏层分别为 256 和 128 个神经元，输出 10 类。

```python
import jax
import jax.numpy as jnp
from jax import random
import optax

def get_mnist_data():
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X = mnist.data.astype('float32') / 255.0
    y = mnist.target.astype('int')
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
    return X_train, y_train, X_test, y_test
```

### 第 2 步：初始化参数

不用类，只写一个返回 pytree 的函数：

```python
def init_params(key):
    k1, k2, k3 = random.split(key, 3)
    scale1 = jnp.sqrt(2.0 / 784)
    scale2 = jnp.sqrt(2.0 / 256)
    scale3 = jnp.sqrt(2.0 / 128)
    params = {
        'layer1': {
            'w': scale1 * random.normal(k1, (784, 256)),
            'b': jnp.zeros(256),
        },
        'layer2': {
            'w': scale2 * random.normal(k2, (256, 128)),
            'b': jnp.zeros(128),
        },
        'layer3': {
            'w': scale3 * random.normal(k3, (128, 10)),
            'b': jnp.zeros(10),
        },
    }
    return params
```

He初始化（He-initialization），手动完成。从一个种子分裂出三个伪随机数生成器（PRNG）键。每个权重都是嵌套字典中的不可变数组。

### 第3步：前向传播（Forward Pass）

```python
def forward(params, x):
    x = jnp.dot(x, params['layer1']['w']) + params['layer1']['b']
    x = jax.nn.relu(x)
    x = jnp.dot(x, params['layer2']['w']) + params['layer2']['b']
    x = jax.nn.relu(x)
    x = jnp.dot(x, params['layer3']['w']) + params['layer3']['b']
    return x

def loss_fn(params, x, y):
    logits = forward(params, x)
    one_hot = jax.nn.one_hot(y, 10)
    return -jnp.mean(jnp.sum(jax.nn.log_softmax(logits) * one_hot, axis=-1))
```

纯函数。参数输入，预测输出。无 `self`，无存储状态。`loss_fn` 从头计算交叉熵——softmax，取对数，负均值。

### 第4步：JIT编译的训练步骤

```python
@jax.jit
def train_step(params, opt_state, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

@jax.jit
def accuracy(params, x, y):
    logits = forward(params, x)
    preds = jnp.argmax(logits, axis=-1)
    return jnp.mean(preds == y)
```

`jax.value_and_grad` 一次计算返回损失值和梯度。`@jax.jit` 装饰器将两个函数编译为 XLA。首次调用后，每个训练步骤都不再调用 Python。

### 第5步：训练循环

```python
optimizer = optax.adam(learning_rate=1e-3)

X_train, y_train, X_test, y_test = get_mnist_data()
X_train, X_test = jnp.array(X_train), jnp.array(X_test)
y_train, y_test = jnp.array(y_train), jnp.array(y_test)

key = random.PRNGKey(0)
params = init_params(key)
opt_state = optimizer.init(params)

batch_size = 128
n_epochs = 10

for epoch in range(n_epochs):
    key, subkey = random.split(key)
    perm = random.permutation(subkey, len(X_train))
    X_shuffled = X_train[perm]
    y_shuffled = y_train[perm]

    epoch_loss = 0.0
    n_batches = len(X_train) // batch_size
    for i in range(n_batches):
        start = i * batch_size
        xb = X_shuffled[start:start + batch_size]
        yb = y_shuffled[start:start + batch_size]
        params, opt_state, loss = train_step(params, opt_state, xb, yb)
        epoch_loss += loss

    train_acc = accuracy(params, X_train[:5000], y_train[:5000])
    test_acc = accuracy(params, X_test, y_test)
    print(f"Epoch {epoch + 1:2d} | Loss: {epoch_loss / n_batches:.4f} | "
          f"Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
```

10个训练周期。测试准确率约为97%。第一周期慢（JIT编译），第2到10周期很快。

请注意缺失的部分：无 `.zero_grad()`，无 `.backward()`，无 `.step()`。整个参数更新是在一个复合函数调用中完成的。梯度计算，通过 Adam 转换，并应用于参数——全部在 `train_step` 内部完成。

## 使用它

### Flax：谷歌标准

Flax 是最常用的 JAX 神经网络库。它重新引入了 `nn.Module`，但状态管理更显式：

```python
import flax.linen as nn

class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(10)(x)
        return x

model = MLP()
params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 784)))
logits = model.apply(params, x_batch)
```

结构和 PyTorch 相同，但 `params` 与模型分离。`model.init()` 用于创建参数。`model.apply(params, x)` 运行前向传播。模型对象没有状态。

### Equinox：更 Python 风格的选择

Equinox（作者 Patrick Kidger）将模型表示为 pytrees：

```python
import equinox as eqx

model = eqx.nn.MLP(
    in_size=784, out_size=10, width_size=256, depth=2,
    activation=jax.nn.relu, key=jax.random.PRNGKey(0)
)
logits = model(x)
```

模型本身是一个 pytree。无需 `.apply()`。参数就是模型的叶子节点。这更符合 JAX 的设计思路。

### Optax：可组合的优化器

Optax 将梯度转换与更新分离：

```python
schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0, peak_value=1e-3,
    warmup_steps=1000, decay_steps=50000
)

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(learning_rate=schedule, weight_decay=0.01),
)
```

梯度裁剪、学习率预热、权重衰减——都作为一链条的转换组合。每个转换接收梯度，修改它，然后传递给下一个。无单一的优化器类。

## 部署它

**安装：**

```bash
pip install jax jaxlib optax flax
```

GPU 支持：

```bash
pip install jax[cuda12]
```

TPU（谷歌云）：

```bash
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

**性能注意事项：**

- 第一次 JIT 调用慢（编译阶段），测性能前先预热。
- 避免在 JIT 内部用 Python 循环 JAX 数组，使用 `jax.lax.scan` 或 `jax.lax.fori_loop`。
- `jax.debug.print()` 可在 JIT 内使用，普通 `print()` 不行。
- 使用 `jax.profiler` 或 TensorBoard 进行分析，XLA 编译可能掩盖瓶颈。
- JAX 默认预分配 GPU 内存的 75%。设置环境变量 `XLA_PYTHON_CLIENT_PREALLOCATE=false` 可禁用。

**检查点保存：**

```python
import orbax.checkpoint as ocp
checkpointer = ocp.PyTreeCheckpointer()
checkpointer.save('/tmp/model', params)
restored = checkpointer.restore('/tmp/model')
```

**本课产出：**
- `outputs/prompt-jax-optimizer.md` —— 选择合适 JAX 优化器配置的提示
- `outputs/skill-jax-patterns.md` —— 涉及 JAX 函数式模式的技能

## 练习

1. 给 MLP 添加 dropout。在 JAX 中，dropout 需要 PRNG 键——在线程中传递一个键给前向传播，并对每个 dropout 层分裂键。比较是否加 dropout 的测试准确率。

2. 使用 `jax.vmap` 对一批 32 张 MNIST 图像计算每个样本的梯度。计算每个样本的梯度范数。哪些样本梯度最大，原因是什么？

3. 用通用的 `mlp_forward(params, x)` 替换手写的前向函数，支持任意层数。用 `jax.tree_leaves` 自动识别深度。

4. 分别用和不用 `@jax.jit` 测试训练步骤的性能。记录各100步耗时。你的硬件上加速有多大？首次调用的编译开销是多少？

5. 通过组合 `optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3))` 实现梯度裁剪。分别带和不带裁剪训练。绘制训练过程中梯度范数变化，观察效果。

## 关键词

| 术语         | 大家说                | 实际含义                                              |
|--------------|-----------------------|-------------------------------------------------------|
| XLA          | “让 JAX 快速的东西”    | Accelerated Linear Algebra —— 一个编译器，可以融合操作并生成优化的 GPU/TPU 内核 |
| JIT          | “即时编译”             | JAX 第一次调用时跟踪函数，编译成 XLA，后续调用用编译版本运行       |
| 纯函数（Pure function） | “无副作用”             | 函数输出仅依赖输入——无全局状态、不变性，无无显式密钥的随机性             |
| vmap         | “自动批处理”           | 将处理单个样本的函数转换为处理批次的函数，无需重写                     |
| pmap         | “自动并行”             | 在多个设备上复制函数并拆分输入批次                                   |
| Pytree       | “嵌套的数组字典”        | 任意嵌套结构的列表、元组、字典和数组，JAX 可遍历并转换                   |
| 跟踪（Tracing） | “记录计算过程”          | JAX 用抽象值执行函数以构建计算图，而不计算真实结果                     |
| 函数式自动微分（Functional autodiff） | “函数的梯度”            | 通过转换函数来计算导数，而不是给张量附加梯度存储                          |
| Optax        | “JAX 的优化库”         | 梯度变换的可组合库——Adam，SGD，裁剪，调度等链式组合                     |
| Flax         | “JAX 的 nn.Module”     | 谷歌的 JAX 神经网络库，添加网络层抽象，同时状态显式管理                   |

## 拓展阅读

- JAX 文档：https://jax.readthedocs.io/ —— 官方文档，包含优质的 grad、jit 和 vmap 教程
- “JAX: composable transformations of Python+NumPy programs”（Bradbury 等，2018）—— 设计理念论文
- Flax 文档：https://flax.readthedocs.io/ —— 谷歌的 JAX 神经网络库
- Patrick Kidger, “Equinox: neural networks in JAX via callable PyTrees and filtered transformations”（2021）—— Flax 的 Pythonic 替代方案
- DeepMind, “Optax: composable gradient transformation and optimisation” —— 通用优化库
- “You Don’t Know JAX”（Colin Raffel，2020）—— 由 T5 作者撰写的 JAX 实用指南，涵盖坑和模式
