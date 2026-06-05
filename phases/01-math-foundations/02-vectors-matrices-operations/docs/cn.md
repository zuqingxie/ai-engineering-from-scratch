# 向量（Vectors）、矩阵（Matrices）与运算

> 每个神经网络其实就是带有额外步骤的矩阵乘法。

**类型：** 构建  
**语言：** Python, Julia  
**前置知识：** 第一阶段，第01课（线性代数直觉）  
**时间：** 大约60分钟

## 学习目标

- 构建一个带有元素级运算、矩阵乘法、转置、行列式和逆矩阵功能的 Matrix 类  
- 区分元素级乘法和矩阵乘法，并说明各自适用场景  
- 使用纯手写的 Matrix 类实现单层密集神经网络层（`relu(W @ x + b)`）  
- 解释广播规则以及神经网络框架中偏置（bias）相加的实现机制  

## 问题描述

你想搭建一个神经网络。你看到如下代码：

```text
output = activation(weights @ input + bias)
```

这里的 `@` 是矩阵乘法符号。`weights` 是一个矩阵，`input` 是一个向量。如果你不知道这些运算做了什么，这行代码就是魔法。如果你知道，那它就是一个层的完整前向传递运算，包含三个步骤。

模型处理的每张图像是一块像素值矩阵。每个词嵌入是一个向量。神经网络的每一层都是矩阵变换。掌握矩阵运算就像掌握变量一样，是你构建 AI 系统的必备语言。

本课将从零开始建立这种运算流利度。

## 基本概念

### 向量（Vectors）：有序数字列表  

向量是有方向和大小的数字列表。在 AI 中，向量表示数据点、特征或参数。

```text
v = [3, 4]        -- 二维向量
w = [1, 0, -2]    -- 三维向量
```

二维向量 `[3, 4]` 在平面上指向坐标 (3, 4)。它的长度（大小）是 5（3-4-5 三角形）。

### 矩阵（Matrices）：数字网格  

矩阵是二维网格，有行和列。一个 m x n 矩阵有 m 行 n 列。

```text
A = | 1  2  3 |     -- 2x3 矩阵（2 行 3 列）
    | 4  5  6 |
```

在神经网络中，权重矩阵将输入向量转换为输出向量。一个具有 784 个输入和 128 个输出的层使用 128x784 权重矩阵。

### 形状的重要性

矩阵乘法有严格规则：`(m x n) @ (n x p) = (m x p)`，中间维度必须匹配。

```text
(128 x 784) @ (784 x 1) = (128 x 1)
  权重矩阵    输入向量       输出向量

中间维度：784 = 784  -- 有效
```

如果你在 PyTorch 遇到形状不匹配错误，就是因为这里不符合规则。

### 运算一览表

| 运算          | 功能描述         | 在神经网络中的用途           |
|---------------|-----------------|-----------------------------|
| 加法          | 元素对应相加     | 为输出添加偏置              |
| 标量乘法      | 缩放所有元素     | 学习率乘以梯度              |
| 矩阵乘法      | 向量变换         | 层的前向传播                |
| 转置          | 行列互换         | 反向传播                    |
| 行列式        | 矩阵的标量总结   | 判断矩阵是否可逆            |
| 逆矩阵        | 撤销变换         | 线性系统求解                |
| 单位矩阵      | 恒等变换         | 初始化、残差连接（ResNet）  |

### 元素级乘法 vs 矩阵乘法

这是新手常犯的错误。

**元素级乘法（Element-wise）：** 逐元素对应相乘，两个矩阵必须同形。

```text
| 1  2 |   | 5  6 |   | 5  12 |
| 3  4 | * | 7  8 | = | 21 32 |
```

**矩阵乘法（Matrix multiplication）：** 计算行与列的点积。中间维度必须一致。

```text
| 1  2 |   | 5  6 |   | 1*5+2*7  1*6+2*8 |   | 19  22 |
| 3  4 | @ | 7  8 | = | 3*5+4*7  3*6+4*8 | = | 43  50 |
```

操作不同，结果不同，规则也不同。

### 广播（Broadcasting）

当你将偏置向量加到输出矩阵时，形状不匹配。广播会自动将小数组自动扩展以匹配大数组。

```text
| 1  2  3 |   +   [10, 20, 30]
| 4  5  6 |

广播后偏置向量扩展到每一行：

| 1  2  3 |   | 10  20  30 |   | 11  22  33 |
| 4  5  6 | + | 10  20  30 | = | 14  25  36 |
```

每个现代框架都会自动完成这个过程。理解广播能避免对此类看似形状错误但代码正常运行情况的困惑。

## 实现

### 第一步：Vector 类

```python
class Vector:
    def __init__(self, data):
        self.data = list(data)
        self.size = len(self.data)

    def __repr__(self):
        return f"Vector({self.data})"

    def __add__(self, other):
        return Vector([a + b for a, b in zip(self.data, other.data)])

    def __sub__(self, other):
        return Vector([a - b for a, b in zip(self.data, other.data)])

    def __mul__(self, scalar):
        return Vector([x * scalar for x in self.data])

    def dot(self, other):
        return sum(a * b for a, b in zip(self.data, other.data))

    def magnitude(self):
        return sum(x ** 2 for x in self.data) ** 0.5
```

### 第二步：核心操作的 Matrix 类

```python
class Matrix:
    def __init__(self, data):
        self.data = [list(row) for row in data]
        self.rows = len(self.data)
        self.cols = len(self.data[0])
        self.shape = (self.rows, self.cols)

    def __repr__(self):
        rows_str = "\n  ".join(str(row) for row in self.data)
        return f"Matrix({self.shape}):\n  {rows_str}"

    def __add__(self, other):
        return Matrix([
            [self.data[i][j] + other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def __sub__(self, other):
        return Matrix([
            [self.data[i][j] - other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def scalar_multiply(self, scalar):
        return Matrix([
            [self.data[i][j] * scalar for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def element_wise_multiply(self, other):
        return Matrix([
            [self.data[i][j] * other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def matmul(self, other):
        return Matrix([
            [
                sum(self.data[i][k] * other.data[k][j] for k in range(self.cols))
                for j in range(other.cols)
            ]
            for i in range(self.rows)
        ])

    def transpose(self):
        return Matrix([
            [self.data[j][i] for j in range(self.rows)]
            for i in range(self.cols)
        ])

    def determinant(self):
        if self.shape == (1, 1):
            return self.data[0][0]
        if self.shape == (2, 2):
            return self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0]
        det = 0
        for j in range(self.cols):
            minor = Matrix([
                [self.data[i][k] for k in range(self.cols) if k != j]
                for i in range(1, self.rows)
            ])
            det += ((-1) ** j) * self.data[0][j] * minor.determinant()
        return det

    def inverse_2x2(self):
        det = self.determinant()
        if det == 0:
            raise ValueError("Matrix is singular, no inverse exists")
        return Matrix([
            [self.data[1][1] / det, -self.data[0][1] / det],
            [-self.data[1][0] / det, self.data[0][0] / det]
        ])

    @staticmethod
    def identity(n):
        return Matrix([
            [1 if i == j else 0 for j in range(n)]
            for i in range(n)
        ])
```

### 第三步：演示效果

```python
A = Matrix([[1, 2], [3, 4]])
B = Matrix([[5, 6], [7, 8]])

print("A + B =", (A + B).data)
print("A @ B =", A.matmul(B).data)
print("A^T =", A.transpose().data)
print("det(A) =", A.determinant())
print("A^-1 =", A.inverse_2x2().data)

I = Matrix.identity(2)
print("A @ A^-1 =", A.matmul(A.inverse_2x2()).data)
```

### 第四步：连接到神经网络

```python
import random

inputs = Matrix([[0.5], [0.8], [0.2]])
weights = Matrix([
    [random.uniform(-1, 1) for _ in range(3)]
    for _ in range(2)
])
bias = Matrix([[0.1], [0.1]])

def relu_matrix(m):
    return Matrix([[max(0, val) for val in row] for row in m.data])

pre_activation = weights.matmul(inputs) + bias
output = relu_matrix(pre_activation)

print(f"输入形状: {inputs.shape}")
print(f"权重形状: {weights.shape}")
print(f"输出形状: {output.shape}")
print(f"输出数据: {output.data}")
```

这是一个单层密集层：`output = relu(W @ x + b)`。每个神经网络中的密集层都执行这一步骤。

## 使用它

NumPy 可以用更少代码并且快几个数量级完成以上工作。

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print("A + B =\n", A + B)
print("A * B (element-wise) =\n", A * B)
print("A @ B (matrix multiply) =\n", A @ B)
print("A^T =\n", A.T)
print("det(A) =", np.linalg.det(A))
print("A^-1 =\n", np.linalg.inv(A))
print("I =\n", np.eye(2))

inputs = np.random.randn(3, 1)
weights = np.random.randn(2, 3)
bias = np.array([[0.1], [0.1]])
output = np.maximum(0, weights @ inputs + bias)

print(f"\n神经网络层：{weights.shape} @ {inputs.shape} = {output.shape}")
print(f"输出：\n{output}")
```

Python 中的 `@` 操作符调用 `__matmul__`，NumPy 用 C 和 Fortran 写的高度优化 BLAS 库实现了这一功能。同样的数学运算，快了 100 倍。

NumPy 中的广播示例：

```python
matrix = np.array([[1, 2, 3], [4, 5, 6]])
bias = np.array([10, 20, 30])
print(matrix + bias)
```

NumPy 自动把一维偏置向量广播到所有行，这就是每个神经网络框架偏置加法的工作方式。

## 部署它

本课生成了一个通过几何直觉教授矩阵运算的提示（prompt）。见 `outputs/prompt-matrix-operations.md`。

这里构建的 Matrix 类是我们在第三阶段，第10课构建迷你神经网络框架的基础。

## 练习题

1. **验证逆矩阵。** 计算 `A @ A.inverse_2x2()`，确认结果是单位矩阵。用三个不同的 2x2 矩阵试试。行列式为零时会发生什么？

2. **实现 3x3 逆矩阵。** 扩展 Matrix 类，使用伴随矩阵法计算 3x3 矩阵的逆。与 NumPy 的 `np.linalg.inv` 结果进行对比测试。

3. **构建两层神经网络。** 只用你写的 Matrix 类（不允许用 NumPy）创建一个两层神经网络：输入层（3）-> 隐藏层（4）-> 输出层（2）。随机初始化权重，完成一次前向传递，并确认形状正确。

## 关键词汇

| 术语           | 常见说法           | 实际含义                                       |
|----------------|-------------------|-----------------------------------------------|
| Vector（向量） | “一支箭”          | 有序数字列表，在 AI 中表示高维空间中的点       |
| Matrix（矩阵） | “数字表格”        | 线性变换，将向量映射到另一个空间               |
| Matrix multiply（矩阵乘法） | “就是数字相乘”    | 计算第一矩阵每行与第二矩阵每列的点积，顺序很重要 |
| Transpose（转置） | “翻转”            | 行列互换，把 m x n 矩阵变为 n x m，反向传播中关键 |
| Determinant（行列式） | “从矩阵算出来的数字” | 测量矩阵放大二维面积或三维体积的程度，0 表示降维 |
| Inverse（逆矩阵） | “撤销矩阵”        | 逆转该变换的矩阵，仅行列式非零时存在           |
| Identity matrix（单位矩阵）  | “无聊的矩阵”       | 乘法中的 1，常用于初始化和残差连接（ResNets）    |
| Broadcasting（广播） | “魔法形状修正”     | 将小数组沿缺失维度重复扩展到大数组相同形状        |
| Element-wise（元素级） | “普通乘法”         | 对应位置两数组元素相乘，两者形状相同或可广播      |

## 深入阅读

- [3Blue1Brown: Essence of Linear Algebra（线性代数的本质）](https://www.3blue1brown.com/topics/linear-algebra) - 对这里涵盖的每个操作的视觉直觉
- [NumPy 文档关于 broadcasting（广播）的说明](https://numpy.org/doc/stable/user/basics.broadcasting.html) - NumPy 遵循的具体规则
- [斯坦福 CS229 Linear Algebra Review（线性代数回顾）](http://cs229.stanford.edu/section/cs229-linalg.pdf) - 面向机器学习的线性代数简明参考
