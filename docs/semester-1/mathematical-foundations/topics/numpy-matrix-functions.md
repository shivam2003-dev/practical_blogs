# 🔢 NumPy Matrix Functions Cheatsheet

---

## 📘 Introduction

NumPy is the fundamental package for scientific computing in Python. This comprehensive cheatsheet covers the most important NumPy functions for matrix operations, essential for machine learning and data science applications.

---

## 📦 Array Creation Functions

### Creating Basic Arrays

| Function | Description | Example |
|----------|-------------|---------|
| `np.array()` | Create array from list or tuple | `np.array([[1, 2], [3, 4]])` |
| `np.zeros()` | Create array filled with zeros | `np.zeros((3, 3))` |
| `np.ones()` | Create array filled with ones | `np.ones((2, 4))` |
| `np.eye()` | Create identity matrix | `np.eye(3)` |
| `np.diag()` | Create diagonal matrix or extract diagonal | `np.diag([1, 2, 3])` |

### Creating Special Matrices

```python
import numpy as np

# Identity matrix
I = np.eye(3)
# [[1., 0., 0.],
#  [0., 1., 0.],
#  [0., 0., 1.]]

# Matrix of zeros
Z = np.zeros((2, 3))
# [[0., 0., 0.],
#  [0., 0., 0.]]

# Matrix of ones
O = np.ones((3, 2))
# [[1., 1.],
#  [1., 1.],
#  [1., 1.]]

# Filled with constant value
F = np.full((2, 2), 7)
# [[7, 7],
#  [7, 7]]
```

### Random Matrix Generation

| Function | Description | Example |
|----------|-------------|---------|
| `np.random.rand()` | Uniform distribution [0, 1) | `np.random.rand(3, 3)` |
| `np.random.randn()` | Standard normal distribution | `np.random.randn(3, 3)` |
| `np.random.randint()` | Random integers | `np.random.randint(0, 10, (3, 3))` |
| `np.random.random()` | Random floats [0, 1) | `np.random.random((3, 3))` |

### Range and Sequence Creation

```python
# Linear spacing
x = np.linspace(0, 10, 5)  # 5 points from 0 to 10
# [0., 2.5, 5., 7.5, 10.]

# Arrange with step
y = np.arange(0, 10, 2)  # From 0 to 10, step 2
# [0, 2, 4, 6, 8]

# Logarithmic spacing
z = np.logspace(0, 3, 4)  # 10^0 to 10^3, 4 points
# [1., 10., 100., 1000.]
```

---

## 🔄 Matrix Operations

### Basic Matrix Operations

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Matrix addition
C = A + B
# [[ 6,  8],
#  [10, 12]]

# Matrix subtraction
D = A - B
# [[-4, -4],
#  [-4, -4]]

# Element-wise multiplication (Hadamard product)
E = A * B
# [[ 5, 12],
#  [21, 32]]

# Matrix multiplication (dot product)
F = np.dot(A, B)  # or A @ B
# [[19, 22],
#  [43, 50]]

# Scalar multiplication
G = 3 * A
# [[ 3,  6],
#  [ 9, 12]]
```

### Transpose and Reshaping

| Function | Description | Example |
|----------|-------------|---------|
| `A.T` or `np.transpose(A)` | Transpose matrix | `A.T` |
| `A.reshape()` | Change shape without changing data | `A.reshape(4, 1)` |
| `A.flatten()` | Flatten to 1D array | `A.flatten()` |
| `A.ravel()` | Flattened view (memory efficient) | `A.ravel()` |

```python
A = np.array([[1, 2, 3], [4, 5, 6]])

# Transpose
AT = A.T
# [[1, 4],
#  [2, 5],
#  [3, 6]]

# Reshape
B = A.reshape(3, 2)
# [[1, 2],
#  [3, 4],
#  [5, 6]]

# Flatten
flat = A.flatten()
# [1, 2, 3, 4, 5, 6]
```

### Matrix Inverse and Determinant

```python
A = np.array([[1, 2], [3, 4]])

# Determinant
det_A = np.linalg.det(A)
# -2.0

# Inverse (if det != 0)
A_inv = np.linalg.inv(A)
# [[-2. ,  1. ],
#  [ 1.5, -0.5]]

# Verify: A × A^(-1) = I
identity = np.dot(A, A_inv)
# [[1., 0.],
#  [0., 1.]]

# Pseudo-inverse (for singular or non-square matrices)
B = np.array([[1, 2], [3, 4], [5, 6]])
B_pinv = np.linalg.pinv(B)
```

### Matrix Power and Exponentiation

```python
A = np.array([[1, 2], [3, 4]])

# Matrix power
A_squared = np.linalg.matrix_power(A, 2)
# [[ 7, 10],
#  [15, 22]]

# Element-wise power
A_elem_squared = np.power(A, 2)  # or A ** 2
# [[ 1,  4],
#  [ 9, 16]]
```

---

## 🔬 Matrix Decomposition

### Eigenvalues and Eigenvectors

```python
A = np.array([[4, 2], [1, 3]])

# Eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)

# eigenvalues: [5., 2.]
# eigenvectors: [[0.89442719, -0.70710678],
#                [0.4472136,   0.70710678]]

# Verify: A v = λ v
lambda1 = eigenvalues[0]
v1 = eigenvectors[:, 0]
result = np.dot(A, v1)  # Should equal lambda1 * v1
```

### Singular Value Decomposition (SVD)

SVD decomposes a matrix $A$ into three matrices:

$$
A = U \Sigma V^T
$$

where:
- $U$ — left singular vectors (orthogonal)
- $\Sigma$ — diagonal matrix of singular values
- $V^T$ — right singular vectors (orthogonal)

```python
A = np.array([[1, 2, 3], [4, 5, 6]])

# SVD decomposition
U, S, VT = np.linalg.svd(A)

# U shape: (2, 2)
# S shape: (2,) - singular values
# VT shape: (3, 3)

# Reconstruct A
Sigma = np.zeros((2, 3))
Sigma[:2, :2] = np.diag(S)
A_reconstructed = U @ Sigma @ VT
```

**Applications of SVD:**
- Dimensionality reduction (PCA)
- Data compression
- Noise reduction
- Recommender systems

### QR Decomposition

Decomposes matrix $A$ into orthogonal matrix $Q$ and upper triangular matrix $R$:

$$
A = QR
$$

```python
A = np.array([[1, 2], [3, 4], [5, 6]])

# QR decomposition
Q, R = np.linalg.qr(A)

# Q is orthogonal: Q^T Q = I
# R is upper triangular

# Verify: A = QR
A_reconstructed = np.dot(Q, R)
```

### Cholesky Decomposition

For symmetric positive-definite matrix $A$:

$$
A = LL^T
$$

where $L$ is lower triangular.

```python
# Must be symmetric positive-definite
A = np.array([[4, 12, -16], [12, 37, -43], [-16, -43, 98]])

# Cholesky decomposition
L = np.linalg.cholesky(A)

# Verify: A = L L^T
A_reconstructed = np.dot(L, L.T)
```

---

## 📐 Linear Algebra Operations

### Solving Linear Systems

Solve $Ax = b$ for $x$:

```python
A = np.array([[3, 1], [1, 2]])
b = np.array([9, 8])

# Solve linear system
x = np.linalg.solve(A, b)
# [2., 3.]

# Verify solution
result = np.dot(A, x)  # Should equal b
```

### Matrix Rank and Norms

```python
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Matrix rank
rank = np.linalg.matrix_rank(A)
# 2 (not full rank)

# Norms
frobenius_norm = np.linalg.norm(A)  # Frobenius norm
l2_norm = np.linalg.norm(A, 2)      # L2 norm (spectral norm)
l1_norm = np.linalg.norm(A, 1)      # L1 norm
inf_norm = np.linalg.norm(A, np.inf) # Infinity norm
```

### Trace and Diagonal Operations

```python
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Trace (sum of diagonal elements)
trace = np.trace(A)
# 15

# Extract diagonal
diag = np.diag(A)
# [1, 5, 9]

# Create diagonal matrix from vector
v = np.array([1, 2, 3])
D = np.diag(v)
# [[1, 0, 0],
#  [0, 2, 0],
#  [0, 0, 3]]
```

---

## 🔀 Reshaping and Manipulation

### Concatenation and Stacking

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Vertical stack (row-wise)
V = np.vstack((A, B))
# [[1, 2],
#  [3, 4],
#  [5, 6],
#  [7, 8]]

# Horizontal stack (column-wise)
H = np.hstack((A, B))
# [[1, 2, 5, 6],
#  [3, 4, 7, 8]]

# Concatenate along axis
C0 = np.concatenate((A, B), axis=0)  # Same as vstack
C1 = np.concatenate((A, B), axis=1)  # Same as hstack

# Block matrix
Block = np.block([[A, B], [B, A]])
# [[1, 2, 5, 6],
#  [3, 4, 7, 8],
#  [5, 6, 1, 2],
#  [7, 8, 3, 4]]
```

### Splitting and Indexing

```python
A = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

# Split horizontally
H_splits = np.hsplit(A, 2)  # Split into 2 parts
# [array([[1, 2], [5, 6], [9, 10]]), array([[3, 4], [7, 8], [11, 12]])]

# Split vertically
V_splits = np.vsplit(A, 3)  # Split into 3 parts

# Advanced indexing
rows = [0, 2]
cols = [1, 3]
submatrix = A[rows, :][:, cols]
# [[ 2,  4],
#  [10, 12]]
```

### Broadcasting and Tiling

```python
# Broadcasting - automatically expand dimensions
A = np.array([[1, 2, 3], [4, 5, 6]])
v = np.array([10, 20, 30])

# Add vector to each row
result = A + v
# [[11, 22, 33],
#  [14, 25, 36]]

# Tile - repeat array
T = np.tile(A, (2, 3))  # Repeat 2 times vertically, 3 times horizontally
# [[1, 2, 3, 1, 2, 3, 1, 2, 3],
#  [4, 5, 6, 4, 5, 6, 4, 5, 6],
#  [1, 2, 3, 1, 2, 3, 1, 2, 3],
#  [4, 5, 6, 4, 5, 6, 4, 5, 6]]

# Repeat elements
R = np.repeat(A, 2, axis=0)  # Repeat each row 2 times
# [[1, 2, 3],
#  [1, 2, 3],
#  [4, 5, 6],
#  [4, 5, 6]]
```

---

## 📊 Statistical Functions

### Basic Statistics

```python
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Mean
mean_all = np.mean(A)         # Mean of all elements: 5.0
mean_rows = np.mean(A, axis=1) # Mean of each row: [2., 5., 8.]
mean_cols = np.mean(A, axis=0) # Mean of each column: [4., 5., 6.]

# Sum
sum_all = np.sum(A)           # Sum of all elements: 45
sum_rows = np.sum(A, axis=1)  # [6, 15, 24]
sum_cols = np.sum(A, axis=0)  # [12, 15, 18]

# Standard deviation
std = np.std(A)               # 2.58...

# Variance
var = np.var(A)               # 6.67...

# Min and Max
min_val = np.min(A)           # 1
max_val = np.max(A)           # 9
min_idx = np.argmin(A)        # Index of min (flattened): 0
max_idx = np.argmax(A)        # Index of max (flattened): 8
```

### Cumulative Operations

```python
A = np.array([[1, 2, 3], [4, 5, 6]])

# Cumulative sum
cumsum = np.cumsum(A)
# [ 1,  3,  6, 10, 15, 21]

cumsum_cols = np.cumsum(A, axis=0)
# [[1, 2, 3],
#  [5, 7, 9]]

# Cumulative product
cumprod = np.cumprod(A, axis=1)
# [[1,   2,   6],
#  [4,  20, 120]]
```

### Correlation and Covariance

```python
X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Covariance matrix
cov_matrix = np.cov(X)
# Each row is an observation

# Correlation coefficient
Y = np.array([[2, 4, 6], [5, 10, 15]])
corr_matrix = np.corrcoef(Y)
# [[ 1.  1.]
#  [ 1.  1.]]  (Perfect correlation)
```

### Percentiles and Quantiles

```python
A = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Percentiles
p25 = np.percentile(A, 25)    # 3.25 (25th percentile)
p50 = np.percentile(A, 50)    # 5.5  (median)
p75 = np.percentile(A, 75)    # 7.75 (75th percentile)

# Quantiles (same as percentile but [0, 1] scale)
q25 = np.quantile(A, 0.25)
q50 = np.quantile(A, 0.5)     # Median
q75 = np.quantile(A, 0.75)

# Median
median = np.median(A)         # 5.5
```

---

## 🎯 Advanced Matrix Operations

### Kronecker Product

The Kronecker product of matrices $A$ and $B$:

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[0, 5], [6, 7]])

# Kronecker product
K = np.kron(A, B)
# [[ 0,  5,  0, 10],
#  [ 6,  7, 12, 14],
#  [ 0, 15,  0, 20],
#  [18, 21, 24, 28]]
```

### Outer Product

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Outer product
outer = np.outer(a, b)
# [[ 4,  5,  6],
#  [ 8, 10, 12],
#  [12, 15, 18]]
```

### Inner and Dot Products

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Inner product (dot product for vectors)
inner = np.inner(a, b)
# 32

# Dot product
dot = np.dot(a, b)
# 32

# Cross product (3D vectors)
c = np.array([1, 0, 0])
d = np.array([0, 1, 0])
cross = np.cross(c, d)
# [0, 0, 1]
```

### Matrix Condition Number

The condition number measures how sensitive a matrix is to changes:

```python
A = np.array([[1, 2], [3, 4]])

# Condition number
cond = np.linalg.cond(A)
# 14.93... (well-conditioned if close to 1)

# A large condition number indicates the matrix is ill-conditioned
# and numerical solutions may be unstable
```

---

## 💡 Practical Examples

### Example 1: Principal Component Analysis (PCA)

```python
# Sample data matrix (samples × features)
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])

# Center the data
X_centered = X - np.mean(X, axis=0)

# Compute covariance matrix
cov_matrix = np.cov(X_centered.T)

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Sort by eigenvalues (descending)
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# Project data onto first principal component
PC1 = eigenvectors[:, 0]
X_pca = X_centered @ PC1.reshape(-1, 1)
```

### Example 2: Linear Regression using Normal Equation

Solve $w = (X^T X)^{-1} X^T y$:

```python
# Feature matrix and target vector
X = np.array([[1, 1], [1, 2], [1, 3], [1, 4]])  # First column is bias
y = np.array([2, 3, 4, 5])

# Normal equation
XTX = X.T @ X
XTy = X.T @ y
w = np.linalg.solve(XTX, XTy)  # More stable than computing inverse

# Or using pseudo-inverse
w_pinv = np.linalg.pinv(X) @ y

# Predictions
y_pred = X @ w
```

### Example 3: Image Compression using SVD

```python
# Simulate grayscale image (or load real image)
image = np.random.rand(100, 100)

# SVD decomposition
U, S, VT = np.linalg.svd(image)

# Keep only top k singular values for compression
k = 20
S_compressed = np.zeros((100, 100))
S_compressed[:k, :k] = np.diag(S[:k])

# Reconstruct compressed image
image_compressed = U @ S_compressed @ VT

# Compression ratio: (100*k + k + k*100) / (100*100)
compression_ratio = (100*k + k + k*100) / (100*100)
# Using 20 components: ~40% of original size
```

---

## 🔧 Performance Tips

### Memory Efficiency

```python
# Use views instead of copies when possible
A = np.array([[1, 2], [3, 4]])

# View (no copy)
B = A.ravel()  # Changes to B affect A

# Copy (independent)
C = A.flatten()  # Changes to C don't affect A
```

### Vectorization

```python
# ❌ Slow: Loop-based computation
A = np.random.rand(1000, 1000)
result = np.zeros((1000, 1000))
for i in range(1000):
    for j in range(1000):
        result[i, j] = A[i, j] ** 2

# ✅ Fast: Vectorized computation
result = A ** 2  # Much faster!
```

### In-place Operations

```python
A = np.array([[1, 2], [3, 4]])

# Regular operation (creates new array)
B = A + 5

# In-place operation (modifies existing array)
A += 5  # More memory efficient
```

---

## 📚 Quick Reference Table

| Category | Key Functions |
|----------|--------------|
| **Creation** | `array`, `zeros`, `ones`, `eye`, `random.rand`, `linspace`, `arange` |
| **Basic Ops** | `+`, `-`, `*`, `@`, `dot`, `transpose`, `reshape` |
| **Linear Algebra** | `inv`, `det`, `solve`, `eig`, `svd`, `qr`, `cholesky` |
| **Decomposition** | `eig`, `svd`, `qr`, `cholesky` |
| **Statistics** | `mean`, `std`, `var`, `min`, `max`, `median`, `percentile` |
| **Manipulation** | `vstack`, `hstack`, `concatenate`, `split`, `tile`, `repeat` |
| **Advanced** | `kron`, `outer`, `inner`, `cross`, `cond`, `matrix_rank` |

---

## 🎓 Summary

NumPy provides comprehensive tools for matrix operations essential in machine learning and scientific computing:

- ✅ Efficient array creation and initialization
- ✅ Complete set of linear algebra operations
- ✅ Matrix decomposition for advanced algorithms (PCA, SVD)
- ✅ Statistical functions for data analysis
- ✅ Flexible reshaping and manipulation
- ✅ Optimized performance through vectorization

**Remember:** Always prefer vectorized operations over loops for better performance!

---

[📚 Back to Mathematical Foundations](../README.md)
