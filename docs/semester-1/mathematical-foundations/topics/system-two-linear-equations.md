# 🧩 System of Two Linear Equations (2 Variables)

---

## 📘 Problem Statement

Given a system of two linear equations with two unknowns $w_1$ and $w_2$:

$$
a_1 w_1 + b_1 w_2 = c_1
$$

$$
a_2 w_1 + b_2 w_2 = c_2
$$

Determine whether:

- ✅ There is a **unique solution**.
- 🔁 There are **infinitely many solutions**.
- ❌ There is **no solution**.

---

## 🧠 Concept

Each equation represents a **line** in 2D space.

| Case | Description | Geometric Meaning |
|------|--------------|-------------------|
| **Unique Solution** | Lines intersect at exactly one point. | Two lines cross once. |
| **Infinite Solutions** | Both equations describe the same line. | Lines are coincident. |
| **No Solution** | Lines are parallel and never meet. | Parallel lines. |

---

## ⚙️ Approach 1 — Determinant (Cramer's Rule)

### Step 1: Form Coefficient Matrix

$$
A =
\begin{bmatrix}
a_1 & b_1 \\
a_2 & b_2
\end{bmatrix}
,\quad
B =
\begin{bmatrix}
c_1 \\
c_2
\end{bmatrix}
$$

### Step 2: Compute Determinant

$$
D = a_1 b_2 - a_2 b_1
$$

### Step 3: Check Cases

| Condition | Result |
|------------|---------|
| $D \neq 0$ | ✅ Unique solution: $w_1 = \frac{c_1 b_2 - c_2 b_1}{D}$, $w_2 = \frac{a_1 c_2 - a_2 c_1}{D}$ |
| $D = 0$ and ratios $\frac{a_1}{a_2} = \frac{b_1}{b_2} = \frac{c_1}{c_2}$ | 🔁 Infinitely many solutions |
| $D = 0$ but constants not in same ratio | ❌ No solution |

---

## ⚙️ Approach 2 — Row Reduction (Gaussian Elimination)

Form **augmented matrix**:

$$
\begin{bmatrix}
a_1 & b_1 & \mid & c_1 \\
a_2 & b_2 & \mid & c_2
\end{bmatrix}
$$

Perform elementary row operations to get **row-echelon form**.

| Form | Meaning |
|-------|----------|
| $[0 \; 0 \mid \text{nonzero}]$ | ❌ No solution |
| $[0 \; 0 \mid 0]$ | 🔁 Infinite solutions |
| No zero rows | ✅ Unique solution |

---

## ⚙️ Approach 3 — Graphical (Slope Comparison)

Each equation represents a line:

$$
L_1: a_1 w_1 + b_1 w_2 = c_1
$$

$$
L_2: a_2 w_1 + b_2 w_2 = c_2
$$

Convert to slope-intercept form:

$$
w_2 = -\frac{a_1}{b_1} w_1 + \frac{c_1}{b_1}
$$

$$
w_2 = -\frac{a_2}{b_2} w_1 + \frac{c_2}{b_2}
$$

| Condition | Interpretation |
|------------|----------------|
| $m_1 \neq m_2$ | ✅ Lines intersect once (unique solution) |
| $m_1 = m_2$ and same intercept | 🔁 Infinite solutions |
| $m_1 = m_2$ and different intercept | ❌ No solution |

---

## ⚙️ Approach 4 — Rank Method (Linear Algebra)

### Step 1: Define Matrices

$$
A =
\begin{bmatrix}
a_1 & b_1 \\
a_2 & b_2
\end{bmatrix}
,\quad
[A|B] =
\begin{bmatrix}
a_1 & b_1 & \mid & c_1 \\
a_2 & b_2 & \mid & c_2
\end{bmatrix}
$$

### Step 2: Compare Ranks

| Condition | Result |
|------------|---------|
| $\text{rank}(A) = \text{rank}([A|B]) = 2$ | ✅ Unique solution |
| $\text{rank}(A) = \text{rank}([A|B]) < 2$ | 🔁 Infinite solutions |
| $\text{rank}(A) < \text{rank}([A|B])$ | ❌ No solution |

---

## 🧮 Example 1 — Infinite Solutions

$$
x + y = 2
$$

$$
2x + 2y = 4
$$

Here, 2nd equation = 2 × (1st equation)

- $\text{rank}(A) = 1$  
- $\text{rank}([A|B]) = 1$  

✅ **Infinitely many solutions**

---

## 🧮 Example 2 — No Solution

$$
x + y = 2
$$

$$
2x + 2y = 5
$$

Here, coefficients have same ratio but constants don't.

- $\text{rank}(A) = 1$  
- $\text{rank}([A|B]) = 2$  

❌ **No solution**

---

## 🧮 Example 3 — Unique Solution

$$
x + y = 3
$$

$$
x - y = 1
$$

Determinant $D = (1)(-1) - (1)(1) = -2 \neq 0$

✅ **Unique solution:**

$$
w_1 = 2, \quad w_2 = 1
$$

---

## 💻 Python Implementation (Determinant Method)

```python
import numpy as np

def solve_linear_2x2(a1, b1, c1, a2, b2, c2):
    D = a1 * b2 - a2 * b1
    if abs(D) > 1e-10:
        w1 = (c1 * b2 - c2 * b1) / D
        w2 = (a1 * c2 - a2 * c1) / D
        return f"Unique solution: w1 = {w1}, w2 = {w2}"
    else:
        if (a1*b2 == a2*b1) and (a1*c2 == a2*c1):
            return "There are infinitely many solutions"
        else:
            return "Intersection does not exist"

print(solve_linear_2x2(1, 1, 3, 1, -1, 1))
```

---

## 💻 Python Implementation (Row Operations Method)

Alternative implementation using row operations (Gaussian elimination):

```python
import numpy as np

def findIntersectionIfExists(e1: np.array, e2: np.array) -> str:
    """
    Solve system of 2 linear equations using row operations.
    
    Args:
        e1: numpy array [a1, b1, c1] representing a1*w1 + b1*w2 = c1
        e2: numpy array [a2, b2, c2] representing a2*w1 + b2*w2 = c2
    
    Returns:
        String describing the solution or solution type
    """
    # Scale e2 to match coefficient of w1 in e1
    alpha = e1[0] / e2[0]
    e2 = e2 * alpha
    
    # Check if equations are identical (infinite solutions)
    if (e2 == e1).all():
        return "There are infinitely many solutions"
    
    # Check if coefficients match but constants differ (no solution)
    elif (e2[:2] == e1[:2]).all():
        return "Intersection does not exist"
    
    # Unique solution: subtract e1 from e2 and solve
    else:
        e2 = e2 - e1
        w2 = e2[2] / e2[1]  # Solve for w2
        w1 = (e1[2] - (e1[1] * w2)) / e1[0]  # Solve for w1
        return '%.3f , %.3f' % (w1, w2)

# Verify solution
e1 = np.array([2, 4, 9])  # 2w1 + 4w2 = 9
e2 = np.array([3, 7, 3])  # 3w1 + 7w2 = 3
print(findIntersectionIfExists(e1, e2))
```

**How it works:**
1. **Scale** the second equation to match the coefficient of $w_1$ in the first equation
2. **Compare** equations to detect infinite or no solution cases
3. **Eliminate** $w_1$ by subtracting equations, then solve for $w_2$ and back-substitute for $w_1$

**Example:**
- Input: $2w_1 + 4w_2 = 9$ and $3w_1 + 7w_2 = 3$
- Output: Unique solution $(w_1, w_2)$

