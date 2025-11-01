# 🔬 Principal Component Analysis (PCA)

---

## ⚙️ What `.fit()` does

**`pca.fit()`** = *"Learn" from the data.*

When you call:

```python
pca.fit(scaled_data)
```

PCA:

- Computes the **mean** of each feature
- Calculates the **covariance matrix**
- Finds the **eigenvectors** and **eigenvalues**
- Selects the **top 2 components** (since `n_components=2`) that explain most variance

👉 In short:  
`fit()` finds the *directions* of maximum variance — i.e., **the new coordinate system (axes)**.

Think of this as **training** PCA on your data.

---

## ⚙️ What `.transform()` does

**`pca.transform()`** = *"Apply" what was learned.*

When you call:

```python
scaled_pca = pca.transform(scaled_data)
```

PCA:

- Projects each data point onto the new **principal components** found during `fit()`
- Gives you new coordinates (in the new 2D PCA space)

👉 In short:  
`transform()` converts your original features into **PCA components** (new columns like PC1, PC2).

---

