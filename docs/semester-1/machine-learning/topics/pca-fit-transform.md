# 🔬 Principal Component Analysis (PCA)

Principal Component Analysis (PCA) is a dimensionality reduction technique that transforms high-dimensional data into a lower-dimensional representation while preserving as much variance as possible.

---

## ⚙️ What `.fit()` does

**`pca.fit()`** = *"Learn" from the data.*

When you call:

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca.fit(scaled_data)
```

PCA performs the following steps:

### 1. **Compute the Mean**
- Calculates the mean of each feature across all samples
- This mean will be used to center the data

### 2. **Calculate the Covariance Matrix**
- Computes how features vary together
- The covariance matrix captures relationships between features

### 3. **Eigenvalue Decomposition**
- Finds the **eigenvectors** and **eigenvalues** of the covariance matrix
- Eigenvectors represent the directions of maximum variance (principal components)
- Eigenvalues represent the amount of variance explained by each component

### 4. **Select Principal Components**
- Selects the top `n_components` eigenvectors (since `n_components=2`, it selects the top 2)
- These are the directions that capture the most variance in the data

👉 **In summary:** `fit()` learns the optimal coordinate system (principal components) from your training data. It finds the directions where your data varies the most.

---

## ⚙️ What `.transform()` does

**`pca.transform()`** = *"Apply" what was learned to new data.*

When you call:

```python
scaled_pca = pca.transform(scaled_data)
```

PCA performs the following:

### 1. **Center the Data**
- Subtracts the mean (learned during `fit()`) from each data point

### 2. **Project onto Principal Components**
- Multiplies each data point by the principal component matrix
- Projects data from the original feature space onto the new PCA space

### 3. **Return Transformed Data**
- Returns new coordinates in the reduced-dimensional space (e.g., PC1, PC2 for 2 components)

👉 **In summary:** `transform()` converts your original features into PCA components (new columns like PC1, PC2). Each data point gets new coordinates in the principal component space.

---

## 🔄 Complete Workflow Example

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Step 1: Prepare data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(original_data)

# Step 2: Fit PCA (learn the components)
pca = PCA(n_components=2)
pca.fit(scaled_data)

# Step 3: Transform data (apply the transformation)
pca_data = pca.transform(scaled_data)

# Optional: Check explained variance
print(f"Explained variance: {pca.explained_variance_ratio_}")
print(f"Total variance explained: {sum(pca.explained_variance_ratio_):.2%}")
```

---

## 📊 Understanding the Results

After fitting PCA, you can access:

- **`pca.components_`**: The principal components (eigenvectors)
- **`pca.explained_variance_ratio_`**: Percentage of variance explained by each component
- **`pca.mean_`**: Mean of each feature (used for centering)

---

## 💡 Key Points

1. **Always scale your data** before applying PCA (e.g., using StandardScaler)
2. **Fit on training data only** - use the same PCA object to transform both training and test data
3. **Choose n_components wisely** - balance between dimensionality reduction and information retention
4. **Explained variance** tells you how much information is preserved in the reduced dimensions

---

## 🎯 Real-World Application

PCA is commonly used for:
- **Data visualization**: Reduce high-dimensional data to 2D or 3D for plotting
- **Feature reduction**: Remove redundant or correlated features
- **Noise reduction**: Eliminate components with low variance (often noise)
- **Preprocessing**: Reduce dimensionality before applying other ML algorithms
