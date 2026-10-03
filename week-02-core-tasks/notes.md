# Week 02: Numerical Computing & Vectorized Operations with NumPy

## 1. Mathematical Foundations & Vectorization
- **L2 Vector Normalization:** Scaled vectors by their Euclidean norm ($\Vert{}v\Vert{}_2 = \sqrt{\sum v_i^2}$) so that the resulting unit vector satisfies $\Vert{}v'\Vert{}_2 = 1$.
- **Feature Standardization (Z-Score):**
  $$z = \frac{x - \mu}{\sigma}$$
  Standardized multi-dimensional arrays along `axis=0` to ensure features with differing variance and physical units reside on an equivalent scale.
- **Striding & Boolean Masking:** Filtered multi-dimensional arrays conditionally without Python loops via vector boolean masks (e.g., separating rainfall vs. dry intervals).

---

## 2. Advanced Exploration: Locality-Sensitive Hashing (LSH)
- Studied angular similarity partitioning inspired by Google's Reformer architecture.
- **Mathematical Workflow:**
  1. Rotated 3D vector embeddings randomly via rotation matrices: $R = R_z(\theta_z) R_y(\theta_y) R_x(\theta_x)$.
  2. Applied L2 normalization using `np.linalg.norm(..., axis=1, keepdims=True)`.
  3. Formed an expanded reference space combining positive and negative basis vectors (`np.concatenate([B, -B], axis=1)`).
  4. Assigned vectors to discrete similarity partitions via `np.argmax(..., axis=1)` to avoid exhaustive $O(N^2)$ pairwise checks.

---

## 3. General Implementation Snippets
```python
import numpy as np

# Vector L2 Normalization
def normalize_vector(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)

# Multi-dimensional Column Standardization
def standardize_matrix(X: np.ndarray) -> np.ndarray:
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    return (X - mu) / sigma