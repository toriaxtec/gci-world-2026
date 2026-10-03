# Week 03: Tabular Preprocessing & Data Cleaning with Pandas

## 1. Relational Joins & Set Operations
- **Join Strategies:**
  - *Inner Join:* Retains observations present in both keys.
  - *Left Outer Join:* Preserves all records from the base dataframe, imputing `NaN` for missing right-hand keys.
  - *Full Outer Join:* Retains the full union across both datasets.
- Handled heterogeneous key identifiers using `left_on` and `right_on` parameters in `pd.merge()`.

---

## 2. Missing Value Strategies & Sampling Bias
- **Listwise Deletion (`dropna()`):** Evaluated the danger of dropping entire rows across high-dimensional datasets. In real-world data, uncalibrated listwise deletion can discard over 75% of observations, introducing severe survivorship and selection bias.
- **Pairwise Deletion & Imputation (`fillna()`):** Selectively isolating target features or imputing sentinel/median values to retain statistical power.

---

## 3. Discretization & Advanced Aggregation
- **Interval Binning (`pd.cut`):** Partitioned continuous ranges into fixed discrete intervals.
- **Quantile Binning (`pd.qcut`):** Divided continuous metrics into equal-frequency quantile groups to prevent long-tail skew.
- **Hierarchical Summaries:** Computed multi-metric descriptive statistics across subgroups using `.groupby()` combined with `.agg(['count', 'mean', 'max', 'min'])`.