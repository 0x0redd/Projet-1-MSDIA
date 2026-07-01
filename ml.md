# ML Table Improvement Guide
## Models to Add, Grids to Define, and How to Frame the Table

> **Context:** Your current ML table has SVM · KNN · Random Forest on
> GLCM / LBP / DWT / HOG / Full(opt) features.
> This guide tells you exactly what to add, why, and with what grid.

---

## 1. The Problem with Your Current Table

| Issue | Detail |
|---|---|
| **KNN is weak** | Lowest performer, no publication angle, distances are sensitive to the HOG 1944-dim space |
| **No gradient boosting** | XGBoost / LightGBM are expected in any 2024+ ML benchmark |
| **No linear baseline** | Logistic Regression gives reviewers a zero-effort reference point |
| **No MLP** | Bridge between ML and DL — expected in a multi-paradigm paper |
| **No feature importance** | RF and XGBoost give it for free — missing a key interpretability argument |
| **No PCA experiment** | HOG is 1944-dim — reviewers will ask "did you try dimensionality reduction?" |

---

## 2. Models to Add (Priority Order)

---

### ✅ MUST ADD — XGBoost

**Why:**
- <cited in BMC Medical Imaging 2024> XGBoost has been directly applied to brain tumor MRI
  classification with GLCM features, achieving F1=0.94 — reviewers will expect it.
- Gradient boosting on tabular/handcrafted features consistently outperforms Random Forest
  on structured data with feature interactions.
- Gives native `feature_importances_` — directly feeds your interpretability section.
- Fast to train on your feature dimensions (max 1944-dim HOG).

**scikit-learn equivalent:** `XGBClassifier` (xgboost library)

**Grid to use:**
```python
"XGBoost": {
    "n_estimators":       [100, 300, 500, 800],
    "max_depth":          [3, 4, 5, 6, 8],
    "learning_rate":      [0.01, 0.05, 0.1, 0.2],
    "subsample":          [0.7, 0.8, 1.0],
    "colsample_bytree":   [0.7, 0.8, 1.0],
    "min_child_weight":   [1, 3, 5],
    "gamma":              [0, 0.1, 0.3],
    "use_label_encoder":  [False],
    "eval_metric":        ["mlogloss"],
}
# Total: 4×5×4×3×3×3×3 = 6,480 candidates
# With 5-fold CV: 32,400 fits
```

**Paper framing:**
> *"XGBoost was included as a modern gradient boosting baseline, given its
> demonstrated effectiveness on handcrafted radiomic features in recent
> brain tumor classification literature."*

---

### ✅ MUST ADD — LightGBM

**Why:**
- <cited in MDPI Diagnostics 2025 review> LightGBM achieves AUC=0.874 on brain MRI
  radiomic features, outperforming XGBoost (0.865) in some configurations.
- 10–20× faster than XGBoost on large feature sets — relevant for your HOG 1944-dim.
- Handles high-dimensional sparse features natively (leaf-wise growth vs level-wise).
- Together with XGBoost, gives you a complete gradient boosting comparison.

**Grid to use:**
```python
"LightGBM": {
    "n_estimators":    [100, 300, 500, 800],
    "max_depth":       [-1, 4, 6, 8],        # -1 = no limit
    "learning_rate":   [0.01, 0.05, 0.1, 0.2],
    "num_leaves":      [15, 31, 63, 127],
    "subsample":       [0.7, 0.8, 1.0],
    "colsample_bytree":[0.7, 0.8, 1.0],
    "min_child_samples":[5, 10, 20],
    "class_weight":    ["balanced", None],
}
# Total: 4×4×4×4×3×3×3×2 = 13,824 candidates
# With 5-fold CV: 69,120 fits
```

**Paper framing:**
> *"LightGBM complements XGBoost by using leaf-wise tree growth,
> which is better suited to high-dimensional feature spaces such as HOG (1944-dim)."*

---

### ✅ MUST ADD — Logistic Regression (with PCA)

**Why:**
- Every ML paper needs a linear baseline. Reviewers expect it.
- On Full(opt) features (GLCM+LBP+DWT+HOG = ~1,987 dims), LR with L2 reg is non-trivial.
- PCA before LR answers the dimensionality question proactively.
- Fast: adds almost zero compute time.
- <MDPI 2025 brain MRI review> LR achieved AUC=0.912 — best overall in one study,
  outperforming LightGBM and XGBoost on radiomic features.

**Grid to use:**
```python
"LR": {
    "C":          [0.001, 0.01, 0.1, 1, 10, 100],
    "penalty":    ["l1", "l2", "elasticnet"],
    "solver":     ["saga"],               # supports all penalties
    "l1_ratio":   [0.1, 0.5, 0.9],       # only for elasticnet
    "max_iter":   [1000],
    "class_weight":["balanced", None],
}
# Total: 6×3×1×3×1×2 = 108 candidates (fast)
# With 5-fold CV: 540 fits

# IMPORTANT: StandardScaler before LR (add to Pipeline)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
pipe = Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression())])
```

**Paper framing:**
> *"Logistic Regression with L2 regularisation provides a linear separability
> baseline and assesses whether the handcrafted feature space is linearly
> separable prior to non-linear classification."*

---

### ✅ MUST ADD — MLP (Multi-Layer Perceptron)

**Why:**
- Acts as the bridge between classical ML and your DL Phase 3.
- <cited in brain tumor MRI classification literature> MLP on handcrafted features
  is a standard comparison model in the domain.
- If MLP on HOG features ≈ CNN+HOG (Tier 1), that is a major finding.
- If MLP << CNN+HOG, it confirms that spatial feature learning (conv layers) matters,
  not just the feature type.
- scikit-learn's `MLPClassifier` — no GPU needed.

**Grid to use:**
```python
"MLP": {
    "hidden_layer_sizes": [
        (256,),
        (512,),
        (256, 128),
        (512, 256),
        (512, 256, 128),
        (1024, 512, 256),
    ],
    "activation":         ["relu", "tanh"],
    "alpha":              [1e-4, 1e-3, 1e-2, 1e-1],   # L2 regularisation
    "learning_rate_init": [1e-3, 5e-4, 1e-4],
    "batch_size":         [32, 64, 128],
    "max_iter":           [500],
    "early_stopping":     [True],
    "validation_fraction":[0.1],
}
# Total: 6×2×4×3×3 = 432 candidates
# With 5-fold CV: 2,160 fits
```

**Paper framing:**
> *"A shallow MLP trained on handcrafted features bridges the classical ML
> and deep learning paradigms, isolating the contribution of spatial
> feature learning (convolutional inductive bias) from raw feature quality."*

---

### ⚠️ OPTIONAL — Extra Ensemble: ExtraTrees + Voting Ensemble

**ExtraTrees (Extremely Randomized Trees):**
- Faster than RF, often comparable accuracy.
- Gives feature importances.
- Adds a data point on the "random threshold" effect in ensembles.

```python
"ExtraTrees": {
    "n_estimators":       [100, 300, 500],
    "max_features":       ["sqrt", "log2", None],
    "min_samples_split":  [2, 5, 10],
    "min_samples_leaf":   [1, 2, 4],
    "criterion":          ["gini", "entropy"],
}
# Total: 3×3×3×3×2 = 162 candidates
```

**Voting Ensemble (no grid search needed):**
```python
from sklearn.ensemble import VotingClassifier
voting = VotingClassifier(
    estimators=[
        ("svm",  best_svm),
        ("xgb",  best_xgb),
        ("lgbm", best_lgbm),
    ],
    voting="soft",   # use predicted probabilities
)
# Train on Full(opt) only — 1 run, no CV sweep
```
> Gives you a "best-of-all" ensemble row in the table for free.

---

### ❌ CUT — Naive Bayes

- Assumes feature independence — violated by GLCM (spatial co-occurrence by definition).
- Not used in recent brain tumor texture literature.
- Adds noise to the table without a defensible justification.

### ❌ CUT — Decision Tree (single)

- Always dominated by RF/XGBoost. No reviewer expects it in 2025.
- Unless you are doing interpretability (single tree as explainable model) — not your angle.

---

## 3. Revised ML Table Structure

```
Rows    = Models × Feature Sets
Columns = F1-macro | Accuracy | Precision | Recall | κ | CI 95% | Time (s)

Feature sets to report per model:
  - Best single feature set  (e.g. HOG for SVM)
  - Full(opt)                (concatenation of all optimal features)
  → 2 rows per model = compact but complete
```

**Target table (values to fill after runs):**

| Model | Feature Set | F1-macro | Accuracy | κ | 95% CI | Time (s) |
|---|---|---|---|---|---|---|
| SVM (RBF) | Full(opt) | **0.9299** | 0.9300 | ? | [?,?] | ? |
| SVM (RBF) | HOG | ? | ? | ? | [?,?] | ? |
| Random Forest | Full(opt) | ? | ? | ? | [?,?] | ? |
| Random Forest | HOG | ? | ? | ? | [?,?] | ? |
| **XGBoost** | Full(opt) | ? | ? | ? | [?,?] | ? |
| **XGBoost** | HOG | ? | ? | ? | [?,?] | ? |
| **LightGBM** | Full(opt) | ? | ? | ? | [?,?] | ? |
| **LightGBM** | HOG | ? | ? | ? | [?,?] | ? |
| **LR + PCA** | Full(opt) | ? | ? | ? | [?,?] | ? |
| **MLP** | Full(opt) | ? | ? | ? | [?,?] | ? |
| **Voting Ensemble** | Full(opt) | ? | ? | ? | [?,?] | ? |
| KNN | Full(opt) | ? | ? | ? | [?,?] | ? |

> Bold = new models to add.
> KNN kept as last row (weakest expected) — move to appendix if it pulls focus.

---

## 4. PCA Experiment (one extra column, not a new model)

Add a PCA pre-processing variant for HOG features only:

```python
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

# HOG is 1944-dim → reduce to 95% variance
pca_svm = Pipeline([
    ("pca", PCA(n_components=0.95, random_state=SEED)),
    ("svm", SVC(kernel="rbf", C=10, gamma="scale")),
])
pca_xgb = Pipeline([
    ("pca", PCA(n_components=0.95, random_state=SEED)),
    ("xgb", XGBClassifier(...)),
])
```

Add as footnote row in table:

| SVM + PCA(95%) | HOG | ? | ? | ? | → compare to SVM on raw HOG |

**Paper framing:**
> *"PCA was applied to the HOG feature vector (1944-dim) to assess
> whether dimensionality reduction improves generalisation.
> The retained variance threshold was set at 95%."*

---

## 5. Feature Importance Plot (free, add to paper)

After fitting XGBoost and RF on Full(opt):

```python
import pandas as pd
import matplotlib.pyplot as plt

feature_names = (
    [f"GLCM_{i}"  for i in range(6)]   +
    [f"LBP_{i}"   for i in range(10)]  +
    [f"DWT_{i}"   for i in range(16)]  +
    [f"HOG_{i}"   for i in range(1944)]+
    [f"Stats_{i}" for i in range(11)]
)

# XGBoost importances
importances = best_xgb.feature_importances_
# Aggregate by descriptor group
groups = {
    "GLCM":  importances[0:6].sum(),
    "LBP":   importances[6:16].sum(),
    "DWT":   importances[16:32].sum(),
    "HOG":   importances[32:1976].sum(),
    "Stats": importances[1976:].sum(),
}
pd.Series(groups).sort_values().plot(kind="barh", color="#0F6E56")
plt.xlabel("Cumulative feature importance")
plt.title("XGBoost feature group importance — Full(opt)")
plt.tight_layout()
plt.savefig("xgb_feature_importance.png", dpi=180)
```

This figure **directly supports your paper's core argument**:
if HOG has the highest group importance → confirms CNN+HOG is the right hybrid choice.

---

## 6. Updated Grid Size Summary (for paper §Methods)

| Model | Candidates | × 5-fold CV = Fits |
|---|---|---|
| SVM (4 kernels) | 903 | 4,515 |
| Random Forest | 14,112 | 70,560 |
| KNN | 384 | 1,920 |
| **XGBoost** | **6,480** | **32,400** |
| **LightGBM** | **13,824** | **69,120** |
| **LR** | **108** | **540** |
| **MLP** | **432** | **2,160** |
| **ExtraTrees** (optional) | 162 | 810 |
| **Total** | **36,405** | **182,025** |

---

## 7. Code Block — Add to Your Notebook

```python
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

NEW_MODEL_GRIDS = {

    "XGBoost": {
        "n_estimators":     [100, 300, 500, 800],
        "max_depth":        [3, 4, 5, 6, 8],
        "learning_rate":    [0.01, 0.05, 0.1, 0.2],
        "subsample":        [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
        "min_child_weight": [1, 3, 5],
        "gamma":            [0, 0.1, 0.3],
    },

    "LightGBM": {
        "n_estimators":      [100, 300, 500, 800],
        "max_depth":         [-1, 4, 6, 8],
        "learning_rate":     [0.01, 0.05, 0.1, 0.2],
        "num_leaves":        [15, 31, 63, 127],
        "subsample":         [0.7, 0.8, 1.0],
        "colsample_bytree":  [0.7, 0.8, 1.0],
        "min_child_samples": [5, 10, 20],
    },

    "LR": [
        Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(
                C=c, penalty=p, solver="saga",
                max_iter=1000, class_weight=cw,
                l1_ratio=l1 if p == "elasticnet" else None
            ))
        ])
        for c   in [0.001, 0.01, 0.1, 1, 10, 100]
        for p   in ["l1", "l2", "elasticnet"]
        for l1  in ([0.1, 0.5, 0.9] if p == "elasticnet" else [None])
        for cw  in ["balanced", None]
    ],

    "MLP": {
        "hidden_layer_sizes": [(256,), (512,), (256,128),
                               (512,256), (512,256,128)],
        "activation":         ["relu", "tanh"],
        "alpha":              [1e-4, 1e-3, 1e-2],
        "learning_rate_init": [1e-3, 5e-4, 1e-4],
        "batch_size":         [32, 64, 128],
        "early_stopping":     [True],
        "max_iter":           [500],
    },

    "ExtraTrees": {
        "n_estimators":      [100, 300, 500],
        "max_features":      ["sqrt", "log2", None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf":  [1, 2, 4],
        "criterion":         ["gini", "entropy"],
    },
}

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
```

---

## 8. What This Changes in Your Paper

| Before | After |
|---|---|
| 3 models (SVM, KNN, RF) | 7 models (+ XGBoost, LightGBM, LR, MLP) |
| No gradient boosting | Full gradient boosting comparison |
| No linear baseline | LR as interpretable lower bound |
| No ML–DL bridge | MLP bridges ML and DL sections |
| No feature importance | XGBoost + RF group importance plot |
| Reviewer question: "why not XGBoost?" | Preemptively answered |
| ML section = weak baseline | ML section = **competitive benchmark** |

---

*Cross-reference: `paper_reduction_guide.md` → §4 ML · `paper_writing_guide.md` → §4.3*