# Paper Structure — ML-Only Focused Paper
## "Handcrafted Feature Optimisation and Classifier Benchmarking for Brain Tumor MRI Classification"

> **Scope decision:** This paper covers ONLY the ML pipeline from your notebook.
> No DL, no VLM. One focused contribution, publishable in 10–12 pages.
>
> **Target journals (Scopus Q1/Q2):**
> - *Expert Systems with Applications* (Elsevier) — IF 8.5
> - *Computers in Biology and Medicine* (Elsevier) — IF 7.7
> - *Biomedical Signal Processing and Control* (Elsevier) — IF 5.1
> - *Applied Soft Computing* (Elsevier) — IF 7.2

---

## Core Argument (one sentence)

> *A classifier-free separability-based parameter search over GLCM, LBP, DWT,
> and HOG descriptors identifies optimal handcrafted features that, when combined
> and fed to a rigorous 8-model benchmark with full statistical validation,
> achieve competitive brain tumor MRI classification without any deep learning.*

---

## Paper Structure — Section by Section

---

## §1 — Abstract *(write last, ~250 words)*

**What to include:**
- Problem: brain tumor MRI classification, 3 classes, clinical stakes
- Gap: most papers use default feature parameters without optimization
- Method: 2-phase pipeline — classifier-free parameter selection → 8-model benchmark
- Key results (fill after runs):
  - Best model + feature set + F1-macro + accuracy
  - Cohen's κ value
  - Statistical significance (McNemar p-value)
- Takeaway: optimised handcrafted features + SVM achieve [X]% accuracy
  without GPU or deep learning

**Template sentence:**
> *"We present a two-phase framework for brain tumor MRI classification
> using exclusively handcrafted features. Phase 1 selects optimal parameters
> for GLCM, LBP, DWT, and HOG descriptors using three classifier-free
> separability metrics — Fisher's Discriminant Ratio, Mutual Information,
> and the Davies–Bouldin Index — avoiding costly full-training sweeps.
> Phase 2 benchmarks eight classifiers (SVM, KNN, Random Forest, XGBoost,
> LightGBM, Logistic Regression, MLP, ExtraTrees) on the optimised feature
> sets. The best configuration achieves F1-macro = [X] with Cohen's κ = [X],
> confirmed statistically significant via McNemar, Wilcoxon, and
> Friedman–Nemenyi tests."*

---

## §2 — Introduction *(~700 words)*

**Paragraph 1 — Clinical motivation:**
- Brain tumors: Glioma, Meningioma, Pituitary — different prognosis, require accurate differentiation
- MRI as gold standard, radiologist workload, automation need

**Paragraph 2 — Existing work:**
- CNN/DL dominates recent literature BUT requires GPU, large data, black-box
- Handcrafted features (texture, shape, frequency) remain clinically interpretable
- Most handcrafted-feature papers use DEFAULT parameters → suboptimal

**Paragraph 3 — The gap (your exact contribution):**
> *"Parameter selection for handcrafted descriptors is typically performed
> heuristically or via exhaustive grid search coupled to a classifier,
> introducing optimisation bias. No study has applied classifier-free
> separability scoring to jointly optimise GLCM, LBP, DWT, and HOG
> parameters before a multi-classifier benchmark on brain tumor MRI."*

**Paragraph 4 — Contributions (bullet list):**
- C1: Classifier-free parameter optimisation via FDR + MI + DBI
  (searches 144 GLCM + 36 LBP + 21 DWT + 12 HOG = **213 configs in Phase 1**)
- C2: 8-classifier benchmark (SVM · KNN · RF · XGBoost · LightGBM · LR · MLP · ExtraTrees)
  with GridSearchCV (**~36,000 candidate configs** in Phase 2)
- C3: Feature-branch importance analysis (RF + XGBoost group importances)
- C4: Full statistical validation (κ · Bootstrap CI · McNemar · Wilcoxon · Friedman–Nemenyi)
- C5: Soft voting ensemble (SVM + XGBoost + LightGBM)

**Paragraph 5 — Organisation:**
> *"The rest of the paper is organised as follows: §2 reviews related work..."*

---

## §3 — Related Work *(~700 words)*

### 3.1 Handcrafted texture features for MRI classification
- GLCM in medical imaging (cite 3–4 papers, 2020–2025)
- LBP for texture in brain MRI (cite 2–3 papers)
- DWT / wavelet for MRI classification (cite 2–3 papers)
- HOG for shape-based medical imaging (cite 2 papers)

### 3.2 Classical ML classifiers in brain tumor classification
- SVM dominance in the field (cite 3 papers — includes your found papers)
- Ensemble methods: RF, XGBoost applied to MRI radiomic features
- MLP as bridge to DL

### 3.3 Parameter selection strategies
- Default parameters: most papers use sklearn defaults
- Grid search coupled to classifier = expensive + overfitting risk
- Separability metrics as alternative (cite 1–2 papers if found)

**Gap closure sentence:**
> *"Despite extensive use of GLCM, LBP, DWT, and HOG in brain tumor MRI
> classification, no published work combines classifier-free separability
> scoring for parameter optimisation with a comprehensive 8-classifier
> benchmark and full statistical validation on the same dataset."*

---

## §4 — Dataset & Preprocessing *(~400 words)*

### 4.1 Dataset
From your notebook (cell 5):

| Property | Value |
|---|---|
| Source | Brain Tumor Detection & Classification (Kaggle) |
| Classes | Glioma · Meningioma · Pituitary Tumor |
| Total images | 1,500 (500 per class — balanced) |
| Split | 80 / 20 train / test (stratified, `random_state=SEED`) |
| Cross-validation | 5-fold StratifiedKFold |

**Include:**
- [ ] Class balance figure (`class_balance_train.png` — cell 54)
- [ ] Sample images figure (3×3 grid: 1 image per class × before/after CLAHE)
- [ ] Note: balanced dataset → SMOTE not required (from cell 54 check)

### 4.2 Preprocessing
From your notebook (cells 5, 7):

| Step | Detail |
|---|---|
| Resize | 128 × 128 pixels |
| Black border crop | `crop_black(img, thr=0.05)` — removes dark frame artifacts |
| Normalisation | CLAHE histogram equalisation (`USE_CLAHE = True`) |
| Format | Grayscale (texture features are grayscale-based) |

**Include:**
- [ ] Before/after CLAHE side-by-side figure (at least 1 per class)
- [ ] 1-sentence justification of CLAHE for MRI

---

## §5 — Feature Extraction & Parameter Selection (Phase 1) *(~900 words)*

> **This is your most original section. Most papers skip this entirely.**

### 5.1 Separability Metrics (classifier-free)
From your notebook (cells 8, 9):

**Three metrics used — describe each:**

| Metric | Formula / Principle | Why used |
|---|---|---|
| **Fisher's Discriminant Ratio (FDR)** | $\text{FDR} = \frac{1}{p}\sum_{j=1}^{p}\frac{\sigma_B^2(j)}{\sigma_W^2(j)+\varepsilon}$ | Between-class / within-class variance ratio |
| **Mutual Information (MI)** | $\text{MI}(X_j; Y) = \sum p(x,y)\log\frac{p(x,y)}{p(x)p(y)}$ | Captures non-linear class dependency |
| **Davies–Bouldin Index (DBI)** | $\text{DBI} = \frac{1}{k}\sum_i\max_{j\neq i}\frac{s_i+s_j}{d_{ij}}$ | Cluster compactness vs. separation (lower = better) |

**Combined score:** composite ranking across the three metrics to select the best parameter configuration per descriptor.

### 5.2 GLCM — Phase 1A
From notebook cell 13:

**Parameter grid searched:**
```
distances  : [1] · [1,2] · [1,2,4] · [1,2,4,8] · [1,3,5] · [2,4,8]  (6 options)
angles     : [0°,45°,90°,135°] subsets                                  (4 options)
levels     : 64 · 128 · 256                                             (3 options)
symmetric  : True · False                                               (2 options)
→ 6 × 4 × 3 × 2 = 144 configurations
```

**Output dimension:** 6 features (contrast, dissimilarity, homogeneity, energy, correlation, ASM)
**Optimal params found:** distances=[1,2,4,8] *(fill after run)*
**Include:** FDR/MI/DBI score plot vs. parameter combinations (from cell 15)

### 5.3 LBP — Phase 1B
From notebook cell 16:

```
P (sampling points) : 8 · 16 · 24 · 32        (4 options)
R (radius)          : 1 · 2 · 3                (3 options)
method              : uniform · nri_uniform · ror  (3 options)
→ 4 × 3 × 3 = 36 configurations
```

**Output dimension:** 10 features (histogram of LBP codes)
**Optimal params found:** P=8, R=1 *(fill after run)*
**Include:** LBP score plot (from cell 18)

### 5.4 DWT — Phase 1C
From notebook cell 19:

```
wavelet family : haar · db2 · db4 · bior1.3 · bior2.2 · sym2 · coif1  (7 options)
decomp. levels : 1 · 2 · 3                                              (3 options)
→ 7 × 3 = 21 configurations
```

**Output dimension:** 16 features (energy + entropy per sub-band)
**Optimal params found:** bior1.3, level=1 *(fill after run)*
**Include:** DWT score plot (from cell 21)

### 5.5 HOG — Phase 1D
From notebook cell 22:

```
orientations   : 6 · 9 · 12        (3 options)
pixels/cell    : 8 · 16            (2 options)
cells/block    : 2 · 3             (2 options)
block_norm     : L2-Hys            (1 option)
→ 3 × 2 × 2 × 1 = 12 configurations
```

**Output dimension:** up to 1,944 features (depends on orientations × cells)
**Optimal params found:** orientations=6 *(fill after run)*
**Include:** HOG score plot (from cell 24)

### 5.6 Phase 1 Summary Table
From notebook cell 26:

| Descriptor | Optimal Parameters | Output Dim | Selection Metric Rank |
|---|---|---|---|
| GLCM | distances=[1,2,4,8] | 6 | FDR=[?] · MI=[?] · DBI=[?] |
| LBP | P=8, R=1 | 10 | FDR=[?] · MI=[?] · DBI=[?] |
| DWT | bior1.3, level=1 | 16 | FDR=[?] · MI=[?] · DBI=[?] |
| HOG | orientations=6 | 1,944 | FDR=[?] · MI=[?] · DBI=[?] |
| Stats | mean, std, skew, kurtosis… | 11 | — |
| **Full(opt)** | all combined | **1,987** | — |

---

## §6 — Feature Cleaning & Dimensionality Handling *(~350 words)*

From notebook cell 29 (`clean_scale` function):

### 6.1 Pipeline applied to all feature sets before training

| Step | Method | Effect |
|---|---|---|
| **1. Constant feature removal** | `VarianceThreshold(0.0)` | Removes zero-variance features |
| **2. Correlation filtering** | Pearson correlation matrix, threshold=0.95 | Removes redundant features |
| **3. Standardisation** | `StandardScaler` (zero mean, unit variance) | Required for SVM, LR, MLP |

**Report the dimension reduction per feature set:**

| Feature Set | Raw dim | After variance filter | After corr filter | Final dim |
|---|---|---|---|---|
| GLCM | 6 | ? | ? | ? |
| LBP | 10 | ? | ? | ? |
| DWT | 16 | ? | ? | ? |
| HOG | 1,944 | ? | ? | ? |
| Full(opt) | 1,987 | ? | ? | ? |

**Note:** Correlation filtering only applied if dim < 5,000 (from your code) → HOG may be exempt.
Mention this as a deliberate design choice.

### 6.2 t-SNE Visualisation
From notebook cell 58:

- t-SNE applied to best feature set (n=500 subsampled training points)
- 2D projection coloured by class label
- **Purpose in paper:** visually confirms class separability in the handcrafted feature space
- **Include:** `tsne_best_feature_set.png`

---

## §7 — Classifier Benchmark (Phase 2) *(~600 words)*

### 7.1 Models and search spaces
From notebook cell 30:

| Model | Key hyperparameters searched | Candidates | × 5-fold |
|---|---|---|---|
| SVM | 4 kernels (linear/rbf/poly/sigmoid), C, γ, degree, coef₀ | 903 | 4,515 |
| KNN | n_neighbors (12), metric (4), weights (2), p (4) | 384 | 1,920 |
| Random Forest | n_estimators, max_depth, max_features, criterion, bootstrap | 14,112 | 70,560 |
| XGBoost | n_estimators, max_depth, lr, subsample, colsample_bytree | 6,480 | 32,400 |
| LightGBM | n_estimators, max_depth, lr, num_leaves, subsample | 13,824 | 69,120 |
| Logistic Reg. | C (6), penalty (3), solver=saga, l1_ratio | 108 | 540 |
| MLP | hidden_layer_sizes (5), activation (2), α (3), lr (3) | 432 | 2,160 |
| ExtraTrees | n_estimators, max_features, min_samples | 162 | 810 |
| **Total** | | **36,405** | **182,025** |

### 7.2 Evaluation protocol
- **GridSearchCV**: scoring = `f1_macro`, refit = True
- **Test evaluation**: accuracy, F1-macro, F1-weighted, precision (macro), recall (macro)
- **All models use same train/test split** (same SEED, same stratified split)

### 7.3 Voting Ensemble
From notebook cell 43:

- Soft voting: SVM + XGBoost + LightGBM on Full(opt)
- Uses `predict_proba` from each model, averages probability vectors
- **No additional hyperparameter search** — uses individually tuned best models

---

## §8 — Feature Importance Analysis *(~400 words)*

### 8.1 RF branch importance
From notebook cell 56:

- RF trained on Full(opt) with best hyperparameters from Phase 2
- `feature_importances_` summed per branch: Stats · GLCM · LBP · DWT · HOG
- **Figure:** `rf_branch_importance.png` — horizontal bar chart
- **Expected finding:** HOG dominates (1,944 dims), but normalised per-dim importance may reveal GLCM as most informative per feature

### 8.2 XGBoost group importance
From notebook cell 42:

- Same group aggregation on XGBoost `feature_importances_`
- **Key table to report:**

| Feature Group | XGBoost Importance | RF Importance | Dims | Importance/Dim |
|---|---|---|---|---|
| Stats | ? | ? | 11 | ? |
| GLCM | ? | ? | 6 | ? |
| LBP | ? | ? | 10 | ? |
| DWT | ? | ? | 16 | ? |
| HOG | ? | ? | 1,944 | ? |

> **The "Importance/Dim" column is KEY** — it shows which descriptor carries
> the most information per feature dimension, regardless of its raw size.
> This is what justifies the paper's contribution.

### 8.3 Connection to Phase 1
> *"The feature group ranked highest by XGBoost and RF importance should
> correspond to the descriptor with the highest Phase 1 separability score.
> Agreement between Phase 1 (unsupervised separability) and Phase 2
> (supervised importance) validates the classifier-free selection strategy."*

---

## §9 — Diagnostics *(~300 words, can be brief)*

### 9.1 Class balance check
From notebook cell 54:

- Imbalance ratio = max(class_count) / min(class_count)
- Dataset is balanced → ratio ≈ 1.0
- Mention: `class_weight='balanced'` tested in LR and RF grids as robustness check

### 9.2 Seed stability
From notebook cell 60:

- Best model re-run with 3 different random seeds (SEED, SEED+1, SEED+2)
- Report: mean ± std of F1-macro across seeds
- **Claim:** *"Results are stable across random splits, confirming the
  findings are not artifacts of a specific data partition."*

| Seed | F1-macro | Accuracy | Best params |
|---|---|---|---|
| SEED | ? | ? | ? |
| SEED+1 | ? | ? | ? |
| SEED+2 | ? | ? | ? |
| **Mean ± Std** | **? ± ?** | **? ± ?** | — |

---

## §10 — Statistical Validation *(~600 words)*

> **Every result claim must be backed by at least one test in this section.**

### 10.1 Cohen's Kappa (κ)
From notebook cell 69:

- Computed for every (Feature Set × Model) on split test
- **Report:** full κ heatmap (`kappa_heatmap.png`)
- **In text:** κ of best model, κ scale interpretation

### 10.2 Bootstrap Confidence Intervals (95% CI)
From notebook cell 71:

- N = 2,000 resamples, joint resampling of (y_true, y_pred) pairs
- **Report:** CI for F1-macro and Accuracy for top-5 models
- **Figure:** error bar plot (`bootstrap_ci_best_fs.png`)

### 10.3 Cross-Validation Fold Statistics
From notebook cell 74:

- Mean ± Std of F1-macro across 5 CV folds per (Feature Set × Model)
- **Report:** models with smallest std = most stable
- **Figure:** bar plot with error bars for Full(opt) (`cv_fold_stats.png`)

### 10.4 McNemar's Test
From notebook cell 77:

- Pairwise comparison of top classifiers on same test set
- Edwards continuity correction applied
- **Report:** matrix of p-values (top-5 models × top-5 models)
- Significant pairs (p < 0.05) highlighted

### 10.5 Wilcoxon Signed-Rank Test
From notebook cell 79:

- Paired non-parametric test across 5 CV fold scores
- Between: best two models on Full(opt)
- **Report:** W-statistic + p-value for each pair

### 10.6 Friedman + Post-Hoc Nemenyi
From notebook cell 81, 82:

- Friedman across all classifiers × 5 folds
- If p < 0.05 → Nemenyi post-hoc (via `scikit_posthocs`)
- **Figure:** Nemenyi heatmap of pairwise p-values OR Critical Difference diagram

### 10.7 Publication-Ready Summary Table
From notebook cell 84:

| Model | Feature Set | F1-macro | Accuracy | Std (CV) | 95% CI | κ | McNemar p |
|---|---|---|---|---|---|---|---|
| SVM | Full(opt) | ? | ? | ? | [?,?] | ? | — |
| XGBoost | Full(opt) | ? | ? | ? | [?,?] | ? | ? |
| LightGBM | Full(opt) | ? | ? | ? | [?,?] | ? | ? |
| RF | Full(opt) | ? | ? | ? | [?,?] | ? | ? |
| MLP | Full(opt) | ? | ? | ? | [?,?] | ? | ? |
| Voting | Full(opt) | ? | ? | ? | [?,?] | ? | ? |

---

## §11 — Results & Discussion *(~800 words)*

### 11.1 Phase 1 — Parameter selection findings
- Which descriptor had the highest separability scores?
- Do FDR, MI, and DBI agree on the ranking?
- How much does the optimal configuration improve over default params?
  → Report: F1-macro with optimal params vs. F1-macro with default params (if testable)

### 11.2 Phase 2 — Classifier comparison
- Which model won? On which feature set?
- Is the winning feature set Full(opt) or a single descriptor?
- Tier the models: top / mid / bottom group

### 11.3 Feature importance findings
- Does XGBoost group importance agree with Phase 1 separability ranking?
- Which descriptor has the highest importance-per-dimension?
- **Paper-worthy sentence:**
  > *"HOG accounts for [X]% of total XGBoost importance, but represents
  > 97.8% of the feature dimensions. On a per-dimension basis, GLCM/Stats
  > contribute [X]× more information per feature, confirming that
  > classifier-free separability scoring correctly identified their value."*

### 11.4 Statistical significance
- Which model differences are statistically significant (McNemar p < 0.05)?
- What is the bootstrap CI spread? Do CIs overlap between models?
- Is the Friedman test significant? What does Nemenyi say?

### 11.5 Limitations
- Single dataset (1,500 images) → external validation needed
- No augmentation → may limit generalisation to unseen scanners
- Handcrafted features assume grayscale → ignores colour/multi-modal information
- Parameter search is sequential per descriptor → joint optimisation may differ

---

## §12 — Conclusion *(~200 words)*

**3 findings to state:**
1. Classifier-free separability scoring (FDR + MI + DBI) successfully selects optimal
   handcrafted feature parameters, validated by agreement with supervised feature importance
2. [Best model] on Full(opt) achieves F1-macro = [X], Cohen's κ = [X],
   confirmed significant via McNemar (p=[X]) and Friedman–Nemenyi
3. HOG dominates in raw importance but GLCM/Stats carry highest per-dimension
   information — justifying feature-selective rather than feature-additive approaches

**Future work (3 bullets max):**
- Apply framework to multi-modal MRI (T1, T2, FLAIR)
- Integrate as a handcrafted branch in a hybrid CNN (→ that's your next paper)
- Extend separability-based search to joint cross-descriptor optimisation

---

## Figures Checklist (max 8 for most journals)

| # | Figure | Source cell | Size |
|---|---|---|---|
| 1 | Sample MRI images × CLAHE pre/post (3×3) | cell 7 | half |
| 2 | Phase 1 separability scores per descriptor (4 subplots) | cells 15,18,21,24 | full |
| 3 | Phase 1 summary — optimal params table (as figure) | cell 26 | half |
| 4 | t-SNE projection of best feature set | cell 58 | half |
| 5 | Phase 2 leaderboard — top-10 bar chart | cell 49 | full |
| 6 | RF + XGBoost feature group importance (side by side) | cells 42, 56 | full |
| 7 | Bootstrap CI error bars — top-5 models | cell 72 | half |
| 8 | Nemenyi post-hoc heatmap OR Kappa heatmap | cells 82, 69 | half |

---

## Page Budget (target: 10–12 pages)

| Section | Pages |
|---|---|
| Abstract | 0.3 |
| Introduction | 1.0 |
| Related work | 1.2 |
| Dataset & preprocessing | 0.6 |
| **Feature extraction & Phase 1** | **1.8** ← most important |
| Feature cleaning | 0.4 |
| Classifier benchmark | 0.8 |
| Feature importance | 0.6 |
| Diagnostics | 0.4 |
| **Statistical validation** | **1.0** |
| Results & discussion | 1.2 |
| Conclusion | 0.3 |
| References (35–45 refs) | 1.4 |
| **Total** | **~11.0** |

---

## Things Still Needed Before Writing

| Item | Status | Priority |
|---|---|---|
| Run all Phase 2 models (XGBoost, LightGBM, LR, MLP, ExtraTrees) | ❌ TODO | 🔴 HIGH |
| Phase 1 separability score outputs (actual metric values) | ❌ TODO | 🔴 HIGH |
| Feature importance values (RF + XGBoost groups) | ❌ TODO | 🔴 HIGH |
| Statistical test outputs (κ, CI, McNemar, Friedman) | ❌ TODO | 🔴 HIGH |
| Seed stability table (3 seeds) | ❌ TODO | 🟡 MEDIUM |
| CLAHE before/after figure | ❌ TODO | 🟡 MEDIUM |
| t-SNE figure | ❌ TODO | 🟡 MEDIUM |
| Default params vs. optimal params comparison | ❌ TODO | 🟡 MEDIUM |

---

*Notebook reference: `Projet7_brainTumorDetection.ipynb` — 86 cells · Phases 1 & 2 only*