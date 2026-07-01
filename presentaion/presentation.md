# Classification d'images médicales par Machine Learning
## Presentation Baseline — Slide-by-Slide Script

> **Purpose of this file.** This is the master baseline for a ≤20-minute talk on classifying brain-tumor MRI with *handcrafted features + classical ML* (no deep learning). It is written to be handed to a slide-generation agent **and** used by the speaker.
>
> **How to read each slide block:**
> - **PROBLÉMATIQUE** — the single question this slide must answer (the reason it exists).
> - **DÉFINITIONS** — terms the audience needs, in plain language.
> - **CONTENU / VISUEL** — what the agent should put on the slide.
> - **NOTES ORATEUR** — what the speaker says out loud (not printed on the slide).
> - **⏱** — target time.
>
> **Legend for data status:**
> - ✅ = real number, confirmed from `statistical_validation_summary.csv` / `phase2_checkpoint_manifest.json`.
> - 🟡 = **PLACEHOLDER** — fill from the files that did *not* upload: `Projet7_brainTumorDetection.html`, `phase1_runs_manifest.csv`, `glcm_history.csv`, `lbp_history.csv`, `dwt_history.csv`, `hog_history.csv`.
>
> **Global facts (confirmed):** 3 classes · 8 feature-sets (branches) · 8 models · 6 eval modes (1 split + 5-fold CV) = **384 runs**. Best result: **Full(opt) + SVM = 92.0% accuracy**.

---

## GRAND PROBLÉMATIQUE (the thesis of the whole talk)

> *Can carefully engineered handcrafted image features, fed to classical machine-learning classifiers, reach clinically-interesting accuracy on brain-tumor MRI — without the data, compute, and opacity of deep learning?*

**Answer the talk delivers:** Yes — **92% accuracy, Cohen's κ = 0.88**, with a fully interpretable, reproducible, low-compute pipeline.

---

# SLIDE 1 — Titre & accroche
**⏱ 0:30 · 1 slide**

- **PROBLÉMATIQUE:** Why should the audience care in the first 30 seconds?
- **CONTENU / VISUEL:**
  - Title: *Classification d'images médicales par Machine Learning — détection de tumeurs cérébrales*
  - Subtitle: *Feature engineering + ML classique vs. Deep Learning*
  - Your name, affiliation, date.
  - One clean MRI image as background.
- **NOTES ORATEUR:** "In the next 20 minutes I'll show that you don't always need deep learning. With the right handcrafted features and classic classifiers we reach 92% on a 3-class brain-tumor problem — and unlike a black-box CNN, we can explain every step."

---

# SLIDE 2 — Machine Learning vs Deep Learning
**⏱ 1:30 · 1 slide**

- **PROBLÉMATIQUE:** Why choose classical ML over deep learning for this problem?
- **DÉFINITIONS:**
  - **Machine Learning classique** — the engineer designs (extracts) the features by hand; the model only learns the decision boundary. Performance is *bounded by feature quality*.
  - **Deep Learning** — the network learns the features *and* the decision jointly, but needs large labelled datasets, GPUs, and yields limited interpretability.
- **CONTENU / VISUEL:** 2-column comparison table.

  | Critère | ML classique (ce projet) | Deep Learning |
  |---|---|---|
  | Features | Conçues à la main (GLCM, LBP, DWT, HOG) | Apprises automatiquement |
  | Données requises | Peu | Beaucoup |
  | Compute | CPU | GPU |
  | Interprétabilité | Élevée | Faible (boîte noire) |
  | Point faible | Qualité des features | Données + coût |

- **NOTES ORATEUR:** "The whole game in classical ML is feature extraction. If the features don't separate the classes, no classifier will save you. So most of this talk is about *building good features* — and proving they're good *before* training anything."

---

# SLIDE 3 — Le dataset
**⏱ 1:00 · 1 slide**

- **PROBLÉMATIQUE:** What exactly are we classifying, and is the problem balanced/hard?
- **DÉFINITIONS:**
  - **Glioma** — tumor arising from glial cells.
  - **Meningioma** — tumor of the meninges (brain/spinal-cord membranes).
  - **Pituitary Tumor** — tumor of the pituitary gland.
- **CONTENU / VISUEL:**
  - 3 classes ✅: **Glioma, Meningioma, Pituitary Tumor**.
  - One representative MRI per class, side by side.
  - Class counts + train/test split 🟡 *(fill from phase1 manifest / notebook)*.
  - Note preprocessing: grayscale, resize to fixed size 🟡.
- **NOTES ORATEUR:** "Three tumor types. Visually they overlap — meningioma and glioma especially — which is exactly why a single feature family won't be enough."

---

# SLIDE 4 — Les 4 familles de features (partie 1)
**⏱ 1:15 · 1 slide**

- **PROBLÉMATIQUE:** What information does each feature family extract from an image?
- **DÉFINITIONS:**
  - **GLCM (Gray-Level Co-occurrence Matrix)** — statistics of how often pairs of gray levels occur at a given distance/angle. Captures **texture**: contrast, homogeneity, energy, correlation.
  - **LBP (Local Binary Pattern)** — thresholds each pixel's neighborhood into a binary code, then histograms it. Captures **local micro-texture** and is illumination-robust.
- **CONTENU / VISUEL:** For GLCM and LBP: 1-line intuition + a small "what it sees" illustration (co-occurrence grid; LBP-coded patch).
- **NOTES ORATEUR:** "GLCM asks: how do gray levels sit next to each other? LBP asks: what's the local pattern around each pixel? Both are texture descriptors, at different scales."

---

# SLIDE 5 — Les 4 familles de features (partie 2)
**⏱ 1:15 · 1 slide**

- **PROBLÉMATIQUE:** (continued) What do the frequency- and shape-based features add?
- **DÉFINITIONS:**
  - **DWT (Discrete Wavelet Transform)** — decomposes the image into multi-scale frequency sub-bands (approximation + horizontal/vertical/diagonal details). Captures **multi-resolution structure**.
  - **HOG (Histogram of Oriented Gradients)** — histograms of gradient orientations over a grid of cells. Captures **shape and edges** (contours of the tumor mass).
- **CONTENU / VISUEL:** For DWT and HOG: 1-line intuition + illustration (wavelet sub-bands; HOG gradient arrows).
- **NOTES ORATEUR:** "DWT looks at structure across scales; HOG captures shape and edges. Spoiler: HOG turns out to be the single strongest family — hold that thought."

---

# SLIDE 6 — Stratégie de benchmark des paramètres (le cœur méthodologique)
**⏱ 1:30 · 1 slide**

- **PROBLÉMATIQUE:** Each feature has knobs (distances, radius, wavelet, cell size…). How do we pick the best parameters **without training a classifier for every combination** (expensive + risk of overfitting)?
- **DÉFINITIONS (les 3 métriques de sélection):**
  - **FDR (Fisher's Discriminant Ratio)** — ratio of between-class variance to within-class variance. **Higher = classes are better separated** along that feature. Supervised, cheap.
  - **MI (Mutual Information)** — how much knowing the feature reduces uncertainty about the class label. **Higher = more informative**; captures non-linear dependence.
  - **DBI (Davies–Bouldin Index)** — average similarity between each cluster and its most similar one. **Lower = tighter, better-separated clusters.**
- **CONTENU / VISUEL:** The selection loop as a diagram:
  `feature params → extract → score with (FDR ↑, MI ↑, DBI ↓) → keep best config`.
- **NOTES ORATEUR:** "Instead of brute-forcing classifiers, we score each parameter configuration directly on class separability — FDR and MI should be high, DBI should be low. This is fast, avoids data leakage, and is model-agnostic."

---

# SLIDE 7 — Meilleurs paramètres par feature + taille du vecteur
**⏱ 1:30 · 1 slide**

- **PROBLÉMATIQUE:** What configuration won for each feature, and how big is each resulting vector?
- **CONTENU / VISUEL:** Table — 🟡 **fill from `glcm/lbp/dwt/hog_history.csv`**:

  | Feature | Best params (🟡) | FDR (🟡) | MI (🟡) | DBI (🟡) | Vector size (🟡) |
  |---|---|---|---|---|---|
  | GLCM | distances=?, angles=?, props=? | ? | ? | ? | ? |
  | LBP | P=?, R=?, method=? | ? | ? | ? | ? |
  | DWT | wavelet=?, level=? | ? | ? | ? | ? |
  | HOG | orientations=?, pixels/cell=?, cells/block=? | ? | ? | ? | ? |

- **NOTES ORATEUR:** "These are the winning configs chosen purely by the three metrics — before any model saw the data." *(Speaker: state the total combined vector length once table is filled.)*

---

# SLIDE 8 — Le pipeline de branches (fusion des features)
**⏱ 1:30 · 1 slide**

- **PROBLÉMATIQUE:** Is one feature family enough, or does combining them help? We test every meaningful combination.
- **DÉFINITIONS:** "Branch" = one feature-set fed to the classifiers. The 8 branches ✅:
  - **A — Stats** — basic statistical features.
  - **B — Tex(opt)** — texture (GLCM + LBP), optimized.
  - **C — DWT(opt)** — wavelet features, optimized.
  - **D — HOG** — gradient/shape features.
  - **A+B** — stats + texture.
  - **B+C(opt)** — texture + wavelet.
  - **A+B+C(opt)** — stats + texture + wavelet.
  - **Full(opt)** — everything fused.
- **CONTENU / VISUEL:** Tree/flow diagram: raw features → 8 branches → classifier bank.
- **NOTES ORATEUR:** "We don't guess which combination is best — we build all 8 branches and let the results decide."

---

# SLIDE 9 — Nettoyage du pipeline
**⏱ 1:00 · 1 slide**

- **PROBLÉMATIQUE:** Fused feature vectors are messy (different scales, redundant/constant columns, NaNs). What breaks a model if left unhandled?
- **DÉFINITIONS / ÉTAPES:**
  - **Standardisation (scaling)** — put all features on the same scale (critical for SVM, KNN, MLP, LR).
  - **NaN / constant / low-variance removal** — drop columns that carry no signal.
  - **De-duplication / correlation pruning** — remove redundant features from fusion.
- **CONTENU / VISUEL:** Before → after: feature-count drop 🟡, "clean matrix" schematic.
- **NOTES ORATEUR:** "Fusion creates redundancy and scale mismatch. Distance- and gradient-based models are very sensitive to this, so cleaning is not optional."

---

# SLIDE 10 — Les modèles & le protocole d'évaluation
**⏱ 1:30 · 1 slide**

- **PROBLÉMATIQUE:** How do we test fairly, and how do we know a score isn't a lucky split?
- **DÉFINITIONS:**
  - **8 classifiers** ✅: **SVM, KNN, LR (Logistic Regression), RF (Random Forest), ExtraTrees, XGBoost, LightGBM, MLP.**
  - **Train/test split** — single hold-out; fast but variance-prone.
  - **5-fold Cross-Validation** — rotate the test fold 5×, average → robust estimate + variance.
- **CONTENU / VISUEL:** Grid "8 branches × 8 models × 6 modes = **384 runs**" ✅. Small 5-fold CV diagram.
- **NOTES ORATEUR:** "Eight models, eight branches, one split plus five folds — 384 total experiments. That's what lets us make statistical claims later instead of just showing one number."

---

# SLIDE 11 — Performance globale (vue d'ensemble)
**⏱ 1:30 · 1 slide**

- **PROBLÉMATIQUE:** Across everything, what's the big picture — which branches and models dominate?
- **CONTENU / VISUEL:** Heatmap (rows = 8 branches, cols = 8 models, color = accuracy) **or** grouped bar chart. Confirmed accuracies (split) to anchor it ✅:
  - Best model **per branch**: A—Stats→SVM 0.787 · B—Tex(opt)→XGBoost 0.827 · C—DWT(opt)→ExtraTrees 0.763 · D—HOG→SVM **0.903** · A+B→ExtraTrees 0.850 · B+C(opt)→XGBoost 0.853 · A+B+C(opt)→RF 0.857 · **Full(opt)→SVM 0.920**.
- **NOTES ORATEUR:** "Two things jump out: SVM and tree-boosters lead, and the two brightest rows are HOG and Full. DWT alone is the weakest at ~76%."

---

# SLIDE 12 — Leaderboard
**⏱ 1:00 · 1 slide**

- **PROBLÉMATIQUE:** What is the single best pipeline — and does fusing everything actually beat the best single feature?
- **CONTENU / VISUEL:** Top-of-leaderboard table ✅ (from `statistical_validation_summary.csv`, sorted by accuracy):

  | Rank | Feature Set | Model | Accuracy | F1-macro | κ (Kappa) |
  |---|---|---|---|---|---|
  | 🥇 1 | **Full(opt)** | **SVM** | **0.920** | **0.9204** | **0.88** |
  | 🥈 2 | D — HOG | SVM | 0.903 | 0.9032 | 0.855 |
  | 🥉 3 | Full(opt) | KNN | 0.900 | 0.9002 | 0.85 |
  | 4 | D — HOG | KNN | 0.897 | 0.8967 | 0.845 |
  | 5 | Full(opt) | LightGBM | 0.893 | 0.8938 | 0.84 |

- **NOTES ORATEUR:** "Here's the punchline and the twist: full fusion wins at 92%, but **HOG alone hits 90.3%**. The extra features buy us less than 2 points. That's a real engineering trade-off — is fusion worth the complexity? We'll test that statistically."

---

# SLIDE 13 — Résultats par feature-set (per-branch plots)
**⏱ 1:00 · 1 slide**

- **PROBLÉMATIQUE:** How stable is each branch across models and CV folds?
- **CONTENU / VISUEL:** Small-multiples: one box/bar plot per branch showing accuracy spread across the 8 models (and/or 5 folds). Highlight Full(opt) and D—HOG.
- **NOTES ORATEUR:** "HOG and Full aren't just high on average — they're tight across models and folds, which means the result is reliable, not a single lucky configuration."

---

# SLIDE 14 — Résumé + test d'inférence
**⏱ 1:30 · 1 slide**

- **PROBLÉMATIQUE:** Does the winning model actually work on a fresh, unseen image?
- **CONTENU / VISUEL:**
  - 3-line recap: best pipeline = **Full(opt) + SVM**, **92% / κ 0.88** ✅.
  - Live (or screenshot) inference: random test MRI → predicted class + confidence → true label. 🟡 *(image + predicted probs)*.
- **NOTES ORATEUR:** "To close the loop: take a random test scan the model never saw, run the full pipeline end-to-end, and here's the prediction with its confidence."

---

# SLIDE 15 — Validation statistique
**⏱ 1:00 · 1 slide**

- **PROBLÉMATIQUE:** Is the winner *significantly* better, or within noise? A single accuracy number proves nothing.
- **DÉFINITIONS (méthodes retenues pour la présentation):**
  - **Bootstrap 95% CI** — resample the test predictions many times to get a confidence interval on the metric. If intervals don't overlap, the difference is real.
  - **Cohen's κ (Kappa)** — agreement corrected for chance; 0.81–1.0 = "almost perfect."
  - **McNemar's test** — paired test on the two top models' errors; answers directly "is fusion worth it vs HOG alone?"
- **CONTENU / VISUEL:** Error-bar chart of accuracy with 95% CI for the top branches ✅:
  - **Full(opt)+SVM: 0.920, 95% CI [0.89, 0.95], CV std ±0.0099, κ = 0.88.**
  - **D—HOG+SVM: 0.903, 95% CI [0.87, 0.94], κ = 0.855.**
  - McNemar Full(opt) vs HOG: 🟡 *(compute p-value; state whether the ~2-pt gain is significant)*.
- **NOTES ORATEUR:** "Kappa of 0.88 is almost-perfect agreement beyond chance. The 95% confidence interval sits clearly above the weaker branches, so the win is real — not a lucky split. The one honest caveat is Full vs HOG: they're close, and McNemar tells us whether fusion is statistically justified."
- **BACKUP (Q&A only):** Friedman + Nemenyi to rank all 8 models simultaneously — mention only if asked.

---

# SLIDE 16 — Conclusion
**⏱ 0:30 · 1 slide (optional, fits in buffer)**

- **PROBLÉMATIQUE:** What should the audience remember tomorrow?
- **CONTENU / VISUEL (3 takeaways):**
  1. Handcrafted features + classical ML → **92% / κ 0.88** on 3-class brain-tumor MRI, fully interpretable.
  2. **Metric-driven feature selection (FDR/MI/DBI)** picks parameters cheaply and without leakage.
  3. **HOG alone is 90.3%** — fusion adds <2 pts; interpretability and simplicity may outweigh it.
- **NOTES ORATEUR:** "Deep learning isn't the only path. With good feature engineering and honest validation, classical ML is competitive, cheaper, and explainable."

---

## TIMING TOTAL
0:30 + 1:30 + 1:00 + 1:15 + 1:15 + 1:30 + 1:30 + 1:30 + 1:00 + 1:30 + 1:30 + 1:00 + 1:00 + 1:30 + 1:00 + 0:30 = **≈ 19:30** (leaves ~30s buffer / Q&A hook). Slide 16 is optional if you run long.

---

## CHECKLIST — data still needed to remove all 🟡
- [ ] `Projet7_brainTumorDetection.html` — preprocessing details, inference screenshot, any figures.
- [ ] `phase1_runs_manifest.csv` — dataset counts, split ratio.
- [ ] `glcm_history.csv`, `lbp_history.csv`, `dwt_history.csv`, `hog_history.csv` — best params, FDR/MI/DBI scores, vector sizes (Slide 7), training curves (Slide 13).
- [ ] McNemar p-value for Full(opt)+SVM vs D—HOG+SVM (Slide 15).

## NOTE FOR THE SLIDE AGENT
- Language of the target deck: **set explicitly** (French / English / bilingual) — this baseline mixes FR headings with EN notes; normalize before generating.
- Keep ~1 slide per minute; do not exceed 6 lines of text per slide — push detail into speaker notes.
- Every slide's "PROBLÉMATIQUE" line is its reason to exist; if a slide can't answer its problematique, cut it.