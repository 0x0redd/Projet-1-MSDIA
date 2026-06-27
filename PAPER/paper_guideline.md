# 📄 Paper Writing Guide — Scopus Publication
## Brain Tumor MRI Classification: Ablation & Interpretability Study

> **Target venues (Scopus-indexed):**
> - *Computers in Biology and Medicine* (Elsevier) — Q1, IF ~7.7
> - *Biomedical Signal Processing and Control* (Elsevier) — Q1, IF ~5.1
> - *Expert Systems with Applications* (Elsevier) — Q1, IF ~8.5
> - *IEEE Journal of Biomedical and Health Informatics* — Q1, IF ~7.7
> - *Applied Soft Computing* (Elsevier) — Q2, IF ~7.2 ← fastest turnaround

---

## ✅ Phase 0 — Before You Write (Pre-writing)

### 0.1 Choose your venue first
- [ ] Pick ONE target journal before writing a single word
- [ ] Download the journal's **author guidelines** (page limits, figure formats, reference style)
- [ ] Read 3–5 recent papers from that journal on medical image classification
- [ ] Note their structure, table style, and how they report statistical tests
- [ ] Check the journal's **Aims & Scope** — confirm "ablation study" / "hybrid DL" fits
- [ ] Verify the journal is **Scopus-indexed** at [scopus.com/sources](https://www.scopus.com/sources)

### 0.2 Define your contribution statement (10 minutes, do this now)
Write one paragraph answering:
- [ ] What is the **problem**? (brain tumor classification, 3 classes, MRI)
- [ ] What is the **gap**? (no systematic ablation of handcrafted features in hybrid DL for this task)
- [ ] What is your **method**? (ML → Tier 0/1/2 DL → VLM, same pipeline, same data)
- [ ] What is the **key finding**? (CNN+HOG beats full fusion; SVM within 0.003 of best DL)
- [ ] What is the **takeaway**? (feature selection matters more than feature aggregation)

### 0.3 Gather all assets before writing
- [ ] All result tables exported from notebooks (F1, accuracy, precision, recall, time)
- [ ] Statistical test outputs (κ, Bootstrap CI, McNemar p-values, Friedman ranks)
- [ ] Grad-CAM visualisations for at least 2 models (CNN+HOG and VGG16)
- [ ] VLM token probability probe plots (before/after LoRA)
- [ ] Phase 2 leaderboard figure (`p2_leaderboard.png`)
- [ ] Confusion matrices for top-3 models + VLM
- [ ] Training loss curves for all DL models

---

## ✅ Phase 1 — Paper Structure & Sections

### §1 — Abstract *(~250 words, write LAST)*
- [ ] Sentence 1–2: Context (brain tumors, MRI, clinical stakes)
- [ ] Sentence 3–4: Problem gap (lack of systematic ablation of feature integration)
- [ ] Sentence 5–7: Method summary (dataset, pipeline, 3 paradigms, 10 models)
- [ ] Sentence 8–9: Key quantitative results (best model, F1, accuracy)
- [ ] Sentence 10: Conclusion / implication

> **Template:**
> *"Accurate classification of brain tumors from MRI is critical for early treatment planning. While deep learning has shown promise, the systematic contribution of handcrafted texture features in hybrid architectures remains underexplored. We present a controlled ablation study comparing [N] models across three paradigms — classical ML, hybrid deep learning, and vision-language models — on a balanced dataset of 1,500 MRI scans (Glioma, Meningioma, Pituitary Tumor). [...] The best model, CNN+HOG (ResNet-18 + HOG branch), achieved F1-macro = 0.933, outperforming both pure CNN backbones and full multi-branch fusion. Statistical validation via Cohen's κ, bootstrap confidence intervals, and McNemar's test confirms the significance of these differences. Our findings demonstrate that targeted feature integration outperforms indiscriminate feature aggregation, and that classical SVM remains competitive with deep models at a fraction of the computational cost."*

---

### §2 — Introduction *(~600–800 words)*
- [ ] **¶1 — Clinical motivation:** brain tumor prevalence, MRI as gold standard, radiologist workload
- [ ] **¶2 — Existing work summary:** CNNs for MRI (VGG, ResNet, EfficientNet), ViT, VLMs
- [ ] **¶3 — The gap:** no paper does a controlled ablation of GLCM/LBP/DWT/HOG in hybrid DL *on this task*
- [ ] **¶4 — Our approach:** 3-paradigm framework (ML → DL → VLM), same data, same split
- [ ] **¶5 — Contributions (bullet list):**
  - [ ] C1: First systematic ablation of handcrafted feature branches in hybrid DL for brain tumor MRI
  - [ ] C2: Multi-paradigm benchmark (ML, DL Tier 0/1/2, VLM) on the same dataset
  - [ ] C3: Rigorous statistical validation (κ, Bootstrap CI, McNemar, Wilcoxon, Friedman–Nemenyi)
  - [ ] C4: Interpretability analysis via Grad-CAM and VLM token probability probes
- [ ] **¶6 — Paper organisation:** "The rest of the paper is organised as follows..."

---

### §3 — Related Work *(~600–900 words)*

#### 3.1 Classical ML for medical image classification
- [ ] GLCM-based texture analysis (cite 3–5 papers)
- [ ] HOG and LBP in medical imaging (cite 2–3 papers)
- [ ] SVM for MRI classification benchmarks (cite 2–3 papers)

#### 3.2 Deep learning for brain tumor classification
- [ ] CNN-based (VGG, ResNet, EfficientNet) — cite seminal + recent
- [ ] Vision Transformer (ViT) for medical imaging
- [ ] Transfer learning in low-data medical settings

#### 3.3 Hybrid / multi-branch architectures
- [ ] Papers combining handcrafted + learned features
- [ ] Feature fusion strategies (concatenation, attention, gating)
- [ ] Note the gap: no controlled ablation of *which* feature matters

#### 3.4 Vision-language models in medical imaging
- [ ] CLIP, LLaVA, Med-PaLM for classification
- [ ] LoRA fine-tuning for domain adaptation
- [ ] Teacher forcing in sequence classification

> **Gap statement to close §3:**
> *"Despite the breadth of work on feature engineering and deep learning separately, no study has systematically ablated the contribution of individual handcrafted feature branches within a unified hybrid architecture, nor compared this approach directly against VLMs on the same benchmark. This work fills that gap."*

---

### §4 — Materials & Methods *(~1,200–1,500 words)*

#### 4.1 Dataset
- [ ] Source, total images: 1,500 (500 × 3 classes)
- [ ] Classes: Glioma, Meningioma, Pituitary Tumor
- [ ] Split: 70 / 10 / 20 (train / val / test) — stratified
- [ ] Class balance: balanced — no augmentation needed for balance
- [ ] Preprocessing: resize to 128×128, CLAHE normalisation

#### 4.2 Phase 1 — Feature extraction & parameter selection
- [ ] GLCM: distances=[1,2,4,8], 6-dim output
- [ ] LBP: P=8, R=1, 10-dim output
- [ ] DWT: bior1.3, level=1, 16-dim output
- [ ] HOG: orientations=6, 1944-dim output
- [ ] Separability metrics used: Fisher's Discriminant Ratio, Mutual Information, Davies–Bouldin Index
- [ ] Justify why classifier-free selection avoids overfitting the parameter search

#### 4.3 Phase 2 — Classical ML benchmark
- [ ] SVM: 4 kernels, full grid (903 candidates)
- [ ] KNN: 384 candidates
- [ ] Random Forest: 14,112 candidates
- [ ] 5-fold stratified CV, GridSearchCV, `n_jobs=-1`
- [ ] Report best config per model

#### 4.4 Phase 3 — Deep learning architectures
- [ ] Tier 0: VGG16, ResNet50, EfficientNet-B0, ViT-B/16 (pretrained ImageNet)
- [ ] Tier 1: ResNet-18 + single handcrafted branch (CNN+GLCM, LBPNet, WaveletCNN, CNN+HOG)
- [ ] Tier 2: MultiBranchFusion, AttentionFusion
- [ ] Fusion strategy description (concatenation after global avg pooling)
- [ ] Training config table (epochs, lr, batch size, AMP, early stopping)

#### 4.5 Phase 4 — Vision-language model (VLM)
- [ ] Base model: Qwen2-VL-2B-Instruct
- [ ] Quantisation: 4-bit NF4 (bitsandbytes)
- [ ] LoRA: r=8, α=16, dropout=0.05, 0.4162% trainable params
- [ ] Teacher forcing: label mask on input_ids, loss on assistant tokens only
- [ ] Prompt template (show the exact template used)

#### 4.6 Ablation design
- [ ] Table: ablation matrix (model × branch removed × metric)
- [ ] Controlled variable: all models trained on same split, same seed
- [ ] Explain what is isolated in each ablation run

#### 4.7 Evaluation metrics
- [ ] F1-macro (primary), Accuracy, Precision, Recall (all macro)
- [ ] Cohen's κ (agreement beyond chance)
- [ ] Bootstrap CI (N=2,000, 95%, paired resampling)
- [ ] McNemar (pairwise), Wilcoxon (paired CV), Friedman+Nemenyi (multi-model)

---

### §5 — Results *(~800–1,000 words + tables/figures)*

#### 5.1 Phase 2 — ML benchmark results
- [ ] Table: SVM / KNN / RF × feature set → F1, Accuracy, κ, Bootstrap CI
- [ ] Best: SVM on Full(opt), F1 = 0.9299
- [ ] Note: SVM significantly outperforms RF (McNemar p-value)

#### 5.2 Phase 3 — DL results
- [ ] Table 12 reproduced (all 10 models, F1, accuracy, time)
- [ ] Key finding: CNN+HOG (F1=0.933) > VGG16 (0.930) > MultiBranchFusion (0.906)
- [ ] Figure: bar chart or radar chart of all models
- [ ] Confusion matrices for top-3 DL models

#### 5.3 Ablation results (core section)
- [ ] Table: feature ablation — CNN+HOG baseline vs. HOG removed vs. GLCM removed, etc.
- [ ] Table: tier comparison — F1 by tier (0 / 1 / 2) with mean ± std
- [ ] Key finding: removing HOG from CNN+HOG causes the largest F1 drop
- [ ] Key finding: Tier 1 outperforms Tier 2 on average

#### 5.4 Phase 4 — VLM results
- [ ] Table 17: per-class precision, recall, F1, support
- [ ] Overall: F1-macro = 0.868, Accuracy = 86.67%
- [ ] Table: token probability probe (before/after LoRA) — CE loss delta
- [ ] Figure: confusion matrix

#### 5.5 Statistical validation
- [ ] Kappa heatmap (Feature Set × Model)
- [ ] Bootstrap CI table for top-5 models
- [ ] McNemar p-values matrix (best 3 models)
- [ ] Friedman test result + Nemenyi post-hoc significance matrix

---

### §6 — Discussion *(~700–900 words)*
- [ ] **6.1 — Why CNN+HOG beats full fusion:** HOG captures global orientation structure (relevant for tumor shape); GLCM/LBP/DWT may add noise in fusion
- [ ] **6.2 — SVM vs DL:** 0.003 F1 difference — discuss cost/benefit of DL in low-data medical settings
- [ ] **6.3 — VLM findings:** 0.868 F1 with no explicit feature engineering — discuss what the model "sees" via token probes
- [ ] **6.4 — Statistical significance:** which differences are real (McNemar p<0.05) and which are within noise (CI overlap)
- [ ] **6.5 — Limitations:**
  - [ ] Single dataset, single institution
  - [ ] VLM limited to 2B parameters (larger models may close the gap)
  - [ ] No external validation set
  - [ ] Grad-CAM not yet quantitatively evaluated

---

### §7 — Conclusion *(~200–250 words)*
- [ ] Restate the problem and approach (2 sentences)
- [ ] State the 3 key findings:
  1. CNN+HOG achieves best F1 (0.933) — targeted feature injection > full fusion
  2. SVM (0.930) remains competitive with SOTA DL at a fraction of the cost
  3. VLM achieves 0.868 without any handcrafted features — promising for zero-shot scenarios
- [ ] Future work:
  - [ ] Apply to multi-modal MRI (T1, T2, FLAIR)
  - [ ] Larger VLM (7B+) with multi-image context
  - [ ] Prospective clinical validation
  - [ ] Attention-based feature weighting in fusion

---

## ✅ Phase 2 — Figures & Tables Checklist

### Tables (mandatory)
- [ ] **Table 1** — Dataset summary (classes, splits, image count)
- [ ] **Table 2** — Feature extraction parameters (optimal values)
- [ ] **Table 3** — ML grid search sizes (SVM/KNN/RF candidates)
- [ ] **Table 4** — ML benchmark results (F1, accuracy, κ, CI per model × feature set)
- [ ] **Table 5** — DL training hyperparameters
- [ ] **Table 6** — DL results (all 10 models — use Table 12 from report)
- [ ] **Table 7** — **Ablation table** (feature removal × F1 delta) ← KEY TABLE
- [ ] **Table 8** — VLM classification report (per-class P/R/F1)
- [ ] **Table 9** — Statistical summary (κ, CI, McNemar, Friedman rank per model)

### Figures (mandatory)
- [ ] **Fig 1** — Overall framework diagram (ML → DL → VLM pipeline)
- [ ] **Fig 2** — Sample MRI images per class (3 × 3 grid, CLAHE pre/post)
- [ ] **Fig 3** — Phase 2 leaderboard bar chart (`p2_leaderboard.png`)
- [ ] **Fig 4** — DL results comparison (grouped bar chart, all 10 models)
- [ ] **Fig 5** — Ablation bar chart (F1 drop per removed feature)
- [ ] **Fig 6** — Grad-CAM heatmaps (CNN+HOG vs VGG16 on same images)
- [ ] **Fig 7** — VLM confusion matrix + token probability probe (before/after)
- [ ] **Fig 8** — Kappa heatmap (Feature Set × Model)
- [ ] **Fig 9** — Bootstrap CI plot (error bars, top models)

### Figure quality
- [ ] All figures at **300 DPI minimum** for submission
- [ ] All text in figures ≥ **8pt** (readable after print scaling)
- [ ] Colorblind-safe palette (avoid red/green only — add pattern/shape encoding)
- [ ] Figures saved as **PDF or EPS** for vector journals (Elsevier accepts both)
- [ ] Each figure has a self-contained caption (reader should not need the body text)

---

## ✅ Phase 3 — Writing Quality Checklist

### Language & style
- [ ] Write in **past tense** for methods and results ("we trained", "the model achieved")
- [ ] Write in **present tense** for claims and discussion ("this suggests", "the results indicate")
- [ ] No first person in abstract ("A systematic comparison was conducted..." not "We compared...")
- [ ] Every acronym defined at first use (MRI, CNN, SVM, HOG, GLCM, LBP, DWT, LoRA, VLM)
- [ ] Consistent notation: F1-macro (not F1_macro, not macro-F1)
- [ ] Numbers: spell out one through nine, use digits for 10+
- [ ] Decimal separator: period (0.933), not comma

### Scientific rigour
- [ ] Every claim backed by a table, figure, or statistical test
- [ ] No "our model is better" without a p-value or CI
- [ ] Reproduce the exact train/val/test split (report seed)
- [ ] State which GPU/CPU was used and training time
- [ ] Code availability statement (GitHub link or "available upon reasonable request")

### Submission requirements (Elsevier standard)
- [ ] Cover letter addressing the editor (2 paragraphs: what you did + why it fits the journal)
- [ ] Highlights: 3–5 bullet points, 85 characters max each
- [ ] Keywords: 5–8 terms (Brain tumor classification; Ablation study; Hybrid CNN; Texture features; LoRA fine-tuning)
- [ ] CRediT author contribution statement
- [ ] Declaration of competing interests
- [ ] Ethics statement (if human data — confirm IRB or dataset provenance)
- [ ] Data availability statement

---

## ✅ Phase 4 — Experiments Still Needed Before Submission

| Experiment | Status | Priority |
|---|---|---|
| Phase 2 extended grid search (76,995 fits) | 🔄 Running | HIGH |
| Ablation: CNN+HOG with HOG removed | ❌ TODO | HIGH |
| Ablation: CNN+HOG with each feature replaced | ❌ TODO | HIGH |
| Grad-CAM on CNN+HOG (top-3 errors + top-3 correct) | ❌ TODO | HIGH |
| Grad-CAM on VGG16 (same images for comparison) | ❌ TODO | MEDIUM |
| SHAP values on SVM Full(opt) features | ❌ TODO | MEDIUM |
| Tier mean ± std table (average over Tier 0 / 1 / 2) | ❌ TODO | MEDIUM |
| McNemar p-value matrix (all top models) | ❌ TODO | HIGH |
| Friedman + Nemenyi post-hoc table | ❌ TODO | HIGH |
| External validation on a second dataset | ❌ TODO | LOW |

---

## ✅ Phase 5 — Submission & Revision

### Before submitting
- [ ] Run manuscript through **Grammarly** or **LanguageTool** (academic mode)
- [ ] Check **iThenticate / Turnitin** similarity (< 15% target)
- [ ] Verify all references exist in Scopus (use [Scopus search](https://scopus.com))
- [ ] Confirm all cited papers are recent enough (majority < 5 years old)
- [ ] Double-check figure/table numbers match citations in text
- [ ] Confirm journal **APC fee** (open access or subscription) before submitting

### Timeline estimate
| Milestone | Estimated time |
|---|---|
| Finish missing experiments | 1–2 weeks |
| Write §4 Methods | 3–4 days |
| Write §5 Results | 2–3 days |
| Write §3 Related work | 3–4 days |
| Write §6 Discussion + §1 Intro + Abstract | 3–4 days |
| Internal review + revision | 1 week |
| **Submission** | **~5–6 weeks total** |

---

## 📌 Key Numbers to Report (from your notebooks)

```
Dataset        : 1,500 images · 500/class · 70/10/20 split
Best ML        : SVM Full(opt)   F1=0.9299  Acc=0.9300
Best DL        : CNN+HOG         F1=0.9331  Acc=0.9333  t=168.4s
Best VLM       : Qwen2-VL+LoRA   F1=0.8679  Acc=0.8667
LoRA params    : 9,232,384 / 2,218,217,984 = 0.4162%
VLM CE before  : Glioma 1.581 · Meningioma 1.459 · Pituitary 1.167
VLM CE after   : Glioma 0.053 · Meningioma 0.099 · Pituitary 0.000
Grid total     : 15,399 candidates × 5 folds = 76,995 fits/feature-set
```

---

*Generated from: Projet7_brainTumorDetection.ipynb · Deep_learning.ipynb · VLM.ipynb*