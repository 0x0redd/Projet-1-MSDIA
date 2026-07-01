# Experiments to run before final submission (ML-only paper)

Scope: Phase 1 + Phase 2 only (`Projet7_brainTumorDetection.ipynb`, `paper_guideline.md`).

## Required notebook runs

1. **Phase 2 remaining models** — XGBoost, LightGBM, LR, MLP, ExtraTrees (all 8 feature sets).
2. **RF completion** — finish remaining feature sets (D, A+B, …, Full(opt)).
3. **Bootstrap CI** — 95%, N=2000 for top-5 Phase~2 models.
4. **McNemar** — KNN Full(opt) vs SVM Full(opt); SVM Full(opt) vs SVM HOG.
5. **Friedman + Nemenyi** — all eight Phase~2 classifiers on CV fold F1.
6. **Soft voting ensemble** — SVM + XGBoost + LightGBM on Full(opt) (cell 43).

## Export pipeline

```powershell
cd "LLM agent"
python scripts/prepare_data.py
python scripts/sync_paper_figures.py
python scripts/tables_to_latex.py
# Edit data/experiments/stats_top5.csv — replace TODO columns after statistical runs
cd ../PAPER/latex
.\build.ps1
```

## Figures synced to `PAPER/latex/figures/`

From `output/`: class balance, phase1 param plots, t-SNE, p2 leaderboard/dashboard, RF branch importance.
From `LLM agent/output/figures/`: leaderboard PDF (optional fallback).

## References

Filtered ML papers: `PAPER/filtered - ML/` (copied via `prepare_data.py` → `data/papers/`).
