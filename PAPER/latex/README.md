# IEEE LaTeX Paper (ML-only: Phase 1 + Phase 2)

**Title:** Handcrafted Feature Optimisation and Classifier Benchmarking for Brain Tumor MRI Classification

## Build

Requires [MiKTeX](https://miktex.org/) or TeX Live (`pdflatex` + `bibtex` on PATH).

```powershell
cd PAPER\latex
.\build.ps1
```

### Sync figures and tables from notebook

```powershell
cd "LLM agent"
python scripts/prepare_data.py
python scripts/sync_paper_figures.py
python scripts/tables_to_latex.py
cd ..\PAPER\latex
.\build.ps1
```

### Word export (.docx)

```powershell
cd PAPER\latex
.\export_word.ps1
```

## Figure sources (ML paper)

| Figure | Source |
|--------|--------|
| Class balance, Phase 1 grids | `output/*.png` |
| t-SNE, Phase 2 leaderboard/dashboard | `output/p2_*.png` |
| RF branch importance | `output/rf_branch_importance.png` |
| Agent charts (optional) | `LLM agent/output/figures/*.pdf` |

Figures are copied to `PAPER/latex/figures/` by `sync_paper_figures.py`.

## References

Filtered ML papers: `PAPER/filtered - ML/` (19 PDFs, indexed via `prepare_data.py`).

## Pre-submission experiments

See [EXPERIMENTS_TODO.md](EXPERIMENTS_TODO.md) for remaining Phase 2 model runs and statistical tests.
