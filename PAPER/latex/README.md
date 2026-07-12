# Research Paper — LaTeX (IEEE Conference)

Two-phase methodology paper: **separability-guided feature configuration** (Phase 1) followed by **statistically validated classifier benchmarking** (Phase 2).

## Title

*Separability-Guided Feature Configuration and Statistically-Validated Classifier Benchmarking for Brain Tumor MRI Classification*

## Build

```powershell
cd PAPER/latex
.\build.ps1          # compile main.pdf
.\build.ps1 -Setup   # first-time MiKTeX package install
```

## Regenerating result tables and figures

```powershell
python PAPER/latex/scripts/gen_results_tables.py
python PAPER/latex/scripts/paper_phase2_viz.py
python PAPER/latex/scripts/export_to_markdown.py
```

`export_to_markdown.py` writes `main.md` from all sections included in `main.tex`.

## File layout

| File | Section |
|------|---------|
| `main.tex` | Preamble, title, keywords, input chain |
| `sections/00_abstract.tex` | Abstract |
| `sections/01_introduction.tex` | Section 1 Introduction (1.1-1.5) — **done** |
| `sections/02_related_work.tex` | Section 2 Related Work (2.1-2.5) — **done** |
| `sections/03_materials_setup.tex` | Section 3 Materials and Experimental Setup — **done** |
| `sections/04_phase1_method.tex` | Section 4 Phase 1 methodology (core novelty) — **done** |
| `sections/05_feature_representation.tex` | Section 5 Feature representation and cleaning — **done** |
| `sections/06_phase2_method.tex` | Section 6 Phase 2 classifier benchmarking — **done** |
| `sections/07_results.tex` | Section 7 Results and Discussion — **done** (populated from benchmark) |
| `sections/08_interpretability.tex` | Section 8 Feature importance — **done** |
| `sections/09_statistics.tex` | Section 9 Statistical validation (RQ4) — **done** |
| `sections/11_limitations.tex` | Section 11 Limitations — **done** |
| `sections/12_conclusion.tex` | Section 12 Conclusion and Future Work (10.1–10.2) — **done** |
| `tables/leaderboard_top10.tex` | Top-10 CV leaderboard (auto-generated) |
| `tables/stats_top5.tex` | Top-5 with $\kappa$ and bootstrap CI |
| `figures/` | Plots copied from `Rapport/figures/` (31 PNG files) |
| `references.bib` | Bibliography |

## Adding your section text

1. Open the matching file under `sections/`.
2. Replace lines containing `\placeholder{...}` with your prose.
3. Keep existing `\cite{...}`, `\ref{...}`, `\input{tables/...}`, and figure environments.
4. Rebuild with `.\build.ps1`.

Previous draft content from `latex-old/` has been migrated where it fit the new structure.

## Narrative flow

Problem -> Gap -> Phase 1 (method) -> Features -> Phase 2 (method) -> Results -> Interpretability -> Statistics -> Discussion -> Limitations -> Conclusion
