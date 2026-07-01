BRAIN TUMOR DETECTION & CLASSIFICATION — LaTeX report
=====================================================

COMPILE
  pdflatex main.tex        (run twice so the table of contents resolves)
  # or:  latexmk -pdf main.tex

FILE LAYOUT
  main.tex                  preamble + \input of every section. Rarely edited.
  results.tex               >>> EDIT ONLY THIS FILE after training/benchmarking <<<
                            All fill-in metrics are \newcommand / \def macros here.
                            Unfilled values show in RED as [CV], [Acc], etc.
  sections/00_abstract.tex
  sections/01_problem.tex
  sections/02_dataset.tex
  sections/03_strategy.tex
  sections/04_descriptors.tex      (GLCM, LBP, DWT, HOG)
  sections/05_hyperparams.tex      (Phase 1 — separability selection)
  sections/06_fusion_benchmark.tex (Phase 2 — fusion + GridSearchCV)
  sections/07_statistical_validation.tex
  sections/08_results_discussion.tex
  sections/09_conclusion.tex

ADDING YOUR RESULTS
  1. Open results.tex.
  2. Replace each \TODO{...} placeholder with your value, e.g.
        \def\svmAcc{0.961}   \def\svmFone{0.958}
        \def\bestModel{SVM}  \def\bestFeatureSet{Full(opt)}
  3. Recompile. The red markers turn into your numbers automatically.

ADDING FIGURES
  Put images in a figs/ folder and uncomment the \includegraphics blocks
  (commented examples are in 02_dataset.tex; add more where the
  "Figure placeholder" boxes are).

NOTES
  - Phase 1 numbers, dataset sizes, and feature dimensions are already filled
    (taken from the notebook) and normally need no change.
  - lmodern + microtype expansion are optional; this compiles on a minimal
    TeX Live too.
