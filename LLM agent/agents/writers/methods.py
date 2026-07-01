from agents.writers.base_writer import BaseWriter

TASK = """Write the Methods section for the ML-only paper (Phase 1 + Phase 2). Subsections:

### Dataset and Preprocessing
- 1,500 MRI images, 3 classes, 80/20 stratified split, 128×128, CLAHE, border crop.

### Separability Metrics (Phase 1)
- FDR, MI, DBI composite score; 213 configs across GLCM/LBP/DWT/HOG.

### Feature Extraction
- Describe GLCM, LBP, DWT, HOG with optimal hyperparameters from experiment data.
- Eight feature sets (A–D and unions); cleaning pipeline (variance, correlation, scaler).

### Phase 2 Classifier Benchmark
- Eight models: SVM, KNN, RF, XGBoost, LightGBM, LR, MLP, ExtraTrees; GridSearchCV.

### Evaluation
- Macro F1 primary; bootstrap CI, McNemar, Friedman planned.

Requirements:
- Use ONLY data from context and experiment records.
- Length: ~700-900 words.
- Start with ## Methods heading."""


def write(context_pkg: dict, cfg: dict, existing_drafts: dict) -> str:
    return BaseWriter().run_task(context_pkg, TASK)
