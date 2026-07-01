from agents.writers.base_writer import BaseWriter

TASK = """Write the Results section for the ML-only paper (Phase 1 + Phase 2):

### Phase 1: Parameter Selection
- Optimal GLCM/LBP/DWT/HOG settings and separability scores from experiment data.

### Phase 2: Classifier Benchmark
- Top configurations on held-out split; KNN Full(opt) vs SVM Full(opt) vs HOG-only.
- Refer to tables; report exact F1-macro and accuracy from experiment data.

### Feature Importance
- RF/XGBoost group importance on Full(opt) if mentioned in data.

### Statistical Validation
- stats_top5 models; note TODO cells for bootstrap/McNemar if not filled.

Requirements:
- Use ONLY numbers from experiment data.
- Length: ~600-800 words.
- Start with ## Results heading."""


def write(context_pkg: dict, cfg: dict, existing_drafts: dict) -> str:
    return BaseWriter().run_task(context_pkg, TASK)
