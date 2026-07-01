from agents.writers.base_writer import BaseWriter

TASK = """Write Phase 2 Results: classifier benchmark on held-out split.

Cover:
- Top-10 (model, feature-set) pairs by macro-F1 from phase2_all_modes_metrics.csv
- KNN Full(opt) best (F1 ~0.946); SVM Full(opt) baseline (F1 ~0.912)
- SVM HOG-only vs Full(opt) delta
- RF partial results if in data

Requirements:
- Use ONLY numbers from experiment data.
- Refer to tables/figures; do not invent XGBoost/LightGBM results if absent.
- ~500-600 words.
- Start with ## Phase 2 Results heading."""


def write(context_pkg: dict, cfg: dict, existing_drafts: dict) -> str:
    return BaseWriter().run_task(context_pkg, TASK)
