from agents.writers.base_writer import BaseWriter

TASK = """Write Feature Importance Analysis section.

Cover:
- RF branch importance on Full(opt): HOG dominates raw dims; GLCM/Stats high per-dimension
- Link to Phase 1 separability scores
- Optional XGBoost group importance if in experiment data

Requirements:
- Ground claims in experiment/metrics data only.
- ~300-400 words.
- Start with ## Feature Importance Analysis heading."""


def write(context_pkg: dict, cfg: dict, existing_drafts: dict) -> str:
    return BaseWriter().run_task(context_pkg, TASK)
