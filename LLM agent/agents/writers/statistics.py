from agents.writers.base_writer import BaseWriter

TASK = """Write Statistical Validation section.

Cover:
- Top Phase 2 models from stats_top5.csv: KNN/SVM Full(opt), KNN/SVM HOG
- Bootstrap CI, Cohen's kappa, McNemar, Friedman — mark TODO if not exported
- Planned pairwise tests: KNN vs SVM Full(opt); SVM Full vs SVM HOG

Requirements:
- Use stats_top5.csv values from experiment data; do not invent p-values.
- ~300-400 words.
- Start with ## Statistical Validation heading."""


def write(context_pkg: dict, cfg: dict, existing_drafts: dict) -> str:
    return BaseWriter().run_task(context_pkg, TASK)
