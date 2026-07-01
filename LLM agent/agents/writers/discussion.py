from agents.writers.base_writer import BaseWriter

TASK = """Write the Discussion section for the ML-only paper. Address:

1. Separability-first parameter selection vs grid-search coupled to one classifier.
2. Why KNN/SVM on HOG or Full(opt) outperform RF on the same features.
3. Modest gain from concatenating all descriptor branches.
4. Limitations: single dataset, incomplete Phase 2 model runs, external validation needed.
5. Practical recommendation for resource-limited clinical prototyping.

Requirements:
- Ground claims in experiment data where numbers are cited.
- Length: ~500-700 words.
- Start with ## Discussion heading."""


def write(context_pkg: dict, cfg: dict, existing_drafts: dict) -> str:
    writer = BaseWriter()
    extra = ""
    if "methods" in existing_drafts:
        extra = "\n\n### Methods excerpt\n" + " ".join(existing_drafts["methods"].split()[:200])
    return writer.run_task(
        {**context_pkg, "paper_chunks": context_pkg.get("paper_chunks", [])},
        TASK + extra,
    )
