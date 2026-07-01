from agents.writers.base_writer import BaseWriter

TASK = """Write Phase 1 Results: parameter selection for GLCM, LBP, DWT, HOG.

Cover:
- 213 configurations evaluated with FDR + MI + DBI composite score
- Optimal params per family from experiment data (phase1_param_search.json)
- LBP/HOG scores near 1.0; GLCM lower; DWT bior1.3 level 1
- Reference Fig phase1 grids and Table phase1

Requirements:
- Use ONLY numbers from experiment data.
- ~400-500 words.
- Start with ## Phase 1 Results heading."""


def write(context_pkg: dict, cfg: dict, existing_drafts: dict) -> str:
    return BaseWriter().run_task(context_pkg, TASK)
