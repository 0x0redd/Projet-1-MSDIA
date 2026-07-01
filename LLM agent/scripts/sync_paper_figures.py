#!/usr/bin/env python3
"""Copy notebook / agent ML figures into PAPER/latex/figures/."""

from __future__ import annotations

import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUTPUT = REPO / "output"
AGENT_FIG = REPO / "LLM agent" / "output" / "figures"
LATEX_FIG = REPO / "PAPER" / "latex" / "figures"

# (source relative to repo, dest filename under latex/figures)
COPY_MAP: list[tuple[Path, str]] = [
    (OUTPUT / "class_balance_train.png", "class_balance_train.png"),
    (OUTPUT / "phase1_summary.png", "phase1_summary.png"),
    (OUTPUT / "glcm_param_search.png", "glcm_param_search.png"),
    (OUTPUT / "lbp_param_search.png", "lbp_param_search.png"),
    (OUTPUT / "dwt_param_search.png", "dwt_param_search.png"),
    (OUTPUT / "hog_param_search.png", "hog_param_search.png"),
    (OUTPUT / "tsne_best_feature_set.png", "tsne_best_feature_set.png"),
    (OUTPUT / "p2_leaderboard.png", "p2_leaderboard.png"),
    (OUTPUT / "p2_overall_dashboard.png", "p2_overall_dashboard.png"),
    # Agent-generated leaderboard (PDF fallback if notebook PNG missing)
    (OUTPUT / "rf_branch_importance.png", "rf_branch_importance.png"),
    (OUTPUT / "descriptor_demo.png", "descriptor_demo.png"),
    (AGENT_FIG / "leaderboard_top10.pdf", "leaderboard_top10.pdf"),
    (AGENT_FIG / "accuracy_comparison.pdf", "accuracy_comparison.pdf"),
]


def main() -> int:
    LATEX_FIG.mkdir(parents=True, exist_ok=True)
    n = 0
    for src, name in COPY_MAP:
        if not src.is_file():
            print(f"[sync_figures] skip (missing): {src.relative_to(REPO)}")
            continue
        dst = LATEX_FIG / name
        shutil.copy2(src, dst)
        print(f"[sync_figures] {src.name} -> figures/{name}")
        n += 1
    print(f"[sync_figures] copied {n} file(s) to {LATEX_FIG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
