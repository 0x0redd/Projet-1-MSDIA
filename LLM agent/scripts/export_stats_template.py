#!/usr/bin/env python3
"""
Export stats_top5.csv template from current phase2/phase3 metrics.

Run full statistical tests in notebooks (bootstrap CI, McNemar, Friedman-Nemenyi)
then replace TODO columns in stats_top5.csv before final tables_to_latex run.

CNN-only baseline: train Tier-1 backbone without handcrafted branch and add row to ablation.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_EXP = PROJECT_ROOT / "data" / "experiments"
OUT = DATA_EXP / "stats_top5.csv"


def main() -> int:
    rows = []
    p3 = DATA_EXP / "phase3_all_metrics.csv"
    p2 = DATA_EXP / "phase2_all_modes_metrics.csv"

    if p3.exists():
        df = pd.read_csv(p3)
        for model, label in [
            ("cnn_hog", "CNN+HOG"),
            ("vgg16", "VGG16"),
            ("attention_fusion", "AttentionFusion"),
        ]:
            sub = df[(df["model"] == model) & (df["eval_mode"] == "split")]
            if not sub.empty:
                rows.append(
                    {
                        "model": label,
                        "f1_macro": sub.iloc[0]["f1_macro"],
                        "ci_low": "TODO",
                        "ci_high": "TODO",
                        "kappa": "TODO",
                        "mcnemar_p": "TODO",
                        "friedman_rank": "TODO",
                    }
                )

    if p2.exists():
        df2 = pd.read_csv(p2)
        sub = df2[
            (df2["eval_mode"] == "split")
            & (df2["model"] == "SVM")
            & (df2["feature_set"].str.contains("Full", na=False))
        ]
        if not sub.empty:
            rows.append(
                {
                    "model": "SVM Full(opt)",
                    "f1_macro": sub.iloc[0]["f1_macro"],
                    "ci_low": "TODO",
                    "ci_high": "TODO",
                    "kappa": "TODO",
                    "mcnemar_p": "TODO",
                    "friedman_rank": "TODO",
                }
            )

    rows.append(
        {
            "model": "Qwen2-VL+LoRA",
            "f1_macro": "TODO",
            "ci_low": "TODO",
            "ci_high": "TODO",
            "kappa": "TODO",
            "mcnemar_p": "TODO",
            "friedman_rank": "TODO",
        }
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"[export_stats] Wrote template -> {OUT}")
    print("[export_stats] Replace TODO after bootstrap/McNemar/Friedman notebook runs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
