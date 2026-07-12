"""
IEEE-friendly Phase 2 figures for the research paper.

Generates compact, readable plots from phase2_benchmark_results.csv
(or leaderbordML.json). Avoids the dense 3-panel-per-feature-set layout.

Usage:
    python PAPER/latex/scripts/paper_phase2_viz.py
    python PAPER/latex/scripts/paper_phase2_viz.py --out PAPER/latex/figures
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CSV = ROOT / "phase2_benchmark_results.csv"
DEFAULT_JSON = ROOT / "output" / "leaderbordML.json"
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "figures"

MODEL_ORDER = [
    "SVM", "KNN", "RF", "XGBoost", "LightGBM", "LR", "MLP", "ExtraTrees",
]
PALETTE = {
    "SVM": "#4C72B0",
    "KNN": "#DD8452",
    "RF": "#55A868",
    "XGBoost": "#C44E52",
    "LightGBM": "#8172B3",
    "LR": "#CCB974",
    "MLP": "#64B5CD",
    "ExtraTrees": "#8C8C8C",
}
FS_ORDER = [
    "A — Stats",
    "B — Tex(opt)",
    "C — DWT(opt)",
    "D — HOG",
    "A+B",
    "B+C(opt)",
    "A+B+C(opt)",
    "Full(opt)",
]
FS_SHORT = {
    "A — Stats": "A",
    "B — Tex(opt)": "B",
    "C — DWT(opt)": "C",
    "D — HOG": "D",
    "A+B": "A+B",
    "B+C(opt)": "B+C",
    "A+B+C(opt)": "A+B+C",
    "Full(opt)": "Full",
}

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)


def _model_color(name: str) -> str:
    return PALETTE.get(name, "#888888")


def load_results(csv_path: Path = DEFAULT_CSV, json_path: Path = DEFAULT_JSON) -> pd.DataFrame:
    if csv_path.exists():
        df = pd.read_csv(csv_path)
    elif json_path.exists():
        df = pd.DataFrame(json.loads(json_path.read_text(encoding="utf-8")))
    else:
        raise FileNotFoundError(f"No results at {csv_path} or {json_path}")
    df["F1-macro"] = df["F1-macro"].astype(float)
    df["Accuracy"] = df["Accuracy"].astype(float)
    return df


def _ordered_fs(df: pd.DataFrame) -> list[str]:
    present = set(df["Feature Set"].unique())
    return [fs for fs in FS_ORDER if fs in present] + sorted(present - set(FS_ORDER))


def plot_f1_heatmap(df: pd.DataFrame, out_path: Path) -> None:
    """Models × feature sets — single overview figure for the paper."""
    fs_list = _ordered_fs(df)
    pivot = pd.DataFrame(
        {
            m: [
                df.loc[(df["Feature Set"] == fs) & (df["Model"] == m), "F1-macro"].max()
                if ((df["Feature Set"] == fs) & (df["Model"] == m)).any()
                else np.nan
                for fs in fs_list
            ]
            for m in MODEL_ORDER
        },
        index=[FS_SHORT.get(fs, fs) for fs in fs_list],
    )

    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    bounds = [0.0, 0.70, 0.80, 0.90, 1.0]
    cmap = plt.get_cmap("YlGnBu", len(bounds) - 1)
    norm = BoundaryNorm(bounds, cmap.N)
    im = ax.imshow(pivot.values, aspect="auto", cmap=cmap, norm=norm)
    ax.set_xticks(range(len(MODEL_ORDER)))
    ax.set_xticklabels(MODEL_ORDER, rotation=30, ha="right")
    ax.set_yticks(range(len(fs_list)))
    ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j]
            if np.isnan(val):
                continue
            ax.text(
                j, i, f"{val:.3f}",
                ha="center", va="center", fontsize=7,
                color="white" if val >= 0.85 else "black",
            )
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Test F1-macro")
    ax.set_title("Phase 2 benchmark: macro-F1 across feature sets and classifiers")
    fig.savefig(out_path)
    plt.close(fig)


def plot_best_per_feature_set(df: pd.DataFrame, out_path: Path) -> None:
    """Best model per feature set — compact horizontal bars."""
    rows = []
    for fs in _ordered_fs(df):
        sub = df[df["Feature Set"] == fs].sort_values("F1-macro", ascending=False).iloc[0]
        rows.append(sub)
    best = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    labels = [FS_SHORT.get(fs, fs) for fs in best["Feature Set"]]
    y = np.arange(len(best))
    colors = [_model_color(m) for m in best["Model"]]
    bars = ax.barh(y, best["F1-macro"], color=colors, edgecolor="white", height=0.65)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0.65, 1.0)
    ax.set_xlabel("Best test F1-macro")
    ax.set_title("Best classifier per feature representation")
    ax.invert_yaxis()
    for bar, (_, row) in zip(bars, best.iterrows()):
        ax.text(
            bar.get_width() + 0.004,
            bar.get_y() + bar.get_height() / 2,
            row["Model"],
            va="center", ha="left", fontsize=8,
        )
    from matplotlib.patches import Patch
    ax.legend(
        handles=[Patch(facecolor=_model_color(m), label=m) for m in MODEL_ORDER],
        loc="lower right", ncol=4, framealpha=0.9,
    )
    fig.savefig(out_path)
    plt.close(fig)


def plot_f1_bars_single_fs(
    df: pd.DataFrame,
    fs_name: str,
    out_path: Path,
    *,
    model_col: str = "Model",
    f1_col: str = "F1-macro",
) -> None:
    """All models on one feature set — sorted horizontal F1 bars only."""
    if model_col in df.columns and f1_col in df.columns and "Feature Set" not in df.columns:
        sub = df.copy()
    else:
        sub = df[df["Feature Set"] == fs_name].copy()
    sub = sub.sort_values(f1_col, ascending=True)
    if sub.empty:
        return

    short = FS_SHORT.get(fs_name, fs_name)
    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    y = np.arange(len(sub))
    colors = [_model_color(m) for m in sub[model_col]]
    bars = ax.barh(y, sub[f1_col], color=colors, edgecolor="white", height=0.62)
    ax.set_yticks(y)
    ax.set_yticklabels(sub[model_col])
    ax.set_xlim(0.65, 1.0)
    ax.set_xlabel("Test F1-macro")
    ax.set_title(f"Classifier comparison — {short}")
    for bar, val in zip(bars, sub[f1_col]):
        ax.text(
            val - 0.01, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", ha="right", fontsize=7, color="white",
        )
    fig.savefig(out_path)
    plt.close(fig)


def plot_leaderboard(df: pd.DataFrame, out_path: Path, top_n: int = 10) -> None:
    """Top-N configurations — horizontal composite score bars."""
    if "Composite" not in df.columns:
        df = df.copy()
        df["Composite"] = df["F1-macro"]
    top = df.sort_values("Composite", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    labels = [
        f"{FS_SHORT.get(r['Feature Set'], r['Feature Set'])} | {r['Model']}"
        for _, r in top.iterrows()
    ]
    y = np.arange(len(top))[::-1]
    colors = [_model_color(m) for m in top["Model"]]
    bars = ax.barh(y, top["F1-macro"], color=colors, edgecolor="white", height=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Test F1-macro")
    ax.set_title(f"Top-{top_n} classifier–feature-set combinations")
    ax.set_xlim(0.82, 0.94)
    for bar, val in zip(bars, top["F1-macro"]):
        ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=7)
    fig.savefig(out_path)
    plt.close(fig)


def crop_confmat_from_grid(
    composite_path: Path,
    out_path: Path,
    model_index: int = 0,
    ncol: int = 4,
    nrow: int = 2,
    top_margin: float = 0.10,
) -> bool:
    """Fallback: extract one panel from the 2×4 confusion-matrix grid."""
    if not composite_path.exists():
        return False
    im = Image.open(composite_path)
    w, h = im.size
    cell_w = w / ncol
    cell_h = (h * (1 - top_margin)) / nrow
    row, col = divmod(model_index, ncol)
    left = int(col * cell_w)
    upper = int(top_margin * h + row * cell_h)
    right = int(left + cell_w)
    lower = int(upper + cell_h)
    im.crop((left, upper, right, lower)).save(out_path)
    return True


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=25, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, f"{cm[i, j]:.2f}",
                ha="center", va="center", fontsize=9,
                color="white" if cm[i, j] > 0.55 else "black",
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(out_path)
    plt.close(fig)


def generate_paper_figures(
    df: pd.DataFrame,
    out_dir: Path,
    confmat_composite: Path | None = None,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    specs = [
        ("p2_f1_heatmap.png", lambda p: plot_f1_heatmap(df, p)),
        ("p2_f1_best_per_featureset.png", lambda p: plot_best_per_feature_set(df, p)),
        ("p2_f1_bars_full.png", lambda p: plot_f1_bars_single_fs(df, "Full(opt)", p)),
        ("p2_f1_bars_hog.png", lambda p: plot_f1_bars_single_fs(df, "D — HOG", p)),
        ("p2_leaderboard.png", lambda p: plot_leaderboard(df, p)),
    ]
    for name, fn in specs:
        path = out_dir / name
        fn(path)
        saved.append(path)

    cm_out = out_dir / "p2_confmat_best.png"
    composite = confmat_composite or ROOT / "output" / "p2_Full(opt)_confmat.png"
    if crop_confmat_from_grid(composite, cm_out, model_index=0):
        saved.append(cm_out)

    # Legacy alias used in some LaTeX paths
    dash = out_dir / "p2_overall_dashboard.png"
    plot_f1_heatmap(df, dash)
    saved.append(dash)

    return saved


def plot_fs_paper(fs_name: str, res_dict: dict, output_dir: Path, y_test, class_names) -> None:
    """
    Notebook helper: one readable figure per feature set (F1 bars + optional best CM).
    Call from the notebook instead of plot_fs() for paper exports.
    """
    from sklearn.metrics import confusion_matrix

    order = [m for m in MODEL_ORDER if m in res_dict]
    if not order:
        return
    safe = fs_name.replace(" ", "_").replace("+", "p").replace("—", "")
    rows = [{"Model": m, "F1-macro": res_dict[m]["f1m"]} for m in order]
    sub = pd.DataFrame(rows)
    plot_f1_bars_single_fs(sub, fs_name, output_dir / f"p2_{safe}_f1_bars.png")

    best_m = max(order, key=lambda m: res_dict[m]["f1m"])
    cm = confusion_matrix(y_test, res_dict[best_m]["y_pred"], normalize="true")
    short = FS_SHORT.get(fs_name, fs_name)
    plot_confusion_matrix(
        cm, class_names,
        output_dir / f"p2_{safe}_confmat_best.png",
        f"{short} — {best_m} (best)",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper-ready Phase 2 figures")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--also-output", type=Path, default=ROOT / "output",
                        help="Also write figures to project output/")
    args = parser.parse_args()

    df = load_results(args.csv, args.json)
    paths = generate_paper_figures(df, args.out)
    if args.also_output:
        generate_paper_figures(df, args.also_output)
    print(f"Generated {len(paths)} figures in {args.out}")
    for p in paths:
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
