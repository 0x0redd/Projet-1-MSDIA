#!/usr/bin/env python3
"""Generate figures/dataset.{pdf,png}: class distribution + sample mosaic.

Data: figures/data/class_distribution.csv (auto-built from images_by_class if missing).
Run: python figures/scripts/gen_dataset_figure.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

FIGURES_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = FIGURES_DIR.parents[1]
DATA_CSV = FIGURES_DIR / "data" / "class_distribution.csv"
OUT_STEM = FIGURES_DIR / "dataset"

sys.path.insert(0, str(FIGURES_DIR))
import plot_defaults  # noqa: E402


def find_images_by_class() -> Path:
    candidates = sorted(REPO_ROOT.rglob("images_by_class"))
    if not candidates:
        raise FileNotFoundError("No images_by_class directory found under repo root.")
    return candidates[0]


def load_paths_and_labels(dataset_dir: Path) -> tuple[list[Path], np.ndarray, list[str]]:
    class_names = sorted(d.name for d in dataset_dir.iterdir() if d.is_dir())
    paths: list[Path] = []
    y: list[int] = []
    for i, name in enumerate(class_names):
        for p in sorted((dataset_dir / name).iterdir()):
            if p.is_file():
                paths.append(p)
                y.append(i)
    return paths, np.array(y), class_names


def build_distribution_csv(dataset_dir: Path) -> tuple[list[dict], list[Path], list[str]]:
    paths, y, class_names = load_paths_and_labels(dataset_dir)
    y_train, _, p_train, _ = train_test_split(
        y, paths, test_size=0.2, random_state=42, stratify=y
    )
    rows = []
    for i, name in enumerate(class_names):
        rows.append(
            {
                "class": name,
                "train": int((y_train == i).sum()),
                "test": int((y == i).sum() - (y_train == i).sum()),
            }
        )
    DATA_CSV.parent.mkdir(parents=True, exist_ok=True)
    with DATA_CSV.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["class", "train", "test"])
        w.writeheader()
        w.writerows(rows)
    return rows, p_train, class_names


def load_distribution() -> tuple[list[dict], list[Path], list[str]]:
    dataset_dir = find_images_by_class()
    paths, y, class_names = load_paths_and_labels(dataset_dir)
    _, _, p_train, _ = train_test_split(
        y, paths, test_size=0.2, random_state=42, stratify=y
    )
    if not DATA_CSV.exists():
        return build_distribution_csv(dataset_dir)

    with DATA_CSV.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return rows, p_train, class_names


def load_grayscale(path: Path, size: int = 128) -> np.ndarray:
    img = Image.open(path).convert("L")
    img = img.resize((size, size), Image.Resampling.LANCZOS)
    return np.asarray(img, dtype=np.float32) / 255.0


def main() -> int:
    plot_defaults.apply()
    rows, p_train, class_names = load_distribution()

    fig = plt.figure(figsize=(10.5, 4.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.28)
    ax_bar = fig.add_subplot(gs[0, 0])

    classes = [r["class"] for r in rows]
    train = [int(r["train"]) for r in rows]
    test = [int(r["test"]) for r in rows]
    x = np.arange(len(classes))
    w = 0.35
    ax_bar.bar(
        x - w / 2,
        train,
        w,
        label="Train",
        color=plot_defaults.COLOURS["train"],
        edgecolor="white",
        linewidth=0.6,
    )
    ax_bar.bar(
        x + w / 2,
        test,
        w,
        label="Test",
        color=plot_defaults.COLOURS["test"],
        edgecolor="white",
        linewidth=0.6,
    )
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(classes, rotation=12, ha="right")
    ax_bar.set_ylabel("Number of samples")
    ax_bar.set_title("Class distribution")
    ax_bar.legend(frameon=False)
    ax_bar.spines[["top", "right"]].set_visible(False)

    mosaic = gs[0, 1].subgridspec(3, 4, wspace=0.04, hspace=0.04)
    rng = np.random.default_rng(42)
    for cls_idx, name in enumerate(class_names):
        cls_paths = [p for p in p_train if p.parent.name == name]
        pick = list(rng.choice(cls_paths, size=4, replace=False))
        for col, path in enumerate(pick):
            ax = fig.add_subplot(mosaic[cls_idx, col])
            ax.imshow(load_grayscale(path), cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(name, fontsize=8)
    fig.text(0.73, 0.96, "Preprocessed train sample mosaic", ha="center", fontsize=11)

    plot_defaults.savefig(fig, str(OUT_STEM))
    plt.close(fig)
    print(f"[gen_dataset_figure] Wrote {OUT_STEM}.pdf / .png")
    print(f"[gen_dataset_figure] Data: {DATA_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
