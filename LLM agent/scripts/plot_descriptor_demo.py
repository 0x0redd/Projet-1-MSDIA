#!/usr/bin/env python3
"""Visualise GLCM, LBP, DWT, and HOG on one MRI slice (Phase 1 optimal params)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pywt
from matplotlib.gridspec import GridSpec
from skimage.feature import graycomatrix, graycoprops, hog, local_binary_pattern
from skimage.transform import resize

REPO = Path(__file__).resolve().parents[2]
CACHE_DIR = REPO / "cache" / "brain_tumor"
PHASE1_JSON = REPO / "LLM agent" / "data" / "experiments" / "phase1_param_search.json"
DEFAULT_OUT = REPO / "output" / "descriptor_demo.png"
LATEX_OUT = REPO / "PAPER" / "latex" / "figures" / "descriptor_demo.png"


def load_sample_image(index: int = 0) -> tuple[np.ndarray, str]:
    """Load one preprocessed slice from the notebook dataset cache."""
    npz_files = sorted(CACHE_DIR.glob("dataset_*.npz"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not npz_files:
        raise FileNotFoundError(f"No dataset_*.npz in {CACHE_DIR}")
    ds = np.load(npz_files[0], allow_pickle=True)
    for xkey, ykey in [
        ("X_train", "y_train"),
        ("X_train_raw", "y_train"),
        ("X_all", "y_all"),
    ]:
        if xkey in ds and ykey in ds:
            X, y = ds[xkey], ds[ykey]
            break
    else:
        raise KeyError(f"Unknown npz layout: {list(ds.keys())}")
    names = ds["class_names"].tolist() if "class_names" in ds else ["Glioma", "Meningioma", "Pituitary Tumor"]
    idx = min(index, len(X) - 1)
    return np.asarray(X[idx], dtype=np.float32), str(names[int(y[idx])])


def load_best_params() -> dict:
    data = json.loads(PHASE1_JSON.read_text(encoding="utf-8"))
    return data["phase1_best"]["best"]


def glcm_panel(ax, img: np.ndarray, params: dict) -> None:
    levels = int(params["levels"])
    distances = list(params["distances"])
    angles = np.asarray(params["angles"], dtype=float)
    symmetric = bool(params["symmetric"])
    q = np.clip(np.floor(img * levels), 0, levels - 1).astype(np.uint8)
    glcm = graycomatrix(
        q, distances=distances[:1], angles=angles[:1],
        levels=levels, symmetric=symmetric, normed=True,
    )
    props = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]
    vals = [float(graycoprops(glcm, p)[0, 0]) for p in props]
    im = ax.imshow(glcm[:, :, 0, 0], cmap="magma", aspect="auto")
    ax.set_title("GLCM (d=1, θ=0°)", fontsize=10, fontweight="bold")
    txt = "\n".join(f"{p[:4]}={v:.3f}" for p, v in zip(props, vals))
    ax.text(
        1.02, 0.5, txt, transform=ax.transAxes, fontsize=7,
        va="center", family="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def lbp_panel(ax, img: np.ndarray, params: dict) -> None:
    P, R = int(params["P"]), int(params["R"])
    method = str(params["method"])
    lbp = local_binary_pattern(img, P=P, R=R, method=method)
    im = ax.imshow(lbp, cmap="jet", vmin=0, vmax=P + 2 if method in ("uniform", "nri_uniform") else None)
    ax.set_title(f"LBP map (P={P}, R={R}, {method})", fontsize=10, fontweight="bold")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def dwt_panel(fig, spec, img: np.ndarray, params: dict) -> None:
    wavelet = str(params["wavelet"])
    level = int(params["level"])
    coeffs = pywt.wavedec2(img, wavelet=wavelet, level=level)
    bands = [("LL", coeffs[0])]
    names = ["LH", "HL", "HH"]
    for i, trip in enumerate(coeffs[1:], start=1):
        for j, band in enumerate(trip):
            bands.append((f"{names[j]}{i}", band))
    n = len(bands)
    inner = spec.subgridspec(1, n, wspace=0.08)
    for k, (name, band) in enumerate(bands):
        ax = fig.add_subplot(inner[k])
        v = np.abs(band)
        ax.imshow(v / (v.max() + 1e-8), cmap="gray")
        ax.set_title(name, fontsize=9)
        ax.axis("off")
    fig.text(
        0.5, 0.18, f"DWT sub-bands ({wavelet}, level={level})",
        ha="center", fontsize=10, fontweight="bold",
    )


def hog_panel(ax, img: np.ndarray, params: dict) -> None:
    ori = int(params["orientations"])
    ppc = tuple(params["pixels_per_cell"])
    cpb = tuple(params["cells_per_block"])
    bn = str(params["block_norm"])
    _, hog_img = hog(
        img, orientations=ori, pixels_per_cell=ppc,
        cells_per_block=cpb, block_norm=bn,
        visualize=True, feature_vector=True,
    )
    hog_img = resize(hog_img, img.shape, anti_aliasing=True, preserve_range=True)
    im = ax.imshow(hog_img, cmap="gray")
    ax.set_title(f"HOG energy (ori={ori}, cell={ppc[0]}×{ppc[1]})", fontsize=10, fontweight="bold")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def build_figure(img: np.ndarray, class_name: str, params: dict, out_path: Path) -> None:
    fig = plt.figure(figsize=(14, 11), facecolor="white")
    gs = GridSpec(3, 3, figure=fig, height_ratios=[1.1, 1, 1], hspace=0.35, wspace=0.25)

    ax0 = fig.add_subplot(gs[0, :])
    ax0.imshow(img, cmap="gray", vmin=0, vmax=1)
    ax0.set_title(f"Preprocessed MRI slice ({class_name}) — 128×128, CLAHE", fontsize=12, fontweight="bold")
    ax0.axis("off")

    ax_glcm = fig.add_subplot(gs[1, 0])
    glcm_panel(ax_glcm, img, params["GLCM"])
    ax_glcm.set_xlabel("Grey level j")
    ax_glcm.set_ylabel("Grey level i")

    ax_lbp = fig.add_subplot(gs[1, 1])
    lbp_panel(ax_lbp, img, params["LBP"])

    ax_hog = fig.add_subplot(gs[1, 2])
    hog_panel(ax_hog, img, params["HOG"])

    dwt_gs = gs[2, :]
    dwt_panel(fig, dwt_gs, img, params["DWT"])

    fig.suptitle(
        "Handcrafted descriptors on one brain-tumour MRI (Phase 1 optimal hyperparameters)",
        fontsize=13, fontweight="bold", y=0.98,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[descriptor_demo] Wrote {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=42, help="Training sample index")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--latex", action="store_true", help="Also copy to PAPER/latex/figures/")
    args = parser.parse_args()

    img, cls = load_sample_image(args.index)
    params = load_best_params()
    build_figure(img, cls, params, args.out)
    if args.latex:
        build_figure(img, cls, params, LATEX_OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
