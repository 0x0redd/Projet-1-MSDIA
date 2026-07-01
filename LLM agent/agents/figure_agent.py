"""Agent D — Figure generation from RAG metrics."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", font="DejaVu Sans")
PALETTE = ["#1D9E75", "#378ADD", "#D85A30", "#7F77DD", "#888780", "#E24B4A"]


class FigureAgent:
    """Produces publication-ready figures with optional error bars and caption sidecars."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.out_dir = Path(cfg["paths"]["figures_dir"])
        if not self.out_dir.is_absolute():
            from agents.paths import resolve_path
            self.out_dir = resolve_path(str(self.out_dir))
        self.out_dir.mkdir(parents=True, exist_ok=True)
        fig_cfg = cfg.get("figures", {})
        self.error_bars = fig_cfg.get("error_bars", True)
        self.caption_suffix = fig_cfg.get("caption_suffix", ".caption.txt")

    def make_figure(
        self,
        figure_type: str,
        metrics: dict,
        section: str,
        *,
        dry_run: bool = False,
    ) -> tuple[str, str | None]:
        if dry_run:
            path = str(self.out_dir / f"{figure_type}_dryrun.pdf")
            caption = self._write_caption(
                figure_type,
                f"[DRY RUN] Would generate {figure_type} for section {section}.",
            )
            return path, caption

        handlers = {
            "accuracy_bar": self._accuracy_bar,
            "roc_curve": self._roc_curve,
            "confusion_matrix": self._confusion_matrix,
            "ablation_heatmap": self._ablation_heatmap,
            "tier_bar": self._tier_bar,
            "leaderboard_top10": self._leaderboard_top10,
            "bootstrap_ci": self._bootstrap_ci_bar,
        }
        if figure_type not in handlers:
            raise ValueError(f"Unknown figure type: {figure_type}")
        return handlers[figure_type](metrics, section)

    def _parse_rows(self, metrics: dict) -> list[dict]:
        rows = []
        for r in metrics.get("records", []):
            d = r["data"]
            if isinstance(d, dict):
                rows.append(d)
        return rows

    def _experiments_dir(self) -> Path:
        from agents.paths import resolve_path

        return resolve_path(self.cfg["paths"]["experiments_dir"])

    def _load_ci_map(self) -> dict[str, tuple[float, float]]:
        """Map model label -> (ci_low, ci_high) macro-F1 from stats_top5.csv."""
        path = self._experiments_dir() / "stats_top5.csv"
        if not path.is_file():
            return {}
        df = pd.read_csv(path)
        out: dict[str, tuple[float, float]] = {}
        for _, row in df.iterrows():
            model = str(row.get("model", ""))
            try:
                lo = float(row["ci_low"])
                hi = float(row["ci_high"])
            except (TypeError, ValueError):
                continue
            if model and lo <= hi:
                out[model] = (lo, hi)
        return out

    def _write_caption(self, stem: str, caption: str, *, meta: dict | None = None) -> str:
        cap_path = self.out_dir / f"{stem}{self.caption_suffix}"
        lines = [caption.strip(), ""]
        if meta:
            lines.append("# metadata")
            lines.append(json.dumps(meta, indent=2))
        cap_path.write_text("\n".join(lines), encoding="utf-8")
        return str(cap_path)

    def _save_figure(
        self,
        fig,
        stem: str,
        caption: str,
        *,
        section: str = "results",
        meta: dict | None = None,
    ) -> tuple[str, str]:
        out = self.out_dir / f"{stem}.pdf"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        cap = self._write_caption(
            stem,
            caption,
            meta={"section": section, "file": out.name, **(meta or {})},
        )
        return str(out), cap

    def _accuracy_bar(self, metrics: dict, section: str) -> tuple[str, str]:
        rows = []
        for d in self._parse_rows(metrics):
            method = d.get("method") or d.get("model")
            acc = d.get("accuracy", d.get("acc"))
            if method is not None and acc is not None:
                rows.append({"method": str(method), "accuracy": float(acc)})

        if not rows:
            return self._placeholder("accuracy_bar", "No accuracy data found in experiments/", section)

        df = pd.DataFrame(rows).drop_duplicates(subset=["method"]).sort_values(
            "accuracy", ascending=True
        )
        ci_map = self._load_ci_map()

        fig, ax = plt.subplots(figsize=(8, max(4, len(df) * 0.35)))
        yvals = df["accuracy"].values * 100
        xerr = None
        if self.error_bars and ci_map:
            err_lo, err_hi = [], []
            for m, v in zip(df["method"], df["accuracy"]):
                if m in ci_map:
                    lo, hi = ci_map[m]
                    err_lo.append(max(0, (v - lo) * 100))
                    err_hi.append(max(0, (hi - v) * 100))
                else:
                    err_lo.append(0)
                    err_hi.append(0)
            if any(err_lo) or any(err_hi):
                xerr = np.array([err_lo, err_hi])

        colors = [PALETTE[i % len(PALETTE)] for i in range(len(df))]
        bars = ax.barh(
            df["method"],
            yvals,
            color=colors,
            edgecolor="white",
            height=0.6,
            xerr=xerr,
            capsize=3 if xerr is not None else 0,
        )
        for bar, val in zip(bars, yvals):
            ax.text(val + 0.3, bar.get_y() + bar.get_height() / 2, f"{val:.1f}%", va="center", fontsize=8)

        ax.set_xlabel("Accuracy (%)", fontsize=10)
        ax.set_xlim(0, 105)
        ax.set_title("Brain tumor MRI classification: method comparison", fontsize=10, pad=8)
        plt.tight_layout()
        caption = (
            "Macro accuracy comparison across Phase~2 classical ML configurations on the "
            "held-out split. Error bars show bootstrap 95\\% CI when available in stats_top5.csv."
        )
        return self._save_figure(fig, "accuracy_comparison", caption, section=section)

    def _bootstrap_ci_bar(self, metrics: dict, section: str) -> tuple[str, str]:
        path = self._experiments_dir() / "stats_top5.csv"
        if not path.is_file():
            return self._placeholder("bootstrap_ci", f"Missing {path.name}", section)
        df = pd.read_csv(path)
        df = df[df["f1_macro"].notna()].copy()
        if df.empty:
            return self._placeholder("bootstrap_ci", "No F1 data in stats_top5.csv", section)

        fig, ax = plt.subplots(figsize=(7, 4))
        labels = df["model"].astype(str)
        f1 = df["f1_macro"].astype(float) * 100
        xerr = None
        if self.error_bars:
            lo = pd.to_numeric(df.get("ci_low"), errors="coerce")
            hi = pd.to_numeric(df.get("ci_high"), errors="coerce")
            if lo.notna().any() and hi.notna().any():
                xerr = np.array([
                    (f1 - lo * 100).fillna(0).values,
                    (hi * 100 - f1).fillna(0).values,
                ])

        ax.barh(labels, f1, color=PALETTE[0], xerr=xerr, capsize=3 if xerr is not None else 0)
        ax.set_xlabel("Macro F1 (%)", fontsize=10)
        ax.set_title("Top models with bootstrap 95\\% CI", fontsize=10)
        plt.tight_layout()
        caption = (
            "Macro-F1 for top Phase~2 models with bootstrap 95\\% confidence intervals "
            "(stats\\_top5.csv). TODO intervals omitted when not yet exported from notebook."
        )
        return self._save_figure(fig, "bootstrap_ci", caption, section=section)

    def _roc_curve(self, metrics: dict, section: str) -> tuple[str, str]:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
        plotted = 0
        for i, d in enumerate(self._parse_rows(metrics)):
            if "fpr" in d and "tpr" in d:
                fpr = np.array(d["fpr"])
                tpr = np.array(d["tpr"])
                auc = d.get("auc", "?")
                label = f"{d.get('method', 'Method')} (AUC={auc})"
                ax.plot(fpr, tpr, color=PALETTE[i % len(PALETTE)], label=label, lw=1.5)
                plotted += 1
        if plotted == 0:
            return self._placeholder("roc_curve", "No fpr/tpr data in experiments/", section)
        ax.set_xlabel("False Positive Rate", fontsize=10)
        ax.set_ylabel("True Positive Rate", fontsize=10)
        ax.set_title("ROC curves", fontsize=10)
        ax.legend(fontsize=8)
        plt.tight_layout()
        return self._save_figure(
            fig, "roc_curves", "ROC curves for best Phase~2/3 models on the held-out split.", section=section
        )

    def _confusion_matrix(self, metrics: dict, section: str) -> tuple[str, str]:
        for d in self._parse_rows(metrics):
            if "confusion_matrix" in d:
                cm = np.array(d["confusion_matrix"])
                labels = d.get("labels", [str(i) for i in range(cm.shape[0])])
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.heatmap(
                    cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels, yticklabels=labels, linewidths=0.5, ax=ax,
                )
                ax.set_xlabel("Predicted", fontsize=10)
                ax.set_ylabel("True", fontsize=10)
                method = d.get("method", "best model")
                ax.set_title(f"Confusion matrix — {method}", fontsize=10)
                plt.tight_layout()
                return self._save_figure(
                    fig, "confusion_matrix",
                    f"Confusion matrix for {method} on the held-out three-class split.",
                    section=section,
                )
        return self._placeholder("confusion_matrix", "No confusion_matrix key in experiments/", section)

    def _ablation_heatmap(self, metrics: dict, section: str) -> tuple[str, str]:
        for d in self._parse_rows(metrics):
            if "ablation" in d:
                abl = d["ablation"]
                df = pd.DataFrame(abl).T
                fig, ax = plt.subplots(figsize=(6, 3))
                sns.heatmap(df, annot=True, fmt=".3f", cmap="YlGn", linewidths=0.5, ax=ax)
                ax.set_title("Ablation study — feature contribution", fontsize=10)
                plt.tight_layout()
                return self._save_figure(
                    fig, "ablation_heatmap",
                    "Feature-branch ablation heatmap (macro-F1 per branch configuration).",
                    section=section,
                )
        return self._placeholder("ablation_heatmap", "No ablation key in experiments/", section)

    def _tier_bar(self, metrics: dict, section: str) -> tuple[str, str]:
        path = self._experiments_dir() / "phase3_all_metrics.csv"
        if not path.exists():
            return self._placeholder("tier_bar", f"Missing {path.name}", section)
        df = pd.read_csv(path)
        split = df[df["eval_mode"] == "split"].copy()
        tier_names = {0: "Tier 0", 1: "Tier 1", 2: "Tier 2"}
        split["label"] = split.apply(
            lambda r: f"{tier_names.get(int(r['tier']), r['tier'])}: {r['model']}", axis=1
        )
        split = split.sort_values("f1_macro", ascending=True)
        colors = [PALETTE[int(t) % len(PALETTE)] for t in split["tier"]]

        fig, ax = plt.subplots(figsize=(8, max(4, len(split) * 0.35)))
        bars = ax.barh(split["label"], split["f1_macro"] * 100, color=colors, height=0.6)
        for bar, val in zip(bars, split["f1_macro"] * 100):
            ax.text(val + 0.2, bar.get_y() + bar.get_height() / 2, f"{val:.1f}%", va="center", fontsize=8)
        ax.set_xlabel("Macro F1 (%)", fontsize=10)
        ax.set_title("Phase 3 models by tier (held-out split)", fontsize=10)
        plt.tight_layout()
        return self._save_figure(
            fig, "tier_bar", "Phase~3 deep learning models grouped by architecture tier.", section=section
        )

    def _leaderboard_top10(self, metrics: dict, section: str) -> tuple[str, str]:
        path = self._experiments_dir() / "phase2_all_modes_metrics.csv"
        if not path.exists():
            return self._placeholder("leaderboard_top10", f"Missing {path.name}", section)
        df = pd.read_csv(path)
        split = df[df["eval_mode"] == "split"].sort_values("f1_macro", ascending=False).head(10)
        split["label"] = split["model"] + " / " + split["feature_set"].str.slice(0, 20)
        yvals = split["f1_macro"].values * 100

        ci_map = self._load_ci_map()
        xerr = None
        if self.error_bars and ci_map:
            err_lo, err_hi = [], []
            for model, fs, v in zip(split["model"], split["feature_set"], split["f1_macro"]):
                key = f"{model} {fs}".strip()
                matched = None
                for k, (lo, hi) in ci_map.items():
                    if model in k or k in key:
                        matched = (lo, hi)
                        break
                if matched:
                    lo, hi = matched
                    err_lo.append(max(0, (v - lo) * 100))
                    err_hi.append(max(0, (hi - v) * 100))
                else:
                    err_lo.append(0)
                    err_hi.append(0)
            if any(err_lo) or any(err_hi):
                xerr = np.array([err_lo, err_hi])

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.barh(
            split["label"], yvals, color=PALETTE[0], height=0.6,
            xerr=xerr, capsize=3 if xerr is not None else 0,
        )
        ax.set_xlabel("Macro F1 (%)", fontsize=10)
        ax.set_title("Phase 2 top-10 configurations (held-out split)", fontsize=10)
        plt.tight_layout()
        caption = (
            "Top-10 Phase~2 (model, feature-set) pairs ranked by macro-F1 on the held-out split. "
            "Horizontal error bars denote bootstrap 95\\% CI when matched in stats\\_top5.csv."
        )
        return self._save_figure(fig, "leaderboard_top10", caption, section=section)

    def _placeholder(self, name: str, msg: str, section: str) -> tuple[str, str]:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.text(0.5, 0.5, f"[{name}]\n{msg}", ha="center", va="center", fontsize=10,
                transform=ax.transAxes, color="gray")
        ax.axis("off")
        return self._save_figure(fig, f"{name}_placeholder", f"Placeholder: {msg}", section=section)
