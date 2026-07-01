"""Seed data/papers and data/experiments from project cache and results."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent

PAPERS_SRC = REPO_ROOT / "PAPER"
DATA_PAPERS = PROJECT_ROOT / "data" / "papers"
DATA_EXP = PROJECT_ROOT / "data" / "experiments"

CACHE_ROOT = REPO_ROOT / "cache" / "brain_tumor"
PHASE_CACHE = CACHE_ROOT / "phase_artifacts"
PHASE3_MANIFEST = PHASE_CACHE / "phase3_dl" / "phase3_manifest.json"
PHASE3_PKL = PHASE_CACHE / "phase3_all_metrics_df.pkl"
PHASE2_PKL = PHASE_CACHE / "phase2_all_metrics_df.pkl"
PHASE2_BENCH = PHASE_CACHE / "phase2_benchmark"
PHASE2_MANIFEST = PHASE2_BENCH / "phase2_checkpoint_manifest.json"
PHASE1_BEST = PHASE_CACHE / "phase1_best.json"
PHASE1_ARCHIVE = PHASE_CACHE / "phase1_full_search_archive"
PHASE1_HISTORY_FILES = {
    "glcm": "glcm_history.csv",
    "lbp": "lbp_history.csv",
    "dwt": "dwt_history.csv",
    "hog": "hog_history.csv",
}

PHASE3_METRICS_CSV = REPO_ROOT / "results" / "phase3_all_metrics.csv"
PHASE2_METRICS_CSV = REPO_ROOT / "phase2_all_modes_metrics.csv"
SPLIT_DIST = REPO_ROOT / "results" / "split_class_distribution.csv"
PHASE3_VS_PHASE2 = REPO_ROOT / "results" / "phase3_vs_phase2_comparison.csv"
RESULTS_DIR = REPO_ROOT / "results"


def copy_papers() -> int:
    """Sync ML-only PDF corpus from PAPER/filtered - ML into data/papers/."""
    DATA_PAPERS.mkdir(parents=True, exist_ok=True)
    ml_dir = PAPERS_SRC / "filtered - ML"
    if not ml_dir.is_dir():
        print(f"[prepare_data] Missing {ml_dir}")
        return 0

    # Remove PDFs no longer in the filtered set
    keep_names = {p.name for p in ml_dir.glob("*.pdf")}
    for old in DATA_PAPERS.glob("*.pdf"):
        if old.name not in keep_names:
            old.unlink()

    count = 0
    for pdf in ml_dir.glob("*.pdf"):
        shutil.copy2(pdf, DATA_PAPERS / pdf.name)
        count += 1

    csv_src = ml_dir / "Exported Items.csv"
    if csv_src.exists():
        shutil.copy2(csv_src, PROJECT_ROOT / "data" / "Exported Items.csv")
    return count


def _tier_label(tier) -> str:
    labels = {0: "Tier0_DL", 1: "Tier1_Hybrid", 2: "Tier2_Fusion"}
    try:
        return labels.get(int(tier), f"Tier{tier}")
    except (TypeError, ValueError):
        return "unknown"


def _load_phase3_pkl() -> pd.DataFrame | None:
    if not PHASE3_PKL.exists():
        return None
    try:
        import joblib

        df = joblib.load(PHASE3_PKL)
        if isinstance(df, pd.DataFrame) and len(df):
            return df.copy()
    except Exception as exc:
        print(f"[prepare_data] Skip {PHASE3_PKL.name}: {exc.__class__.__name__}: {exc}")
    return None


def _load_phase3_manifest() -> pd.DataFrame | None:
    """Latest run per (tier, model, eval_mode) from phase3_manifest.json."""
    if not PHASE3_MANIFEST.exists():
        return None
    try:
        manifest = json.loads(PHASE3_MANIFEST.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[prepare_data] Skip manifest: {exc}")
        return None

    runs = manifest.get("runs", [])
    if not runs:
        return None

    df = pd.DataFrame(runs)
    if "saved_at_utc" in df.columns:
        df = df.sort_values("saved_at_utc")
    df = df.groupby(["tier", "model", "eval_mode"], as_index=False).tail(1)

    rows = []
    for _, row in df.iterrows():
        acc = row.get("test_acc", row.get("acc"))
        f1 = row.get("test_f1_macro", row.get("f1_macro"))
        if pd.isna(acc):
            continue
        rows.append({
            "tier": row.get("tier"),
            "model": row.get("model"),
            "eval_mode": row.get("eval_mode", "split"),
            "fold": row.get("fold", ""),
            "acc": float(acc),
            "f1_macro": float(f1) if pd.notna(f1) else None,
            "precision_macro": row.get("precision_macro"),
            "recall_macro": row.get("recall_macro"),
            "fit_time_s": row.get("fit_time_s"),
            "source": "cache/phase3_manifest.json",
        })
    return pd.DataFrame(rows) if rows else None


def load_phase3_metrics() -> pd.DataFrame:
    """Prefer cache/brain_tumor, fallback to results/phase3_all_metrics.csv."""
    df = _load_phase3_pkl()
    if df is not None:
        df = df.copy()
        df["source"] = "cache/phase3_all_metrics_df.pkl"
        print(f"[prepare_data] Phase 3 metrics from pickle ({len(df)} rows)")
        return df

    df = _load_phase3_manifest()
    if df is not None:
        print(f"[prepare_data] Phase 3 metrics from manifest ({len(df)} rows)")
        return df

    if PHASE3_METRICS_CSV.exists():
        df = pd.read_csv(PHASE3_METRICS_CSV)
        df["source"] = "results/phase3_all_metrics.csv"
        print(f"[prepare_data] Phase 3 metrics from CSV ({len(df)} rows)")
        return df

    print("[prepare_data] No Phase 3 metrics found")
    return pd.DataFrame()


def _latest_phase1_run_id() -> str | None:
    if PHASE1_BEST.exists():
        try:
            data = json.loads(PHASE1_BEST.read_text(encoding="utf-8"))
            rid = data.get("phase1_run_id")
            if rid:
                return str(rid)
        except Exception:
            pass
    return None


def copy_phase1_histories() -> int:
    """Copy GLCM/LBP/DWT/HOG param-search tables into data/experiments/."""
    count = 0
    if not PHASE1_ARCHIVE.is_dir():
        return 0
    for extractor, fname in PHASE1_HISTORY_FILES.items():
        src = PHASE1_ARCHIVE / fname
        if src.exists():
            shutil.copy2(src, DATA_EXP / fname)
            count += 1
    return count


def build_phase1_summary() -> dict:
    """Best param-search row per extractor (GLCM, LBP, DWT, HOG) from cache archive."""
    summary: dict = {
        "extractors": {},
        "source_dir": str(PHASE1_ARCHIVE),
    }

    if PHASE1_BEST.exists():
        try:
            summary["phase1_best"] = json.loads(PHASE1_BEST.read_text(encoding="utf-8"))
        except Exception:
            pass

    latest_run = _latest_phase1_run_id()
    if latest_run:
        summary["latest_run_id"] = latest_run

    for extractor, fname in PHASE1_HISTORY_FILES.items():
        path = PHASE1_ARCHIVE / fname
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if df.empty or "score" not in df.columns:
            continue

        sub = df
        if latest_run and "phase1_run_id" in df.columns:
            run_sub = df[df["phase1_run_id"] == latest_run]
            if not run_sub.empty:
                sub = run_sub

        best_idx = sub["score"].idxmax()
        best_row = sub.loc[best_idx].to_dict()
        for key, val in list(best_row.items()):
            if pd.isna(val):
                best_row[key] = None

        top5 = (
            sub.sort_values("score", ascending=False)
            .head(5)
            .to_dict(orient="records")
        )
        summary["extractors"][extractor] = {
            "history_file": fname,
            "n_configs": int(len(sub)),
            "best": best_row,
            "top5": top5,
        }

    if summary.get("phase1_best", {}).get("scores"):
        summary["ablation"] = {
            ext: {"score": float(score)}
            for ext, score in summary["phase1_best"]["scores"].items()
        }

    return summary


def _load_phase2_manifest_df() -> pd.DataFrame | None:
    """Phase 2 checkpoint manifest — latest entry per (model, feature_set, eval_mode)."""
    if not PHASE2_MANIFEST.exists():
        return None
    try:
        manifest = json.loads(PHASE2_MANIFEST.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[prepare_data] Skip phase2 manifest: {exc}")
        return None

    runs = manifest.get("runs", [])
    if not runs:
        return None

    df = pd.DataFrame(runs)
    if "saved_at_utc" in df.columns:
        df = df.sort_values("saved_at_utc")
    df = df.groupby(["model", "feature_set", "eval_mode"], as_index=False).tail(1)

    rows = []
    for _, row in df.iterrows():
        rows.append({
            "eval_mode": str(row.get("eval_mode", "")),
            "feature_set": str(row.get("feature_set", "")),
            "model": str(row.get("model", "")),
            "acc": float(row.get("acc", float("nan"))),
            "f1_macro": float(row.get("f1_macro", float("nan"))),
            "best_cv": float(row.get("best_cv", float("nan"))),
            "saved_at_utc": row.get("saved_at_utc"),
            "source": "cache/phase2_checkpoint_manifest.json",
        })
    return pd.DataFrame(rows) if rows else None


def build_phase2_summary() -> dict:
    """Full Phase 2 manifest plus leaderboard from latest split runs."""
    if not PHASE2_MANIFEST.exists():
        return {}

    try:
        manifest = json.loads(PHASE2_MANIFEST.read_text(encoding="utf-8"))
    except Exception:
        return {}

    df = _load_phase2_manifest_df()
    leaderboard = []
    if df is not None and not df.empty:
        split = df[df["eval_mode"] == "split"] if "eval_mode" in df.columns else df
        if not split.empty:
            split = split.sort_values("acc", ascending=False)
            leaderboard = split.to_dict(orient="records")

    return {
        "manifest_path": str(PHASE2_MANIFEST),
        "total_runs": len(manifest.get("runs", [])),
        "unique_jobs": len(df) if df is not None else 0,
        "split_leaderboard": leaderboard,
        "runs": manifest.get("runs", []),
    }


def _load_phase2_meta() -> pd.DataFrame | None:
    if not PHASE2_BENCH.is_dir():
        return None
    rows = []
    for meta_path in sorted(PHASE2_BENCH.glob("*.meta.json")):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        rows.append({
            "eval_mode": str(meta.get("eval_mode", "")),
            "feature_set": str(meta.get("feature_set", "")),
            "model": str(meta.get("model", "")),
            "acc": float(meta.get("acc", float("nan"))),
            "f1_macro": float(meta.get("f1_macro", float("nan"))),
            "best_cv": float(meta.get("best_cv", float("nan"))),
            "source": f"cache/phase2_benchmark/{meta_path.name}",
        })
    return pd.DataFrame(rows) if rows else None


def _load_phase2_pkl() -> pd.DataFrame | None:
    if not PHASE2_PKL.exists():
        return None
    try:
        import joblib

        df = joblib.load(PHASE2_PKL)
        if isinstance(df, pd.DataFrame) and len(df):
            out = df.copy()
            out["source"] = "cache/phase2_all_metrics_df.pkl"
            return out
    except Exception as exc:
        print(f"[prepare_data] Skip {PHASE2_PKL.name}: {exc.__class__.__name__}: {exc}")
    return None


def load_phase2_metrics() -> pd.DataFrame:
    """Prefer phase2_checkpoint_manifest.json, then meta.json, then CSV."""
    df = _load_phase2_manifest_df()
    if df is not None:
        print(f"[prepare_data] Phase 2 metrics from checkpoint manifest ({len(df)} rows)")
        return df

    df = _load_phase2_meta()
    if df is not None:
        print(f"[prepare_data] Phase 2 metrics from cache meta ({len(df)} rows)")
        return df

    df = _load_phase2_pkl()
    if df is not None:
        print(f"[prepare_data] Phase 2 metrics from pickle ({len(df)} rows)")
        return df

    if PHASE2_METRICS_CSV.exists():
        df = pd.read_csv(PHASE2_METRICS_CSV)
        df["source"] = "phase2_all_modes_metrics.csv"
        print(f"[prepare_data] Phase 2 metrics from CSV ({len(df)} rows)")
        return df

    print("[prepare_data] No Phase 2 metrics found")
    return pd.DataFrame()


def build_results_csv() -> pd.DataFrame:
    rows: list[dict] = []

    df3 = load_phase3_metrics()
    if not df3.empty:
        split_df = df3[df3["eval_mode"] == "split"] if "eval_mode" in df3.columns else df3
        for _, row in split_df.iterrows():
            tier = row.get("tier", "")
            model = str(row.get("model", "unknown"))
            acc = row.get("acc", row.get("test_acc"))
            f1 = row.get("f1_macro", row.get("test_f1_macro"))
            rows.append({
                "method": f"{_tier_label(tier)}_{model}",
                "accuracy": float(acc),
                "f1": float(f1) if pd.notna(f1) else None,
                "tier": tier,
                "eval_mode": row.get("eval_mode", "split"),
                "source": row.get("source", "phase3"),
            })

    df2 = load_phase2_metrics()
    if not df2.empty:
        if "best_cv" in df2.columns and "feature_set" in df2.columns:
            idx = df2.groupby(["feature_set", "model"])["best_cv"].idxmax()
            best = df2.loc[idx]
        else:
            best = df2.drop_duplicates(subset=["feature_set", "model"], keep="first")
        for _, row in best.iterrows():
            acc = row.get("best_cv", row.get("acc", 0))
            rows.append({
                "method": f"ML_{row.get('feature_set', '')}_{row.get('model', '')}",
                "accuracy": float(acc),
                "f1": float(row.get("f1_macro", 0)) if pd.notna(row.get("f1_macro")) else None,
                "tier": "phase1_ml",
                "eval_mode": row.get("eval_mode", "cv"),
                "source": row.get("source", "phase2"),
            })

    return pd.DataFrame(rows)


def _class_names() -> list[str]:
    if PHASE1_BEST.exists():
        try:
            data = json.loads(PHASE1_BEST.read_text(encoding="utf-8"))
            names = data.get("class_names")
            if names:
                return list(names)
        except Exception:
            pass
    return ["Glioma", "Meningioma", "Pituitary Tumor"]


def _load_confusion_csv(path: Path) -> dict:
    cm_df = pd.read_csv(path, index_col=0)
    labels = list(cm_df.columns)
    return {
        "method": path.stem,
        "labels": labels,
        "confusion_matrix": cm_df.values.tolist(),
    }


def build_all_confusion_matrices() -> list[dict]:
    """Load all phase3 confusion matrices from results/ (written by training cache pipeline)."""
    matrices = []
    for path in sorted(RESULTS_DIR.glob("phase3_cm_*_split.csv")):
        try:
            matrices.append(_load_confusion_csv(path))
        except Exception as exc:
            print(f"[prepare_data] Skip CM {path.name}: {exc}")
    return matrices


def build_confusion_json(phase3_df: pd.DataFrame, matrices: list[dict]) -> dict | None:
    if not matrices:
        return None

    best_acc = -1.0
    best_entry = matrices[0]

    if not phase3_df.empty and "eval_mode" in phase3_df.columns:
        split_df = phase3_df[phase3_df["eval_mode"] == "split"]
        if not split_df.empty:
            acc_col = "acc" if "acc" in split_df.columns else "test_acc"
            best_row = split_df.loc[split_df[acc_col].idxmax()]
            model = str(best_row["model"])
            tier = int(best_row["tier"])
            tag = f"phase3_cm_{tier}_{model}_split"
            for m in matrices:
                if tag in m["method"] or m["method"].replace("_", "") == tag.replace("_", ""):
                    best_entry = m
                    best_acc = float(best_row[acc_col])
                    break
            else:
                for m in matrices:
                    if model in m["method"]:
                        best_entry = m
                        best_acc = float(best_row[acc_col])
                        break

    out = dict(best_entry)
    out["accuracy"] = best_acc if best_acc >= 0 else None
    out["labels"] = out.get("labels") or _class_names()
    return out


def build_dataset_stats(phase3_df: pd.DataFrame) -> dict:
    stats: dict = {
        "task": "brain_tumor_mri_classification",
        "classes": _class_names(),
        "modalities": ["MRI"],
        "pipeline": "Phase1_ML -> Phase3_DL_Tier0_1_2 -> VLM",
        "cache_root": str(CACHE_ROOT),
    }

    if PHASE1_BEST.exists():
        try:
            stats["phase1_best"] = json.loads(PHASE1_BEST.read_text(encoding="utf-8"))
        except Exception:
            pass

    phase1_summary = build_phase1_summary()
    if phase1_summary.get("extractors"):
        stats["phase1_param_search"] = phase1_summary

    phase2_summary = build_phase2_summary()
    if phase2_summary:
        stats["phase2_checkpoint_summary"] = {
            "total_runs": phase2_summary.get("total_runs"),
            "unique_jobs": phase2_summary.get("unique_jobs"),
            "split_leaderboard_top5": phase2_summary.get("split_leaderboard", [])[:5],
        }

    if PHASE3_MANIFEST.exists():
        try:
            man = json.loads(PHASE3_MANIFEST.read_text(encoding="utf-8"))
            stats["phase3_manifest_run_count"] = len(man.get("runs", []))
        except Exception:
            pass

    if not phase3_df.empty:
        stats["phase3_metrics_rows"] = len(phase3_df)
        split = phase3_df[phase3_df["eval_mode"] == "split"] if "eval_mode" in phase3_df.columns else phase3_df
        acc_col = "acc" if "acc" in split.columns else "test_acc"
        if acc_col in split.columns and not split.empty:
            best = split.loc[split[acc_col].idxmax()]
            stats["best_split_model"] = {
                "tier": int(best.get("tier", -1)),
                "model": str(best.get("model", "")),
                "accuracy": float(best[acc_col]),
                "f1_macro": float(best.get("f1_macro", best.get("test_f1_macro", 0))),
            }

    if SPLIT_DIST.exists():
        df = pd.read_csv(SPLIT_DIST)
        stats["split_distribution"] = df.to_dict(orient="records")

    if PHASE3_VS_PHASE2.exists():
        df = pd.read_csv(PHASE3_VS_PHASE2)
        stats["phase3_vs_phase2"] = df.to_dict(orient="records")

    return stats


def main() -> None:
    DATA_EXP.mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "output" / "sections").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "output" / "figures").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "db" / "chroma").mkdir(parents=True, exist_ok=True)

    n_papers = copy_papers()
    n_phase1 = copy_phase1_histories()

    phase1_summary = build_phase1_summary()
    if phase1_summary.get("extractors"):
        with open(DATA_EXP / "phase1_param_search.json", "w", encoding="utf-8") as f:
            json.dump(phase1_summary, f, indent=2, default=str)

    phase2_summary = build_phase2_summary()
    if phase2_summary:
        shutil.copy2(PHASE2_MANIFEST, DATA_EXP / "phase2_checkpoint_manifest.json")
        with open(DATA_EXP / "phase2_summary.json", "w", encoding="utf-8") as f:
            json.dump(phase2_summary, f, indent=2, default=str)

    phase3_df = load_phase3_metrics()
    if not phase3_df.empty:
        phase3_df.to_csv(DATA_EXP / "phase3_all_metrics.csv", index=False)

    phase2_df = load_phase2_metrics()
    if not phase2_df.empty:
        phase2_df.to_csv(DATA_EXP / "phase2_all_modes_metrics.csv", index=False)

    if PHASE3_MANIFEST.exists():
        shutil.copy2(PHASE3_MANIFEST, DATA_EXP / "phase3_manifest.json")

    results_df = build_results_csv()
    results_path = DATA_EXP / "results.csv"
    results_df.to_csv(results_path, index=False)

    stats = build_dataset_stats(phase3_df)
    with open(DATA_EXP / "dataset_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    all_cms = build_all_confusion_matrices()
    if all_cms:
        with open(DATA_EXP / "confusion_matrices.json", "w", encoding="utf-8") as f:
            json.dump(all_cms, f, indent=2)

    confusion = build_confusion_json(phase3_df, all_cms)
    if confusion:
        with open(DATA_EXP / "confusion.json", "w", encoding="utf-8") as f:
            json.dump(confusion, f, indent=2)

    if PHASE3_VS_PHASE2.exists():
        shutil.copy2(PHASE3_VS_PHASE2, DATA_EXP / "phase3_vs_phase2_comparison.csv")

    print(f"[prepare_data] Copied {n_papers} PDF(s) -> {DATA_PAPERS}")
    if n_phase1:
        print(f"[prepare_data] Copied {n_phase1} Phase 1 history CSV(s) (GLCM/LBP/DWT/HOG)")
    if phase1_summary.get("extractors"):
        print(f"[prepare_data] Wrote phase1_param_search.json ({len(phase1_summary['extractors'])} extractors)")
    if phase2_summary:
        print(
            f"[prepare_data] Wrote phase2_summary.json + manifest "
            f"({phase2_summary.get('unique_jobs', 0)} unique jobs)"
        )
    print(f"[prepare_data] Wrote {len(results_df)} metric rows -> {results_path}")
    print(f"[prepare_data] Wrote dataset_stats.json (cache: {CACHE_ROOT.exists()})")
    if all_cms:
        print(f"[prepare_data] Wrote confusion_matrices.json ({len(all_cms)} matrices)")
    if confusion:
        print(f"[prepare_data] Wrote confusion.json ({confusion['method']})")


if __name__ == "__main__":
    main()
