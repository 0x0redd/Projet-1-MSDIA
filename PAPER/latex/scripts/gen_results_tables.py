"""Generate LaTeX tables from benchmark artifacts."""
import csv
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CSV_PATH = ROOT / "statistical_validation_summary.csv"
JSON_PATH = ROOT / "output" / "leaderbordML.json"
PHASE1_PATH = ROOT / "cache" / "brain_tumor" / "phase_artifacts" / "phase1_best.json"
OUT_DIR = Path(__file__).resolve().parents[1] / "tables"


def fmt(x, nd=4):
    return f"{x:.{nd}f}"


def load_csv():
    rows = []
    with CSV_PATH.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def classifier_averages(rows):
    agg = defaultdict(lambda: {"acc": [], "prec": [], "rec": [], "f1": []})
    for r in rows:
        m = r["Model"]
        agg[m]["acc"].append(float(r["Accuracy"]))
        agg[m]["prec"].append(float(r["Precision"]))
        agg[m]["rec"].append(float(r["Recall"]))
        agg[m]["f1"].append(float(r["F1-macro"]))
    order = ["SVM", "KNN", "LightGBM", "MLP", "XGBoost", "ExtraTrees", "RF", "LR"]
    lines = []
    for m in order:
        d = agg[m]
        et = "ET" if m == "ExtraTrees" else m
        lines.append(
            f"{et:8s} & {fmt(st.mean(d['acc']))} & {fmt(st.mean(d['prec']))} & "
            f"{fmt(st.mean(d['rec']))} & {fmt(st.mean(d['f1']))} \\\\"
        )
    return lines


def feature_set_averages(rows):
    order = [
        ("A — Stats", "A"),
        ("B — Tex(opt)", "B"),
        ("C — DWT(opt)", "C"),
        ("D — HOG", "D"),
        ("A+B", "A+B"),
        ("B+C(opt)", "B+C"),
        ("A+B+C(opt)", "A+B+C"),
        ("Full(opt)", "Full(opt)"),
    ]
    agg = defaultdict(lambda: {"acc": [], "f1": []})
    for r in rows:
        fs = r["Feature Set"]
        agg[fs]["acc"].append(float(r["Accuracy"]))
        agg[fs]["f1"].append(float(r["F1-macro"]))
    lines = []
    for fs_key, label in order:
        d = agg[fs_key]
        lines.append(
            f"{label:10s} & {fmt(st.mean(d['acc']))} & {fmt(st.mean(d['f1']))} \\\\"
        )
    return lines


def leaderboard_top10(json_rows):
    lines = []
    for i, e in enumerate(json_rows[:10], 1):
        fs = e["Feature Set"].replace("—", "---")
        model = e["Model"]
        if model == "ExtraTrees":
            model = "ET"
        lines.append(
            f"{i} & {fs} & {model} & {fmt(e['Accuracy'])} & {fmt(e['F1-macro'])} \\\\"
        )
    return lines


def leaderboard_top10_with_ci(csv_rows, json_rows):
  """Top-10 from JSON with kappa and F1 CI from CSV."""
  lookup = {(r["Feature Set"], r["Model"]): r for r in csv_rows}
  lines = []
  for i, e in enumerate(json_rows[:10], 1):
    fs = e["Feature Set"]
    model = e["Model"]
    r = lookup.get((fs, model))
    fs_tex = fs.replace("—", "---")
    m = "ET" if model == "ExtraTrees" else model
    if r:
      kappa = r["κ (Kappa)"]
      ci_lo = r["F1 CI Low"]
      ci_hi = r["F1 CI High"]
      lines.append(
        f"{i} & {fs_tex} & {m} & {fmt(float(r['Accuracy']))} & "
        f"{fmt(float(r['F1-macro']))} & {kappa} & "
        f"[{ci_lo}, {ci_hi}] \\\\"
      )
    else:
      lines.append(
        f"{i} & {fs_tex} & {m} & {fmt(e['Accuracy'])} & "
        f"{fmt(e['F1-macro'])} & -- & -- \\\\"
      )
  return lines


def phase1_table(phase1):
    best = phase1["best"]
    scores = phase1["scores"]
    glcm = best["GLCM"]
    lbp = best["LBP"]
    dwt = best["DWT"]
    hog = best["HOG"]
    sym = "symmetric" if glcm["symmetric"] else "asymmetric"
    dists = ",".join(str(d) for d in glcm["distances"])
    ppc = hog["pixels_per_cell"]
    cpb = hog["cells_per_block"]
    return [
        f"GLCM & dist. $\\{{{dists}\\}}$, "
        f"$1$ angle ($0^{{\\circ}}$), & $6$ & {fmt(scores['glcm'])} \\\\",
        f"     & ${glcm['levels']}$ levels, {sym} & & \\\\",
        f"LBP  & $P={lbp['P']}$, $R={lbp['R']}$, {lbp['method']} & "
        f"${lbp['P'] + 2}$ & {fmt(scores['lbp'])} \\\\",
        f"DWT  & \\texttt{{{dwt['wavelet']}}}, level ${dwt['level']}$ & $16$ & {fmt(scores['dwt'])} \\\\",
        f"HOG  & ${hog['orientations']}$ orient., "
        f"${ppc[0]}\\times{ppc[1]}$ cell, & "
        f"$1176$ & {fmt(scores['hog'])} \\\\",
        f"     & ${cpb[0]}\\times{cpb[1]}$ block, "
        f"{hog['block_norm']} & & \\\\",
    ]


def main():
    rows = load_csv()
    with JSON_PATH.open(encoding="utf-8") as f:
        lb = json.load(f)
    with PHASE1_PATH.open(encoding="utf-8") as f:
        phase1 = json.load(f)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Leaderboard top-10 with CI
    lb_lines = leaderboard_top10_with_ci(rows, lb)
    (OUT_DIR / "leaderboard_top10.tex").write_text(
        "% Auto-generated — do not edit by hand\n"
        "\\begin{table}[!t]\n\\centering\n"
        "\\caption{Top-10 classifier--feature-set combinations ranked by macro-F1 "
        "(5-fold stratified CV). $\\kappa$: Cohen's kappa; F1 CI: bootstrap 95\\% interval.}\n"
        "\\label{tab:leaderboard}\n"
        "\\renewcommand{\\arraystretch}{1.15}\n"
        "\\footnotesize\n"
        "\\setlength{\\tabcolsep}{3pt}\n"
        "\\begin{tabular}{@{}rllcccl@{}}\n\\toprule\n"
        "\\textbf{Rank} & \\textbf{Feature set} & \\textbf{Clf.} & \\textbf{Acc.} & "
        "\\textbf{F1} & $\\boldsymbol{\\kappa}$ & \\textbf{F1 CI} \\\\\n\\midrule\n"
        + "\n".join(lb_lines)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n",
        encoding="utf-8",
    )

    # Stats summary top-5 for statistics section
    top5_lines = []
    for i, e in enumerate(lb[:5], 1):
        fs = e["Feature Set"]
        model = e["Model"]
        r = next(x for x in rows if x["Feature Set"] == fs and x["Model"] == model)
        fs_tex = fs.replace("—", "---")
        m = "ET" if model == "ExtraTrees" else model
        top5_lines.append(
            f"{fs_tex} & {m} & {fmt(float(r['Accuracy']))} & {fmt(float(r['F1-macro']))} & "
            f"{r['κ (Kappa)']} & [{r['F1 CI Low']}, {r['F1 CI High']}] \\\\"
        )
    (OUT_DIR / "stats_top5.tex").write_text(
        "% Auto-generated — do not edit by hand\n"
        "\\begin{table}[!t]\n\\centering\n"
        "\\caption{Top-5 configurations with Cohen's $\\kappa$ and bootstrap F1 "
        "confidence intervals (5-fold CV).}\n"
        "\\label{tab:stats_top5}\n"
        "\\renewcommand{\\arraystretch}{1.2}\n"
        "\\begin{tabular}{@{}llcccc@{}}\n\\toprule\n"
        "\\textbf{Feature set} & \\textbf{Clf.} & \\textbf{Acc.} & \\textbf{F1} & "
        "$\\boldsymbol{\\kappa}$ & \\textbf{95\\% CI} \\\\\n\\midrule\n"
        + "\n".join(top5_lines)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n",
        encoding="utf-8",
    )

  # Phase1 params
    p1_lines = phase1_table(phase1)
    (OUT_DIR / "phase1_params.tex").write_text(
        "% Auto-generated — do not edit by hand\n"
        "\\begin{table}[!t]\n\\centering\n"
        "\\caption{Phase 1 optimal descriptor configurations. Composite scores are "
        "normalized within each family.}\n"
        "\\label{tab:phase1_generated}\n"
        "\\renewcommand{\\arraystretch}{1.3}\n"
        "\\begin{tabular}{@{}llcc@{}}\n\\toprule\n"
        "\\textbf{Descriptor} & \\textbf{Optimal parameters} & \\textbf{Dim.} & "
        "\\textbf{Score} \\\\\n\\midrule\n"
        + "\n".join(p1_lines)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n",
        encoding="utf-8",
    )

    print("Generated tables in", OUT_DIR)
    print("Classifier averages:")
    for line in classifier_averages(rows):
        print(line)
    print("Feature set averages:")
    for line in feature_set_averages(rows):
        print(line)


if __name__ == "__main__":
    main()
