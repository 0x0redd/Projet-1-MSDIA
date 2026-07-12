"""Insert paper figure cell into Projet7 notebook."""
import json
from pathlib import Path

nb_path = Path(__file__).resolve().parents[3] / "Projet7_brainTumorDetection.ipynb"
nb = json.loads(nb_path.read_text(encoding="utf-8"))

note = (
    "# NOTE: plot_fs() is for notebook exploration (dense multi-panel output).\n"
    "# For the IEEE paper, run the next cell or:\n"
    "#   python PAPER/latex/scripts/paper_phase2_viz.py\n\n"
)
cell45 = nb["cells"][45]
src = "".join(cell45["source"])
if not src.startswith("# NOTE: plot_fs"):
    cell45["source"] = [note + src]

new_source = '''# Paper-ready Phase 2 figures (IEEE)
from pathlib import Path
import sys
_paper_scripts = Path("PAPER/latex/scripts").resolve()
if str(_paper_scripts) not in sys.path:
    sys.path.insert(0, str(_paper_scripts))
from paper_phase2_viz import generate_paper_figures, load_results, plot_fs_paper

PAPER_FIG_DIR = Path("PAPER/latex/figures")
PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)

_results_csv = Path("phase2_benchmark_results.csv")
if _results_csv.exists():
    _df = load_results(_results_csv)
    generate_paper_figures(_df, PAPER_FIG_DIR)
    generate_paper_figures(_df, OUTPUT_DIR)
    print("Paper suite saved to PAPER/latex/figures/ and output/")

if P2:
    for fs_name, res_dict in P2.items():
        if not res_dict:
            continue
        plot_fs_paper(fs_name, res_dict, PAPER_FIG_DIR, y_test, class_names)
        plot_fs_paper(fs_name, res_dict, OUTPUT_DIR, y_test, class_names)
    print("Per-set paper bars/confmat saved")
else:
    print("P2 empty — run Phase 2 split cells for per-set confusion matrices")
'''

if not any("paper_phase2_viz" in "".join(c.get("source", [])) for c in nb["cells"]):
    nb["cells"].insert(
        46,
        {
            "cell_type": "code",
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in new_source.split("\n")],
        },
    )

nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("Notebook updated:", nb_path)
