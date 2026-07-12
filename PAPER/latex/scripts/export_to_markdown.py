"""Export PAPER/latex paper to main.md."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"

SECTIONS = [
    "sections/00_abstract.tex",
    "sections/01_introduction.tex",
    "sections/02_related_work.tex",
    "sections/03_materials_setup.tex",
    "sections/04_phase1_method.tex",
    "sections/05_feature_representation.tex",
    "sections/06_phase2_method.tex",
    "sections/07_results.tex",
    "sections/08_interpretability.tex",
    "sections/09_statistics.tex",
    "sections/11_limitations.tex",
    "sections/12_conclusion.tex",
]


def strip_comments(text: str) -> str:
    lines = []
    for line in text.splitlines():
        if line.lstrip().startswith("%"):
            continue
        lines.append(re.sub(r"(?<!\\)%.*", "", line))
    return "\n".join(lines)


def expand_inputs(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        path = ROOT / match.group(1).replace("/", "\\")
        if path.exists():
            return strip_comments(path.read_text(encoding="utf-8"))
        return f"<!-- missing: {match.group(1)} -->"

    return re.sub(r"\\input\{([^}]+)\}", repl, text)


def latex_table_to_md(block: str) -> str:
    rows = []
    for line in block.splitlines():
        line = line.strip()
        if not line or line.startswith("\\"):
            if any(x in line for x in ("toprule", "midrule", "bottomrule", "hline")):
                continue
            if line.startswith("\\caption"):
                cap = re.search(r"\\caption\{([^}]+)\}", line)
                if cap:
                    rows.append(f"*{cap.group(1)}*\n")
            continue
        if "&" in line:
            cells = [c.strip().rstrip("\\") for c in line.split("&")]
            rows.append("| " + " | ".join(cells) + " |")
    if not rows:
        return block
    header = rows[0]
    sep = "| " + " | ".join(["---"] * header.count("|")) + " |"
    return "\n".join([header, sep] + rows[1:])


def latex_to_md(text: str) -> str:
    text = strip_comments(text)
    text = expand_inputs(text)

    # Tables
    def table_repl(match: re.Match[str]) -> str:
        return "\n" + latex_table_to_md(match.group(0)) + "\n"

    text = re.sub(
        r"\\begin\{table\*?\}.*?\\end\{table\*?\}",
        table_repl,
        text,
        flags=re.S,
    )

    # Figures -> markdown image refs
    text = re.sub(
        r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}",
        r"![](figures/\1.png)",
        text,
    )
    text = re.sub(r"\\caption\{([^}]+)\}", r"*\1*", text)

    # Headings
    text = re.sub(
        r"\\subsubsection\*?\{([^}]+)\}",
        r"#### \1\n",
        text,
    )
    text = re.sub(
        r"\\subsection\*?\{([^}]+)\}",
        r"### \1\n",
        text,
    )
    text = re.sub(
        r"\\section\*?\{([^}]+)\}",
        r"## \1\n",
        text,
    )
    text = re.sub(r"\\paragraph\{([^}]+)\}", r"**\1.** ", text)

    # Lists
    text = re.sub(r"\\begin\{itemize\}|\\begin\{enumerate\}", "", text)
    text = re.sub(r"\\end\{itemize\}|\\end\{enumerate\}", "", text)
    text = re.sub(r"\\item\s+", "- ", text)

    # Inline formatting / refs
    text = re.sub(r"\\label\{[^}]+\}", "", text)
    text = re.sub(r"\\cite\{([^}]+)\}", r"[\1]", text)
    text = re.sub(r"\\ref\{([^}]+)\}", r"(\1)", text)
    text = re.sub(r"\\textbf\{([^}]+)\}", r"**\1**", text)
    text = re.sub(r"\\textit\{([^}]+)\}", r"*\1*", text)
    text = re.sub(r"\\emph\{([^}]+)\}", r"*\1*", text)
    text = re.sub(r"\\texttt\{([^}]+)\}", r"`\1`", text)
    # Greek / common math symbols before command stripping
    greek = {
        "kappa": "κ", "alpha": "α", "chi": "χ", "rho": "ρ",
        "mu": "μ", "sigma": "σ", "theta": "θ", "pi": "π",
    }
    for name, sym in greek.items():
        text = re.sub(rf"\\{name}\b", sym, text)
    text = re.sub(r"\\mathrm\{([^}]+)\}", r"\1", text)
    text = re.sub(r"\\text\{([^}]+)\}", r"\1", text)
    text = re.sub(r"\\TBD\b", "[TBD]", text)
    text = re.sub(r"~", " ", text)

    # Math environments
    def eq_repl(match: re.Match[str]) -> str:
        body = match.group(1).strip()
        body = re.sub(r"\\\\", " \\\\ ", body)
        return f"\n$$\n{body}\n$$\n"

    text = re.sub(r"\\begin\{equation\*?\}(.*?)\\end\{equation\*?\}", eq_repl, text, flags=re.S)
    text = re.sub(r"\\begin\{align\*?\}(.*?)\\end\{align\*?\}", eq_repl, text, flags=re.S)
    text = re.sub(r"\\\[([\s\S]*?)\\\]", eq_repl, text)

    # Strip remaining LaTeX commands (keep content inside braces handled above)
    text = re.sub(r"\\[a-zA-Z]+\*?", "", text)

    text = re.sub(r"\\begin\{(figure\*?|subfigure|tabular|center)\}.*?\\end\{\1\}", "", text, flags=re.S)
    text = re.sub(r"\\begin\{(figure\*?|subfigure|tabular|center)\}|\\end\{(figure\*?|subfigure|tabular|center)\}", "", text)
    text = re.sub(r"\\centering|\\toprule|\\midrule|\\bottomrule|\\hline", "", text)
    text = re.sub(r"\\renewcommand\{[^}]+\}\{[^}]+\}", "", text)
    text = re.sub(r"\\arraystretch\{[^}]+\}", "", text)
    text = re.sub(r"\\small|\\footnotesize|\\setlength\{[^}]+\}\{[^}]+\}", "", text)

    text = text.replace("---", "—").replace("--", "–")
    text = re.sub(r"\{,\}", ",", text)
    text = re.sub(r"\$([^$]+)\$", lambda m: f"${m.group(1).strip()}$", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def main() -> None:
    header = """# Handcrafted Feature Optimisation and Classifier Benchmarking for Brain Tumor MRI Classification

**Authors:** Ferrah Othmane, Ilham El Ouariachi  
**Affiliation:** Department of Computer Science, University of Moulay Ismail, Morocco  
**Email:** o.ferrah@edu.umi.ac.ma

**Keywords:** Brain Tumor MRI, Handcrafted Features, Feature Optimization, Texture Analysis, GLCM, LBP, DWT, HOG, Machine Learning, Classifier Benchmarking, Statistical Validation

---

## Abstract

"""
    parts = [header]
    for sec in SECTIONS:
        path = ROOT / sec
        if not path.exists():
            continue
        body = latex_to_md(path.read_text(encoding="utf-8"))
        if sec.endswith("00_abstract.tex"):
            parts.append(body + "\n\n---\n")
        else:
            parts.append(body + "\n")

    out = ROOT / "main.md"
    out.write_text("\n".join(parts), encoding="utf-8")
    print(f"Wrote {out} ({out.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
