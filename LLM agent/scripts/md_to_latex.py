#!/usr/bin/env python3
"""Convert agent section Markdown drafts to LaTeX fragments for PAPER/latex/sections/."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_SECTIONS_IN = PROJECT_ROOT / "output" / "sections"
DEFAULT_SECTIONS_OUT = PROJECT_ROOT.parent / "PAPER" / "latex" / "sections"

SECTION_MAP = {
    "abstract": "00_abstract.tex",
    "introduction": "01_introduction.tex",
    "related_work": "02_related_work.tex",
    "methods": "03_methods.tex",
    "results": "05_results_dl.tex",  # results prose merges into DL section
    "discussion": "09_discussion.tex",
    "conclusion": "10_conclusion.tex",
}

LATEX_SPECIAL = str.maketrans(
    {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
)


def escape_latex(text: str) -> str:
    return text.translate(LATEX_SPECIAL)


def md_bold_to_latex(text: str) -> str:
    return re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", text)


def md_headers_to_latex(text: str) -> str:
    lines = []
    for line in text.splitlines():
        if line.startswith("### "):
            lines.append(f"\\subsubsection{{{escape_latex(line[4:].strip())}}}")
        elif line.startswith("## "):
            title = line[3:].strip()
            if title.lower() in ("methods", "results", "discussion", "conclusion", "related work", "introduction", "abstract"):
                lines.append(f"% Section heading managed by main.tex / file name")
            else:
                lines.append(f"\\subsection{{{escape_latex(title)}}}")
        elif line.startswith("# "):
            lines.append(f"% {line[2:].strip()}")
        else:
            lines.append(line)
    return "\n".join(lines)


def md_citations_to_latex(text: str) -> str:
    return re.sub(
        r"\[([A-Za-z][^\]]+?,\s*\d{4})\]",
        lambda m: f"\\cite{{{m.group(1).split(',')[0].strip().replace(' ', '')}}}",
        text,
    )


def convert_md(md: str) -> str:
    md = re.sub(r"<!--.*?-->", "", md, flags=re.DOTALL)
    md = md_headers_to_latex(md)
    md = md_bold_to_latex(md)
  # Keep markdown tables as-is for manual conversion; prose only
    paragraphs = []
    for block in re.split(r"\n\n+", md):
        block = block.strip()
        if not block:
            continue
        if block.startswith("\\") or block.startswith("%"):
            paragraphs.append(block)
        elif block.startswith("|"):
            paragraphs.append("% TODO: convert markdown table\n" + block)
        else:
            paragraphs.append(escape_latex(block))
    return "\n\n".join(paragraphs)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sections", type=Path, default=DEFAULT_SECTIONS_IN)
    parser.add_argument("--out", type=Path, default=DEFAULT_SECTIONS_OUT)
    args = parser.parse_args()

    if not args.sections.is_dir():
        print(f"[md_to_latex] Missing {args.sections}")
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    count = 0
    for stem, out_name in SECTION_MAP.items():
        src = args.sections / f"{stem}.md"
        if not src.exists():
            continue
        body = convert_md(src.read_text(encoding="utf-8"))
        header = "% RAG-generated — review before submit\n"
        dest = args.out / out_name
        existing = dest.read_text(encoding="utf-8") if dest.exists() else ""
        # Prepend agent prose; keep \input tables from scaffold if present
        inputs = re.findall(r"\\input\{[^}]+\}", existing)
        figs = re.findall(r"\\begin\{figure.*?\n\\end\{figure\}", existing, re.DOTALL)
        suffix = ""
        if inputs:
            suffix += "\n\n" + "\n".join(inputs)
        if figs:
            suffix += "\n\n" + "\n\n".join(figs)
        dest.write_text(header + body + suffix, encoding="utf-8")
        print(f"[md_to_latex] {src.name} -> {out_name}")
        count += 1

    print(f"[md_to_latex] Converted {count} section(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
