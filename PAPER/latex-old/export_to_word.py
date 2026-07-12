#!/usr/bin/bin python3
"""
Flatten IEEE LaTeX sources and export main.docx via pandoc (pypandoc-binary).

Usage (from PAPER/latex):
  python export_to_word.py
  python export_to_word.py -o main.docx
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

LATEX_DIR = Path(__file__).resolve().parent
REPO_ROOT = LATEX_DIR.parent.parent
BUILD_DIR = LATEX_DIR / "word_build"
MEDIA_DIR = BUILD_DIR / "media"

GRAPHIC_DIRS = [
    LATEX_DIR / "figures",
    REPO_ROOT / "figures",
    LATEX_DIR.parent / "LLM agent" / "output" / "figures",
]

INPUT_RE = re.compile(r"\\input\{([^}]+)\}")
INCLUDE_RE = re.compile(
    r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}",
    re.IGNORECASE,
)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def resolve_input_path(base: Path, target: str) -> Path:
    target = target.strip()
    if not target.endswith(".tex"):
        target += ".tex"
    candidates = [
        base.parent / target,
        LATEX_DIR / target,
    ]
    for cand in candidates:
        if cand.is_file():
            return cand
    raise FileNotFoundError(f"Cannot resolve \\input{{{target}}}")


def flatten_inputs(content: str, base: Path, seen: set[Path] | None = None) -> str:
    seen = seen or set()

    def repl(match: re.Match[str]) -> str:
        path = resolve_input_path(base, match.group(1))
        if path in seen:
            return f"% skipped duplicate input: {path.name}"
        seen.add(path)
        nested = read_text(path)
        return flatten_inputs(nested, path, seen)

    return INPUT_RE.sub(repl, content)


def find_figure(src: str) -> Path | None:
    name = Path(src).name
    exts = [""] + [".pdf", ".png", ".jpg", ".jpeg", ".PNG", ".PDF"]
    for folder in GRAPHIC_DIRS:
        if not folder.is_dir():
            continue
        for ext in exts:
            cand = folder / f"{name}{ext}" if ext else folder / name
            if cand.is_file():
                return cand
    return None


def pdf_to_png(pdf_path: Path, png_path: Path, dpi: int = 180) -> None:
    import fitz  # pymupdf

    doc = fitz.open(pdf_path)
    page = doc[0]
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    pix.save(str(png_path))
    doc.close()


def prepare_media(content: str) -> str:
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    mapping: dict[str, str] = {}

    def repl(match: re.Match[str]) -> str:
        key = match.group(1)
        if key in mapping:
            rel = mapping[key]
            return rf"\includegraphics[width=\linewidth]{{{rel}}}"

        src = find_figure(key)
        if src is None:
            print(f"  [warn] figure not found: {key}")
            return match.group(0)

        out_name = re.sub(r"[^\w.\-]+", "_", Path(key).name) + ".png"
        out_path = MEDIA_DIR / out_name
        rel = f"media/{out_name}"

        if src.suffix.lower() == ".pdf":
            try:
                pdf_to_png(src, out_path)
            except Exception as exc:
                print(f"  [warn] PDF->PNG failed for {src.name}: {exc}")
                shutil.copy2(src, MEDIA_DIR / src.name)
                rel = f"media/{src.name}"
        else:
            shutil.copy2(src, out_path)

        mapping[key] = rel
        print(f"  [media] {src.name} -> {rel}")
        return rf"\includegraphics[width=\linewidth]{{{rel}}}"

    return INCLUDE_RE.sub(repl, content)


def pandoc_sanitize(content: str) -> str:
    """Strip IEEE-only commands; simplify tables for pandoc."""
    content = re.sub(r"\\bibliographystyle\{[^}]+\}", "", content)
    content = re.sub(r"\\bibliography\{[^}]+\}", "", content)
    content = re.sub(r"\\graphicspath\{[^}]+\}", "", content)
    content = re.sub(r"\\providecommand\{[^}]+\}\{[^}]*\}", "", content)
    content = re.sub(r"\\(toprule|midrule|bottomrule)", r"\\hline", content)
    content = re.sub(r"\\begin\{figure\}\[!t\]", r"\\begin{figure}", content)
    content = re.sub(r"\\begin\{table\}\[!t\]", r"\\begin{table}", content)
    content = re.sub(r"\\label\{[^}]+\}", "", content)
    content = re.sub(r"\\IEEEauthorblockN\{([^}]+)\}", r"\\textbf{\1}", content)
    content = re.sub(r"\\IEEEauthorblockA\{([^}]+)\}", r"\1", content)
    content = content.replace("\\&", "&")
    return content


def extract_braced_command(tex: str, command: str) -> tuple[str, str]:
    """Return (inner_content, tex_with_command_removed) for \\command{...}."""
    needle = f"\\{command}" + "{"
    start = tex.find(needle)
    if start == -1:
        return "", tex
    i = start + len(needle)
    depth = 1
    while i < len(tex) and depth:
        ch = tex[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        i += 1
    inner = tex[start + len(needle) : i - 1]
    remainder = tex[:start] + tex[i:]
    return inner, remainder


def latex_author_to_text(author_tex: str) -> str:
    author_tex = re.sub(r"\\IEEEauthorblockN\{([^}]+)\}", r"\1", author_tex)
    author_tex = re.sub(r"\\IEEEauthorblockA\{([^}]+)\}", r"\1", author_tex)
    author_tex = author_tex.replace("\\\\", "\n")
    return normalize_whitespace(author_tex)


def extract_title_author(main_tex: str) -> tuple[str, str]:
    title_inner, rest = extract_braced_command(main_tex, "title")
    author_inner, _ = extract_braced_command(main_tex, "author")
    title = normalize_whitespace(title_inner)
    author = latex_author_to_text(author_inner)
    return title, author


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def bbl_to_plain_text(bbl: str) -> str:
    """Convert main.bbl to readable plain text for Word."""
    lines: list[str] = []
    for raw in bbl.splitlines():
        line = raw.strip()
        if not line or line.startswith("\\providecommand"):
            continue
        if line.startswith("\\bibitem"):
            lines.append("")
            continue
        if line.startswith("\\begin{") or line.startswith("\\end{"):
            continue
        if line.startswith("\\BIB"):
            continue
        line = re.sub(r"\\url\{([^}]+)\}", r"\1", line)
        line = re.sub(r"\\emph\{([^}]+)\}", r"\1", line)
        line = re.sub(r"\\textit\{([^}]+)\}", r"\1", line)
        line = re.sub(r"\\newblock", " ", line)
        line = re.sub(r"\\[A-Za-z]+\{([^}]*)\}", r"\1", line)
        line = re.sub(r"\\[A-Za-z]+", "", line)
        line = normalize_whitespace(line)
        if line:
            lines.append(line)
    return "\n".join(lines).strip()


def remove_command(tex: str, command: str) -> str:
    _, tex = extract_braced_command(tex, command)
    return tex


def remove_latex_comments(text: str) -> str:
    """Drop full-line % comments; keep lines that contain math/content."""
    lines = []
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("%") and not stripped.startswith("%!"):
            continue
        lines.append(line)
    return "\n".join(lines)


def build_flat_tex() -> Path:
    main_tex = read_text(LATEX_DIR / "main.tex")
    title, author = extract_title_author(main_tex)
    author_tex = author.replace("%", "").strip()
    if not author_tex:
        author_tex = "Author Name \\\\ Department, Institution \\\\ author@institution.edu"

    body = main_tex
    body = re.sub(r"\\documentclass\[journal\]\{IEEEtran\}", "", body)
    body = re.sub(r"\\usepackage(?:\[[^\]]*\])?\{[^}]+\}", "", body)
    for cmd in ("title", "author"):
        body = remove_command(body, cmd)
    body = remove_command(body, "graphicspath")
    body = re.sub(r"\\begin\{document\}", "", body)
    body = re.sub(r"\\end\{document\}", "", body)
    body = re.sub(r"\\maketitle", "", body)

    body = flatten_inputs(body, LATEX_DIR / "main.tex")
    body = prepare_media(body)
    body = remove_latex_comments(body)
    body = pandoc_sanitize(body)

    abstract_m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", body, re.DOTALL)
    keywords_m = re.search(
        r"\\begin\{IEEEkeywords\}(.*?)\\end\{IEEEkeywords\}", body, re.DOTALL
    )
    abstract = abstract_m.group(1).strip() if abstract_m else ""
    keywords = keywords_m.group(1).strip() if keywords_m else ""
    body = re.sub(r"\\begin\{abstract\}.*?\\end\{abstract\}", "", body, flags=re.DOTALL)
    body = re.sub(
        r"\\begin\{IEEEkeywords\}.*?\\end\{IEEEkeywords\}", "", body, flags=re.DOTALL
    )

    bbl_path = LATEX_DIR / "main.bbl"
    bib_path = LATEX_DIR / "references.bib"
    refs_block = ""
    if not bib_path.is_file() and bbl_path.is_file():
        references_md = bbl_to_plain_text(read_text(bbl_path))
        refs_block = f"""
\\section*{{References}}
\\begin{{verbatim}}
{references_md}
\\end{{verbatim}}
"""

    flat = f"""\\documentclass[11pt]{{article}}
\\usepackage[utf8]{{inputenc}}
\\usepackage{{graphicx}}
\\usepackage{{booktabs}}
\\usepackage{{amsmath}}
\\usepackage{{hyperref}}

\\title{{{title}}}
\\author{{{author_tex}}}

\\begin{{document}}
\\maketitle

\\begin{{abstract}}
{remove_latex_comments(abstract)}
\\end{{abstract}}

\\noindent\\textbf{{Keywords:}} {keywords}

\\bigskip

{body.strip()}
{refs_block}
\\end{{document}}
"""
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    flat_path = BUILD_DIR / "paper_flat.tex"
    flat_path.write_text(flat, encoding="utf-8")
    return flat_path


def export_docx(flat_tex: Path, output: Path) -> Path:
    try:
        import pypandoc
    except ImportError as exc:
        raise SystemExit(
            "Install: pip install pypandoc-binary"
        ) from exc

    extra_args = [
        f"--resource-path={BUILD_DIR}",
        f"--resource-path={LATEX_DIR}",
        "--standalone",
    ]
    bib = LATEX_DIR / "references.bib"
    if bib.is_file():
        extra_args.extend([
            f"--bibliography={bib}",
            "--citeproc",
        ])

    def _convert(dest: Path) -> None:
        print(f"[export] pandoc -> {dest.name}")
        pypandoc.convert_file(
            str(flat_tex),
            "docx",
            outputfile=str(dest),
            format="latex",
            extra_args=extra_args,
        )

    try:
        _convert(output)
        return output
    except (OSError, RuntimeError) as exc:
        msg = str(exc).lower()
        locked = "permission denied" in msg or "being used" in msg
        if output.name == "main.docx" and locked:
            alt = output.with_name("main_export.docx")
            print(f"[export] Could not write {output.name} (file may be open in Word); using {alt.name}")
            _convert(alt)
            return alt
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description="Export LaTeX paper to Word (.docx)")
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=LATEX_DIR / "main.docx",
        help="Output .docx path (default: PAPER/latex/main.docx)",
    )
    args = parser.parse_args()

    print("[export] Flattening LaTeX sources...")
    flat = build_flat_tex()
    print(f"  Wrote: {flat}")

    output = args.output.resolve()
    try:
        written = export_docx(flat, output)
    except (OSError, RuntimeError) as exc:
        print(f"[export] pandoc error: {exc}", file=sys.stderr)
        print("[export] Flat TeX kept at:", flat, file=sys.stderr)
        return 1

    print(f"[export] Success: {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
