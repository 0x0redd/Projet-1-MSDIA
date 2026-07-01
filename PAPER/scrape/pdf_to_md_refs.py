#!/usr/bin/env python3
"""
Convert a PDF to Markdown and extract its reference list to JSON.

Outputs (next to the PDF by default):
  <name>.md                 — full document as Markdown
  <name>_references.json    — references with title and link(s) separated

Usage:
  python pdf_to_md_refs.py "path/to/paper.pdf"
  python pdf_to_md_refs.py "path/to/paper.pdf" -o ./output_dir
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any
from urllib.parse import unquote

DOI_RE = re.compile(r"10\.\d{4,9}/[^\s\],;)>\]]+", re.IGNORECASE)
URL_RE = re.compile(r"https?://[^\s\],;)>\]]+", re.IGNORECASE)
REF_START_RE = re.compile(r"^\d{1,3}\.\s+(?=[A-Za-z\"'(\[])")
REF_SPLIT_RE = re.compile(r"(?m)^(\d{1,3})\.\s+(?=[A-Za-z\"'(\[])")
MD_REF_LINE_RE = re.compile(r"^\s*-\s*\[\s*(\d{1,3})\s*\]\s+(.*)$")
BRACKET_REF_LINE_RE = re.compile(r"^\s*\[\s*(\d{1,3})\s*\]\s+(.*)$")
AUTHOR_BIO_STOP_RE = re.compile(
    r"received the (M\.Sc|Ph\.D|B\.Sc)|is currently (an Assistant Professor|a Professor)",
    re.IGNORECASE,
)
PAGE_NOISE_RE = re.compile(
    r"^\*\*==> picture|^VOLUME\s+\d+|^\d{4,5}$|^[A-Z]\.\s+.+et al\.:\s",
    re.IGNORECASE,
)


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def clean_url(url: str) -> str:
    url = url.replace(" ", "")
    url = re.sub(r"[.,;]+$", "", url)
    return unquote(url)


def normalize_doi(raw: str) -> str:
    doi = raw.lower().strip()
    doi = doi.replace("https://doi.org/", "")
    doi = doi.replace("http://doi.org/", "")
    doi = doi.replace("doi:", "").strip()
    doi = re.sub(r"\.?(pmid|pmcid):.*$", "", doi, flags=re.IGNORECASE)
    doi = re.sub(r"[.,;]+$", "", doi)
    return doi


def extract_dois(text: str) -> list[str]:
    compact = re.sub(r"\s+", "", text.replace("\n", " "))
    found: list[str] = []
    for match in DOI_RE.findall(compact):
        doi = normalize_doi(match)
        if doi and doi not in found:
            found.append(doi)
    # "doi: 10.xxx" without URL
    for match in re.findall(r"doi:\s*(10\.\d{4,9}/[^\s\],;)]+)", text, re.IGNORECASE):
        doi = normalize_doi(match)
        if doi and doi not in found:
            found.append(doi)
    return found


def extract_urls(text: str) -> list[str]:
    compact = text.replace("\n", " ")
    found: list[str] = []
    for match in URL_RE.findall(compact):
        url = clean_url(match)
        if url not in found:
            found.append(url)
    return found


def preprocess_references_text(raw: str) -> str:
    """Merge wrapped lines so DOIs/URLs are not split into fake ref numbers."""
    lines = raw.splitlines()
    merged: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if REF_START_RE.match(stripped):
            merged.append(stripped)
        elif merged:
            merged[-1] += " " + stripped
        else:
            merged.append(stripped)
    return "\n".join(merged)


def strip_links_for_title(text: str) -> str:
    """Reference title = citation text without URLs / DOIs / access noise."""
    t = re.sub(r"https?://\S+", "", text)
    t = re.sub(r"doi:\s*10\.\d{4,9}/\S+", "", t, flags=re.IGNORECASE)
    t = re.sub(r"10\.\d{4,9}/\S+", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\b(PMID|PMCID):\s*\S+", "", t, flags=re.IGNORECASE)
    return normalize_whitespace(t)


def build_link_fields(dois: list[str], urls: list[str]) -> tuple[str | None, list[dict[str, str]]]:
    """Primary link + structured link list."""
    links: list[dict[str, str]] = []
    for doi in dois:
        links.append({"type": "doi", "url": f"https://doi.org/{doi}", "value": doi})
    for url in urls:
        if not any(x["url"] == url for x in links):
            links.append({"type": "url", "url": url, "value": url})

    primary: str | None = None
    if dois:
        primary = f"https://doi.org/{dois[0]}"
    elif urls:
        primary = urls[0]
    return primary, links


def find_references_section(markdown: str) -> tuple[str, int | None]:
    """
    Return (references_body, start_line_index).
    Looks for a References / Bibliography heading, else scans the tail of the doc.
    """
    lines = markdown.splitlines()
    heading_re = re.compile(
        r"^\s*#{0,3}\s*\*?\*?(references|bibliography|works cited|literature cited)\*?\*?\s*\.?\s*$",
        re.IGNORECASE,
    )

    start_idx: int | None = None
    for i, line in enumerate(lines):
        plain = re.sub(r"[#*_`]", "", line).strip()
        if heading_re.match(plain) or heading_re.match(line.strip()):
            start_idx = i + 1
            break

    if start_idx is None:
        tail = "\n".join(lines[max(0, int(len(lines) * 0.65)) :])
        md_ref_count = len(MD_REF_LINE_RE.findall(tail))
        num_ref_count = len(REF_SPLIT_RE.findall(tail))
        if md_ref_count >= 5 or num_ref_count >= 5:
            start_idx = max(0, int(len(lines) * 0.65))

    if start_idx is None:
        return "", None

    body_lines = lines[start_idx:]
    # Stop at obvious post-reference sections
    stop_re = re.compile(
        r"^\s*#{0,3}\s*(appendix|supplementary|acknowledg|author contributions)\b",
        re.IGNORECASE,
    )
    kept: list[str] = []
    for line in body_lines:
        stripped = line.strip()
        if stop_re.match(stripped) or stop_re.match(re.sub(r"[#*_`]", "", line).strip()):
            break
        if AUTHOR_BIO_STOP_RE.search(stripped):
            break
        kept.append(line)

    return "\n".join(kept).strip(), start_idx


def _ref_entry(number: int, body: str) -> dict[str, Any] | None:
    text = normalize_whitespace(body)
    if len(text) < 15:
        return None
    dois = extract_dois(body)
    urls = extract_urls(body)
    primary, links = build_link_fields(dois, urls)
    return {
        "number": number,
        "title": strip_links_for_title(text),
        "link": primary,
        "links": links,
        "raw": text,
    }


def parse_markdown_bracket_references(raw: str) -> list[dict[str, Any]]:
    """IEEE / pandoc style: '- [1] Author..., doi: ...'"""
    refs: list[dict[str, Any]] = []
    current_num: int | None = None
    current_parts: list[str] = []

    def flush() -> None:
        nonlocal current_num, current_parts
        if current_num is None or not current_parts:
            return
        entry = _ref_entry(current_num, " ".join(current_parts))
        if entry:
            refs.append(entry)
        current_num = None
        current_parts = []

    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if AUTHOR_BIO_STOP_RE.search(stripped):
            break
        m = MD_REF_LINE_RE.match(stripped) or BRACKET_REF_LINE_RE.match(stripped)
        if m:
            flush()
            current_num = int(m.group(1))
            current_parts = [m.group(2).strip()]
        elif current_num is not None:
            if PAGE_NOISE_RE.match(stripped):
                continue
            current_parts.append(stripped)

    flush()
    return refs


def parse_references_from_text(raw: str) -> list[dict[str, Any]]:
    if not raw.strip():
        return []

    md_refs = parse_markdown_bracket_references(raw)
    if len(md_refs) >= 3:
        return md_refs

    preprocessed = preprocess_references_text(raw)
    parts = REF_SPLIT_RE.split(preprocessed)
    if len(parts) < 3:
        return md_refs

    refs: list[dict[str, Any]] = []
    it = iter(parts[1:])
    for num_str, body in zip(it, it, strict=False):
        entry = _ref_entry(int(num_str), body)
        if entry:
            refs.append(entry)
    return refs or md_refs


def pdf_to_markdown(pdf_path: Path) -> str:
    try:
        import pymupdf4llm
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: pip install pymupdf pymupdf4llm"
        ) from exc

    return pymupdf4llm.to_markdown(str(pdf_path))


def convert_pdf(
    pdf_path: Path,
    output_dir: Path | None = None,
) -> dict[str, Path]:
    pdf_path = pdf_path.resolve()
    if not pdf_path.is_file():
        raise FileNotFoundError(pdf_path)

    out_dir = (output_dir or pdf_path.parent).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = pdf_path.stem
    md_path = out_dir / f"{stem}.md"
    json_path = out_dir / f"{stem}_references.json"

    print(f"Converting PDF to Markdown: {pdf_path.name}")
    markdown = pdf_to_markdown(pdf_path)
    md_path.write_text(markdown, encoding="utf-8")
    print(f"  Saved: {md_path}")

    ref_raw, ref_line = find_references_section(markdown)
    references = parse_references_from_text(ref_raw)

    payload = {
        "source_pdf": str(pdf_path),
        "markdown_file": str(md_path),
        "references_section_line": ref_line,
        "references_found": len(references),
        "references": references,
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Extracted {len(references)} references -> {json_path}")

    if not references:
        print(
            "  Warning: no numbered references parsed. "
            "Check the .md file — the References heading may differ."
        )

    return {"markdown": md_path, "references": json_path}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="PDF to Markdown + references JSON (title and link separated)."
    )
    parser.add_argument("pdf", type=Path, help="Input PDF path")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: same folder as the PDF)",
    )
    args = parser.parse_args(argv)

    try:
        convert_pdf(args.pdf, args.output_dir)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
