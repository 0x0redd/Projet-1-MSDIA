#!/usr/bin/env python3
"""Build references.bib from PAPER/scrape/References.txt."""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import requests

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
REFS_FILE = PROJECT_ROOT.parent / "PAPER" / "scrape" / "References.txt"
DEFAULT_OUT = PROJECT_ROOT.parent / "PAPER" / "latex" / "references.bib"
PAPERS_DIR = PROJECT_ROOT / "data" / "papers"

REF_START_RE = re.compile(r"^\d{1,3}\.\s+(?=[A-Za-z\"'])")
REF_SPLIT_RE = re.compile(r"(?m)^(\d{1,3})\.\s+(?=[A-Za-z\"'])")
DOI_RE = re.compile(r"10\.\d{4,9}/[^\s\],;)]+", re.IGNORECASE)
URL_RE = re.compile(r"https?://[^\s\],;)]+", re.IGNORECASE)


def preprocess_references_text(raw: str) -> str:
    lines = raw.splitlines()
    merged: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.lower() == "references":
            continue
        if REF_START_RE.match(stripped):
            merged.append(stripped)
        elif merged:
            merged[-1] += " " + stripped
        else:
            merged.append(stripped)
    return "\n".join(merged)


def normalize_doi(raw: str) -> str:
    doi = raw.lower().strip()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        doi = doi.replace(prefix, "")
    doi = re.sub(r"\.?(pmid|pmcid):.*$", "", doi, flags=re.IGNORECASE)
    return re.sub(r"[.,;]+$", "", doi)


def extract_dois(text: str) -> list[str]:
    compact = re.sub(r"\s+", "", text)
    found = []
    for match in DOI_RE.findall(compact):
        doi = normalize_doi(match)
        if doi and doi not in found:
            found.append(doi)
    return found


def make_cite_key(num: int, text: str, doi: str | None) -> str:
    if doi:
        slug = doi.split("/")[-1].replace(".", "")[:24]
        return f"ref{num:02d}_{slug}"
    author = re.match(r"^([A-Za-zÀ-ÿ\-]+)", text)
    year_m = re.search(r"\b(19|20)\d{2}\b", text)
    author_part = author.group(1) if author else "unknown"
    year_part = year_m.group(0) if year_m else "nd"
    return f"ref{num:02d}_{author_part}{year_part}"


def crossref_lookup(doi: str) -> dict | None:
    try:
        r = requests.get(
            f"https://api.crossref.org/works/{doi}",
            headers={"User-Agent": "Projet-MSDIA-BibGen/1.0"},
            timeout=20,
        )
        if r.ok:
            return r.json().get("message")
    except requests.RequestException:
        pass
    return None


def format_authors(cr: dict) -> str:
    authors = cr.get("author", [])
    parts = []
    for a in authors[:6]:
        family = a.get("family", "")
        given = a.get("given", "")
        if family:
            parts.append(f"{family}, {given}".strip(", "))
    if len(authors) > 6:
        parts.append("others")
    return " and ".join(parts) if parts else "Unknown"


def entry_from_crossref(key: str, doi: str, cr: dict, fallback_text: str) -> str:
    title = (cr.get("title") or [fallback_text[:120]])[0]
    journal = (cr.get("container-title") or [""])[0]
    year = ""
    for field in ("published-print", "published-online", "created"):
        if field in cr and "date-parts" in cr[field]:
            year = str(cr[field]["date-parts"][0][0])
            break
    authors = format_authors(cr)
    etype = "@article"
    if cr.get("type") in ("proceedings-article", "paper-conference"):
        etype = "@inproceedings"
    lines = [
        f"{etype}{{{key},",
        f"  author = {{{authors}}},",
        f"  title = {{{title}}},",
    ]
    if journal:
        if etype == "@inproceedings":
            lines.append(f"  booktitle = {{{journal}}},")
        else:
            lines.append(f"  journal = {{{journal}}},")
    if year:
        lines.append(f"  year = {{{year}}},")
    lines.append(f"  doi = {{{doi}}},")
    lines.append("}")
    return "\n".join(lines)


def entry_manual(key: str, text: str, doi: str | None, url: str | None) -> str:
    year_m = re.search(r"\b(19|20)\d{2}\b", text)
    year = year_m.group(0) if year_m else ""
    lines = [
        f"@misc{{{key},",
        f"  note = {{{text[:500]}}},",
    ]
    if year:
        lines.append(f"  year = {{{year}}},")
    if doi:
        lines.append(f"  doi = {{{doi}}},")
    if url:
        lines.append(f"  url = {{{url}}},")
    lines.append("}")
    return "\n".join(lines)


def parse_references(path: Path) -> list[tuple[int, str]]:
    raw = preprocess_references_text(path.read_text(encoding="utf-8", errors="replace"))
    parts = REF_SPLIT_RE.split(raw)
    refs = []
    it = iter(parts[1:])
    for num_str, body in zip(it, it, strict=False):
        refs.append((int(num_str), body.strip()))
    return refs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refs", type=Path, default=REFS_FILE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--use-crossref", action="store_true", help="Fetch metadata from CrossRef (slow)")
    parser.add_argument("--max", type=int, default=0, help="Max entries (0 = all)")
    args = parser.parse_args()

    if not args.refs.exists():
        print(f"[bib] Missing {args.refs}")
        return 1

    entries = []
    refs = parse_references(args.refs)
    if args.max:
        refs = refs[: args.max]

    indexed_pdfs = {p.stem.lower() for p in PAPERS_DIR.glob("*.pdf")} if PAPERS_DIR.is_dir() else set()

    for num, text in refs:
        lower = text.lower()
        if "kindly provide" in lower or "incomplete reference" in lower:
            continue
        dois = extract_dois(text)
        doi = dois[0] if dois else None
        urls = URL_RE.findall(text)
        url = urls[0] if urls else None
        key = make_cite_key(num, text, doi)

        if args.use_crossref and doi:
            cr = crossref_lookup(doi)
            time.sleep(0.15)
            if cr:
                entries.append(entry_from_crossref(key, doi, cr, text))
                continue
        entries.append(entry_manual(key, text, doi, url))

    header = "% Auto-generated by references_to_bib.py\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(header + "\n\n".join(entries) + "\n", encoding="utf-8")
    print(f"[bib] Wrote {len(entries)} entries -> {args.out}")
    if indexed_pdfs:
        print(f"[bib] {len(indexed_pdfs)} PDFs in data/papers/ for citation validation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
