#!/usr/bin/env python3
"""Build references.bib from Zotero Exported Items.csv (filtered ML corpus)."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
REPO_ROOT = PROJECT_ROOT.parent

sys.path.insert(0, str(PROJECT_ROOT))
from agents.citation_keys import make_cite_key, normalize_doi  # noqa: E402

DEFAULT_CSV = REPO_ROOT / "PAPER" / "filtered - ML" / "Exported Items.csv"
DEFAULT_OUT = REPO_ROOT / "PAPER" / "latex" / "references.bib"
CITATIONS_OUT = PROJECT_ROOT / "data" / "citations.bib"


def format_authors(raw: str) -> str:
    if not raw.strip():
        return "Unknown"
    parts = [a.strip() for a in raw.split(";") if a.strip()]
    formatted: list[str] = []
    for part in parts:
        # Institution-style author lines break IEEEtran.bst comma parsing
        if part.count(",") >= 2 or re.search(
            r"\b(University|Institute|Faculty|Department|Laboratory)\b", part, re.I
        ):
            formatted.append(f"{{{part}}}")
        else:
            formatted.append(part)
    return " and ".join(formatted)


def bib_escape(text: str) -> str:
    return text.replace("{", "\\{").replace("}", "\\}").strip()


def entry_type(item_type: str) -> str:
    if item_type.lower() in ("conferencepaper", "proceedingspaper"):
        return "@inproceedings"
    return "@article"


def build_entry(row: dict) -> str:
    doi_raw = (row.get("DOI") or "").strip()
    doi = normalize_doi(doi_raw) if doi_raw else None
    title = bib_escape(row.get("Title") or "Untitled")
    authors = format_authors(row.get("Author") or "")
    year = (row.get("Publication Year") or "").strip()
    journal = bib_escape(row.get("Publication Title") or "")
    pages = (row.get("Pages") or "").strip()
    volume = (row.get("Volume") or "").strip()
    number = (row.get("Issue") or "").strip()
    conference = bib_escape(row.get("Conference Name") or row.get("Meeting Name") or "")
    item_type = row.get("Item Type") or "journalArticle"

    key = make_cite_key("zotero", title=title, authors=authors, year=year, doi=doi)
    etype = entry_type(item_type)

    lines = [
        f"{etype}{{{key},",
        f"  author = {{{authors}}},",
        f"  title = {{{title}}},",
    ]
    if etype == "@inproceedings":
        booktitle = conference or journal
        if booktitle:
            lines.append(f"  booktitle = {{{booktitle}}},")
    elif journal:
        lines.append(f"  journal = {{{journal}}},")
    if year:
        lines.append(f"  year = {{{year}}},")
    if volume:
        lines.append(f"  volume = {{{volume}}},")
    if number:
        lines.append(f"  number = {{{number}}},")
    if pages:
        lines.append(f"  pages = {{{pages}}},")
    if doi:
        lines.append(f"  doi = {{{doi}}},")
    lines.append("}")
    return key, "\n".join(lines)


def read_zotero_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def main() -> int:
    parser = argparse.ArgumentParser(description="Zotero CSV → BibTeX (IEEE cite keys from DOI)")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--also", type=Path, default=CITATIONS_OUT, help="Copy for LLM agent validation")
    args = parser.parse_args()

    if not args.csv.exists():
        print(f"[zotero-bib] Missing {args.csv}")
        return 1

    rows = read_zotero_csv(args.csv)
    entries: list[str] = []
    keys: list[str] = []
    seen_keys: set[str] = set()

    for row in rows:
        if not (row.get("Title") or "").strip():
            continue
        key, block = build_entry(row)
        if key in seen_keys:
            suffix = re.sub(r"[^a-z0-9]", "", (row.get("Key") or "dup").lower())[:6]
            block = block.replace(f"{{{key},", f"{{{key}_{suffix},", 1)
            key = f"{key}_{suffix}"
        seen_keys.add(key)
        keys.append(key)
        entries.append(block)

    header = (
        "% Auto-generated from Zotero Exported Items.csv\n"
        "% Cite keys: DOI slug via agents.citation_keys.make_cite_key\n"
    )
    body = header + "\n\n".join(entries) + "\n"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(body, encoding="utf-8")
    print(f"[zotero-bib] Wrote {len(entries)} entries -> {args.out}")

    if args.also:
        args.also.parent.mkdir(parents=True, exist_ok=True)
        args.also.write_text(body, encoding="utf-8")
        print(f"[zotero-bib] Mirrored -> {args.also}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
