"""Derive BibTeX-style citation keys from PDF metadata and text."""

from __future__ import annotations

import re
from pathlib import Path

DOI_RE = re.compile(r"10\.\d{4,9}/[^\s\],;)<>\"']+", re.IGNORECASE)
YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")


def normalize_doi(raw: str) -> str:
    doi = raw.lower().strip()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        doi = doi.replace(prefix, "")
    return re.sub(r"[.,;]+$", "", doi)


def extract_dois(text: str, limit: int = 3) -> list[str]:
    found: list[str] = []
    compact = re.sub(r"\s+", " ", text[:8000])
    for match in DOI_RE.findall(compact):
        doi = normalize_doi(match)
        if doi and doi not in found:
            found.append(doi)
        if len(found) >= limit:
            break
    return found


def slugify_filename(name: str, max_len: int = 28) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", Path(name).stem.lower()).strip("_")
    return slug[:max_len] or "unknown"


def make_cite_key(
    source: str,
    *,
    title: str | None = None,
    authors: str | None = None,
    year: str | None = None,
    doi: str | None = None,
) -> str:
    """Build a stable cite key aligned with references_to_bib.py conventions."""
    stem = Path(source).stem
    if re.match(r"^ref[\d_]", stem, re.IGNORECASE):
        return stem.replace("-", "_").replace(" ", "_")

    if doi:
        slug = doi.split("/")[-1].replace(".", "").replace("-", "")[:24]
        return f"ref_{slug}"

    author_part = "unknown"
    if authors:
        first = re.split(r"[,;]| and ", authors.strip())[0].strip()
        m = re.match(r"^([A-Za-zÀ-ÿ\-]+)", first)
        if m:
            author_part = m.group(1).lower()
    elif title:
        m = re.match(r"^([A-Za-zÀ-ÿ\-]+)", title.strip())
        if m:
            author_part = m.group(1).lower()
    else:
        author_part = slugify_filename(source).split("_")[0]

    year_part = year or "nd"
    if not year:
        for blob in (title or "", source):
            ym = YEAR_RE.search(blob)
            if ym:
                year_part = ym.group(0)
                break

    return f"ref_{author_part}{year_part}"


def parse_pdf_metadata(meta: dict | None, source: str, body_text: str) -> dict:
    """Merge pypdf/pdfplumber metadata with heuristics from body text."""
    meta = meta or {}
    title = (meta.get("/Title") or meta.get("Title") or "").strip()
    authors = (meta.get("/Author") or meta.get("Author") or "").strip()
    subject = (meta.get("/Subject") or meta.get("Subject") or "").strip()

    creation = meta.get("/CreationDate") or meta.get("CreationDate") or ""
    year = None
    ym = re.search(r"D:(\d{4})", str(creation))
    if ym:
        year = ym.group(1)
    if not year:
        ym = YEAR_RE.search(body_text[:3000])
        if ym:
            year = ym.group(0)

    if not title and subject:
        title = subject

    dois = extract_dois(body_text)
    doi = dois[0] if dois else None

    citation_key = make_cite_key(
        source, title=title or None, authors=authors or None, year=year, doi=doi
    )

    return {
        "title": title,
        "authors": authors,
        "year": year,
        "doi": doi,
        "citation_key": citation_key,
    }
