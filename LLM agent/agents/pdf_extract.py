"""PDF text + table extraction via pdfplumber (fallback: pypdf)."""

from __future__ import annotations

from pathlib import Path

from agents.citation_keys import parse_pdf_metadata


def _table_to_text(table: list[list]) -> str:
    if not table:
        return ""
    lines = ["[TABLE]"]
    for row in table:
        cells = [str(c or "").strip().replace("\n", " ") for c in row]
        if any(cells):
            lines.append(" | ".join(cells))
    return "\n".join(lines)


def extract_pdf(path: str | Path) -> tuple[str | None, dict]:
    """
    Extract plain text, detected tables, and bibliographic metadata.

    Returns (combined_text, metadata_dict) or (None, {}) on failure.
    """
    path = Path(path)
    if not path.is_file() or path.read_bytes()[:4] != b"%PDF":
        return None, {}

    try:
        import pdfplumber
    except ImportError:
        return _extract_pypdf_fallback(path)

    parts: list[str] = []
    pdf_meta: dict = {}
    n_tables = 0

    try:
        with pdfplumber.open(path) as pdf:
            pdf_meta = pdf.metadata or {}
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                if page_text.strip():
                    parts.append(page_text)

                for table in page.extract_tables() or []:
                    tbl = _table_to_text(table)
                    if tbl:
                        parts.append(tbl)
                        n_tables += 1
    except Exception as exc:
        print(f"[pdf_extract] pdfplumber failed on {path.name}: {exc}")
        return _extract_pypdf_fallback(path)

    body = "\n\n".join(parts).strip()
    if not body:
        print(f"[pdf_extract] pdfplumber empty text on {path.name}, trying pypdf")
        return _extract_pypdf_fallback(path)

    bib = parse_pdf_metadata(pdf_meta, path.name, body)
    bib["source"] = path.name
    bib["tables_detected"] = n_tables
    bib["extractor"] = "pdfplumber"
    return body, bib


def _extract_pypdf_fallback(path: Path) -> tuple[str | None, dict]:
    try:
        from pypdf import PdfReader
        from pypdf.errors import PdfReadError, PdfStreamError

        reader = PdfReader(str(path), strict=False)
        if not reader.pages:
            return None, {}
        body = "\n".join(page.extract_text() or "" for page in reader.pages).strip()
        if not body:
            return None, {}
        meta = reader.metadata or {}
        pdf_meta = {
            "/Title": getattr(meta, "title", None),
            "/Author": getattr(meta, "author", None),
            "/Subject": getattr(meta, "subject", None),
            "/CreationDate": getattr(meta, "creation_date", None),
        }
        bib = parse_pdf_metadata(pdf_meta, path.name, body)
        bib["source"] = path.name
        bib["tables_detected"] = 0
        bib["extractor"] = "pypdf"
        return body, bib
    except (PdfReadError, PdfStreamError, OSError, ValueError) as exc:
        print(f"[pdf_extract] pypdf fallback failed on {path.name}: {exc}")
        return None, {}
