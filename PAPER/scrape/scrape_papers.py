#!/usr/bin/env python3
"""
Scrape and download open-access PDFs from PAPER/scrape/References.txt.

Uses Unpaywall + Semantic Scholar + direct URL resolution (legal OA sources only).
Outputs PDFs to PAPER/scrape/downloads/ and a manifest JSON/CSV for follow-up.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable
from urllib.parse import unquote, urlparse

import requests

SCRIPT_DIR = Path(__file__).resolve().parent
REFS_FILE = SCRIPT_DIR / "References.txt"
DOWNLOAD_DIR = SCRIPT_DIR / "downloads"
PAPER_DIR = SCRIPT_DIR.parent
MANIFEST_JSON = SCRIPT_DIR / "download_manifest.json"
MANIFEST_CSV = SCRIPT_DIR / "download_manifest.csv"

DOI_RE = re.compile(r"10\.\d{4,9}/[^\s\],;)]+", re.IGNORECASE)
# Reference lines start with "N. Author..." — not bare DOI suffixes like "110247."
REF_START_RE = re.compile(r"^\d{1,3}\.\s+(?=[A-Za-z\"'])")
REF_SPLIT_RE = re.compile(r"(?m)^(\d{1,3})\.\s+(?=[A-Za-z\"'])")
URL_RE = re.compile(r"https?://[^\s\],;)]+", re.IGNORECASE)

SKIP_URL_SUBSTRINGS = (
    "kaggle.com",
    "cancer.net",
    "docs.opencv.org",
    "scikit-image.org",
    "scikit-learn.org",
    "scikit- learn",
    "christophm.github.io",
    "github.io/interpretable",
    "10.34740/kaggle",
)

SKIP_TEXT_SUBSTRINGS = (
    "kindly provide reference",
    "packt publishing",
    "dataset.",
    "documentation",
    "statistics.",
    "retrieved may",
    "accessed ",
    "n.d.).",
)

USER_AGENT = (
    "Projet-MSDIA-PaperScraper/1.0 "
    "(academic research; mailto:research@example.com)"
)


@dataclass
class Reference:
    number: int
    text: str
    dois: list[str] = field(default_factory=list)
    urls: list[str] = field(default_factory=list)
    kind: str = "unknown"
    skip_reason: str | None = None


@dataclass
class DownloadResult:
    ref_number: int
    title_hint: str
    kind: str
    status: str
    source: str | None = None
    doi: str | None = None
    url: str | None = None
    file_path: str | None = None
    error: str | None = None


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


def preprocess_references_text(raw: str) -> str:
    """Merge wrapped lines so DOIs/URLs are not split into fake ref numbers."""
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


def extract_dois(text: str) -> list[str]:
    compact = re.sub(r"\s+", "", text.replace("\n", " "))
    found = []
    for match in DOI_RE.findall(compact):
        doi = normalize_doi(match)
        if doi and doi not in found:
            found.append(doi)
    return found


def extract_urls(text: str) -> list[str]:
    compact = text.replace("\n", " ")
    found = []
    for match in URL_RE.findall(compact):
        url = clean_url(match)
        if url not in found:
            found.append(url)
    return found


def classify_reference(ref: Reference) -> Reference:
    lower = ref.text.lower()
    if any(s in lower for s in SKIP_TEXT_SUBSTRINGS):
        ref.kind = "skip"
        ref.skip_reason = "non-paper reference (book/docs/statistics/note)"
        return ref

    for url in ref.urls:
        if any(s in url.lower() for s in SKIP_URL_SUBSTRINGS):
            ref.kind = "skip"
            ref.skip_reason = "non-paper URL (dataset/docs/web page)"
            return ref

    for doi in ref.dois:
        if "kaggle" in doi:
            ref.kind = "skip"
            ref.skip_reason = "dataset DOI"
            return ref

    if ref.dois:
        ref.kind = "paper"
        return ref

    if ref.urls:
        ref.kind = "paper_url"
        return ref

    ref.kind = "manual"
    ref.skip_reason = "no DOI or URL found — add link manually"
    return ref


def parse_references(path: Path) -> list[Reference]:
    raw = preprocess_references_text(
        path.read_text(encoding="utf-8", errors="replace")
    )
    parts = REF_SPLIT_RE.split(raw)
    refs: list[Reference] = []

    # split returns: [preamble, num1, text1, num2, text2, ...]
    it = iter(parts[1:])
    for num_str, body in zip(it, it, strict=False):
        text = normalize_whitespace(body)
        ref = Reference(
            number=int(num_str),
            text=text,
            dois=extract_dois(body),
            urls=extract_urls(body),
        )
        refs.append(classify_reference(ref))
    return refs


def title_hint(text: str, max_len: int = 80) -> str:
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"10\.\d{4,9}/\S+", "", text)
    text = normalize_whitespace(text)
    if len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text


def safe_filename(ref_num: int, doi: str | None, url: str | None) -> str:
    if doi:
        slug = doi.replace("/", "_").replace(".", "-")
    elif url:
        slug = re.sub(r"[^\w\-]+", "_", urlparse(url).path)[:60]
    else:
        slug = "unknown"
    return f"ref_{ref_num:02d}_{slug}.pdf"


class PaperDownloader:
    def __init__(self, email: str, timeout: int = 45, delay: float = 1.0):
        self.email = email
        self.timeout = timeout
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})

    def _sleep(self) -> None:
        if self.delay > 0:
            time.sleep(self.delay)

    def unpaywall_pdf(self, doi: str) -> str | None:
        self._sleep()
        try:
            resp = self.session.get(
                f"https://api.unpaywall.org/v2/{doi}",
                params={"email": self.email},
                timeout=self.timeout,
            )
            if not resp.ok:
                return None
            data = resp.json()
            loc = data.get("best_oa_location") or {}
            return loc.get("url_for_pdf") or loc.get("url")
        except requests.RequestException:
            return None

    def semantic_scholar_pdf(self, doi: str) -> str | None:
        self._sleep()
        try:
            resp = self.session.get(
                f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}",
                params={"fields": "openAccessPdf,title"},
                timeout=self.timeout,
            )
            if not resp.ok:
                return None
            oa = resp.json().get("openAccessPdf") or {}
            return oa.get("url")
        except requests.RequestException:
            return None

    def crossref_title_lookup(self, query: str) -> str | None:
        self._sleep()
        try:
            resp = self.session.get(
                "https://api.crossref.org/works",
                params={"query.title": query[:200], "rows": 1},
                timeout=self.timeout,
            )
            if not resp.ok:
                return None
            items = resp.json().get("message", {}).get("items", [])
            if not items:
                return None
            doi = items[0].get("DOI")
            return normalize_doi(doi) if doi else None
        except requests.RequestException:
            return None

    def resolve_pdf_url(self, ref: Reference) -> tuple[str | None, str | None, str | None]:
        """Return (pdf_url, doi_used, source_name)."""
        doi = ref.dois[0] if ref.dois else None

        if not doi and ref.kind == "manual":
            doi = self.crossref_title_lookup(ref.text)
            if doi:
                ref.dois = [doi]
                ref.kind = "paper"

        if doi:
            for name, fn in (
                ("unpaywall", self.unpaywall_pdf),
                ("semantic_scholar", self.semantic_scholar_pdf),
            ):
                pdf_url = fn(doi)
                if pdf_url:
                    return pdf_url, doi, name

        for url in ref.urls:
            if any(s in url.lower() for s in SKIP_URL_SUBSTRINGS):
                continue
            if url.lower().endswith(".pdf"):
                return url, doi, "direct_pdf_url"
            # Research Square / preprint hosts sometimes expose /pdf
            if "researchsquare.com" in url.lower() or "rs." in url.lower():
                if "/v" in url and not url.endswith("/pdf"):
                    return url.rstrip("/") + "/pdf", doi, "researchsquare_guess"

        return None, doi, None

    def _candidate_pdf_urls(self, url: str) -> list[str]:
        parsed = urlparse(url)
        candidates = [url]
        if "mdpi.com" in parsed.netloc:
            base = url.split("?")[0].rstrip("/")
            if not base.endswith("/pdf"):
                candidates.insert(0, base + "/pdf")
            article = base.replace("/pdf", "")
            candidates.append(article)
        if url.endswith("/pdf"):
            candidates.append(url.rsplit("/pdf", 1)[0])
        # de-duplicate preserving order
        seen: set[str] = set()
        ordered: list[str] = []
        for item in candidates:
            if item not in seen:
                seen.add(item)
                ordered.append(item)
        return ordered

    def download_pdf(self, url: str, dest: Path) -> bool:
        for candidate in self._candidate_pdf_urls(url):
            if self._try_download_once(candidate, dest):
                return True
        return False

    def _try_download_once(self, url: str, dest: Path) -> bool:
        self._sleep()
        headers = {"Accept": "application/pdf,*/*"}
        parsed = urlparse(url)
        if parsed.netloc:
            headers["Referer"] = f"{parsed.scheme}://{parsed.netloc}/"

        try:
            # Warm up publisher session (helps MDPI / Elsevier gateways).
            if "mdpi.com" in parsed.netloc and "/pdf" in url:
                landing = url.split("/pdf")[0]
                self.session.get(landing, timeout=self.timeout)

            resp = self.session.get(
                url,
                timeout=self.timeout,
                allow_redirects=True,
                stream=True,
                headers=headers,
            )
            if not resp.ok:
                return False

            content_type = (resp.headers.get("Content-Type") or "").lower()
            chunks: list[bytes] = []
            size = 0
            for chunk in resp.iter_content(chunk_size=65536):
                if not chunk:
                    continue
                chunks.append(chunk)
                size += len(chunk)
                if size > 12:
                    break

            if not chunks:
                return False

            head = b"".join(chunks)[:8]
            if head[:4] != b"%PDF" and "pdf" not in content_type:
                return False

            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as f:
                for part in chunks:
                    f.write(part)
                for chunk in resp.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)
            if dest.stat().st_size <= 1024:
                dest.unlink(missing_ok=True)
                return False
            return self._validate_pdf(dest)
        except (requests.RequestException, OSError):
            return False

    def _validate_pdf(self, path: Path) -> bool:
        """Reject truncated downloads that pass the %PDF header check."""
        try:
            from pypdf import PdfReader
            from pypdf.errors import PdfReadError, PdfStreamError

            reader = PdfReader(str(path), strict=False)
            if not reader.pages:
                path.unlink(missing_ok=True)
                return False
            return True
        except (PdfReadError, PdfStreamError, OSError, ValueError):
            path.unlink(missing_ok=True)
            return False


def write_manifest(results: list[DownloadResult]) -> None:
    MANIFEST_JSON.write_text(
        json.dumps([asdict(r) for r in results], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    fields = list(asdict(results[0]).keys()) if results else []
    if fields:
        with open(MANIFEST_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in results:
                writer.writerow(asdict(row))


def filter_refs(refs: Iterable[Reference], only: set[int] | None) -> list[Reference]:
    refs = list(refs)
    if only:
        return [r for r in refs if r.number in only]
    return refs


def main() -> int:
    parser = argparse.ArgumentParser(description="Download OA papers from References.txt")
    parser.add_argument(
        "--email",
        required=True,
        help="Email for Unpaywall API (required by their terms of service)",
    )
    parser.add_argument(
        "--refs",
        help="Comma-separated reference numbers to process (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DOWNLOAD_DIR,
        help="Folder for downloaded PDFs",
    )
    parser.add_argument(
        "--copy-to-paper",
        action="store_true",
        help="Also copy successful PDFs into PAPER/ for prepare_data.py",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse references and print plan without downloading",
    )
    parser.add_argument(
        "--lookup-missing-doi",
        action="store_true",
        help="Try CrossRef title search for refs without DOI/URL",
    )
    parser.add_argument("--delay", type=float, default=1.0, help="Seconds between API calls")
    parser.add_argument("--timeout", type=int, default=45, help="HTTP timeout seconds")
    args = parser.parse_args()

    if not REFS_FILE.exists():
        print(f"[scrape] Missing {REFS_FILE}")
        return 1

    only = None
    if args.refs:
        only = {int(x.strip()) for x in args.refs.split(",") if x.strip()}

    refs = filter_refs(parse_references(REFS_FILE), only)
    print(f"[scrape] Parsed {len(refs)} reference(s) from {REFS_FILE.name}")

    if args.dry_run:
        for ref in refs:
            print(
                f"  #{ref.number:02d} [{ref.kind}] "
                f"dois={ref.dois or '-'} urls={len(ref.urls)} "
                f"{title_hint(ref.text, 60)}"
            )
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    downloader = PaperDownloader(email=args.email, timeout=args.timeout, delay=args.delay)
    results: list[DownloadResult] = []
    ok = 0

    for ref in refs:
        hint = title_hint(ref.text)
        if ref.kind == "skip":
            results.append(
                DownloadResult(
                    ref_number=ref.number,
                    title_hint=hint,
                    kind=ref.kind,
                    status="skipped",
                    error=ref.skip_reason,
                )
            )
            print(f"[scrape] #{ref.number:02d} skipped — {ref.skip_reason}")
            continue

        if ref.kind == "manual" and not args.lookup_missing_doi:
            results.append(
                DownloadResult(
                    ref_number=ref.number,
                    title_hint=hint,
                    kind=ref.kind,
                    status="manual",
                    error=ref.skip_reason,
                )
            )
            print(f"[scrape] #{ref.number:02d} manual — no DOI/URL")
            continue

        pdf_url, doi, source = downloader.resolve_pdf_url(ref)
        if not pdf_url:
            results.append(
                DownloadResult(
                    ref_number=ref.number,
                    title_hint=hint,
                    kind=ref.kind,
                    status="not_found",
                    doi=doi,
                    error="No open-access PDF found (paywalled or unavailable)",
                )
            )
            print(f"[scrape] #{ref.number:02d} not found — {hint[:50]}")
            continue

        dest = args.output_dir / safe_filename(ref.number, doi, pdf_url)
        if dest.exists() and dest.stat().st_size > 1024:
            status = "cached"
            ok += 1
            file_path = str(dest)
            print(f"[scrape] #{ref.number:02d} cached — {dest.name}")
        elif downloader.download_pdf(pdf_url, dest):
            status = "downloaded"
            ok += 1
            file_path = str(dest)
            print(f"[scrape] #{ref.number:02d} downloaded via {source} — {dest.name}")
            if args.copy_to_paper:
                paper_dest = PAPER_DIR / dest.name
                paper_dest.write_bytes(dest.read_bytes())
        else:
            status = "failed"
            file_path = None
            if dest.exists():
                dest.unlink(missing_ok=True)
            print(f"[scrape] #{ref.number:02d} failed — {pdf_url}")

        results.append(
            DownloadResult(
                ref_number=ref.number,
                title_hint=hint,
                kind=ref.kind,
                status=status,
                source=source,
                doi=doi,
                url=pdf_url,
                file_path=file_path,
            )
        )

    write_manifest(results)
    downloaded = sum(1 for r in results if r.status in {"downloaded", "cached"})
    print(
        f"[scrape] Done: {downloaded} PDF(s) ready, "
        f"{len(results) - downloaded} skipped/failed/manual"
    )
    print(f"[scrape] Manifest -> {MANIFEST_JSON.name}, {MANIFEST_CSV.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
