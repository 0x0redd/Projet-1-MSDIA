"""Per-section checkpointing via output/state.json."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class PaperState:
    """Tracks section/figure progress for resume-after-interrupt workflows."""

    def __init__(self, path: Path):
        self.path = path
        self.data: dict[str, Any] = self._default()
        if path.is_file():
            self.load()

    @staticmethod
    def _default() -> dict[str, Any]:
        return {
            "version": 1,
            "sections": {},
            "figures": {},
            "rag_call_count": 0,
            "dry_run": False,
            "updated_at": None,
        }

    def load(self) -> None:
        try:
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            self.data = self._default()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.path.write_text(json.dumps(self.data, indent=2, ensure_ascii=False), encoding="utf-8")

    def section_status(self, section: str) -> str:
        return self.data["sections"].get(section, {}).get("status", "pending")

    def mark_section(
        self,
        section: str,
        status: str,
        *,
        output_path: str | None = None,
        rag_queries: list[str] | None = None,
        dry_run: bool = False,
    ) -> None:
        entry: dict[str, Any] = {
            "status": status,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "dry_run": dry_run,
        }
        if output_path:
            entry["output"] = output_path
        if rag_queries is not None:
            entry["rag_queries"] = rag_queries
        self.data["sections"][section] = entry
        self.save()

    def mark_figure(self, figure_type: str, path: str, *, caption_path: str | None = None) -> None:
        entry: dict[str, Any] = {
            "path": path,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        if caption_path:
            entry["caption"] = caption_path
        self.data["figures"][figure_type] = entry
        self.save()

    def increment_rag_calls(self, n: int = 1) -> None:
        self.data["rag_call_count"] = int(self.data.get("rag_call_count", 0)) + n
        self.save()

    def should_skip_section(self, section: str, *, force: bool = False) -> bool:
        if force:
            return False
        entry = self.data["sections"].get(section, {})
        if entry.get("dry_run"):
            return False  # always rewrite dry-run placeholders
        return entry.get("status") in ("drafted", "reviewed", "final")
