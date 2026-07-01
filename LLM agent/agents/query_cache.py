"""Disk cache for RAG query results."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path


class QueryCache:
    def __init__(self, cache_dir: Path, enabled: bool = True):
        self.cache_dir = cache_dir
        self.enabled = enabled
        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, query: str, kind: str) -> Path:
        digest = hashlib.sha256(f"{kind}:{query.strip().lower()}".encode()).hexdigest()[:20]
        return self.cache_dir / f"{kind}_{digest}.json"

    def get(self, query: str, kind: str) -> dict | None:
        if not self.enabled:
            return None
        path = self._path(query, kind)
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["_cache"] = {"hit": True, "path": str(path)}
            return payload
        except (json.JSONDecodeError, OSError):
            return None

    def set(self, query: str, kind: str, result: dict) -> None:
        if not self.enabled:
            return
        path = self._path(query, kind)
        payload = {
            **result,
            "_cached_at": time.time(),
            "_query": query,
            "_kind": kind,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
