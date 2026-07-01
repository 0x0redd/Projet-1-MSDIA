"""Structured logging for GraphRAG retrieval calls."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any


def setup_rag_logging(log_dir: Path, level: str = "INFO") -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("graphrag.rag")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler(log_dir / "rag_calls.log", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


class RAGCallLogger:
    """Logs each RAG call to rag_calls.log and append-only rag_calls.jsonl."""

    def __init__(self, log_dir: Path, level: str = "INFO"):
        self.log_dir = log_dir
        self.logger = setup_rag_logging(log_dir, level)
        self.jsonl_path = log_dir / "rag_calls.jsonl"

    def log_call(
        self,
        *,
        kind: str,
        query: str,
        cache_hit: bool,
        n_candidates: int,
        n_returned: int,
        elapsed_ms: float,
        extra: dict[str, Any] | None = None,
    ) -> None:
        record = {
            "kind": kind,
            "query": query,
            "cache_hit": cache_hit,
            "n_candidates": n_candidates,
            "n_returned": n_returned,
            "elapsed_ms": round(elapsed_ms, 2),
            **(extra or {}),
        }
        self.logger.info(
            f"{kind} | cache={'HIT' if cache_hit else 'MISS'} | "
            f"candidates={n_candidates} returned={n_returned} | {elapsed_ms:.1f}ms | "
            f"q={query[:80]!r}"
        )
        with open(self.jsonl_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    class Timer:
        def __init__(self):
            self._t0 = time.perf_counter()

        def elapsed_ms(self) -> float:
            return (time.perf_counter() - self._t0) * 1000
