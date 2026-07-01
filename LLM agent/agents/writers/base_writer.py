"""Shared LLM call logic for section writers."""

from __future__ import annotations

from pathlib import Path

import ollama
import yaml

from agents.paths import resolve_config

DENSITY_PROMPTS = {
    "sparse": """
CONTEXT DENSITY: SPARSE (few retrieved chunks).
- Be conservative; cite only what appears in context.
- Flag gaps explicitly with [DATA MISSING] rather than filling with general knowledge.
- Prefer shorter paragraphs; do not invent related work.""",
    "moderate": """
CONTEXT DENSITY: MODERATE.
- Balance breadth and depth; cite multiple sources where available.
- Tie each claim to a chunk or experiment record.""",
    "dense": """
CONTEXT DENSITY: DENSE (many retrieved chunks).
- Synthesise across sources; avoid repeating the same citation twice in one paragraph.
- Prioritise highest-scoring chunks; merge overlapping evidence.
- Keep prose tight — do not restate every chunk.""",
}


class BaseWriter:
    def __init__(self, config_path: str | Path | None = None):
        cfg_path = resolve_config(config_path)
        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        self.cfg = cfg
        self.model = cfg["llm"]["model"]
        self.temp = cfg["llm"]["temperature"]
        self.ctx_win = cfg["llm"]["context_window"]
        self.config_path = cfg_path
        paper = cfg.get("paper", {})
        self.paper_title = paper.get("title", "Research paper")

    def _system_prompt(self, density: str) -> str:
        density_block = DENSITY_PROMPTS.get(density, DENSITY_PROMPTS["moderate"])
        return f"""You are a scientific writing assistant for the research paper:
"{self.paper_title}".

STRICT RULES:
1. Write in formal, precise academic English. No colloquialisms.
2. Every number MUST come from the EXPERIMENT DATA block below — never invent metrics.
3. Cite ONLY using \\cite{{key}} with keys exactly as shown in context.
4. NEVER use numbered references [1], [2] or a References section at the end.
5. NEVER invent author names, paper titles, or datasets not in context.
6. NEVER write "In conclusion" at the end of Related Work.
7. Structure paragraphs: claim → evidence → interpretation.
8. If experiment data is empty, write "[DATA MISSING]" instead of guessing.
9. Output Markdown body only — no meta-commentary.
{density_block}"""

    @staticmethod
    def context_density(n_chunks: int, n_metrics: int = 0) -> str:
        score = n_chunks + (1 if n_metrics else 0)
        if score <= 2:
            return "sparse"
        if score <= 8:
            return "moderate"
        return "dense"

    def call_llm(self, user_prompt: str, *, density: str = "moderate", dry_run: bool = False) -> str:
        if dry_run:
            preview = user_prompt[:400].replace("\n", " ")
            return (
                f"<!-- DRY RUN: LLM call skipped for section draft -->\n\n"
                f"**[DRY RUN]** Would call `{self.model}` with {density} context.\n\n"
                f"Prompt preview: {preview}…"
            )

        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": self._system_prompt(density)},
                {"role": "user", "content": user_prompt},
            ],
            options={
                "temperature": self.temp,
                "num_ctx": self.ctx_win,
                "repeat_penalty": self.cfg["llm"].get("repeat_penalty", 1.1),
            },
        )
        return response["message"]["content"]

    def format_context(self, chunks: list[dict], *, max_chunks: int | None = None) -> str:
        if not chunks:
            return "## Retrieved context\n\n[No paper chunks retrieved]\n"

        density = self.context_density(len(chunks))
        limit = max_chunks
        if limit is None:
            limit = {"sparse": 6, "moderate": 12, "dense": 8}[density]

        lines = [f"## Retrieved context ({density}, top {min(limit, len(chunks))})\n"]
        for i, chunk in enumerate(chunks[:limit], 1):
            meta = (
                f"Key: {chunk.get('citation_key', '?')} | "
                f"Source: {chunk.get('source', '?')} | "
                f"Score: {chunk.get('score', '?')}"
            )
            if chunk.get("title"):
                meta += f" | Title: {chunk['title'][:80]}"
            lines.append(f"[{i}] {meta}\n{chunk['text']}\n")
        return "\n".join(lines)

    def format_metrics(self, records: list[dict]) -> str:
        if not records:
            return ""
        lines = ["## Experiment data\n"]
        for r in records:
            lines.append(f"Source: {r['source']}\n{r['data']}\n")
        return "\n".join(lines)

    def build_prompt(
        self,
        context_pkg: dict,
        task: str,
        *,
        dry_run: bool = False,
    ) -> tuple[str, str]:
        """Return (full_prompt, density) for adaptive LLM calls."""
        chunks = context_pkg.get("paper_chunks", [])
        metrics = context_pkg.get("metric_records", [])
        density = self.context_density(len(chunks), len(metrics))
        context = self.format_context(chunks)
        metrics_block = self.format_metrics(metrics)
        prompt = f"{context}\n\n{metrics_block}\n\n## Task\n{task}"
        return prompt, density

    def run_task(self, context_pkg: dict, task: str) -> str:
        """Adaptive prompt + optional dry-run for section writers."""
        dry_run = bool(context_pkg.get("dry_run", False))
        prompt, density = self.build_prompt(context_pkg, task, dry_run=dry_run)
        return self.call_llm(prompt, density=density, dry_run=dry_run)
