"""Agent B — Cursor Orchestrator."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import typer
import yaml
from rich.console import Console

from agents.figure_agent import FigureAgent
from agents.paths import PROJECT_ROOT, resolve_config, resolve_path
from agents.rag_agent import GraphRAGAgent
from agents.state import PaperState
from agents.writers import get_writer
from agents.writers.base_writer import BaseWriter

console = Console()
app = typer.Typer(help="GraphRAG paper writing orchestrator")


SECTION_QUERIES = {
    "introduction": [
        "brain tumor MRI classification glioma meningioma pituitary motivation",
        "handcrafted texture features GLCM LBP HOG DWT separability",
        "classical machine learning SVM brain tumor limited data",
    ],
    "related_work": [
        "brain tumor MRI classification machine learning texture features",
        "LBP GLCM HOG DWT medical image feature extraction",
        "SVM random forest XGBoost radiomics brain MRI",
        "parameter selection grid search separability metrics",
    ],
    "methods": [
        "brain tumor dataset glioma meningioma pituitary MRI split",
        "LBP GLCM HOG DWT parameter grid search feature extraction",
        "SVM KNN random forest XGBoost texture features benchmark",
        "stratified cross validation FDR mutual information Davies-Bouldin",
    ],
    "phase1_results": [
        "GLCM LBP DWT HOG optimal parameters separability score",
        "Phase 1 parameter search Fisher discriminant mutual information",
    ],
    "results_ml": [
        "phase2 SVM KNN random forest Full opt F1 macro accuracy",
        "classical ML leaderboard feature set comparison",
    ],
    "results": [
        "accuracy F1 comparison classical ML brain tumor Phase 2",
        "phase2 SVM feature set accuracy leaderboard",
        "McNemar Friedman statistical test model comparison",
    ],
    "interpretability": [
        "random forest feature importance GLCM LBP HOG branch",
        "XGBoost feature group importance brain MRI texture",
    ],
    "statistics": [
        "bootstrap confidence interval McNemar brain tumor classification",
        "Cohen kappa Friedman Nemenyi classifier comparison",
    ],
    "discussion": [
        "why handcrafted features competitive brain MRI limited data",
        "feature selection aggregation brain tumor MRI classical ML",
        "separability metrics vs grid search classifier bias",
    ],
    "conclusion": [
        "future work handcrafted features brain MRI external validation",
        "limitations single dataset brain tumor classification study",
    ],
    "abstract": [
        "main contribution brain tumor MRI classical ML separability benchmark",
    ],
}


class CursorAgent:
    def __init__(
        self,
        config_path: str | Path | None = None,
        *,
        dry_run: bool = False,
        force: bool = False,
    ):
        cfg_path = resolve_config(config_path)
        with open(cfg_path, encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)
        self.config_path = cfg_path
        self.dry_run = dry_run
        self.force = force
        self.cfg.setdefault("runtime", {})["dry_run"] = dry_run

        self.rag = GraphRAGAgent(cfg_path)
        self.writer = BaseWriter(cfg_path)

        order = self.cfg.get("agents", {}).get("section_order")
        self.SECTION_ORDER = order if order else [
            "introduction", "related_work", "methods", "results",
            "discussion", "conclusion", "abstract",
        ]

        state_path = resolve_path(
            self.cfg.get("paths", {}).get("state_file", "output/state.json")
        )
        self.state_mgr = PaperState(state_path)
        self.state = {s: self.state_mgr.section_status(s) for s in self.SECTION_ORDER}
        self.drafts: dict[str, str] = {}
        self.figures: dict[str, str] = {}

        sections_dir = resolve_path(self.cfg["paths"]["sections_dir"])
        figures_dir = resolve_path(self.cfg["paths"]["figures_dir"])
        sections_dir.mkdir(parents=True, exist_ok=True)
        figures_dir.mkdir(parents=True, exist_ok=True)

        self._load_existing_drafts(sections_dir)

    def _load_existing_drafts(self, sections_dir: Path) -> None:
        for section in self.SECTION_ORDER:
            path = sections_dir / f"{section}.md"
            if path.is_file():
                self.drafts[section] = path.read_text(encoding="utf-8")

    def index_knowledge_base(self, *, rebuild: bool = False) -> None:
        if self.dry_run:
            console.print("[yellow]DRY RUN: skipping index (use without --dry-run to ingest)[/]")
            return
        if rebuild:
            console.print("[bold yellow]Rebuilding ChromaDB collections...[/]")
            self.rag.rebuild_collections()
            cache_dir = resolve_path(self.cfg.get("cache", {}).get("query_cache_dir", "db/query_cache"))
            if cache_dir.is_dir():
                import shutil
                shutil.rmtree(cache_dir)
                console.print(f"[dim]Cleared query cache: {cache_dir}[/]")
        console.print("[bold cyan]Indexing papers (pdfplumber + metadata)...[/]")
        self.rag.ingest_papers(self.cfg["paths"]["papers_dir"])
        console.print("[bold cyan]Indexing experiments...[/]")
        self.rag.ingest_experiments(self.cfg["paths"]["experiments_dir"])
        console.print("[bold green]Knowledge base ready.[/]")

    def write_section(self, section: str) -> None:
        if section not in self.SECTION_ORDER:
            raise ValueError(f"Unknown section: {section}")

        if self.state_mgr.should_skip_section(section, force=self.force):
            console.print(f"[dim]Skipping {section} (already {self.state_mgr.section_status(section)})[/]")
            if section in self.drafts:
                return

        mode = "[DRY RUN] " if self.dry_run else ""
        console.print(f"\n[bold yellow]{mode}Cursor -> writing: {section}[/]")

        queries = SECTION_QUERIES.get(section, [
            f"brain tumor MRI {section.replace('_', ' ')}",
            f"classical machine learning {section.replace('_', ' ')}",
        ])
        context_pkg = self._gather_context(queries, section)
        context_pkg["dry_run"] = self.dry_run

        if not context_pkg["paper_chunks"] and section not in ("abstract", "results", "results_ml"):
            console.print(f"[red]WARNING: No relevant context found for {section}.[/]")

        try:
            writer_fn = get_writer(section)
        except KeyError:
            raise ValueError(
                f"No writer registered for section '{section}'. "
                f"Add agents/writers/{section}.py or update config section_order."
            )

        draft = writer_fn(context_pkg, self.cfg, self.drafts)

        if self.cfg["agents"]["citation_validation"] and not self.dry_run:
            draft = self._validate_citations(draft, context_pkg)

        out_path = resolve_path(self.cfg["paths"]["sections_dir"]) / f"{section}.md"
        out_path.write_text(draft, encoding="utf-8")
        self.drafts[section] = draft
        self.state[section] = "drafted"
        self.state_mgr.mark_section(
            section,
            "drafted",
            output_path=str(out_path),
            rag_queries=queries,
            dry_run=self.dry_run,
        )
        console.print(f"[green]Saved: {out_path}[/]")

    def request_figure(self, figure_type: str, section: str, query: str) -> str:
        mode = "[DRY RUN] " if self.dry_run else ""
        console.print(f"\n[bold yellow]{mode}Cursor -> figure: {figure_type}[/]")
        metrics = self.rag.query_metrics(query)
        self.state_mgr.increment_rag_calls()
        fig_agent = FigureAgent(self.cfg)
        path, caption_path = fig_agent.make_figure(
            figure_type=figure_type,
            metrics=metrics,
            section=section,
            dry_run=self.dry_run,
        )
        self.figures[figure_type] = path
        self.state_mgr.mark_figure(figure_type, path, caption_path=caption_path)
        console.print(f"[green]Figure saved: {path}[/]")
        if caption_path:
            console.print(f"[green]Caption saved: {caption_path}[/]")
        return path

    def write_all(self) -> None:
        if not self.dry_run:
            self.index_knowledge_base()
        for section in self.SECTION_ORDER:
            try:
                self.write_section(section)
            except KeyError:
                console.print(f"[yellow]Skipping unknown writer section: {section}[/]")
        self.request_figure("leaderboard_top10", "results_ml", "phase2 split leaderboard F1")
        self.request_figure("accuracy_bar", "results_ml", "accuracy comparison methods brain tumor")
        if not self.dry_run:
            self.assemble_paper()
            self.assemble_latex()

    def assemble_latex(self) -> None:
        import subprocess

        os.chdir(PROJECT_ROOT)
        py = sys.executable
        scripts = PROJECT_ROOT / "scripts"
        console.print("[bold cyan]Generating LaTeX tables...[/]")
        subprocess.run([py, str(scripts / "tables_to_latex.py")], check=True)
        console.print("[bold cyan]Generating references.bib...[/]")
        zotero_csv = PROJECT_ROOT.parent / "PAPER" / "filtered - ML" / "Exported Items.csv"
        if zotero_csv.exists():
            subprocess.run([py, str(scripts / "zotero_export_to_bib.py")], check=True)
        else:
            subprocess.run([py, str(scripts / "references_to_bib.py")], check=True)
        if self.cfg.get("paths", {}).get("convert_agent_md_to_latex", False):
            console.print("[bold cyan]Converting sections MD -> LaTeX...[/]")
            subprocess.run([py, str(scripts / "md_to_latex.py")], check=True)
        else:
            console.print(
                "[dim]Skipping md_to_latex (convert_agent_md_to_latex=false). "
                "Using existing PAPER/latex/sections/*.tex[/]"
            )
        latex_dir = resolve_path(self.cfg["paths"].get("latex_dir", "../PAPER/latex/"))
        console.print(f"[bold green]LaTeX ready: {latex_dir}[/]")

    def assemble_paper(self) -> None:
        out = resolve_path(self.cfg["paths"]["output_dir"]) / "paper_final.md"
        out.parent.mkdir(parents=True, exist_ok=True)

        for section in self.SECTION_ORDER:
            sec_path = resolve_path(self.cfg["paths"]["sections_dir"]) / f"{section}.md"
            if sec_path.exists() and section not in self.drafts:
                self.drafts[section] = sec_path.read_text(encoding="utf-8")

        paper = self._paper_header()
        for section in self.SECTION_ORDER:
            if section in self.drafts:
                paper += f"\n\n---\n\n{self.drafts[section]}"
        out.write_text(paper, encoding="utf-8")
        console.print(f"\n[bold green]Paper assembled: {out}[/]")

    def _gather_context(self, queries: list[str], section: str) -> dict:
        max_calls = self.cfg["agents"]["max_rag_calls_per_section"]
        paper_chunks = []
        metric_records = []
        seen_texts: set[str] = set()

        for query in queries[:max_calls]:
            result = self.rag.query_papers(query)
            self.state_mgr.increment_rag_calls()
            for chunk in result["chunks"]:
                if chunk["text"] not in seen_texts:
                    paper_chunks.append(chunk)
                    seen_texts.add(chunk["text"])
            if section in (
                "results", "results_ml", "discussion", "abstract", "methods",
                "statistics", "phase1_results", "interpretability",
            ):
                m_result = self.rag.query_metrics(query)
                self.state_mgr.increment_rag_calls()
                metric_records.extend(m_result["records"])

        return {
            "section": section,
            "paper_chunks": paper_chunks,
            "metric_records": metric_records,
            "queries": queries,
            "dry_run": self.dry_run,
        }

    def _validate_citations(self, draft: str, context_pkg: dict) -> str:
        cites_in_draft = set(re.findall(r"\\cite\{([^}]+)\}", draft))
        valid_keys = {c["citation_key"] for c in context_pkg["paper_chunks"]}
        warnings = []
        for cite in cites_in_draft:
            if cite not in valid_keys and not any(cite.lower() in k.lower() for k in valid_keys):
                warnings.append(
                    f"<!-- UNVERIFIED CITE: \\cite{{{cite}}} — confirm against data/papers/ -->"
                )
        if warnings:
            draft += "\n\n" + "\n".join(warnings)
        return draft

    def _paper_header(self) -> str:
        p = self.cfg["paper"]
        return f"""# {p['title']}

**Authors:** {', '.join(p['authors'])}

---
"""


def _agent(dry_run: bool = False, force: bool = False) -> CursorAgent:
    return CursorAgent(dry_run=dry_run, force=force)


@app.command("index")
def cmd_index(
    rebuild: bool = typer.Option(False, "--rebuild", help="Drop and recreate Chroma collections"),
) -> None:
    """Build ChromaDB index from data/papers and data/experiments."""
    os.chdir(PROJECT_ROOT)
    _agent().index_knowledge_base(rebuild=rebuild)


@app.command("write-section")
def cmd_write_section(
    section: str,
    dry_run: bool = typer.Option(False, "--dry-run", help="RAG only; skip LLM and indexing"),
    force: bool = typer.Option(False, "--force", help="Rewrite even if checkpoint exists"),
) -> None:
    """Write a single paper section."""
    os.chdir(PROJECT_ROOT)
    _agent(dry_run=dry_run, force=force).write_section(section)


@app.command("figure")
def cmd_figure(
    figure_type: str,
    section: str = "results",
    query: str = "accuracy comparison methods",
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Generate a figure from experiment metrics."""
    os.chdir(PROJECT_ROOT)
    _agent(dry_run=dry_run).request_figure(figure_type, section, query)


@app.command("assemble")
def cmd_assemble() -> None:
    """Merge section drafts into output/paper_final.md."""
    os.chdir(PROJECT_ROOT)
    _agent().assemble_paper()


@app.command("bib")
def cmd_bib() -> None:
    """Regenerate references.bib from Zotero Exported Items.csv."""
    import subprocess

    os.chdir(PROJECT_ROOT)
    subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "zotero_export_to_bib.py")], check=True)


@app.command("assemble-latex")
def cmd_assemble_latex() -> None:
    """Generate LaTeX tables, bib, and convert section drafts to .tex."""
    os.chdir(PROJECT_ROOT)
    _agent().assemble_latex()


@app.command("draft-sections")
def cmd_draft_sections() -> None:
    """Refresh tables and use data-grounded LaTeX section drafts (no LLM)."""
    os.chdir(PROJECT_ROOT)
    from scripts.draft_sections_from_data import main as draft_main

    draft_main()


@app.command("tables")
def cmd_tables() -> None:
    """Generate PAPER/latex/tables/*.tex from experiment data."""
    os.chdir(PROJECT_ROOT)
    from scripts.tables_to_latex import main as tables_main

    tables_main()


@app.command("write-all")
def cmd_write_all(
    dry_run: bool = typer.Option(False, "--dry-run"),
    force: bool = typer.Option(False, "--force"),
) -> None:
    """Full pipeline: index, all sections, figures, assemble markdown + LaTeX."""
    os.chdir(PROJECT_ROOT)
    _agent(dry_run=dry_run, force=force).write_all()


@app.command("prepare-data")
def cmd_prepare_data() -> None:
    """Seed data/papers and data/experiments from the main project."""
    os.chdir(PROJECT_ROOT)
    from scripts.prepare_data import main as prepare_main

    prepare_main()


@app.command("status")
def cmd_status() -> None:
    """Show output/state.json checkpoint summary."""
    os.chdir(PROJECT_ROOT)
    agent = _agent()
    console.print(f"[bold]State file:[/] {agent.state_mgr.path}")
    for sec, info in agent.state_mgr.data.get("sections", {}).items():
        console.print(f"  {sec}: {info.get('status', '?')}")
    console.print(f"RAG calls logged: {agent.state_mgr.data.get('rag_call_count', 0)}")


def main() -> None:
    os.chdir(PROJECT_ROOT)
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    app()


if __name__ == "__main__":
    main()
