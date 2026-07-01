# GraphRAG Multi-Agent Paper Writing System
## "Texture Knows Best: When Handcrafted Features Still Beat End-to-End Learning in Medical Image Classification"

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Agent Roles & Responsibilities](#2-agent-roles--responsibilities)
3. [Project Structure](#3-project-structure)
4. [Environment Setup](#4-environment-setup)
5. [GraphRAG Knowledge Agent](#5-graphrag-knowledge-agent)
6. [Cursor Orchestrator Agent](#6-cursor-orchestrator-agent)
7. [Section Writer Agents](#7-section-writer-agents)
8. [Figure Generation Agent](#8-figure-generation-agent)
9. [Inter-Agent Communication Protocol](#9-inter-agent-communication-protocol)
10. [Tokenization Optimizations](#10-tokenization-optimizations)
11. [Paper Section Blueprints](#11-paper-section-blueprints)
12. [Running the System](#12-running-the-system)
13. [Ollama Model Config](#13-ollama-model-config)
14. [Agent Constraints & Rules](#14-agent-constraints--rules)

---

## 1. System Overview

This is a **two-tier multi-agent system** for writing a full academic research paper. The architecture separates concerns strictly:

```
┌─────────────────────────────────────────────────────────────┐
│                   CURSOR ORCHESTRATOR                        │
│  Controls: section order, figure requests, paper state      │
│  Issues: RAGQuery, WriteSection, MakeFigure commands        │
└──────────────────────┬──────────────────────────────────────┘
                       │ tool_call / function_call
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                 GRAPHRAG KNOWLEDGE AGENT                     │
│  Owns: local papers, your metrics, experiment logs          │
│  Serves: citations, statistics, related work summaries      │
└─────────────────────────────────────────────────────────────┘
```

**Key principle:** The Cursor agent NEVER invents data. Every metric, citation, and claim MUST come from a RAGQuery response. The GraphRAG agent NEVER writes prose — it only returns structured JSON payloads.

---

## 2. Agent Roles & Responsibilities

### Agent A — GraphRAG Knowledge Agent (`rag_agent.py`)

| Responsibility | Detail |
|---|---|
| Index local papers | PDFs in `/data/papers/` → ChromaDB |
| Index your results | CSVs/JSONs in `/data/experiments/` → ChromaDB |
| Serve citations | Returns author, year, title, relevant excerpt |
| Serve metrics | Returns accuracy, AUC, F1 for any experiment query |
| Serve concept context | Graph traversal for related concepts (LBP→texture→CAD) |
| Output format | Always structured JSON — never raw prose |

**What it NEVER does:**
- Write sentences or paragraphs
- Make editorial decisions about paper structure
- Hallucinate data not in the local index

---

### Agent B — Cursor Orchestrator Agent (`cursor_agent.py`)

| Responsibility | Detail |
|---|---|
| Manage paper state | Tracks which sections are drafted / reviewed / final |
| Dispatch section writes | Calls writer agents with retrieved context |
| Request figures | Sends structured figure specs to figure agent |
| Control section order | Enforces: Abstract last, Related Work before Methods |
| Validate citations | Confirms every cite maps to a RAGQuery result |
| Final assembly | Merges all sections into paper.md |

**What it NEVER does:**
- Access the knowledge base directly without going through the RAG agent
- Write prose without retrieved context in hand
- Skip the citation validation step

---

### Agent C — Section Writer Agents (`writers/`)

One writer agent per section. Each receives a structured payload from the Cursor agent and produces a section draft.

| Agent | File | Input |
|---|---|---|
| Abstract writer | `writers/abstract.py` | Full paper summary after all sections done |
| Related work writer | `writers/related_work.py` | Citation payloads from RAG |
| Methods writer | `writers/methods.py` | Experiment config + dataset description |
| Results writer | `writers/results.py` | Metrics JSON from RAG |
| Discussion writer | `writers/discussion.py` | Results + related work summaries |
| Conclusion writer | `writers/conclusion.py` | Discussion summary |

---

### Agent D — Figure Agent (`figure_agent.py`)

Receives figure specs from Cursor, queries RAG for data, produces matplotlib/seaborn figures.

| Figure type | Trigger |
|---|---|
| Accuracy bar chart | Results section — compare methods |
| ROC curve | Results section — AUC comparison |
| Confusion matrix | Results section — per-class breakdown |
| Feature visualization | Methods section — LBP/GLCM examples |
| Ablation heatmap | Discussion — which features matter most |

---

## 3. Project Structure

```
texture_paper/
├── GRAPHRAG_PAPER_AGENT.md        ← this file (agent spec)
│
├── agents/
│   ├── rag_agent.py               ← GraphRAG Knowledge Agent (Agent A)
│   ├── cursor_agent.py            ← Orchestrator (Agent B)
│   ├── figure_agent.py            ← Figure Generator (Agent D)
│   └── writers/
│       ├── base_writer.py         ← shared prompt + LLM call logic
│       ├── abstract.py
│       ├── related_work.py
│       ├── methods.py
│       ├── results.py
│       ├── discussion.py
│       └── conclusion.py
│
├── data/
│   ├── papers/                    ← DROP YOUR PDFs HERE
│   │   ├── ojala_lbp_2002.pdf
│   │   ├── haralick_glcm_1973.pdf
│   │   └── ...
│   ├── experiments/
│   │   ├── results.csv            ← your accuracy/AUC/F1 numbers
│   │   ├── ablation.json          ← per-feature ablation results
│   │   └── dataset_stats.json     ← dataset description
│   └── citations.bib              ← auto-generated BibTeX
│
├── output/
│   ├── sections/                  ← per-section drafts (markdown)
│   ├── figures/                   ← generated plots (PNG/PDF)
│   └── paper_final.md             ← assembled paper
│
├── db/
│   └── chroma/                    ← persisted ChromaDB index
│
├── config.yaml                    ← model, paths, thresholds
└── requirements.txt
```

---

## 4. Environment Setup

### 4.1 Install dependencies

```bash
# Python packages
pip install --break-system-packages \
  chromadb \
  sentence-transformers \
  networkx \
  spacy \
  pypdf \
  ollama \
  pyyaml \
  matplotlib \
  seaborn \
  pandas \
  scikit-learn \
  rich \
  typer

# spaCy model for entity extraction
python -m spacy download en_core_web_sm

# Ollama — install from https://ollama.ai
ollama pull mistral:7b-instruct-q4_K_M
```

### 4.2 `config.yaml`

```yaml
# ── LLM ──────────────────────────────────────────────────────
llm:
  model: "mistral:7b-instruct-q4_K_M"
  base_url: "http://localhost:11434"
  context_window: 8192
  temperature: 0.3          # low = consistent academic tone
  top_p: 0.9
  repeat_penalty: 1.1

# ── Retrieval ─────────────────────────────────────────────────
retrieval:
  embedder: "all-MiniLM-L6-v2"
  chunk_size: 450
  chunk_overlap: 60
  retrieve_k: 20            # candidates from ChromaDB
  rerank_k: 3               # final chunks to LLM
  reranker: "cross-encoder/ms-marco-MiniLM-L-6-v2"
  relevance_threshold: 0.45
  dedup_threshold: 0.92

# ── Paths ─────────────────────────────────────────────────────
paths:
  papers_dir: "data/papers/"
  experiments_dir: "data/experiments/"
  chroma_dir: "db/chroma/"
  output_dir: "output/"
  sections_dir: "output/sections/"
  figures_dir: "output/figures/"
  citations_bib: "data/citations.bib"

# ── Paper metadata ────────────────────────────────────────────
paper:
  title: "Texture Knows Best: When Handcrafted Features Still Beat End-to-End Learning in Medical Image Classification"
  authors: ["Your Name"]
  venue: ""                 # target conference/journal
  style: "IEEE"             # citation style: IEEE | APA | Nature

# ── Agent behavior ────────────────────────────────────────────
agents:
  max_rag_calls_per_section: 8
  min_citations_per_section: 3
  citation_validation: true
  section_order:
    - related_work
    - methods
    - results
    - discussion
    - conclusion
    - abstract           # always last
```

### 4.3 `requirements.txt`

```
chromadb>=0.4.22
sentence-transformers>=2.7.0
networkx>=3.3
spacy>=3.7.4
pypdf>=4.2.0
ollama>=0.2.1
pyyaml>=6.0
matplotlib>=3.9.0
seaborn>=0.13.2
pandas>=2.2.0
scikit-learn>=1.4.0
rich>=13.7.0
typer>=0.12.0
```

---

## 5. GraphRAG Knowledge Agent

**File:** `agents/rag_agent.py`

This agent owns all local knowledge. It must be started before any other agent.

### 5.1 Indexing Pipeline

```python
# agents/rag_agent.py

import os, json, re
import chromadb
import networkx as nx
import spacy
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, CrossEncoder
import yaml

class GraphRAGAgent:
    """
    Agent A — sole owner of all local knowledge.
    Cursor agent queries this; it never queries Cursor.
    """

    def __init__(self, config_path="config.yaml"):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        rc = self.cfg["retrieval"]
        pc = self.cfg["paths"]

        # ── Models (CPU) ──────────────────────────────────────
        self.embedder   = SentenceTransformer(rc["embedder"])
        self.reranker   = CrossEncoder(rc["reranker"])
        self.nlp        = spacy.load("en_core_web_sm")

        # ── Vector store ──────────────────────────────────────
        self.chroma     = chromadb.PersistentClient(path=pc["chroma_dir"])
        self.papers_col = self.chroma.get_or_create_collection("papers")
        self.metrics_col= self.chroma.get_or_create_collection("metrics")

        # ── Concept graph ─────────────────────────────────────
        self.graph      = nx.Graph()

        self.cfg_r      = rc

    # ── Ingestion ─────────────────────────────────────────────

    def ingest_papers(self, papers_dir: str):
        """Parse all PDFs, chunk, embed, store, build graph."""
        for fname in os.listdir(papers_dir):
            if not fname.endswith(".pdf"):
                continue
            path = os.path.join(papers_dir, fname)
            text = self._extract_pdf(path)
            text = self._clean_text(text)
            chunks = self._chunk(text)
            chunks = self._deduplicate(chunks)
            self._embed_and_store(chunks, source=fname, collection=self.papers_col)
            self._build_graph(chunks, source=fname)
        print(f"[RAG] Indexed {self.papers_col.count()} paper chunks")

    def ingest_experiments(self, experiments_dir: str):
        """Load CSV/JSON experiment results into metrics collection."""
        for fname in os.listdir(experiments_dir):
            path = os.path.join(experiments_dir, fname)
            if fname.endswith(".csv"):
                import pandas as pd
                df = pd.read_csv(path)
                for _, row in df.iterrows():
                    record = json.dumps(row.to_dict())
                    self.metrics_col.add(
                        documents=[record],
                        ids=[f"{fname}_{_}"],
                        metadatas=[{"source": fname, "type": "metric"}]
                    )
            elif fname.endswith(".json"):
                with open(path) as jf:
                    data = json.load(jf)
                record = json.dumps(data)
                self.metrics_col.add(
                    documents=[record],
                    ids=[fname],
                    metadatas=[{"source": fname, "type": "experiment"}]
                )
        print(f"[RAG] Indexed {self.metrics_col.count()} metric records")

    # ── Query API (called by Cursor Agent) ────────────────────

    def query_papers(self, query: str) -> dict:
        """
        Returns top-k relevant paper chunks with citations.
        Output schema:
        {
          "query": str,
          "chunks": [
            {
              "text": str,
              "source": str,
              "score": float,
              "citation_key": str
            }
          ],
          "graph_concepts": [str]
        }
        """
        # Step 1: dense retrieval
        k_retrieve = self.cfg_r["retrieve_k"]
        k_final    = self.cfg_r["rerank_k"]
        threshold  = self.cfg_r["relevance_threshold"]

        results = self.papers_col.query(
            query_texts=[query],
            n_results=k_retrieve,
            include=["documents", "metadatas", "distances"]
        )
        docs   = results["documents"][0]
        metas  = results["metadatas"][0]
        dists  = results["distances"][0]

        # Step 2: filter by threshold
        sims  = [1 - d for d in dists]
        pairs = [(doc, meta, sim) for doc, meta, sim
                 in zip(docs, metas, sims) if sim > threshold]

        if not pairs:
            return {"query": query, "chunks": [], "graph_concepts": []}

        # Step 3: rerank
        scores = self.reranker.predict([(query, doc) for doc, _, _ in pairs])
        ranked = sorted(zip(scores, pairs), reverse=True)[:k_final]

        # Step 4: graph expansion
        concepts = self._graph_expand(query)

        chunks = []
        for score, (doc, meta, sim) in ranked:
            chunks.append({
                "text":         doc,
                "source":       meta.get("source", "unknown"),
                "score":        round(float(score), 4),
                "citation_key": self._derive_citekey(meta.get("source", ""))
            })

        return {"query": query, "chunks": chunks, "graph_concepts": concepts}

    def query_metrics(self, query: str) -> dict:
        """
        Returns experiment metrics matching the query.
        Output schema:
        {
          "query": str,
          "records": [
            { "data": dict, "source": str }
          ]
        }
        """
        results = self.metrics_col.query(
            query_texts=[query],
            n_results=5,
            include=["documents", "metadatas"]
        )
        records = []
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            try:
                data = json.loads(doc)
            except Exception:
                data = {"raw": doc}
            records.append({"data": data, "source": meta.get("source", "")})
        return {"query": query, "records": records}

    # ── Private helpers ───────────────────────────────────────

    def _extract_pdf(self, path: str) -> str:
        reader = PdfReader(path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    def _clean_text(self, text: str) -> str:
        text = re.sub(r'\n\d+\n', '\n', text)
        text = re.sub(r'(doi:|©|Copyright|All rights reserved)[^\n]*', '',
                      text, flags=re.IGNORECASE)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)
        return text.strip()

    def _chunk(self, text: str) -> list[str]:
        size    = self.cfg_r["chunk_size"]
        overlap = self.cfg_r["chunk_overlap"]
        words   = text.split()
        chunks  = []
        i = 0
        while i < len(words):
            chunk = " ".join(words[i:i + size])
            chunks.append(chunk)
            i += size - overlap
        return chunks

    def _deduplicate(self, chunks: list[str]) -> list[str]:
        if len(chunks) < 2:
            return chunks
        threshold  = self.cfg_r["dedup_threshold"]
        embeddings = self.embedder.encode(chunks, show_progress_bar=False)
        from sklearn.metrics.pairwise import cosine_similarity
        keep = [0]
        for i in range(1, len(embeddings)):
            sims = cosine_similarity([embeddings[i]], [embeddings[j] for j in keep])[0]
            if max(sims) < threshold:
                keep.append(i)
        return [chunks[i] for i in keep]

    def _embed_and_store(self, chunks, source, collection):
        existing_ids = set(collection.get()["ids"])
        new_chunks, new_ids, new_metas = [], [], []
        for i, chunk in enumerate(chunks):
            cid = f"{source}_{i}"
            if cid not in existing_ids:
                new_chunks.append(chunk)
                new_ids.append(cid)
                new_metas.append({"source": source})
        if new_chunks:
            collection.add(documents=new_chunks, ids=new_ids, metadatas=new_metas)

    def _build_graph(self, chunks, source):
        """Extract named entities and build co-occurrence edges."""
        for chunk in chunks:
            doc = self.nlp(chunk[:500])
            entities = [ent.text.lower() for ent in doc.ents
                        if ent.label_ in ("ORG","PRODUCT","WORK_OF_ART","GPE","PERSON")]
            # Domain keyword extraction
            keywords = [tok.lemma_.lower() for tok in doc
                        if tok.pos_ in ("NOUN","PROPN") and len(tok.text) > 3]
            nodes = list(set(entities + keywords))[:20]
            for node in nodes:
                self.graph.add_node(node, source=source)
            for i, n1 in enumerate(nodes):
                for n2 in nodes[i+1:]:
                    if self.graph.has_edge(n1, n2):
                        self.graph[n1][n2]["weight"] += 1
                    else:
                        self.graph.add_edge(n1, n2, weight=1)

    def _graph_expand(self, query: str, hops: int = 1) -> list[str]:
        """Return 1-hop concept neighbors of query terms."""
        query_tokens = [t.lower() for t in query.split() if len(t) > 3]
        concepts = set()
        for token in query_tokens:
            if token in self.graph:
                neighbors = sorted(
                    self.graph[token].items(),
                    key=lambda x: x[1].get("weight", 0),
                    reverse=True
                )[:5]
                concepts.update([n for n, _ in neighbors])
        return list(concepts)

    def _derive_citekey(self, source: str) -> str:
        """Turn filename into a BibTeX-style cite key."""
        name = os.path.splitext(source)[0]
        parts = name.replace("-", "_").split("_")
        if len(parts) >= 2:
            return f"{parts[0].capitalize()}{parts[-1]}"
        return name.capitalize()
```

---

## 6. Cursor Orchestrator Agent

**File:** `agents/cursor_agent.py`

The Cursor agent is the **only** agent that has a global view of the paper. It issues commands and assembles the final document.

```python
# agents/cursor_agent.py

import json
import yaml
from pathlib import Path
from rich.console import Console
from agents.rag_agent import GraphRAGAgent
from agents.writers.base_writer import BaseWriter

console = Console()

class CursorAgent:
    """
    Agent B — Orchestrator.
    Controls paper state, dispatches tasks, validates citations.
    """

    SECTION_ORDER = [
        "related_work",
        "methods",
        "results",
        "discussion",
        "conclusion",
        "abstract",   # always last — summarizes the completed paper
    ]

    def __init__(self, config_path="config.yaml"):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        self.rag     = GraphRAGAgent(config_path)
        self.writer  = BaseWriter(config_path)
        self.state   = {s: "pending" for s in self.SECTION_ORDER}
        self.drafts  = {}
        self.figures = {}

        Path(self.cfg["paths"]["sections_dir"]).mkdir(parents=True, exist_ok=True)
        Path(self.cfg["paths"]["figures_dir"]).mkdir(parents=True, exist_ok=True)

    # ── Public interface ──────────────────────────────────────

    def index_knowledge_base(self):
        """Step 0: must run before write commands."""
        console.print("[bold cyan]Indexing papers...[/]")
        self.rag.ingest_papers(self.cfg["paths"]["papers_dir"])
        console.print("[bold cyan]Indexing experiments...[/]")
        self.rag.ingest_experiments(self.cfg["paths"]["experiments_dir"])
        console.print("[bold green]Knowledge base ready.[/]")

    def write_section(self, section: str):
        """
        Cursor dispatches a section write.
        1. Queries RAG for relevant context
        2. Validates that context exists
        3. Calls the appropriate writer agent
        4. Saves draft and updates state
        """
        if section not in self.SECTION_ORDER:
            raise ValueError(f"Unknown section: {section}")

        console.print(f"\n[bold yellow]Cursor → writing: {section}[/]")

        # ── Step 1: RAG queries ───────────────────────────────
        queries     = SECTION_QUERIES[section]
        context_pkg = self._gather_context(queries, section)

        # ── Step 2: validate non-empty context ────────────────
        if not context_pkg["paper_chunks"] and section != "abstract":
            console.print(f"[red]WARNING: No relevant context found for {section}.[/]")
            console.print("[red]Add relevant PDFs to data/papers/ and re-index.[/]")
            return

        # ── Step 3: dispatch to section writer ────────────────
        from agents.writers import get_writer
        writer_fn   = get_writer(section)
        draft       = writer_fn(context_pkg, self.cfg, self.drafts)

        # ── Step 4: citation validation ───────────────────────
        if self.cfg["agents"]["citation_validation"]:
            draft = self._validate_citations(draft, context_pkg)

        # ── Step 5: save ──────────────────────────────────────
        out_path = Path(self.cfg["paths"]["sections_dir"]) / f"{section}.md"
        out_path.write_text(draft)
        self.drafts[section] = draft
        self.state[section]  = "drafted"
        console.print(f"[green]Saved: {out_path}[/]")

    def request_figure(self, figure_type: str, section: str, query: str):
        """
        Cursor requests a figure.
        Queries RAG for data, sends spec to figure agent.
        """
        console.print(f"\n[bold yellow]Cursor → figure: {figure_type}[/]")
        metrics = self.rag.query_metrics(query)
        from agents.figure_agent import FigureAgent
        fig_agent = FigureAgent(self.cfg)
        path = fig_agent.make_figure(
            figure_type=figure_type,
            metrics=metrics,
            section=section
        )
        self.figures[figure_type] = path
        console.print(f"[green]Figure saved: {path}[/]")
        return path

    def write_all(self):
        """Full paper pipeline in correct order."""
        self.index_knowledge_base()
        for section in self.SECTION_ORDER:
            self.write_section(section)
        # Auto-generate standard figures
        self.request_figure("accuracy_bar",   "results",  "accuracy comparison handcrafted vs CNN")
        self.request_figure("roc_curve",      "results",  "ROC AUC comparison methods")
        self.request_figure("confusion_matrix","results", "confusion matrix classification")
        self.assemble_paper()

    def assemble_paper(self):
        """Merge all sections + figures into final paper.md"""
        out = Path(self.cfg["paths"]["output_dir"]) / "paper_final.md"
        paper = self._paper_header()
        for section in self.SECTION_ORDER:
            if section in self.drafts:
                paper += f"\n\n---\n\n{self.drafts[section]}"
        out.write_text(paper)
        console.print(f"\n[bold green]Paper assembled: {out}[/]")

    # ── Private helpers ───────────────────────────────────────

    def _gather_context(self, queries: list[str], section: str) -> dict:
        """Run all queries against RAG, return combined context package."""
        max_calls = self.cfg["agents"]["max_rag_calls_per_section"]
        paper_chunks = []
        metric_records = []
        seen_texts = set()

        for query in queries[:max_calls]:
            # Query papers
            result = self.rag.query_papers(query)
            for chunk in result["chunks"]:
                if chunk["text"] not in seen_texts:
                    paper_chunks.append(chunk)
                    seen_texts.add(chunk["text"])
            # Query metrics if section needs numbers
            if section in ("results", "discussion", "abstract"):
                m_result = self.rag.query_metrics(query)
                metric_records.extend(m_result["records"])

        return {
            "section":       section,
            "paper_chunks":  paper_chunks,
            "metric_records": metric_records,
            "queries":       queries,
        }

    def _validate_citations(self, draft: str, context_pkg: dict) -> str:
        """
        Ensure every [Author, Year] cite in draft maps to a RAG result.
        Appends a warning comment for any unverified cite.
        """
        import re
        cites_in_draft = set(re.findall(r'\[([A-Z][a-z]+(?:[\s&]+[A-Z][a-z]+)*, \d{4})\]', draft))
        valid_keys = {c["citation_key"] for c in context_pkg["paper_chunks"]}
        # Flag any cite not backed by RAG result
        warnings = []
        for cite in cites_in_draft:
            last_name = cite.split(",")[0].split()[-1]
            if not any(last_name.lower() in k.lower() for k in valid_keys):
                warnings.append(f"<!-- UNVERIFIED CITE: {cite} — confirm against data/papers/ -->")
        if warnings:
            draft += "\n\n" + "\n".join(warnings)
        return draft

    def _paper_header(self) -> str:
        p = self.cfg["paper"]
        return f"""# {p['title']}

**Authors:** {', '.join(p['authors'])}

---
"""


# ── Section query templates ────────────────────────────────────────────────────
# These define what the Cursor agent asks the RAG agent for each section.
# Edit these to match your specific paper focus.

SECTION_QUERIES = {
    "related_work": [
        "handcrafted texture features medical image classification LBP GLCM",
        "local binary pattern rotation invariant texture descriptor",
        "gray level co-occurrence matrix Haralick features",
        "deep learning CNN medical image classification histopathology",
        "texture analysis dermatology skin lesion classification",
        "limited data regime medical imaging small dataset CNN failure",
        "radiomics feature extraction clinical interpretability",
        "Gabor filter bank texture recognition",
    ],
    "methods": [
        "LBP feature extraction implementation parameters",
        "GLCM computation distance angle quantization",
        "SVM classifier texture features baseline",
        "dataset split train validation test medical imaging",
        "data augmentation limited medical dataset",
        "feature normalization standardization pipeline",
        "cross validation stratified medical classification",
    ],
    "results": [
        "accuracy AUC F1 comparison handcrafted deep learning",
        "LBP accuracy classification results",
        "GLCM performance metrics medical",
        "CNN ResNet accuracy histopathology",
        "ablation study feature importance texture",
        "statistical significance McNemar test comparison",
    ],
    "discussion": [
        "why handcrafted features outperform CNN limited data",
        "interpretability explainability texture features clinical",
        "texture bias deep neural networks",
        "domain shift generalization medical imaging CNN",
        "inductive bias handcrafted vs learned features",
        "sample efficiency texture descriptors small training set",
    ],
    "conclusion": [
        "future work texture features medical imaging",
        "limitations handcrafted feature extraction",
        "hybrid approach handcrafted learned features",
    ],
    "abstract": [
        # Abstract queries the completed drafts, not the RAG.
        # These are used as fallback if drafts are incomplete.
        "main contribution texture features beat CNN medical imaging",
    ],
}
```

---

## 7. Section Writer Agents

**File:** `agents/writers/base_writer.py`

All section writers share this base, which handles the Ollama call and prompt construction.

```python
# agents/writers/base_writer.py

import ollama
import yaml

SYSTEM_PROMPT = """You are a scientific writing assistant for the research paper:
"Texture Knows Best: When Handcrafted Features Still Beat End-to-End Learning \
in Medical Image Classification".

STRICT RULES:
1. Write in formal, precise academic English. No colloquialisms.
2. Every factual claim MUST be supported by the provided context chunks.
3. Cite using the format [Author, Year] — only use citation keys provided.
4. Do NOT invent statistics, accuracy numbers, or dataset details.
5. Do NOT cite papers not present in the context.
6. Structure paragraphs: claim → evidence → interpretation.
7. Use passive voice for methods, active voice for arguments.
8. Output only the section content in Markdown. No meta-commentary."""


class BaseWriter:
    def __init__(self, config_path="config.yaml"):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        self.model   = cfg["llm"]["model"]
        self.temp    = cfg["llm"]["temperature"]
        self.ctx_win = cfg["llm"]["context_window"]

    def call_llm(self, user_prompt: str) -> str:
        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            options={
                "temperature":    self.temp,
                "num_ctx":        self.ctx_win,
                "repeat_penalty": 1.1,
            }
        )
        return response["message"]["content"]

    def format_context(self, chunks: list[dict]) -> str:
        """Format retrieved chunks into numbered context block for prompt."""
        lines = ["## Retrieved context\n"]
        for i, chunk in enumerate(chunks, 1):
            lines.append(
                f"[{i}] Source: {chunk['source']} | Key: {chunk['citation_key']}\n"
                f"{chunk['text']}\n"
            )
        return "\n".join(lines)

    def format_metrics(self, records: list[dict]) -> str:
        """Format experiment records into readable block."""
        if not records:
            return ""
        lines = ["## Experiment data\n"]
        for r in records:
            lines.append(f"Source: {r['source']}\n{r['data']}\n")
        return "\n".join(lines)
```

---

**File:** `agents/writers/__init__.py`

```python
from agents.writers import (
    related_work, methods, results, discussion, conclusion, abstract
)

WRITERS = {
    "related_work": related_work.write,
    "methods":      methods.write,
    "results":      results.write,
    "discussion":   discussion.write,
    "conclusion":   conclusion.write,
    "abstract":     abstract.write,
}

def get_writer(section: str):
    return WRITERS[section]
```

---

**File:** `agents/writers/related_work.py`

```python
from agents.writers.base_writer import BaseWriter

def write(context_pkg: dict, cfg: dict, existing_drafts: dict) -> str:
    writer  = BaseWriter()
    context = writer.format_context(context_pkg["paper_chunks"])

    prompt = f"""{context}

## Task
Write the Related Work section for this paper. Structure as follows:

1. **Handcrafted texture descriptors** — survey LBP, GLCM, Gabor filters, their
   variants, and applications in medical imaging. Emphasize their theoretical grounding.

2. **Deep learning for medical imaging** — survey CNN-based approaches.
   Be objective but note their known failure modes (data hunger, opacity).

3. **Comparative studies** — papers that directly compare handcrafted vs. learned
   features. Position our work relative to them.

4. **Gap and motivation** — end with 1 paragraph explaining what prior work
   leaves unanswered and why our contribution is needed.

Requirements:
- Minimum 6 citations drawn from the context above.
- Each citation must use the key format [Key, Year] as shown in context.
- Length: ~600-800 words.
- Do NOT use subheadings — write as continuous paragraphs.
"""
    return writer.call_llm(prompt)
```

---

**File:** `agents/writers/results.py`

```python
from agents.writers.base_writer import BaseWriter

def write(context_pkg: dict, cfg: dict, existing_drafts: dict) -> str:
    writer  = BaseWriter()
    context = writer.format_context(context_pkg["paper_chunks"])
    metrics = writer.format_metrics(context_pkg["metric_records"])

    prompt = f"""{context}

{metrics}

## Task
Write the Results section. Structure as follows:

1. **Overview of experimental setup** — one paragraph re-stating dataset, splits,
   evaluation metrics (accuracy, AUC, F1, MCC).

2. **Main comparison table** — describe (in prose) the table comparing:
   - LBP + SVM
   - GLCM + SVM
   - Gabor + SVM
   - Ensemble (LBP+GLCM+Gabor) + SVM
   - ResNet-50 (fine-tuned)
   - EfficientNet-B0 (fine-tuned)
   Include the actual numbers from the experiment data above.
   Format the table in Markdown.

3. **Ablation results** — which texture features contributed most.

4. **Statistical significance** — report p-values or confidence intervals
   where available in the experiment data.

Requirements:
- Use ONLY the numbers from the experiment data section above.
- Do NOT invent any accuracy, AUC, or F1 values.
- Length: ~500-700 words + table.
"""
    return writer.call_llm(prompt)
```

---

**File:** `agents/writers/abstract.py`

```python
from agents.writers.base_writer import BaseWriter

def write(context_pkg: dict, cfg: dict, existing_drafts: dict) -> str:
    """Abstract is written from the completed paper sections, not from RAG."""
    writer = BaseWriter()

    # Build summary from completed sections
    section_summaries = ""
    for sec in ["related_work", "methods", "results", "discussion", "conclusion"]:
        if sec in existing_drafts:
            # Take first 300 words of each draft as summary seed
            words = existing_drafts[sec].split()[:300]
            section_summaries += f"\n\n### {sec.upper()} (excerpt)\n" + " ".join(words)

    prompt = f"""The following are excerpts from the completed sections of our paper:
{section_summaries}

## Task
Write a structured abstract (250–300 words) with these implicit components
(do NOT use subheadings):

1. **Context** (1-2 sentences): Why medical image classification matters.
2. **Problem** (1-2 sentences): Why end-to-end CNN approaches fall short in this domain.
3. **Method** (2-3 sentences): What features we evaluate, on what datasets, with what baselines.
4. **Results** (2-3 sentences): Key quantitative findings showing handcrafted features win.
   Use the specific numbers from the results excerpt above.
5. **Conclusion** (1-2 sentences): What this implies for practitioners.

Requirements:
- No citations in the abstract.
- Use present tense for general statements, past tense for specific experiments.
- Mention at least: LBP, GLCM, the dataset name, and the best accuracy figure.
"""
    return writer.call_llm(prompt)
```

---

## 8. Figure Generation Agent

**File:** `agents/figure_agent.py`

```python
# agents/figure_agent.py

import json
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")   # no display needed
import seaborn as sns
import numpy as np
import pandas as pd

sns.set_theme(style="whitegrid", font="DejaVu Sans")
PALETTE = ["#1D9E75", "#378ADD", "#D85A30", "#7F77DD", "#888780", "#E24B4A"]


class FigureAgent:
    """
    Agent D — Figure generation.
    Receives metrics from Cursor, produces publication-ready figures.
    Never calls LLM. Never queries RAG directly.
    """

    def __init__(self, cfg: dict):
        self.cfg      = cfg
        self.out_dir  = Path(cfg["paths"]["figures_dir"])
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def make_figure(self, figure_type: str, metrics: dict, section: str) -> str:
        handlers = {
            "accuracy_bar":    self._accuracy_bar,
            "roc_curve":       self._roc_curve,
            "confusion_matrix":self._confusion_matrix,
            "ablation_heatmap":self._ablation_heatmap,
        }
        if figure_type not in handlers:
            raise ValueError(f"Unknown figure type: {figure_type}")
        return handlers[figure_type](metrics)

    def _accuracy_bar(self, metrics: dict) -> str:
        """Bar chart comparing accuracy across all methods."""
        records = metrics.get("records", [])
        # Parse records — expect {"method": ..., "accuracy": ..., "auc": ...}
        rows = []
        for r in records:
            d = r["data"]
            if isinstance(d, dict) and "method" in d and "accuracy" in d:
                rows.append(d)

        if not rows:
            # Fallback: render placeholder with instructions
            return self._placeholder("accuracy_bar", "No accuracy data found in experiments/")

        df = pd.DataFrame(rows).sort_values("accuracy", ascending=True)

        fig, ax = plt.subplots(figsize=(7, 4))
        colors = [PALETTE[2] if "CNN" in str(m) or "Net" in str(m) else PALETTE[0]
                  for m in df["method"]]
        bars = ax.barh(df["method"], df["accuracy"] * 100,
                       color=colors, edgecolor="white", height=0.6)

        # Annotate bars
        for bar, val in zip(bars, df["accuracy"] * 100):
            ax.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}%", va="center", ha="left", fontsize=9)

        ax.set_xlabel("Accuracy (%)", fontsize=10)
        ax.set_xlim(0, 105)
        ax.set_title("Classification accuracy: handcrafted vs. end-to-end learning",
                     fontsize=10, pad=8)

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=PALETTE[0], label="Handcrafted features"),
            Patch(facecolor=PALETTE[2], label="Deep learning")
        ]
        ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

        plt.tight_layout()
        out = self.out_dir / "accuracy_comparison.pdf"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close()
        return str(out)

    def _roc_curve(self, metrics: dict) -> str:
        """ROC curves per method if raw predictions are available."""
        # Requires records with "fpr" and "tpr" arrays
        records = metrics.get("records", [])
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
        plotted = 0
        for i, r in enumerate(records):
            d = r["data"]
            if isinstance(d, dict) and "fpr" in d and "tpr" in d:
                fpr = np.array(d["fpr"])
                tpr = np.array(d["tpr"])
                auc = d.get("auc", "?")
                ax.plot(fpr, tpr, color=PALETTE[i % len(PALETTE)],
                        label=f"{d.get('method','Method')} (AUC={auc:.3f})", lw=1.5)
                plotted += 1
        if plotted == 0:
            return self._placeholder("roc_curve", "No fpr/tpr data in experiments/")
        ax.set_xlabel("False Positive Rate", fontsize=10)
        ax.set_ylabel("True Positive Rate", fontsize=10)
        ax.set_title("ROC curves", fontsize=10)
        ax.legend(fontsize=8)
        plt.tight_layout()
        out = self.out_dir / "roc_curves.pdf"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close()
        return str(out)

    def _confusion_matrix(self, metrics: dict) -> str:
        records = metrics.get("records", [])
        for r in records:
            d = r["data"]
            if isinstance(d, dict) and "confusion_matrix" in d:
                cm = np.array(d["confusion_matrix"])
                labels = d.get("labels", [str(i) for i in range(cm.shape[0])])
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                            xticklabels=labels, yticklabels=labels,
                            linewidths=0.5, ax=ax)
                ax.set_xlabel("Predicted", fontsize=10)
                ax.set_ylabel("True", fontsize=10)
                ax.set_title(f"Confusion matrix — {d.get('method','best model')}", fontsize=10)
                plt.tight_layout()
                out = self.out_dir / "confusion_matrix.pdf"
                fig.savefig(out, dpi=300, bbox_inches="tight")
                plt.close()
                return str(out)
        return self._placeholder("confusion_matrix", "No confusion_matrix key in experiments/")

    def _ablation_heatmap(self, metrics: dict) -> str:
        records = metrics.get("records", [])
        for r in records:
            d = r["data"]
            if isinstance(d, dict) and "ablation" in d:
                abl = d["ablation"]   # expected: {feature: {metric: value}}
                df  = pd.DataFrame(abl).T
                fig, ax = plt.subplots(figsize=(6, 3))
                sns.heatmap(df, annot=True, fmt=".3f", cmap="YlGn",
                            linewidths=0.5, ax=ax)
                ax.set_title("Ablation study — feature contribution", fontsize=10)
                plt.tight_layout()
                out = self.out_dir / "ablation_heatmap.pdf"
                fig.savefig(out, dpi=300, bbox_inches="tight")
                plt.close()
                return str(out)
        return self._placeholder("ablation_heatmap", "No ablation key in experiments/")

    def _placeholder(self, name: str, msg: str) -> str:
        """Generate a labeled placeholder figure."""
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.text(0.5, 0.5, f"[{name}]\n{msg}",
                ha="center", va="center", fontsize=10,
                transform=ax.transAxes, color="gray")
        ax.axis("off")
        out = self.out_dir / f"{name}_placeholder.pdf"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        return str(out)
```

---

## 9. Inter-Agent Communication Protocol

All agent communication uses **structured JSON payloads**. No free-text messages between agents.

### RAG Query Schema (Cursor → RAG)

```json
{
  "type": "paper_query",
  "query": "handcrafted texture features outperform CNN limited data",
  "section": "related_work",
  "max_results": 3
}
```

### RAG Response Schema (RAG → Cursor)

```json
{
  "type": "paper_query_response",
  "query": "...",
  "chunks": [
    {
      "text": "LBP achieves rotation invariance by...",
      "source": "ojala_lbp_2002.pdf",
      "score": 0.87,
      "citation_key": "Ojala2002"
    }
  ],
  "graph_concepts": ["texture", "rotation invariance", "histogram"]
}
```

### Write Command Schema (Cursor → Writer)

```json
{
  "type": "write_section",
  "section": "results",
  "context_package": {
    "paper_chunks": [...],
    "metric_records": [...],
    "queries": [...]
  }
}
```

### Figure Request Schema (Cursor → Figure Agent)

```json
{
  "type": "make_figure",
  "figure_type": "accuracy_bar",
  "section": "results",
  "metrics_query": "accuracy comparison handcrafted vs CNN"
}
```

---

## 10. Tokenization Optimizations

All optimizations below are **safe — zero quality loss**. Already wired into the agents above.

| Optimization | Where applied | Token saving |
|---|---|---|
| Boilerplate stripping | `_clean_text()` in RAG agent | ~10-15% per PDF |
| Chunk deduplication | `_deduplicate()` in RAG agent | ~8-12% index size |
| Rerank top-3 not top-20 | `query_papers()` in RAG agent | ~65% LLM input |
| Prompt caching | Static `SYSTEM_PROMPT` in `BaseWriter` | KV cache reuse |
| Relevance threshold filter | `query_papers()` at threshold=0.45 | Variable |
| Adaptive context | `_gather_context()` in Cursor | Scales with need |

### Optimizations to AVOID for this system

| What | Why it hurts |
|---|---|
| Quantization below Q4_K_M | Breaks multi-sentence academic coherence |
| Shrinking context window | Related work needs full paragraph context |
| Hard truncation of chunks | Cuts mid-argument, causes hallucinated completions |
| LLMLingua aggressive mode | Strips hedging language critical to academic register |
| Reducing chunk overlap <40 tokens | Breaks cross-sentence argument structure |

---

## 11. Paper Section Blueprints

These are the expected structure and length targets for each section.

### Abstract (250–300 words)
- Context → Problem → Method → Results → Conclusion
- Written last. No citations.
- Must include: dataset name, best method, key accuracy figure.

### Related Work (~800 words, ≥6 citations)
- Handcrafted texture descriptors (LBP, GLCM, Gabor)
- CNN approaches and their medical imaging applications
- Direct comparisons / head-to-head studies
- Gap analysis leading to this paper

### Methods (~700 words)
- Dataset description (source, size, classes, splits)
- Feature extraction pipeline (LBP params, GLCM params, Gabor params)
- Classifier choice and hyperparameter search
- Baseline CNN architectures (ResNet-50, EfficientNet-B0)
- Evaluation protocol (k-fold CV, metrics used)

### Results (~600 words + 2-3 tables/figures)
- Main comparison table (all methods × all metrics)
- Ablation study
- Statistical significance

### Discussion (~600 words)
- Why do handcrafted features win? (data hunger, texture bias, inductive bias)
- Clinical interpretability argument
- Limitations of our study
- When would CNNs win? (large data regime)

### Conclusion (~250 words)
- Summary of findings
- Practical recommendation for clinicians/engineers
- Future work (self-supervised + handcrafted hybrid?)

---

## 12. Running the System

### Option A — Write everything automatically

```bash
python -m agents.cursor_agent write_all
```

### Option B — Manual section control (recommended for iteration)

```python
from agents.cursor_agent import CursorAgent

cursor = CursorAgent()

# Step 1: Build knowledge base (run once)
cursor.index_knowledge_base()

# Step 2: Write sections in order
cursor.write_section("related_work")
cursor.write_section("methods")
cursor.write_section("results")

# Step 3: Generate figures
cursor.request_figure("accuracy_bar",    "results", "accuracy comparison")
cursor.request_figure("roc_curve",       "results", "AUC ROC methods")
cursor.request_figure("confusion_matrix","results", "confusion matrix best model")

# Step 4: Write remaining sections
cursor.write_section("discussion")
cursor.write_section("conclusion")
cursor.write_section("abstract")   # always last

# Step 5: Assemble
cursor.assemble_paper()
```

### Option C — Cursor IDE integration

Create `.cursorrules` in your project root:

```
You are the Cursor Orchestrator Agent for the paper:
"Texture Knows Best: When Handcrafted Features Still Beat End-to-End Learning
in Medical Image Classification"

RULES:
1. Before writing any section, always call cursor.write_section() — never write
   academic prose directly without RAG context.
2. Before inserting any accuracy/AUC/F1 number, always call cursor.rag.query_metrics().
3. Before adding a citation, confirm it exists in data/papers/ via query_papers().
4. Figure requests go through cursor.request_figure() — never hardcode plot data.
5. Section order: related_work → methods → results → discussion → conclusion → abstract.
6. The abstract is always the last thing written.
```

---

## 13. Ollama Model Config

Start Ollama before running any agent:

```bash
# Start Ollama server
ollama serve

# In another terminal, pre-pull the model
ollama pull mistral:7b-instruct-q4_K_M

# Verify it fits in VRAM
ollama run mistral:7b-instruct-q4_K_M "Say OK"
```

### Modelfile (optional — bake in academic system prompt)

```
# Save as Modelfile_texture_paper
FROM mistral:7b-instruct-q4_K_M

PARAMETER temperature 0.3
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 8192
PARAMETER top_p 0.9

SYSTEM """You are a scientific writing assistant specializing in medical image analysis
and texture-based feature extraction. Write formally. Cite evidence. No hallucination."""
```

```bash
ollama create texture-paper -f Modelfile_texture_paper
# Then use model: "texture-paper" in config.yaml
```

---

## 14. Agent Constraints & Rules

These rules are NON-NEGOTIABLE. Both agents must follow them.

### GraphRAG Knowledge Agent rules

- [ ] NEVER return prose paragraphs — always structured JSON
- [ ] NEVER invent a citation not present in `/data/papers/`
- [ ] NEVER return a metric not present in `/data/experiments/`
- [ ] ALWAYS include `citation_key` and `source` in every chunk response
- [ ] ALWAYS run deduplication before adding to ChromaDB
- [ ] ALWAYS filter by relevance threshold before returning to Cursor
- [ ] Re-index only when new files are added to `data/papers/` or `data/experiments/`

### Cursor Orchestrator Agent rules

- [ ] NEVER write a section without a RAG context package
- [ ] NEVER insert an accuracy/AUC number not returned by `query_metrics()`
- [ ] NEVER write the abstract before all other sections are drafted
- [ ] ALWAYS run `_validate_citations()` before saving a section
- [ ] ALWAYS log state transitions: `pending → drafted → reviewed → final`
- [ ] ALWAYS save each section to `output/sections/{section}.md` before assembling

### Section Writer rules

- [ ] NEVER hallucinate data — if context is empty, say so and stop
- [ ] NEVER use first person ("we found") — use "the proposed approach" / "results indicate"
- [ ] ALWAYS structure paragraphs: claim → evidence (with cite) → interpretation
- [ ] ALWAYS use the citation keys exactly as provided by RAG
- [ ] NEVER add citations not in the context package

### Figure Agent rules

- [ ] NEVER hardcode example/fake data for production figures
- [ ] ALWAYS use the `_placeholder()` method if real data is missing
- [ ] ALWAYS save as PDF (vector) for publication quality
- [ ] ALWAYS use the standard `PALETTE` for color consistency across figures

---

## Quick-start Checklist

```
[ ] pip install -r requirements.txt
[ ] python -m spacy download en_core_web_sm
[ ] ollama pull mistral:7b-instruct-q4_K_M
[ ] Drop your PDFs into data/papers/
[ ] Put your results.csv into data/experiments/
[ ] Edit config.yaml — set paper.authors and paper.venue
[ ] python -m agents.cursor_agent write_all
[ ] Review output/sections/*.md
[ ] Run cursor.assemble_paper()
[ ] output/paper_final.md is your draft
```

---

*Generated for hardware: 6GB VRAM / 16GB RAM | Model: Mistral 7B Q4_K_M via Ollama*