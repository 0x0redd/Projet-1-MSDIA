"""Agent A — GraphRAG Knowledge Agent."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import chromadb
import networkx as nx
import spacy
import yaml
from sentence_transformers import CrossEncoder, SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from agents.citation_keys import make_cite_key
from agents.paths import resolve_config, resolve_path
from agents.pdf_extract import extract_pdf
from agents.query_cache import QueryCache
from agents.rag_logging import RAGCallLogger


class GraphRAGAgent:
    """Sole owner of all local knowledge. Returns structured JSON only."""

    def __init__(self, config_path: str | Path | None = None):
        cfg_path = resolve_config(config_path)
        with open(cfg_path, encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        rc = self.cfg["retrieval"]
        pc = self.cfg["paths"]
        runtime = self.cfg.get("runtime", {})

        self.embedder = SentenceTransformer(rc["embedder"])
        self.reranker = CrossEncoder(rc["reranker"])
        self.nlp = spacy.load("en_core_web_sm")

        chroma_dir = str(resolve_path(pc["chroma_dir"]))
        Path(chroma_dir).mkdir(parents=True, exist_ok=True)
        self.chroma = chromadb.PersistentClient(path=chroma_dir)
        self.papers_col = self.chroma.get_or_create_collection(
            "papers", metadata={"hnsw:space": "cosine"}
        )
        self.metrics_col = self.chroma.get_or_create_collection(
            "metrics", metadata={"hnsw:space": "cosine"}
        )

        self.graph = nx.Graph()
        self.cfg_r = rc
        self._doc_meta: dict[str, dict] = {}

        cache_cfg = self.cfg.get("cache", {})
        cache_dir = resolve_path(cache_cfg.get("query_cache_dir", "db/query_cache"))
        self.query_cache = QueryCache(
            cache_dir,
            enabled=cache_cfg.get("query_cache_enabled", True),
        )

        log_cfg = self.cfg.get("logging", {})
        log_dir = resolve_path(log_cfg.get("rag_log_dir", "output/logs"))
        self.rag_logger = RAGCallLogger(log_dir, log_cfg.get("level", "INFO"))

        self.dry_run = bool(runtime.get("dry_run", False))

    def rebuild_collections(self) -> None:
        """Drop and recreate Chroma collections (required after embedding metric fix)."""
        for name in ("papers", "metrics"):
            try:
                self.chroma.delete_collection(name)
                print(f"[RAG] Dropped collection: {name}")
            except Exception:
                pass
        self.papers_col = self.chroma.get_or_create_collection(
            "papers", metadata={"hnsw:space": "cosine"}
        )
        self.metrics_col = self.chroma.get_or_create_collection(
            "metrics", metadata={"hnsw:space": "cosine"}
        )
        self.graph = nx.Graph()
        self._doc_meta = {}

    def ingest_papers(self, papers_dir: str | None = None) -> None:
        papers_dir = str(resolve_path(papers_dir or self.cfg["paths"]["papers_dir"]))
        if not os.path.isdir(papers_dir):
            print(f"[RAG] Papers dir missing: {papers_dir}")
            return

        for fname in os.listdir(papers_dir):
            if not fname.lower().endswith(".pdf"):
                continue
            path = os.path.join(papers_dir, fname)
            text, bib = extract_pdf(path)
            if not text or not text.strip():
                print(f"[RAG] Skipped (unreadable or empty): {fname}")
                continue

            self._doc_meta[fname] = bib
            text = self._clean_text(text)
            chunks = self._chunk(text)
            chunks = self._deduplicate(chunks)
            self._embed_and_store(chunks, source=fname, collection=self.papers_col, bib=bib)
            self._build_graph(chunks, source=fname)

        meta_path = resolve_path(self.cfg.get("cache", {}).get("doc_meta_path", "db/doc_meta.json"))
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(self._doc_meta, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[RAG] Indexed {self.papers_col.count()} paper chunks ({len(self._doc_meta)} PDFs)")

    def ingest_experiments(self, experiments_dir: str | None = None) -> None:
        experiments_dir = str(resolve_path(experiments_dir or self.cfg["paths"]["experiments_dir"]))
        if not os.path.isdir(experiments_dir):
            print(f"[RAG] Experiments dir missing: {experiments_dir}")
            return

        import pandas as pd

        existing_ids = set(self.metrics_col.get()["ids"])
        for fname in os.listdir(experiments_dir):
            path = os.path.join(experiments_dir, fname)
            if fname.endswith(".csv"):
                df = pd.read_csv(path)
                for idx, row in df.iterrows():
                    record = json.dumps(row.to_dict())
                    rid = f"{fname}_{idx}"
                    if rid not in existing_ids:
                        emb = self.embedder.encode(record, show_progress_bar=False).tolist()
                        self.metrics_col.add(
                            documents=[record],
                            ids=[rid],
                            metadatas=[{"source": fname, "type": "metric"}],
                            embeddings=[emb],
                        )
            elif fname.endswith(".json"):
                with open(path, encoding="utf-8") as jf:
                    data = json.load(jf)
                record = json.dumps(data)
                if fname not in existing_ids:
                    emb = self.embedder.encode(record, show_progress_bar=False).tolist()
                    self.metrics_col.add(
                        documents=[record],
                        ids=[fname],
                        metadatas=[{"source": fname, "type": "experiment"}],
                        embeddings=[emb],
                    )

        print(f"[RAG] Indexed {self.metrics_col.count()} metric records")

    def query_papers(self, query: str) -> dict:
        cached = self.query_cache.get(query, "papers")
        if cached is not None:
            self.rag_logger.log_call(
                kind="papers",
                query=query,
                cache_hit=True,
                n_candidates=cached.get("_n_candidates", 0),
                n_returned=len(cached.get("chunks", [])),
                elapsed_ms=0.0,
            )
            return {k: v for k, v in cached.items() if not k.startswith("_")}

        timer = self.rag_logger.Timer()
        k_retrieve = self.cfg_r["retrieve_k"]
        k_final = self.cfg_r["rerank_k"]
        min_rerank = float(self.cfg_r.get("min_rerank_score", -4.0))

        if self.papers_col.count() == 0:
            result = {"query": query, "chunks": [], "graph_concepts": []}
            self._finish_paper_query(query, result, timer, 0)
            return result

        q_emb = self.embedder.encode(query, show_progress_bar=False).tolist()
        results = self.papers_col.query(
            query_embeddings=[q_emb],
            n_results=min(k_retrieve, self.papers_col.count()),
            include=["documents", "metadatas", "distances"],
        )
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0]

        pairs = [
            (doc, meta, self._distance_to_similarity(d))
            for doc, meta, d in zip(docs, metas, dists)
        ]
        n_candidates = len(pairs)

        if not pairs:
            result = {"query": query, "chunks": [], "graph_concepts": []}
            self._finish_paper_query(query, result, timer, n_candidates)
            return result

        scores = self.reranker.predict([(query, doc) for doc, _, _ in pairs])
        ranked = sorted(zip(scores, pairs), key=lambda item: float(item[0]), reverse=True)
        ranked = [(s, p) for s, p in ranked if float(s) >= min_rerank][:k_final]
        concepts = self._graph_expand(query)

        chunks = []
        for score, (doc, meta, sim) in ranked:
            source = meta.get("source", "unknown")
            cite_key = meta.get("citation_key") or self._derive_citekey(source)
            chunks.append({
                "text": doc,
                "source": source,
                "score": round(float(score), 4),
                "vector_sim": round(float(sim), 4),
                "citation_key": cite_key,
                "title": meta.get("title", ""),
                "authors": meta.get("authors", ""),
                "year": meta.get("year", ""),
                "doi": meta.get("doi", ""),
                "chunk_type": meta.get("chunk_type", "text"),
            })

        result = {"query": query, "chunks": chunks, "graph_concepts": concepts}
        self._finish_paper_query(query, result, timer, n_candidates)
        return result

    def query_metrics(self, query: str) -> dict:
        cached = self.query_cache.get(query, "metrics")
        if cached is not None:
            self.rag_logger.log_call(
                kind="metrics",
                query=query,
                cache_hit=True,
                n_candidates=cached.get("_n_candidates", 0),
                n_returned=len(cached.get("records", [])),
                elapsed_ms=0.0,
            )
            return {k: v for k, v in cached.items() if not k.startswith("_")}

        timer = self.rag_logger.Timer()
        if self.metrics_col.count() == 0:
            result = {"query": query, "records": []}
            self._finish_metrics_query(query, result, timer, 0)
            return result

        results = self.metrics_col.query(
            query_embeddings=[self.embedder.encode(query, show_progress_bar=False).tolist()],
            n_results=min(5, self.metrics_col.count()),
            include=["documents", "metadatas"],
        )
        records = []
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            try:
                data = json.loads(doc)
            except json.JSONDecodeError:
                data = {"raw": doc}
            records.append({"data": data, "source": meta.get("source", "")})

        result = {"query": query, "records": records}
        self._finish_metrics_query(query, result, timer, len(records))
        return result

    def _finish_paper_query(self, query: str, result: dict, timer, n_candidates: int) -> None:
        payload = {**result, "_n_candidates": n_candidates}
        self.query_cache.set(query, "papers", payload)
        self.rag_logger.log_call(
            kind="papers",
            query=query,
            cache_hit=False,
            n_candidates=n_candidates,
            n_returned=len(result.get("chunks", [])),
            elapsed_ms=timer.elapsed_ms(),
            extra={"graph_concepts": len(result.get("graph_concepts", []))},
        )

    def _finish_metrics_query(self, query: str, result: dict, timer, n_candidates: int) -> None:
        payload = {**result, "_n_candidates": n_candidates}
        self.query_cache.set(query, "metrics", payload)
        self.rag_logger.log_call(
            kind="metrics",
            query=query,
            cache_hit=False,
            n_candidates=n_candidates,
            n_returned=len(result.get("records", [])),
            elapsed_ms=timer.elapsed_ms(),
        )

    @staticmethod
    def _distance_to_similarity(dist: float) -> float:
        if dist <= 0:
            return 1.0
        if dist <= 1.0:
            return max(0.0, 1.0 - dist)
        return max(0.0, 1.0 - (dist**2) / 2.0)

    def _clean_text(self, text: str) -> str:
        text = re.sub(r"\n\d+\n", "\n", text)
        text = re.sub(
            r"(doi:|©|Copyright|All rights reserved)[^\n]*",
            "",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
        return text.strip()

    def _chunk(self, text: str) -> list[str]:
        size = self.cfg_r["chunk_size"]
        overlap = self.cfg_r["chunk_overlap"]
        words = text.split()
        chunks = []
        i = 0
        while i < len(words):
            chunk = " ".join(words[i : i + size])
            chunks.append(chunk)
            i += size - overlap
        return chunks

    def _deduplicate(self, chunks: list[str]) -> list[str]:
        if len(chunks) < 2:
            return chunks
        threshold = self.cfg_r["dedup_threshold"]
        embeddings = self.embedder.encode(chunks, show_progress_bar=False)
        keep = [0]
        for i in range(1, len(embeddings)):
            sims = cosine_similarity([embeddings[i]], [embeddings[j] for j in keep])[0]
            if max(sims) < threshold:
                keep.append(i)
        return [chunks[i] for i in keep]

    def _embed_and_store(self, chunks, source, collection, bib: dict | None = None) -> None:
        bib = bib or {}
        existing_ids = set(collection.get()["ids"])
        new_chunks, new_ids, new_metas = [], [], []
        cite_key = bib.get("citation_key") or make_cite_key(source)

        for i, chunk in enumerate(chunks):
            cid = f"{source}_{i}"
            if cid not in existing_ids:
                new_chunks.append(chunk)
                new_ids.append(cid)
                new_metas.append({
                    "source": source,
                    "citation_key": cite_key,
                    "title": bib.get("title", "") or "",
                    "authors": bib.get("authors", "") or "",
                    "year": str(bib.get("year", "") or ""),
                    "doi": bib.get("doi", "") or "",
                    "chunk_type": "table" if chunk.startswith("[TABLE]") else "text",
                })
        if new_chunks:
            embeddings = self.embedder.encode(new_chunks, show_progress_bar=False).tolist()
            collection.add(
                documents=new_chunks,
                ids=new_ids,
                metadatas=new_metas,
                embeddings=embeddings,
            )

    def _build_graph(self, chunks, source) -> None:
        for chunk in chunks:
            doc = self.nlp(chunk[:500])
            entities = [
                ent.text.lower()
                for ent in doc.ents
                if ent.label_ in ("ORG", "PRODUCT", "WORK_OF_ART", "GPE", "PERSON")
            ]
            keywords = [
                tok.lemma_.lower()
                for tok in doc
                if tok.pos_ in ("NOUN", "PROPN") and len(tok.text) > 3
            ]
            nodes = list(set(entities + keywords))[:20]
            for node in nodes:
                self.graph.add_node(node, source=source)
            for i, n1 in enumerate(nodes):
                for n2 in nodes[i + 1 :]:
                    if self.graph.has_edge(n1, n2):
                        self.graph[n1][n2]["weight"] += 1
                    else:
                        self.graph.add_edge(n1, n2, weight=1)

    def _graph_expand(self, query: str, hops: int = 1) -> list[str]:
        query_tokens = [t.lower() for t in query.split() if len(t) > 3]
        concepts: set[str] = set()
        for token in query_tokens:
            if token in self.graph:
                neighbors = sorted(
                    self.graph[token].items(),
                    key=lambda x: x[1].get("weight", 0),
                    reverse=True,
                )[:5]
                concepts.update(n for n, _ in neighbors)
        return list(concepts)

    def _derive_citekey(self, source: str) -> str:
        if source in self._doc_meta:
            return self._doc_meta[source].get("citation_key") or make_cite_key(source)
        return make_cite_key(source)
