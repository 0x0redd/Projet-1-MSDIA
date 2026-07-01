from agents.writers.base_writer import BaseWriter


def write(context_pkg: dict, cfg: dict, existing_drafts: dict) -> str:
    writer = BaseWriter()
    dry_run = bool(context_pkg.get("dry_run", False))

    section_summaries = ""
    for sec in [
        "introduction", "related_work", "methods", "results", "discussion", "conclusion",
    ]:
        if sec in existing_drafts:
            words = existing_drafts[sec].split()[:300]
            section_summaries += f"\n\n### {sec.upper()} (excerpt)\n" + " ".join(words)

    task = f"""Excerpts from completed sections:
{section_summaries}

Write a structured abstract (250-300 words) with implicit components (no subheadings):

1. Context: brain tumour MRI classification importance.
2. Problem: default feature parameters and limited multi-classifier benchmarks.
3. Method: two-phase ML pipeline, 1,500 balanced slices, eight classifiers.
4. Results: best F1-macro and accuracy from excerpts.
5. Conclusion: practical takeaway for classical ML on handcrafted features.

Requirements:
- No citations.
- Start with ## Abstract heading."""

    return writer.call_llm(task, density="dense" if section_summaries else "sparse", dry_run=dry_run)
