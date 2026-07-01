from agents.writers.base_writer import BaseWriter


def write(context_pkg: dict, cfg: dict, existing_drafts: dict) -> str:
    writer = BaseWriter()
    dry_run = bool(context_pkg.get("dry_run", False))

    section_summaries = ""
    for sec in ["results", "discussion"]:
        if sec in existing_drafts:
            words = existing_drafts[sec].split()[:250]
            section_summaries += f"\n\n### {sec.upper()} (excerpt)\n" + " ".join(words)

    task = f"""The following are excerpts from completed sections:
{section_summaries}

Write the Conclusion (~200-250 words):

1. Summarize two-phase ML pipeline and best Phase 2 result (KNN/SVM Full(opt)).
2. Practical recommendation for handcrafted features + classical classifiers.
3. Future work: complete eight-model benchmark, statistical tests, external validation.

Requirements:
- No new citations.
- Use specific numbers from excerpts only.
- Start with ## Conclusion heading."""

    return writer.call_llm(task, density="moderate", dry_run=dry_run)
