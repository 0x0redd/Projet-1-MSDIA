from agents.writers.base_writer import BaseWriter

CORE_ARGUMENT = (
    "Classifier-free separability scoring for GLCM, LBP, DWT, and HOG combined "
    "with an eight-model classical ML benchmark achieves strong brain tumour MRI "
    "classification without deep learning."
)


def write(context_pkg: dict, cfg: dict, existing_drafts: dict) -> str:
    task = f"""Write the Introduction for the ML-only paper.

Core argument (state clearly):
{CORE_ARGUMENT}

Structure:
1. Clinical motivation — glioma, meningioma, pituitary on MRI.
2. Handcrafted texture features vs deep learning on limited data.
3. Gap: no classifier-free joint optimisation + multi-classifier benchmark.
4. Contributions: Phase 1 separability search, Phase 2 eight-model benchmark,
   feature importance, statistical validation.
5. Paper organisation pointer.

Requirements:
- Length: ~400-500 words.
- Start with ## Introduction heading.
- Cite related work from context where appropriate."""

    return BaseWriter().run_task(context_pkg, task)
