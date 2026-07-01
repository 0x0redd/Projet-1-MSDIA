from agents.writers.base_writer import BaseWriter

TASK = """Write the Related Work section for a brain tumor MRI classification paper (3 classes:
glioma, meningioma, pituitary). Structure as follows:

1. **Handcrafted texture and radiomics** — LBP, GLCM, HOG, DWT and their use in
   medical image analysis and brain tumor detection.

2. **Classical ML classifiers** — SVM, ensemble methods on radiomic features;
   parameter selection strategies.

3. **Parameter selection and separability metrics** — grid search vs classifier-free scoring.

4. **Gap and motivation** — no study combines classifier-free GLCM/LBP/DWT/HOG optimisation
   with an eight-model benchmark and full statistical validation on the same split.

Requirements:
- Use ### subheadings aligned with the structure above.
- Minimum 6 \\cite{ref...} citations using ONLY citation_key values from context chunks.
- FORBIDDEN: numbered refs [1], fake author lists, invented papers, References section.
- Compare prior work to OUR two-phase ML pipeline (Phase 1 + Phase 2).
- Length: ~700-900 words.
- Start with ## Related Work heading only."""


def write(context_pkg: dict, cfg: dict, existing_drafts: dict) -> str:
    return BaseWriter().run_task(context_pkg, TASK)
