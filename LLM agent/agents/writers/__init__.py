from agents.writers import (
    abstract,
    conclusion,
    discussion,
    introduction,
    interpretability,
    methods,
    phase1_results,
    related_work,
    results,
    results_ml,
    statistics,
)

WRITERS = {
    "introduction": introduction.write,
    "related_work": related_work.write,
    "methods": methods.write,
    "phase1_results": phase1_results.write,
    "results_ml": results_ml.write,
    "results": results.write,
    "interpretability": interpretability.write,
    "statistics": statistics.write,
    "discussion": discussion.write,
    "conclusion": conclusion.write,
    "abstract": abstract.write,
}


def get_writer(section: str):
    if section not in WRITERS:
        raise KeyError(f"Unknown section: {section}")
    return WRITERS[section]
