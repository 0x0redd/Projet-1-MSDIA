#!/usr/bin/env python3
"""
Write data-grounded LaTeX section drafts to PAPER/latex/sections/.

Bypasses the LLM for prose quality: all numbers come directly from
data/experiments/*.csv and JSON. Run after prepare-data:

    python scripts/draft_sections_from_data.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
LATEX_SECTIONS = PROJECT_ROOT.parent / "PAPER" / "latex" / "sections"


def main() -> int:
    # Regenerate tables first
    subprocess.run([sys.executable, str(SCRIPT_DIR / "tables_to_latex.py")], check=True)

    # Section .tex files are maintained as data-grounded drafts in PAPER/latex/sections/.
    # This script re-syncs tables and prints a reminder.
    n = len(list(LATEX_SECTIONS.glob("*.tex")))
    print(f"[draft_sections] {n} section files in {LATEX_SECTIONS}")
    print("[draft_sections] Tables refreshed. Edit sections/*.tex or re-run this after metric updates.")
    print("[draft_sections] Build: cd PAPER/latex && .\\build.ps1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
