"""Generate Phase 2 feature-cleaning figures for the rapport."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[1]

sets = ["A", "B", "C", "D", "A+B", "B+C", "A+B+C", "Full"]
raw = np.array([11, 20, 16, 1944, 31, 36, 47, 1991])
after_var = raw.copy()
cleaned = np.array([9, 18, 12, 490, 27, 30, 36, 526])
kept_pct = cleaned / raw * 100

plt.rcParams.update({"font.size": 10})

# Figure: dimensions before / after
fig, ax = plt.subplots(figsize=(10, 4.5))
x = np.arange(len(sets))
w = 0.25
ax.bar(x - w, raw, w, label="Brut", color="#2E5A88")
ax.bar(x, after_var, w, label="Après filtre variance", color="#4A7FB5")
ax.bar(x + w, cleaned, w, label="Nettoyé", color="#1F6F54")
ax.set_xticks(x)
ax.set_xticklabels(sets)
ax.set_ylabel("Nombre de caractéristiques")
ax.set_title("Dimensions des ensembles avant et après nettoyage")
ax.legend(loc="upper right", fontsize=9)
ax.set_yscale("symlog", linthresh=20)
fig.tight_layout()
fig.savefig(OUT / "feature_cleaning_dims.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# Figure: conservation ratio
fig, ax = plt.subplots(figsize=(9, 4.2))
colors = ["#1F6F54" if p >= 75 else "#E0A85B" if p >= 50 else "#C44E52" for p in kept_pct]
bars = ax.bar(sets, kept_pct, color=colors, edgecolor="white")
ax.bar_label(bars, fmt="%.1f%%", fontsize=8, padding=2)
ax.set_ylim(0, 105)
ax.set_ylabel("Ratio de conservation (%)")
ax.set_title("Pourcentage de caractéristiques conservées après nettoyage")
ax.axhline(50, color="gray", ls="--", lw=0.8, alpha=0.5)
fig.tight_layout()
fig.savefig(OUT / "feature_cleaning_kept_ratio.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print("Saved:", OUT / "feature_cleaning_dims.png")
print("Saved:", OUT / "feature_cleaning_kept_ratio.png")
