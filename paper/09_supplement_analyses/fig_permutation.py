"""counterfactual transcriptome permutation.

"""
import os, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from figstyle import set_style, save_fig, PALETTE
set_style()

RESULTS = "results"
df = pd.read_csv(os.path.join(RESULTS, "permutation_pairwise.csv"))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 3.0),
                               gridspec_kw=dict(width_ratios=[1.4, 1]))

# ---- Panel A: scatter original vs permuted, two swap conditions ----
auc0 = roc_auc_score(df["y"], df["original"])
for scheme, col, lbl in [("same_tissue", PALETTE["green_mid"], "Same-tissue swap"),
                         ("cross_tissue", PALETTE["green_dark"], "Cross-tissue swap")]:
    rho = spearmanr(df["original"], df[scheme]).correlation
    ax1.scatter(df["original"], df[scheme], s=6, alpha=0.35, color=col,
                edgecolors="none", label=f"{lbl}  (ρ={rho:.2f})")
ax1.plot([0, 1], [0, 1], ls="--", lw=0.8, color="black")
ax1.set_xlabel("Prediction with true transcriptome")
ax1.set_ylabel("Prediction with swapped transcriptome")
ax1.set_xlim(0, 1); ax1.set_ylim(0, 1)
ax1.set_aspect("equal")
ax1.legend(loc="upper left", markerscale=2)
ax1.set_title("Counterfactual permutation", fontsize=8)

# ---- Panel B: ΔAUROC bars ----
schemes, deltas, colors = [], [], []
for scheme, col, lbl in [("same_tissue", PALETTE["green_mid"], "Same-tissue"),
                         ("cross_tissue", PALETTE["green_dark"], "Cross-tissue")]:
    auc = roc_auc_score(df["y"], df[scheme])
    schemes.append(lbl); deltas.append(auc - auc0); colors.append(col)
x = np.arange(len(schemes))
ax2.bar(x, deltas, color=colors, edgecolor="black", linewidth=0.6, width=0.6)
ax2.axhline(0, color="black", lw=0.8)
ax2.set_xticks(x); ax2.set_xticklabels(schemes)
ax2.set_ylabel("ΔAUROC vs. true sample")
ax2.set_title("Performance drop under swap", fontsize=8)
# annotate
for xi, d in zip(x, deltas):
    ax2.text(xi, d - 0.005 if d < 0 else d + 0.005, f"{d:+.3f}",
             ha="center", va="top" if d < 0 else "bottom", fontsize=6.5)

fig.tight_layout()
save_fig(fig, os.path.join(RESULTS, "Fig_permutation"))
print(f"AUROC(original) = {auc0:.3f}")
