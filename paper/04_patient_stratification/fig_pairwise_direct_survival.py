"""
Fig — LS cohort survival with direct PAIRWISE stratification (new analysis line).

Companion to fig_survival.py (expression LPS held-out). Patients are assigned
BTKiCombo(+/-) from PAIRWISE drug-combination predictions (Ward clustering on
training folds; held-out assigned by nearest combo centroid).

INPUT (from survival_analysis/run_pairwise_direct_survival.py):
  results/survival/pairwise_direct_cox_forest.csv
  results/survival/pairwise_direct_km.csv
  results/survival/pairwise_direct_logrank.csv

Usage:
  python supplement_codes/fig_pairwise_direct_survival.py
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figstyle import PALETTE, save_fig, set_style

set_style()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SURV = os.path.join(ROOT, "results", "survival")
COX_CSV = os.path.join(SURV, "pairwise_direct_cox_forest.csv")
KM_CSV = os.path.join(SURV, "pairwise_direct_km.csv")
LR_CSV = os.path.join(SURV, "pairwise_direct_logrank.csv")
FIG_DIR = os.path.join(SURV, "figures")
OUT_BASE = os.path.join(FIG_DIR, "Fig_LS_PAIRWISE_direct")

SPEC_LABELS = {
    "unadjusted": "Unadjusted",
    "+age_stage": "+ age + stage",
    "+IPI": "+ IPI",
    "+full": "+ full (COO + subtype)",
}
GROUP_STYLE = {
    "sig_pos": ("BTKiCombo(+)", PALETTE["drug"]),
    "sig_neg": ("BTKiCombo(−)", PALETTE["neutral"]),
}


def _require(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing {path}. Run: python survival_analysis/run_pairwise_direct_survival.py"
        )


def fmt_p(p):
    return f"p = {p:.1e}" if p < 1e-3 else f"p = {p:.3f}"


def panel_forest(ax, cox):
    cox = cox[cox["model_spec"].isin(SPEC_LABELS)]
    order = [s for s in SPEC_LABELS if s in set(cox["model_spec"])]
    cox = cox.set_index("model_spec").loc[order].reset_index()
    y = np.arange(len(cox))[::-1]
    for yi, row in zip(y, cox.itertuples(index=False)):
        sig = row.p_value < 0.05
        color = PALETTE["warn"] if sig else PALETTE["neutral"]
        ax.plot([row.ci_low, row.ci_high], [yi, yi], "-", color=color, lw=1.2, zorder=2)
        ax.plot(row.HR, yi, "o", color=color, ms=5, zorder=3)
        ax.annotate(
            f"{row.HR:.2f} ({row.ci_low:.2f}–{row.ci_high:.2f})  {fmt_p(row.p_value)}",
            xy=(row.ci_high, yi), xytext=(4, 0), textcoords="offset points",
            va="center", ha="left", fontsize=6,
            style="italic" if sig else "normal",
        )
    ax.axvline(1.0, ls="--", lw=0.8, color="#444444", zorder=1)
    ax.set_xscale("log")
    ax.set_xticks([0.3, 0.5, 1.0, 2.0])
    ax.set_xticklabels(["0.3", "0.5", "1.0", "2.0"])
    ax.set_xlim(0.3, 3.2)
    ax.set_yticks(y)
    ax.set_yticklabels([SPEC_LABELS[s] for s in cox["model_spec"]])
    ax.set_xlabel("Hazard ratio, BTKiCombo(+) vs (−)  (PAIRWISE OOF)")
    ax.set_title("Cox (PFS), direct PAIRWISE stratification", fontsize=8)


def panel_km(ax, km, lr):
    t_max = km["time"].max()
    for grp, (label, color) in GROUP_STYLE.items():
        sub = km[km["group"] == grp].sort_values("time")
        if sub.empty:
            continue
        t = np.concatenate([[0.0], sub["time"].values])
        s = np.concatenate([[1.0], sub["survival"].values])
        ax.step(t, s, where="post", color=color, lw=1.8, label=label, zorder=3)
    txt = f"consensus log-rank {fmt_p(float(lr['logrank_p'].iloc[0]))}"
    if "logrank_p_mean" in lr.columns:
        txt += f"\n(5×CV mean p = {float(lr['logrank_p_mean'].iloc[0]):.1e})"
    ax.text(0.97, 0.95, txt, transform=ax.transAxes, ha="right", va="top",
            fontsize=6, style="italic")
    x_lo = -0.16 * t_max
    ax.set_xlim(x_lo, t_max * 1.02)
    ax.set_ylim(0, 1.02)
    ax.tick_params(labelbottom=False)
    ax.set_ylabel("PFS probability")
    ax.set_title("LS — PAIRWISE-defined BTKiCombo(+) vs (−)", fontsize=8)
    ax.legend(loc="lower left", fontsize=6, frameon=False)
    return t_max


def at_risk_table(ax, km, t_max):
    ticks = np.linspace(0, t_max, 6)
    x_lo = -0.16 * t_max
    ax.set_xlim(x_lo, t_max * 1.02)
    ax.set_ylim(0, 1)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.set_yticks([])
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:.1f}" for t in ticks])
    ax.set_xlabel("Progression-free survival (years)")
    short = {"sig_pos": "(+)", "sig_neg": "(−)"}
    for j, (grp, (label, color)) in enumerate(GROUP_STYLE.items()):
        sub = km[km["group"] == grp].sort_values("time")
        yj = 0.66 - j * 0.40
        ax.text(x_lo, yj, short[grp], color=color, fontsize=6, fontweight="bold",
                ha="left", va="center")
        for tk in ticks:
            prior = sub[sub["time"] <= tk]
            nrisk = int(prior["at_risk"].iloc[-1]) if len(prior) else (
                int(sub["at_risk"].iloc[0]) if len(sub) else 0)
            ax.text(tk, yj, str(nrisk), fontsize=6, ha="center", va="center")
    ax.text(0.5, 1.18, "Number at risk", fontsize=6, ha="center",
            va="bottom", transform=ax.transAxes, style="italic")


def main():
    for p in (COX_CSV, KM_CSV, LR_CSV):
        _require(p)
    cox = pd.read_csv(COX_CSV)
    km = pd.read_csv(KM_CSV)
    lr = pd.read_csv(LR_CSV)
    os.makedirs(FIG_DIR, exist_ok=True)
    fig = plt.figure(figsize=(8.5, 4.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[4, 1.1], width_ratios=[1.05, 1],
                          hspace=0.45, wspace=0.45)
    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_risk = fig.add_subplot(gs[1, 1])
    panel_forest(ax_a, cox)
    t_max = panel_km(ax_b, km, lr)
    at_risk_table(ax_risk, km, t_max)
    for ax, lbl in [(ax_a, "a"), (ax_b, "b")]:
        ax.text(-0.18, 1.04, lbl, transform=ax.transAxes, fontsize=11,
                fontweight="bold", va="bottom", ha="right")
    save_fig(fig, OUT_BASE)


if __name__ == "__main__":
    main()
