"""
figstyle.py
"""
import matplotlib as mpl
import matplotlib.pyplot as plt

# ---- color palette (consistent across all revision figures) ----
PALETTE = {
    "pairwise":   "#C8A87E",   # former transynergy_liu (TranSynergy arch), now --model pairwise
    "drug":       "#E8A33D",   # orange — chemical / drug stream
    "transcriptome": "#4CA66B",# green — sample / transcriptome stream
    "dti":        "#5B7FB9",   # blue  — drug-target / network stream
    "neutral":    "#8C8C8C",   # grey  — baselines / reference
    "warn":       "#C0392B",   # red   — leakage / warning
    # graduated greens for swap conditions
    "green_light":"#A8D5B5",
    "green_mid":  "#5FB07E",
    "green_dark": "#2E7D4F",
    # baseline model colors (6 competitors in muted tones)
    "deepsynergy_preuer": "#7E8AA2",
    "matchmaker_brahim":  "#A2A77E",
    "multitaskdnn_kim":   "#A27E94",
    "deepdds_wang":       "#7EA29C",
    "TGSynergy":          "#9C8AA2",
    "transynergy_liu":    "#C0392B",   # former pairwise arch, now --model transynergy_liu
}

# pretty display names for models
MODEL_NAMES = {
    "pairwise":           "PAIRWISE",       # public name; TranSynergy-style architecture
    "deepsynergy_preuer": "DeepSynergy",
    "matchmaker_brahim":  "MatchMaker",
    "multitaskdnn_kim":   "Multitask-DNN",
    "deepdds_wang":       "DeepDDS",
    "TGSynergy":          "TGSynergy",
    "transynergy_liu":    "TranSynergy",
    "LR": "LR", "RF": "RF", "XGBOOST": "XGBoost", "ERT": "ERT",
}

def set_style():
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 600,                 # journal print quality
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "legend.frameon": False,
        "lines.linewidth": 1.2,
        "lines.markersize": 4,
        "pdf.fonttype": 42,                 # editable text in Illustrator
        "ps.fonttype": 42,
    })

def save_fig(fig, path_noext):
    """Save both a vector PDF (for the journal) and a PNG (for quick view)."""
    fig.savefig(f"{path_noext}.pdf", bbox_inches="tight", transparent=True)
    fig.savefig(f"{path_noext}.png", bbox_inches="tight", dpi=200)
    print(f"saved: {path_noext}.pdf / .png")
