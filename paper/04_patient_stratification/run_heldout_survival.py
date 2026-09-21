#!/usr/bin/env python3
"""
Leakage-free held-out validation of BTKiCombo(+/-) stratification (reviewer R1.4 +
circularity concern) plus Task 9 Cox re-analysis.

Design (breaks the circularity the reviewer flagged):
  Repeated stratified 5-fold CV over LS-cohort patients (seeds 1..N_SEEDS).
  Within each fold:
    * BTKiCombo(+/-) groups are defined by Ward clustering of PAIRWISE drug-
      combination predictions on the TRAINING patients ONLY.
    * A 50-gene LPS classifier is trained on the training patients' expression.
    * The frozen classifier predicts the HELD-OUT patients' group from their
      GENE EXPRESSION ALONE (never their PAIRWISE predictions).
  Out-of-fold predictions are pooled -> every patient gets a leakage-free label.
  Survival (KM + log-rank) and Cox models (unadjusted -> +age+stage -> +IPI ->
  +full COO/genetic-subtype) are evaluated on these held-out labels, repeated
  across seeds to report distribution (not a single lucky split).

Clinical covariates: NEJM Schmitz et al. 2018 appendix Tab S9 (IPI, stage, age,
gender, COO, genetic subtype, PFS time/status).
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.model_selection import StratifiedKFold

from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.exceptions import ConvergenceWarning

from lps_classifier import LPSClassifier

warnings.simplefilter("ignore", ConvergenceWarning)
warnings.simplefilter("ignore", RuntimeWarning)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "lstaudt_Data")
RESULTS = os.path.realpath(os.path.join(HERE, "..", "results"))
SURV_OUT = os.path.join(RESULTS, "survival")

N_SEEDS = 5
N_FOLDS = 5
N_TOP_COMBOS = 30
N_LPS_GENES = 50
POS, NEG = "Eff", "Noneff"

MODEL_SPECS = {
    "unadjusted": ["group"],
    "+age_stage": ["group", "age", "stage"],
    "+IPI": ["group", "age", "stage", "ipi"],
    "+full": ["group", "age", "stage", "ipi", "coo_GCB", "coo_Unclass",
              "sub_MCD", "sub_BN2", "sub_EZB", "sub_N1"],
}


def load_data():
    expr = pd.read_csv(os.path.join(DATA, "orginal_lstaudt_DLBC_exp.csv"), index_col=0)
    expr.index = expr.index.astype(str)
    expr = expr[~expr.index.duplicated()]

    mshap = pd.read_csv(
        os.path.join(HERE, "cooclassifier-main", "mshap_lstaudt.csv"), index_col=0
    )
    mshap = mshap.replace([np.inf, -np.inf], np.nan)
    mshap = mshap.dropna(how="all")                       # drop all-NaN combos
    mshap = mshap.apply(lambda r: r.fillna(r.mean()), axis=1)  # impute residual NaN

    xl = pd.ExcelFile(os.path.join(DATA, "nejmoa1801445_appendix_2.xlsx"))
    s9 = xl.parse("Tab S9 Characteristics DLBCL", header=0)
    s9 = s9.rename(columns={
        s9.columns[0]: "subject",
        "Gene Expression Subgroup": "coo",
        "Genetic Subtype": "subtype",
        "Gender": "gender",
        "Age": "age",
        "Ann Arbor Stage": "stage",
        "IPI Range": "ipi_range",
        "IPI Group": "ipi_group",
        "Progression_Free Survival _PFS_ Time _yrs": "pfs_time",
        "Progression_Free Survival _PFS_ Status_ 0 No Progressoin_ 1 Progression": "pfs_event",
        "Included in Survival Analysis": "included",
    })

    clin = s9[["subject", "coo", "subtype", "gender", "age", "stage",
               "ipi_range", "ipi_group", "pfs_time", "pfs_event", "included"]].copy()
    clin = clin.set_index("subject")
    clin["age"] = pd.to_numeric(clin["age"], errors="coerce")
    clin["stage"] = pd.to_numeric(clin["stage"], errors="coerce").round()
    clin["ipi"] = pd.to_numeric(clin["ipi_range"], errors="coerce")
    clin.loc[clin["ipi"] > 5, "ipi"] = np.nan  # codes 12/15 == missing
    clin["gender"] = clin["gender"].map({"M": 1, "F": 0})
    clin["pfs_time"] = pd.to_numeric(clin["pfs_time"], errors="coerce")
    clin["pfs_event"] = pd.to_numeric(clin["pfs_event"], errors="coerce")
    # COO / subtype dummies (ABC and Other = references)
    clin["coo_GCB"] = (clin["coo"] == "GCB").astype(int)
    clin["coo_Unclass"] = (clin["coo"] == "Unclass").astype(int)
    for sub in ["MCD", "BN2", "EZB", "N1"]:
        clin[f"sub_{sub}"] = (clin["subtype"] == sub).astype(int)
    return expr, mshap, clin


def cluster_train_labels(mshap: pd.DataFrame, train_samples) -> pd.Series:
    """BTKiCombo(+/-) by Ward clustering of PAIRWISE combo preds on TRAIN only."""
    sub = mshap[train_samples]
    top = sub.var(axis=1).sort_values(ascending=False).head(N_TOP_COMBOS).index
    feat = sub.loc[top].T.values            # samples x top-combos
    Z = linkage(feat, method="ward")
    lab = fcluster(Z, t=2, criterion="maxclust")
    grp_mean = {g: feat[lab == g].mean() for g in np.unique(lab)}
    pos_cluster = max(grp_mean, key=grp_mean.get)  # higher predicted synergy = (+)
    labels = pd.Series(
        [POS if g == pos_cluster else NEG for g in lab], index=train_samples
    )
    return labels


def oof_labels_for_seed(expr, mshap, samples, events, seed):
    """Return out-of-fold (group label, continuous LPS) for every patient.

    The continuous score is the per-patient LPS predicted from expression alone in
    the fold where the patient was held out (no outcome information involved).
    """
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    out = pd.Series(index=samples, dtype=object)
    lps = pd.Series(index=samples, dtype=float)
    samples = np.array(samples)
    for tr_idx, te_idx in skf.split(samples, events):
        tr, te = list(samples[tr_idx]), list(samples[te_idx])
        labels = cluster_train_labels(mshap, tr)
        clf = LPSClassifier(n_genes=N_LPS_GENES).fit(expr[tr], labels)
        pred = clf.predict(expr[te])
        out.loc[te] = pred["group"].values
        lps.loc[te] = pred["lps"].values
    return out, lps


def fit_cox(df: pd.DataFrame, covariates, primary="group"):
    cols = ["pfs_time", "pfs_event"] + covariates
    d = df[cols].dropna()
    if d["pfs_event"].sum() < 5 or (primary == "group" and d["group"].nunique() < 2):
        return None
    cph = CoxPHFitter(penalizer=0.1)
    try:
        cph.fit(d, duration_col="pfs_time", event_col="pfs_event")
    except Exception:
        return None
    if primary not in cph.params_.index:
        return None
    s = cph.summary.loc[primary]
    return dict(HR=float(s["exp(coef)"]), ci_low=float(s["exp(coef) lower 95%"]),
                ci_high=float(s["exp(coef) upper 95%"]), p_value=float(s["p"]), n=len(d))


def run():
    os.makedirs(SURV_OUT, exist_ok=True)
    expr, mshap, clin = load_data()

    samples = [s for s in mshap.columns if s in expr.columns and s in clin.index]
    base = clin.loc[samples]
    ok = base["pfs_time"].notna() & base["pfs_event"].notna() & (base["pfs_time"] > 0)
    samples = [s for s, v in zip(samples, ok) if v]
    events = clin.loc[samples, "pfs_event"].astype(int).values
    print(f"Analyzable LS patients (expr & preds & PFS & clinical): {len(samples)} "
          f"(events={events.sum()})")

    cox_rows, lr_rows, cont_rows = [], [], []
    oof_by_seed = {}
    for seed in range(1, N_SEEDS + 1):
        oof, lps = oof_labels_for_seed(expr, mshap, samples, events, seed)
        oof_by_seed[seed] = oof
        df = clin.loc[samples].copy()
        df["group"] = (oof == POS).astype(float)
        df.loc[oof == "unclassified", "group"] = np.nan
        df["lps_z"] = (lps - lps.mean()) / lps.std()   # continuous score (per SD)
        classified = df.dropna(subset=["group"])
        n_pos = int((classified["group"] == 1).sum())
        n_neg = int((classified["group"] == 0).sum())

        cl = classified.dropna(subset=["pfs_time", "pfs_event"])
        lr = logrank_test(
            cl.loc[cl.group == 1, "pfs_time"], cl.loc[cl.group == 0, "pfs_time"],
            cl.loc[cl.group == 1, "pfs_event"], cl.loc[cl.group == 0, "pfs_event"],
        )
        lr_rows.append(dict(seed=seed, logrank_p=float(lr.p_value),
                            n_pos=n_pos, n_neg=n_neg,
                            n_unclassified=int((oof == "unclassified").sum())))
        for spec, covs in MODEL_SPECS.items():
            res = fit_cox(df, covs)
            if res:
                cox_rows.append(dict(seed=seed, model_spec=spec, **res))
            cres = fit_cox(df, ["lps_z" if c == "group" else c for c in covs],
                           primary="lps_z")
            if cres:
                cont_rows.append(dict(seed=seed, model_spec="continuous_" + spec, **cres))
        print(f"  seed {seed}: +={n_pos} -={n_neg} unclassified="
              f"{int((oof=='unclassified').sum())} logrank p={lr.p_value:.4g}")

    cox_df = pd.DataFrame(cox_rows)
    lr_df = pd.DataFrame(lr_rows)
    cont_df = pd.DataFrame(cont_rows)
    cox_df.to_csv(os.path.join(SURV_OUT, "heldout_cox_by_seed.csv"), index=False)
    lr_df.to_csv(os.path.join(SURV_OUT, "heldout_logrank_by_seed.csv"), index=False)
    cont_df.to_csv(os.path.join(SURV_OUT, "heldout_continuous_by_seed.csv"), index=False)

    # Per-patient OOF labels across all seeds (audit trail for consensus).
    labels = pd.DataFrame({f"seed{s}": oof_by_seed[s] for s in oof_by_seed}, index=samples)
    labels.index.name = "sample"
    labels.to_csv(os.path.join(SURV_OUT, "heldout_oof_labels.csv"))

    # Canonical Fig-9 forest = across-seed mean (geometric mean for HR/CI).
    summ = (cox_df.assign(lHR=np.log(cox_df.HR), llo=np.log(cox_df.ci_low),
                          lhi=np.log(cox_df.ci_high))
            .groupby("model_spec")
            .agg(lHR=("lHR", "mean"), llo=("llo", "mean"), lhi=("lhi", "mean"),
                 p_value=("p_value", "median"), n=("n", "median")).reset_index())
    order = {k: i for i, k in enumerate(MODEL_SPECS)}
    summ = summ.sort_values("model_spec", key=lambda s: s.map(order))
    forest = pd.DataFrame({
        "model_spec": summ.model_spec,
        "HR": np.exp(summ.lHR), "ci_low": np.exp(summ.llo),
        "ci_high": np.exp(summ.lhi), "p_value": summ.p_value,
    })
    forest.to_csv(os.path.join(RESULTS, "cox_forest.csv"), index=False)

    # Consensus group = majority vote across seeds (ties -> unclassified).
    pos_votes = (labels == POS).sum(axis=1)
    neg_votes = (labels == NEG).sum(axis=1)
    consensus = pd.Series("unclassified", index=samples)
    consensus[pos_votes > neg_votes] = POS
    consensus[neg_votes > pos_votes] = NEG
    dfc = clin.loc[samples].copy()
    dfc["group"] = (consensus == POS).astype(float)
    dfc.loc[consensus == "unclassified", "group"] = np.nan
    clc = dfc.dropna(subset=["group", "pfs_time", "pfs_event"])
    n_pos_c = int((clc.group == 1).sum())
    n_neg_c = int((clc.group == 0).sum())
    print(f"  consensus (majority of {N_SEEDS} seeds): +={n_pos_c} -={n_neg_c} "
          f"unclassified={int((consensus=='unclassified').sum())}")

    # Canonical KM from consensus grouping + at-risk.
    km_rows = []
    for g, name in [(1, "sig_pos"), (0, "sig_neg")]:
        sub = clc[clc.group == g]
        kmf = KaplanMeierFitter().fit(sub["pfs_time"], sub["pfs_event"])
        et = kmf.event_table
        sf = kmf.survival_function_[kmf.survival_function_.columns[0]]
        for t in sf.index:
            km_rows.append(dict(time=float(t), group=name, survival=float(sf.loc[t]),
                                at_risk=int(et.loc[t, "at_risk"]) if t in et.index else np.nan))
    pd.DataFrame(km_rows).to_csv(os.path.join(RESULTS, "km_curve.csv"), index=False)

    # Per-seed KM curves (for the repeatability overlay in panel b).
    seed_rows = []
    for seed in range(1, N_SEEDS + 1):
        oof = oof_by_seed[seed]
        dfs = clin.loc[samples].copy()
        dfs["group"] = (oof == POS).astype(float)
        dfs.loc[oof == "unclassified", "group"] = np.nan
        cls = dfs.dropna(subset=["group", "pfs_time", "pfs_event"])
        for g, name in [(1, "sig_pos"), (0, "sig_neg")]:
            sub = cls[cls.group == g]
            if sub.empty:
                continue
            kmf = KaplanMeierFitter().fit(sub["pfs_time"], sub["pfs_event"])
            sf = kmf.survival_function_[kmf.survival_function_.columns[0]]
            for t in sf.index:
                seed_rows.append(dict(seed=seed, time=float(t), group=name,
                                      survival=float(sf.loc[t])))
    pd.DataFrame(seed_rows).to_csv(os.path.join(RESULTS, "km_curve_seeds.csv"), index=False)

    lrc = logrank_test(clc.loc[clc.group == 1, "pfs_time"], clc.loc[clc.group == 0, "pfs_time"],
                       clc.loc[clc.group == 1, "pfs_event"], clc.loc[clc.group == 0, "pfs_event"])
    pd.DataFrame([dict(logrank_p=float(lrc.p_value),
                       logrank_p_mean=float(lr_df.logrank_p.mean()),
                       logrank_p_sd=float(lr_df.logrank_p.std()),
                       grouping="consensus_majority")]
                 ).to_csv(os.path.join(RESULTS, "km_logrank.csv"), index=False)

    _summary(forest, lr_df, cox_df, cont_df)


def _summary(forest, lr_df, cox_df, cont_df=None):
    print("\n================ HELD-OUT VALIDATION SUMMARY ================")
    print(f"Repeated {N_SEEDS}x stratified {N_FOLDS}-fold OOF; clustering+LPS fit on "
          f"train folds only; survival on held-out predictions.")
    print(f"\nLog-rank (held-out OOF): p = {lr_df.logrank_p.mean():.4g} "
          f"± {lr_df.logrank_p.std():.2g} (median {lr_df.logrank_p.median():.4g})")
    print("\nCox HR for BTKiCombo(+) vs (-), across-seed mean [95% CI], median p:")
    for _, r in forest.iterrows():
        print(f"  {r.model_spec:<12} HR={r.HR:.2f} [{r.ci_low:.2f}, {r.ci_high:.2f}]  p={r.p_value:.3g}")
    if cont_df is not None and len(cont_df):
        print("\nCox HR for continuous LPS (per +1 SD), across-seed mean [95% CI], median p:")
        g = (cont_df.assign(lHR=np.log(cont_df.HR), llo=np.log(cont_df.ci_low),
                            lhi=np.log(cont_df.ci_high))
             .groupby("model_spec")
             .agg(lHR=("lHR", "mean"), llo=("llo", "mean"), lhi=("lhi", "mean"),
                  p_value=("p_value", "median")).reset_index())
        cont_order = ["continuous_" + s for s in MODEL_SPECS]
        g = g.set_index("model_spec").reindex(cont_order).dropna().reset_index()
        for _, r in g.iterrows():
            print(f"  {r.model_spec:<22} HR={np.exp(r.lHR):.2f} "
                  f"[{np.exp(r.llo):.2f}, {np.exp(r.lhi):.2f}]  p={r.p_value:.3g}")
    print(f"\nWrote: {os.path.join(RESULTS,'cox_forest.csv')}, km_curve.csv, km_logrank.csv")
    print(f"       {SURV_OUT}/heldout_cox_by_seed.csv, heldout_logrank_by_seed.csv")


if __name__ == "__main__":
    run()
