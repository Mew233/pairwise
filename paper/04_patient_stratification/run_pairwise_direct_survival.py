#!/usr/bin/env python3
"""
Direct PAIRWISE stratification survival on the LS cohort (new analysis line).

Unlike run_heldout_survival.py (expression LPS proxy), this line assigns each
patient's BTKiCombo(+/-) group from PAIRWISE drug-combination predictions
(mshap) alone:

  OOF mode (default, recommended):
    Within each CV fold, Ward clustering on TRAINING patients' top-variance
    combos defines (+)/(-). Held-out patients are assigned by nearest cluster
    centroid in the same combo feature space (their own PAIRWISE preds).

  Full mode (exploratory):
    Ward clustering on the entire analyzable cohort; survival on those labels.
    Outcomes never enter clustering, but this is not cross-validated.

Continuous score (both modes): standardized mean of the fold-specific top-30
combo predictions (train stats applied to held-out in OOF).

Outputs mirror run_heldout_survival.py under results/survival/pairwise_direct_*.
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

from run_heldout_survival import (
    DATA,
    MODEL_SPECS,
    N_FOLDS,
    N_SEEDS,
    N_TOP_COMBOS,
    POS,
    NEG,
    SURV_OUT,
    cluster_train_labels,
    fit_cox,
    load_data,
)

warnings.simplefilter("ignore", ConvergenceWarning)
warnings.simplefilter("ignore", RuntimeWarning)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PREFIX = "pairwise_direct"


def _top_combo_features(mshap: pd.DataFrame, samples) -> tuple[pd.Index, np.ndarray]:
    sub = mshap[samples]
    top = sub.var(axis=1).sort_values(ascending=False).head(N_TOP_COMBOS).index
    feat = sub.loc[top].T.values.astype(float)
    return top, feat


def assign_test_by_centroids(feat_tr: np.ndarray, lab: np.ndarray,
                             feat_te: np.ndarray) -> np.ndarray:
    """Nearest-centroid assignment in combo-prediction space."""
    out = np.empty(len(feat_te), dtype=object)
    clusters = np.unique(lab)
    centroids = {g: feat_tr[lab == g].mean(axis=0) for g in clusters}
    grp_mean = {g: feat_tr[lab == g].mean() for g in clusters}
    pos_cluster = max(grp_mean, key=grp_mean.get)
    for i, row in enumerate(feat_te):
        dists = {g: np.linalg.norm(row - centroids[g]) for g in clusters}
        nearest = min(dists, key=dists.get)
        out[i] = POS if nearest == pos_cluster else NEG
    return out


def pairwise_continuous_score(mshap: pd.DataFrame, top, samples,
                            ref_mean: float | None = None,
                            ref_std: float | None = None) -> pd.Series:
    raw = mshap.loc[top, samples].mean(axis=0).astype(float)
    mu = raw.mean() if ref_mean is None else ref_mean
    sd = raw.std() if ref_std is None else ref_std
    sd = sd if sd and sd > 0 else 1.0
    return (raw - mu) / sd


def oof_pairwise_for_seed(mshap, samples, events, seed):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    groups = pd.Series(index=samples, dtype=object)
    scores = pd.Series(index=samples, dtype=float)
    samples_arr = np.array(samples)
    for tr_idx, te_idx in skf.split(samples_arr, events):
        tr = list(samples_arr[tr_idx])
        te = list(samples_arr[te_idx])
        top, feat_tr = _top_combo_features(mshap, tr)
        Z = linkage(feat_tr, method="ward")
        lab = fcluster(Z, t=2, criterion="maxclust")
        feat_te = mshap.loc[top, te].T.values.astype(float)
        pred = assign_test_by_centroids(feat_tr, lab, feat_te)
        groups.loc[te] = pred
        raw_tr = mshap.loc[top, tr].mean(axis=0).astype(float)
        mu, sd = float(raw_tr.mean()), float(raw_tr.std())
        scores.loc[te] = pairwise_continuous_score(mshap, top, te, mu, sd).values
    return groups, scores


def full_cohort_pairwise(mshap, samples):
    labels = cluster_train_labels(mshap, samples)
    top, _ = _top_combo_features(mshap, samples)
    scores = pairwise_continuous_score(mshap, top, samples)
    return labels, scores


def _cox_tiers(df, time_col, event_col, dich_primary="group", cont_primary="pairwise_z"):
    cox_rows, cont_rows = [], []
    for spec, covs in MODEL_SPECS.items():
        d = df.copy()
        d["group"] = df["group"]
        res = fit_cox(d, covs, primary=dich_primary)
        if res:
            cox_rows.append(dict(model_spec=spec, **res))
        ccovs = ["pairwise_z" if c == "group" else c for c in covs]
        cres = fit_cox(d, ccovs, primary=cont_primary)
        if cres:
            cont_rows.append(dict(model_spec="continuous_" + spec, **cres))
    return cox_rows, cont_rows


def _km_and_logrank(df, time_col="pfs_time", event_col="pfs_event"):
    cl = df.dropna(subset=["group", time_col, event_col])
    lr = logrank_test(
        cl.loc[cl.group == 1, time_col], cl.loc[cl.group == 0, time_col],
        cl.loc[cl.group == 1, event_col], cl.loc[cl.group == 0, event_col],
    )
    rows = []
    for g, name in [(1, "sig_pos"), (0, "sig_neg")]:
        sub = cl[cl.group == g]
        kmf = KaplanMeierFitter().fit(sub[time_col], sub[event_col])
        et = kmf.event_table
        sf = kmf.survival_function_[kmf.survival_function_.columns[0]]
        for t in sf.index:
            rows.append(dict(
                time=float(t), group=name, survival=float(sf.loc[t]),
                at_risk=int(et.loc[t, "at_risk"]) if t in et.index else np.nan,
            ))
    return pd.DataFrame(rows), float(lr.p_value)


def run(mode: str = "oof"):
    os.makedirs(SURV_OUT, exist_ok=True)
    _, mshap, clin = load_data()
    samples = [s for s in mshap.columns if s in clin.index]
    base = clin.loc[samples]
    ok = base["pfs_time"].notna() & base["pfs_event"].notna() & (base["pfs_time"] > 0)
    samples = [s for s, v in zip(samples, ok) if v]
    events = clin.loc[samples, "pfs_event"].astype(int).values
    print(f"LS direct PAIRWISE stratification ({mode}): n={len(samples)} "
          f"(events={events.sum()})")

    if mode == "full":
        oof, pscore = full_cohort_pairwise(mshap, samples)
        df = clin.loc[samples].copy()
        df["group"] = (oof == POS).astype(float)
        df["pairwise_z"] = pscore.values
        cox_rows, cont_rows = _cox_tiers(df, "pfs_time", "pfs_event")
        cox_df = pd.DataFrame(cox_rows)
        cont_df = pd.DataFrame(cont_rows)
        km, lp = _km_and_logrank(df)
        cox_df.to_csv(os.path.join(SURV_OUT, f"{OUT_PREFIX}_cox_full.csv"), index=False)
        cont_df.to_csv(os.path.join(SURV_OUT, f"{OUT_PREFIX}_continuous_full.csv"), index=False)
        km.to_csv(os.path.join(SURV_OUT, f"{OUT_PREFIX}_km_full.csv"), index=False)
        pd.DataFrame([dict(logrank_p=lp, mode="full", n=len(df),
                           n_pos=int((df.group == 1).sum()),
                           n_neg=int((df.group == 0).sum()))]).to_csv(
            os.path.join(SURV_OUT, f"{OUT_PREFIX}_logrank_full.csv"), index=False)
        _print_summary(cox_df, cont_df, lp, mode="full")
        return

    cox_rows, cont_rows, lr_rows = [], [], []
    oof_by_seed = {}
    for seed in range(1, N_SEEDS + 1):
        oof, pscore = oof_pairwise_for_seed(mshap, samples, events, seed)
        oof_by_seed[seed] = oof
        df = clin.loc[samples].copy()
        df["group"] = (oof == POS).astype(float)
        df["pairwise_z"] = pscore.values
        n_pos = int((df.group == 1).sum())
        n_neg = int((df.group == 0).sum())
        _, lp = _km_and_logrank(df)
        lr_rows.append(dict(seed=seed, logrank_p=lp, n_pos=n_pos, n_neg=n_neg))
        cr, ctr = _cox_tiers(df, "pfs_time", "pfs_event")
        for r in cr:
            cox_rows.append(dict(seed=seed, **r))
        for r in ctr:
            cont_rows.append(dict(seed=seed, **r))
        print(f"  seed {seed}: +={n_pos} -={n_neg} logrank p={lp:.4g}")

    cox_df = pd.DataFrame(cox_rows)
    cont_df = pd.DataFrame(cont_rows)
    lr_df = pd.DataFrame(lr_rows)
    cox_df.to_csv(os.path.join(SURV_OUT, f"{OUT_PREFIX}_cox_by_seed.csv"), index=False)
    cont_df.to_csv(os.path.join(SURV_OUT, f"{OUT_PREFIX}_continuous_by_seed.csv"), index=False)
    lr_df.to_csv(os.path.join(SURV_OUT, f"{OUT_PREFIX}_logrank_by_seed.csv"), index=False)

    labels = pd.DataFrame({f"seed{s}": oof_by_seed[s] for s in oof_by_seed}, index=samples)
    labels.index.name = "sample"
    labels.to_csv(os.path.join(SURV_OUT, f"{OUT_PREFIX}_oof_labels.csv"), index=False)

    forest = _aggregate_forest(cox_df)
    forest.to_csv(os.path.join(SURV_OUT, f"{OUT_PREFIX}_cox_forest.csv"), index=False)
    cont_forest = _aggregate_forest(cont_df)
    cont_forest.to_csv(os.path.join(SURV_OUT, f"{OUT_PREFIX}_continuous_forest.csv"), index=False)

    pos_votes = (labels == POS).sum(axis=1)
    neg_votes = (labels == NEG).sum(axis=1)
    consensus = pd.Series(NEG, index=samples)
    consensus[pos_votes > neg_votes] = POS
    consensus[neg_votes > pos_votes] = NEG
    consensus[pos_votes == neg_votes] = np.nan

    dfc = clin.loc[samples].copy()
    dfc["group"] = (consensus == POS).astype(float)
    dfc.loc[consensus.isna(), "group"] = np.nan
    dfc = dfc.dropna(subset=["group"])
    pscore_full = pd.concat(
        [oof_pairwise_for_seed(mshap, samples, events, s)[1] for s in range(1, N_SEEDS + 1)],
        axis=1,
    ).mean(axis=1)
    dfc["pairwise_z"] = pscore_full.loc[dfc.index].values
    km, lp_c = _km_and_logrank(dfc)
    km.to_csv(os.path.join(SURV_OUT, f"{OUT_PREFIX}_km.csv"), index=False)
    pd.DataFrame([dict(
        logrank_p=lp_c,
        logrank_p_mean=float(lr_df.logrank_p.mean()),
        logrank_p_sd=float(lr_df.logrank_p.std()),
        grouping="consensus_majority",
        n=len(dfc),
        n_pos=int((dfc.group == 1).sum()),
        n_neg=int((dfc.group == 0).sum()),
    )]).to_csv(os.path.join(SURV_OUT, f"{OUT_PREFIX}_logrank.csv"), index=False)

    _print_summary(forest, cont_forest, lr_df.logrank_p.mean(), mode="oof",
                   lr_sd=lr_df.logrank_p.std())


def _aggregate_forest(cox_df: pd.DataFrame) -> pd.DataFrame:
    summ = (cox_df.assign(lHR=np.log(cox_df.HR), llo=np.log(cox_df.ci_low),
                          lhi=np.log(cox_df.ci_high))
            .groupby("model_spec")
            .agg(lHR=("lHR", "mean"), llo=("llo", "mean"), lhi=("lhi", "mean"),
                 p_value=("p_value", "median"), n=("n", "median"))
            .reset_index())
    order = {k: i for i, k in enumerate(MODEL_SPECS)}
    cont_order = {f"continuous_{k}": i for i, k in enumerate(MODEL_SPECS)}
    order.update(cont_order)
    summ = summ.sort_values("model_spec", key=lambda s: s.map(order))
    return pd.DataFrame({
        "model_spec": summ.model_spec,
        "HR": np.exp(summ.lHR),
        "ci_low": np.exp(summ.llo),
        "ci_high": np.exp(summ.lhi),
        "p_value": summ.p_value,
        "n": summ.n,
    })


def _print_summary(dich, cont, lp, mode, lr_sd=None):
    print(f"\n================ LS DIRECT PAIRWISE ({mode}) ================")
    print(f"Log-rank: p = {lp:.4g}" + (f" ± {lr_sd:.2g}" if lr_sd else ""))
    print("\nCox HR, dichotomized BTKiCombo(+) vs (-):")
    for _, r in dich.iterrows():
        print(f"  {r.model_spec:<14} HR={r.HR:.2f} [{r.ci_low:.2f}, {r.ci_high:.2f}] "
              f"p={r.p_value:.3g} (n={int(r.n)})")
    print("\nCox HR, continuous PAIRWISE combo score (per +1 SD):")
    for _, r in cont.iterrows():
        print(f"  {r.model_spec:<22} HR={r.HR:.2f} [{r.ci_low:.2f}, {r.ci_high:.2f}] "
              f"p={r.p_value:.3g} (n={int(r.n)})")
    print(f"\nWrote: {SURV_OUT}/{OUT_PREFIX}_*.csv")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="LS survival with direct PAIRWISE stratification")
    p.add_argument("--mode", choices=("oof", "full"), default="oof",
                   help="oof = CV-held-out PAIRWISE labels (default); full = whole-cohort cluster")
    run(mode=p.parse_args().mode)
