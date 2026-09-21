"""
Fig 5b recomputation — largest connected-component size of the BTK-combination
partner-drug PRIMARY TARGETS on the STRING PPI, with two nulls:

  (i)  Uniform random sampling       — draw |seeds| genes uniformly from the
       DPI target universe; recompute largest-CC on the same subnetwork.
  (ii) Degree-preserving rewiring    — double_edge_swap on the DPI-induced
       STRING subnetwork (every node's degree preserved exactly), seed set
       fixed; recompute largest-CC.

Statistic: size (node count) of the LARGEST connected component of the subgraph
induced by the seed genes.  Network: STRING (confidence >= CUTOFF), restricted
to the DPI target universe (results/proessed_dpi.csv row genes).

Outputs (results/):
    fig5b_cc_null_uniform.csv   null_value (N_PERM rows)
    fig5b_cc_null_degree.csv    null_value (N_PERM rows)
    fig5b_cc_observed.csv       observed, k_seeds, cutoff, uniform_mean/sd/z/p,
                                degree_mean/sd/z/p

Usage:  python scripts/eval_fig5b_cc_null.py --n_perm 1000 --cutoff 0.5
"""
from __future__ import annotations

import argparse
import os

import networkx as nx
import numpy as np
import pandas as pd

ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
DPI_CSV = os.path.join(ROOT, "results", "proessed_dpi.csv")
STRING = os.path.join(ROOT, "data", "cell_line_data", "PPI", "string_network")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_UNIFORM = os.path.join(OUT_DIR, "fig5b_cc_null_uniform.csv")
OUT_DEGREE = os.path.join(OUT_DIR, "fig5b_cc_null_degree.csv")
OUT_OBS = os.path.join(OUT_DIR, "fig5b_cc_observed.csv")

# Primary targets labelled in Fig 5a (positively-predicted BTK-combination
# partner drugs), mapped to Entrez gene IDs.
SEED_TARGETS = {
    "WEE1": 7465, "MTOR": 2475, "FGFR1": 2260, "BCL2": 596, "MCL1": 4170,
    "CSNK2A1": 1457, "CDK4": 1019, "PRKDC": 5591, "PARP1": 142, "IGF1R": 3480,
    "ERBB2": 2064, "AKT1": 207, "NAMPT": 10135, "SRC": 6714, "CHEK1": 1111,
    "CHEK2": 11200, "CDK9": 1025, "ALK": 238, "JAK2": 3717, "AURKA": 6790,
    "ATR": 545, "BTK": 695,
}


def largest_cc_size(net, nodes):
    present = [n for n in nodes if n in net]
    if not present:
        return 0
    sub = net.subgraph(present)
    return len(max(nx.connected_components(sub), key=len))


def empirical_p(nulls, observed):
    nulls = np.asarray(nulls, dtype=float)
    return (np.sum(nulls >= observed) + 1) / (len(nulls) + 1)


def build_subnetwork(cutoff):
    G = nx.read_edgelist(STRING, delimiter="\t", nodetype=int,
                         data=(("weight", float),))
    H = nx.Graph()
    H.add_nodes_from(G.nodes())
    H.add_edges_from((u, v) for u, v, d in G.edges(data=True) if d["weight"] >= cutoff)
    dpi_genes = set(int(x) for x in pd.read_csv(DPI_CSV, index_col=0).index)
    P = H.subgraph(sorted(dpi_genes & set(H.nodes()))).copy()
    return P


def run(n_perm=1000, cutoff=0.5, seed=0):
    P = build_subnetwork(cutoff)
    nodes = np.array(sorted(P.nodes()))
    seeds = list(SEED_TARGETS.values())
    seeds_present = [s for s in seeds if s in P]
    k = len(seeds_present)
    observed = largest_cc_size(P, seeds)
    print(f"subnetwork (STRING>={cutoff}, DPI universe): "
          f"{P.number_of_nodes()} nodes, {P.number_of_edges()} edges")
    print(f"seeds: {len(seeds)} total, {k} present; observed largest-CC = {observed}")

    # (i) uniform null
    rng = np.random.default_rng(seed)
    uniform = np.empty(n_perm)
    for i in range(n_perm):
        draw = rng.choice(nodes, size=k, replace=False)
        uniform[i] = largest_cc_size(P, draw)
    pd.DataFrame({"null_value": uniform}).to_csv(OUT_UNIFORM, index=False)
    print(f"uniform null done: mean={uniform.mean():.2f} sd={uniform.std():.2f}")

    # (ii) degree-preserving null
    deg0 = sorted(d for _, d in P.degree())
    seed_set = set(seeds_present)
    nswap = P.number_of_edges()
    degree = np.empty(n_perm)
    rng2 = np.random.default_rng(seed + 1)
    for i in range(n_perm):
        g = P.copy()
        try:
            nx.double_edge_swap(g, nswap=nswap, max_tries=nswap * 20,
                                seed=int(rng2.integers(0, 2**31 - 1)))
        except nx.NetworkXAlgorithmError:
            degree[i] = observed
            continue
        if i == 0:
            assert sorted(d for _, d in g.degree()) == deg0, "degree sequence changed!"
        degree[i] = largest_cc_size(g, seed_set)
        if (i + 1) % 50 == 0:
            print(f"  degree null {i + 1}/{n_perm}", flush=True)
    pd.DataFrame({"null_value": degree}).to_csv(OUT_DEGREE, index=False)
    print(f"degree null done: mean={degree.mean():.2f} sd={degree.std():.2f}")

    def z(x):
        return (observed - x.mean()) / x.std(ddof=0) if x.std(ddof=0) > 0 else np.inf

    obs_df = pd.DataFrame({
        "observed": [observed], "k_seeds": [k], "cutoff": [cutoff],
        "uniform_mean": [uniform.mean()], "uniform_sd": [uniform.std(ddof=0)],
        "uniform_z": [z(uniform)], "uniform_p": [empirical_p(uniform, observed)],
        "degree_mean": [degree.mean()], "degree_sd": [degree.std(ddof=0)],
        "degree_z": [z(degree)], "degree_p": [empirical_p(degree, observed)],
    })
    obs_df.to_csv(OUT_OBS, index=False)
    print(obs_df.to_string(index=False))
    return obs_df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_perm", type=int, default=1000)
    ap.add_argument("--cutoff", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run(n_perm=args.n_perm, cutoff=args.cutoff, seed=args.seed)
