"""
Task 7 — PPI connectivity null models for drug-target seeds.

Statistic: number of edges in the subgraph induced by the seed gene set
within the CCLE-intersected STRING PPI (fixed parent graph for uniform null;
rewired parent for degree-preserving null).

Null models:
  (i)  Uniform random sampling — draw |seeds| genes uniformly from parent nodes.
  (ii) Degree-preserving rewiring — double_edge_swap on parent, seeds fixed.

Outputs (under paper/03_network_pathway/results/):
    rwr_null_uniform.csv       — null_value (1000 rows)
    rwr_null.csv               — degree-preserving null_value (1000 rows)
    rwr_observed.csv           — observed_value, empirical_p_uniform,
                                 empirical_p_degree

Usage:
    python eval_rwr_null.py
    python eval_rwr_null.py --skip_degree --n_null 1000
"""
from __future__ import annotations

import argparse
import os
import sys

import networkx as nx
import numpy as np
import pandas as pd

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
UNIFORM_CSV = os.path.join(OUT_DIR, "rwr_null_uniform.csv")
DEGREE_CSV = os.path.join(OUT_DIR, "rwr_null.csv")
OBS_CSV = os.path.join(OUT_DIR, "rwr_observed.csv")
os.makedirs(OUT_DIR, exist_ok=True)


def _load_parent_network_and_seeds():
    network = nx.read_edgelist(
        os.path.join(ROOT_DIR, "data", "cell_line_data", "PPI", "string_network"),
        delimiter="\t",
        nodetype=int,
        data=(("weight", float),),
    )
    data_dicts = np.load(
        os.path.join(ROOT_DIR, "data", "cell_line_data", "CCLE", "input_cellline_data.npy"),
        allow_pickle=True,
    ).item()
    ccle = set(int(g) for g in data_dicts["exp"].index)
    parent = network.subgraph([n for n in network.nodes if n in ccle]).copy()

    drug_target = pd.read_csv(
        os.path.join(ROOT_DIR, "results", "proessed_dpi.csv"), index_col=0
    )
    seeds = [int(g) for g in drug_target.index if int(g) in parent]
    return parent, seeds


def _build_adjacency(network):
    adj = {n: set() for n in network.nodes}
    for u, v in network.edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def seed_edge_count(network, node_set, adj=None):
    if not isinstance(node_set, set):
        node_set = set(node_set)
    if adj is not None:
        return sum(len(adj[u] & node_set) for u in node_set) // 2
    return sum(1 for u, v in network.edges if u in node_set and v in node_set)


def empirical_p(nulls, observed, one_sided_greater=True):
    nulls = np.asarray(nulls, dtype=float)
    if one_sided_greater:
        return (np.sum(nulls >= observed) + 1) / (len(nulls) + 1)
    return (np.sum(nulls <= observed) + 1) / (len(nulls) + 1)


def uniform_random_null(parent, n_seeds, n_null=1000, seed=0):
    rng = np.random.default_rng(seed)
    nodes = np.array(list(parent.nodes))
    adj = _build_adjacency(parent)
    print(
        f"uniform null: parent {parent.number_of_nodes()} nodes, "
        f"{parent.number_of_edges()} edges; sample {n_seeds} genes; {n_null} draws"
    )
    nulls = []
    for i in range(n_null):
        sample = set(rng.choice(nodes, size=n_seeds, replace=False).tolist())
        nulls.append(seed_edge_count(parent, sample, adj=adj))
        if (i + 1) % 100 == 0:
            print(f"  uniform progress: {i + 1}/{n_null}", flush=True)
    return np.array(nulls, dtype=float)


def degree_preserving_null(parent, seeds, n_null=1000, nswap_factor=0.01, seed=0):
    rng = np.random.default_rng(seed)
    seed_set = set(seeds)
    adj = _build_adjacency(parent)
    observed = seed_edge_count(parent, seed_set, adj=adj)
    n_edges = max(parent.number_of_edges(), 1)
    nswap = max(int(nswap_factor * n_edges), 500)
    print(
        f"degree null: parent {parent.number_of_nodes()} nodes, {n_edges} edges; "
        f"{len(seeds)} seeds; observed seed-edges={observed}; "
        f"{n_null} draws, nswap={nswap}"
    )
    nulls = []
    for i in range(n_null):
        g = parent.copy()
        try:
            nx.double_edge_swap(
                g,
                nswap=nswap,
                max_tries=nswap * 10,
                seed=int(rng.integers(0, 2**31 - 1)),
            )
        except nx.NetworkXAlgorithmError:
            nulls.append(observed)
            continue
        nulls.append(seed_edge_count(g, seed_set))
        if (i + 1) % 50 == 0:
            print(f"  degree progress: {i + 1}/{n_null}")
    return observed, np.array(nulls, dtype=float)


def run(
    n_null=1000,
    seed=0,
    nswap_factor=0.01,
    skip_uniform=False,
    skip_degree=False,
):
    os.makedirs(TASK7_DIR, exist_ok=True)
    parent, seeds = _load_parent_network_and_seeds()
    seed_set = set(seeds)
    adj = _build_adjacency(parent)
    observed = seed_edge_count(parent, seed_set, adj=adj)

    if skip_uniform and os.path.exists(UNIFORM_CSV):
        uniform_nulls = pd.read_csv(UNIFORM_CSV)["null_value"].values
        print(f"Loaded existing uniform null: {UNIFORM_CSV}")
    else:
        uniform_nulls = uniform_random_null(parent, len(seeds), n_null=n_null, seed=seed)
        pd.DataFrame({"null_value": uniform_nulls}).to_csv(UNIFORM_CSV, index=False)
        print(f"Wrote {UNIFORM_CSV} ({len(uniform_nulls)} rows)")

    if skip_degree and os.path.exists(DEGREE_CSV):
        degree_nulls = pd.read_csv(DEGREE_CSV)["null_value"].values
        print(f"Loaded existing degree null: {DEGREE_CSV}")
    else:
        observed, degree_nulls = degree_preserving_null(
            parent, seeds, n_null=n_null, nswap_factor=nswap_factor, seed=seed
        )
        pd.DataFrame({"null_value": degree_nulls}).to_csv(DEGREE_CSV, index=False)
        print(f"Wrote {DEGREE_CSV} ({len(degree_nulls)} rows)")

    p_uniform = empirical_p(uniform_nulls, observed)
    p_degree = empirical_p(degree_nulls, observed)
    obs_df = pd.DataFrame(
        {
            "observed_value": [observed],
            "empirical_p_uniform": [p_uniform],
            "empirical_p_degree": [p_degree],
        }
    )
    obs_df.to_csv(OBS_CSV, index=False)

    print(f"observed seed-edges: {observed}")
    print(
        f"uniform null mean±sd: {uniform_nulls.mean():.1f} ± {uniform_nulls.std():.1f}; "
        f"empirical_p={p_uniform:.4f}"
    )
    print(
        f"degree null mean±sd: {degree_nulls.mean():.1f} ± {degree_nulls.std():.1f}; "
        f"empirical_p={p_degree:.4f}"
    )
    print(f"Wrote {OBS_CSV}")
    return obs_df


def main():
    parser = argparse.ArgumentParser(description="Task 7 PPI connectivity null models")
    parser.add_argument("--n_null", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nswap_factor", type=float, default=0.01)
    parser.add_argument("--skip_uniform", action="store_true")
    parser.add_argument("--skip_degree", action="store_true")
    args = parser.parse_args()
    run(
        n_null=args.n_null,
        seed=args.seed,
        nswap_factor=args.nswap_factor,
        skip_uniform=args.skip_uniform,
        skip_degree=args.skip_degree,
    )


if __name__ == "__main__":
    main()
