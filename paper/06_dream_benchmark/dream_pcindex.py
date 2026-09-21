"""
Faithful Python port of the official NCI-DREAM SubChallenge2 scoring script
(NCI_DREAMSubchallenge2_ScoringScript.m). Probabilistic concordance index (pc-index)
with the Gaussian-error weighting and 10,000-permutation null p-value.
"""
import numpy as np, glob, os, re
_HERE = os.path.dirname(os.path.abspath(__file__))
from math import erf

GOLD = os.path.join(_HERE, "data", "NCI-DREAMSubchallenge2_GoldStandard.txt")

def canon(a, b):
    a, b = a.strip(), b.strip()
    x, y = sorted([a, b])
    return f"{x}_{y}"

def read_gold(path=GOLD):
    drug1, drug2, eob, err = [], [], [], []
    with open(path, encoding="latin-1") as f:
        raw = f.read()
    # file uses \r as line sep, with a header line
    lines = [l for l in re.split(r'[\r\n]+', raw) if l.strip()]
    for line in lines[1:]:
        p = line.split('\t')
        if len(p) < 4: continue
        drug1.append(p[0]); drug2.append(p[1]); eob.append(float(p[2])); err.append(float(p[3]))
    eob = np.array(eob); err = np.array(err)
    eob = eob * -1.0                      # script negates EoB
    idx = np.argsort(eob, kind='stable')  # ascending
    eob, err = eob[idx], err[idx]
    pairs = [canon(drug1[i], drug2[i]) for i in idx]
    return pairs, eob, err

def probability_matrix(x, xstd):
    x = x.reshape(-1,1); n = len(x)
    X = x - x.T
    S = np.sqrt(xstd.reshape(-1,1)**2 + xstd.reshape(1,-1)**2)
    return 0.5*(1 + np.vectorize(erf)(X/S))

def concordance(x, y, p_matrix):
    x = np.asarray(x,float).reshape(-1,1); y = np.asarray(y,float).reshape(-1,1)
    n = len(x)
    X = x - x.T; Y = y - y.T
    C = (np.sign(X) == np.sign(Y)).astype(float)
    pT = p_matrix.T
    C = C*(1-pT) + (1-C)*pT
    tril = np.tril(C, -1)
    return tril.sum()/n/(n-1)*2

def read_pred(path):
    """Returns dict canonical_pair -> rank (rank 1 = most synergistic)."""
    d = {}
    with open(path, encoding="latin-1") as f:
        raw = f.read()
    lines = [l for l in re.split(r'[\r\n]+', raw) if l.strip()]
    for line in lines[1:]:
        parts = line.split(',')
        if len(parts) < 2: continue
        combo, rank = parts[0], parts[1]
        combo = combo.replace(' & ', '_')
        a = combo.split('_')
        if len(a) != 2: continue
        d[canon(a[0], a[1])] = int(float(rank))
    return d

def score(pred_path, gold_pairs, eob, err, p_matrix, n_perm=10000, seed=0):
    pred = read_pred(pred_path)
    ranks_pred = np.array([pred[p] for p in gold_pairs], float)
    gold_ranks = np.arange(1, len(gold_pairs)+1, dtype=float)
    wci = concordance(ranks_pred, gold_ranks, p_matrix)
    rng = np.random.default_rng(seed)
    null = np.array([concordance(rng.permutation(len(gold_pairs))+1, gold_ranks, p_matrix)
                     for _ in range(n_perm)])
    pv = np.mean(null >= wci)
    return wci, pv

if __name__ == "__main__":
    gold_pairs, eob, err = read_gold()
    pmat = probability_matrix(eob, err)
    print(f"Gold standard: {len(gold_pairs)} pairs. Most synergistic: {gold_pairs[0]}")
    results = []
    for f in sorted(glob.glob(os.path.join(_HERE, "data", "Team Predictions", "Rank*.csv")),
                    key=lambda s:int(re.search(r'Rank(\d+)',s).group(1))):
        wci, pv = score(f, gold_pairs, eob, err, pmat, n_perm=2000)
        results.append((os.path.basename(f), wci, pv))
    results.sort(key=lambda r:-r[1])
    print(f"\n{'file':12s}{'pc-index':>10s}{'p-value':>10s}")
    for name, wci, pv in results:
        print(f"{name:12s}{wci:10.4f}{pv:10.4f}")
    print(f"\nBest pc-index = {results[0][1]:.4f} (expected DIGRE/winner ~0.613)")
    print(f"Worst pc-index = {results[-1][1]:.4f} (expected ~0.42)")

# ---------------------------------------------------------------------------
# Public convenience API (use this on PAIRWISE / baseline outputs)
# ---------------------------------------------------------------------------
def evaluate(pred_csv, gold_path=GOLD, n_perm=10000, seed=0):
    """Score one prediction CSV ('Drug A & Drug B,rank' rows, rank1=most synergistic).
    Returns (pc_index, p_value). Metric identical to the official DREAM script."""
    gold_pairs, eob, err = read_gold(gold_path)
    pmat = probability_matrix(eob, err)
    return score(pred_csv, gold_pairs, eob, err, pmat, n_perm=n_perm, seed=seed)
