# Benchmark Study: Drug Synergy Prediction on NCI-DREAM SC2 / OCI-LY3

## Overview

Five methods — **PAIRWISE**, **SynGen**, **RACS**, **DrugComboRanker**, **DIGRE** — are compared on the
NCI-DREAM Sub-challenge 2 benchmark: 14 compounds, 91 drug-drug combinations, tested on the OCI-LY-3
DLBCL cell line. The gold standard labels 23/91 pairs as synergistic (EoB > 0).

Each method was evaluated under **two matched protocols**: pair-level 5-fold CV, and leave-one-drug-out,
where the held-out pairs contain a drug the model never encountered during adaptation.

**Key finding:** Under a matched leave-one-drug-out protocol the methods land in a narrow band
(pooled AUROC 0.76-0.80). PAIRWISE ranks first on pc-index, AUROC and AUPRC (0.626 / 0.801 / 0.727,
p = 1e-4), with RACS and SynGen close behind; DIGRE and DrugComboRanker do not separate from chance on
pc-index. Given 91 pairs and 23 positives, the intervals overlap and the leading methods are best
described as comparable.

---

## Protocols

| Protocol | Training signal | Eval | n pairs | Key test |
|----------|----------------|-----|---------|----------|
| Matched L2R | All methods get identical supervised L2R | 5-fold pair CV | ~72/19 | What is each method's representation worth when adapted identically? |
| Leave-one-drug-out (LODO) | Matched L2R, all pairs involving one drug held out | 14 folds | ~78/13 | Can the model generalize to an unseen drug? |

The table below reports the LODO protocol, which is the one the manuscript cites. Note that
the manuscript Methods still describe the earlier two-stage IC20 transfer for this benchmark;
that text needs to be replaced with the matched-adaptation LODO description.

**Scoring:** Official DREAM probabilistic concordance index (pc-index, 10,000 permutations, random = 0.5),
AUROC, AUPRC (classes imbalanced: 23/91 positives).

---

## Results

### Leave-one-drug-out (LODO) — matched L2R, 14 folds

| Method | pc-index | p-value | AUROC | AUPRC |
|--------|---------:|--------:|------:|------:|
| PAIRWISE | **0.626** | 0.0001 | **0.801** | **0.727**  |
| RACS | 0.597 | 0.002 | 0.758 | 0.506 |
| SynGen | 0.588 | 0.005 | 0.778 | 0.665 |
| DIGRE | 0.490 | 0.606 | 0.772 | 0.269 |
| DrugComboRanker | 0.441 | 0.957 | 0.757 | 0.235 |

LODO is the most stringent test — every held-out fold contains pairs with a drug the model never saw
at training, so pair non-independence cannot inflate the score. On the pooled out-of-fold ranking
PAIRWISE has the highest pc-index (0.626), AUROC (0.801) and AUPRC (0.727); RACS and SynGen follow,
while DIGRE and DrugComboRanker do not separate from chance on pc-index (p = 0.61 and 0.96).

With 91 pairs and 23 positives the confidence intervals are wide and overlap between the leading
methods, so PAIRWISE, RACS and SynGen should be read as comparable on this benchmark rather than as a
strict ordering.



## Interpretation

1. **Matched adaptation is what makes the comparison fair.** Every method is given the same
   OCI-LY3 adaptation budget and the same features (IC20 potency, transcriptomic concordance), so the
   remaining differences reflect the representations rather than unequal access to target-domain data.

2. **Prior biological knowledge transfers across cell lines.** DIGRE (gene-network diffusion) and
   SynGen (transcriptomic signature matching) carry mechanism priors that apply to a cell line they
   were not built on; PAIRWISE instead transfers a representation learned from large-scale
   DrugComb/p13 training.

3. **Generalizing to an unseen drug is the binding constraint.** Under LODO every fold scores pairs
   containing a compound absent from adaptation, and all methods lose ground relative to pair-level
   CV — the drop is broad rather than specific to one method, indicating shared sensitivity to
   drug-specific patterns.

4. **The benchmark is small.** 91 pairs and 23 positives put wide, overlapping intervals on every
   estimate; this benchmark supports "comparable to the DREAM-challenge methods", not a claim of a
   decisive margin.
