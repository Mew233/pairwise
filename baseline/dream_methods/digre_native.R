#!/usr/bin/env Rscript
# Export DIGRE native pair scores at a given fold for the semi-supervised pipeline.
#
# This script reproduces the official NCI-DREAM winning DIGRE predictions on the
# OCI-LY3 14-drug / 91-pair benchmark using the DIGREsyn R package. Because
# installing KEGGgraph (and its graphviz system dependency) is non-trivial, we
# source the DIGRE algorithm directly from a checkout of
# https://github.com/Minzhe/DIGREsyn and stub the single KEGGgraph helper DIGRE
# actually calls (translateGeneID2KEGGID, which is a one-line paste()). All
# other dependencies (org.Hs.eg.db, preprocessCore, AnnotationDbi, BiocGenerics)
# are required and must be installed.
args <- commandArgs(trailingOnly = TRUE)
fold <- if (length(args) >= 1) as.numeric(args[1]) else 0.6
DIGRESYN_PATH <- if (length(args) >= 2) args[2] else "/tmp/DIGREsyn"

# --- 1. Verify the algorithm source is available -----------------------------
if (!file.exists(file.path(DIGRESYN_PATH, "R", "DIGREsyn.R"))) {
  message("DIGREsyn source not found at ", DIGRESYN_PATH,
          "\nClone it first: git clone https://github.com/Minzhe/DIGREsyn.git")
  quit(status = 1)
}

# --- 2. Stub the only KEGGgraph symbol DIGRE actually uses -------------------
# translateGeneID2KEGGID is a trivial prefix-paste in KEGGgraph (see
# KEGGgraph/R/annotation.R); defining it here avoids the full KEGGgraph + graphviz
# install. Other KEGGgraph functions (layoutGraph, renderGraph, getRgraphvizEdgeNames)
# are only used by DIGRE's plotting helpers, which we do not call.
translateGeneID2KEGGID <- function(x, organism = "hsa") {
  paste(organism, ":", x, sep = "")
}

# --- 3. Source the DIGRE algorithm and load the bundled OCI-LY3 data ---------
# preprocessCore's normalize.quantiles is referenced without a namespace prefix
# inside DIGRE's profileGeneExp; expose it explicitly.
suppressMessages({
  library(preprocessCore)
  library(org.Hs.eg.db)
})
source(file.path(DIGRESYN_PATH, "R", "DIGREsyn.R"))
for (d in c("geneExp.demo", "doseRes.demo", "geneNetLymph", "CGP.mat", "KEGGnet.mat")) {
  load(file.path(DIGRESYN_PATH, "data", paste0(d, ".rda")), .GlobalEnv)
}

# --- 4. Run DIGRE end-to-end --------------------------------------------------
geneExpDiff <- profileGeneExp(geneExp = geneExp.demo)
geneNetLymph.mat <- constGeneNet(geneNet = geneNetLymph)
pred <- DIGREscore(geneExpDiff = geneExpDiff, doseRes = doseRes.demo,
                   pathway = "GeneNet", geneNet = geneNetLymph.mat, fold = fold)
sr <- pred$scoreRank

# --- 5. Write the score table to the location digre_native.py expects --------
# Drug names contain commas (e.g. "H-7, Dihydrochloride"), so quoting is mandatory.
out <- data.frame(A = sr$drugA, B = sr$drugB, score = sr$Score)
if (!dir.exists("results")) dir.create("results", recursive = TRUE)
write.csv(out, "results/digre_native_scores.csv", row.names = FALSE, quote = TRUE)
cat("Wrote", nrow(out), "DIGRE native scores to results/digre_native_scores.csv\n")
