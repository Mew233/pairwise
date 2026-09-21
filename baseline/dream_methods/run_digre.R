#!/usr/bin/env Rscript
# DIGRE on DREAM OCI-LY3.
#
#   Rscript run_digre.R legacy     # zero-shot ranking CSV (~pc-index 0.613)
#   Rscript digre_native.R 0.6     # native scores for Python semi-sup (preferred)
#
# Semi-supervised calibration: python run_semisupervised.py

args <- commandArgs(trailingOnly = TRUE)

if (length(args) >= 1 && args[1] == "legacy") {
  if (!requireNamespace("DIGREsyn", quietly = TRUE)) {
    cat("DIGREsyn not installed.\n")
    quit(status = 1)
  }
  library(DIGREsyn)
  geneExpDiff <- profileGeneExp(geneExp = geneExp.demo)
  geneNetLymph.mat <- constGeneNet(geneNet = geneNetLymph)
  pred <- DIGREscore(geneExpDiff = geneExpDiff, doseRes = doseRes.demo,
                     pathway = "GeneNet", geneNet = geneNetLymph.mat, fold = 0.6)
  sr <- pred$scoreRank
  sr <- sr[order(sr$Rank), ]
  out <- data.frame(`Drug Combination` = paste(sr$drugA, sr$drugB, sep = " & "),
                    Rank = sr$Rank, check.names = FALSE)
  write.csv(out, "digre_OCI-LY3_ranking.csv", row.names = FALSE, quote = FALSE)
  cat("Wrote digre_OCI-LY3_ranking.csv\n")
} else {
  cat("Usage: Rscript run_digre.R legacy\n")
  cat("For semi-sup native scores: Rscript digre_native.R <fold>\n")
  cat("Full pipeline: python run_semisupervised.py\n")
}
