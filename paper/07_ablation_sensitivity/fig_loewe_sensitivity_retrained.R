#!/usr/bin/env Rscript
# ---- 0. Dependencies (explicit; avoids meta-package version drift) ----------
suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(scales)
})

# ---- 1. Configuration -------------------------------------------------------
# 1a. Canonical model identifiers, display labels, and a FIXED plotting order
#     (PAIRWISE first; baselines by descending overall AUROC). Fixing the order
#     here — rather than deriving it at run time — keeps the figure byte-stable.
MODEL_LEVELS <- c("pairwise", "transynergy_liu", "multitaskdnn_kim",
                  "deepdds_wang", "deepsynergy_preuer", "matchmaker_brahim")
MODEL_LABELS <- c(
  pairwise           = "PAIRWISE",
  transynergy_liu    = "TranSynergy",
  multitaskdnn_kim   = "MultiTaskDNN",
  deepdds_wang       = "DeepDDS",
  deepsynergy_preuer = "DeepSynergy",
  matchmaker_brahim  = "MatchMaker"
)
# Colourblind-safe palette; PAIRWISE emphasised in deep red.
MODEL_COLORS <- c(
  pairwise           = "#C0392B",  # deep red  (emphasis)
  transynergy_liu    = "#1F4E79",  # dark blue
  multitaskdnn_kim   = "#2E8B57",  # sea green
  deepdds_wang       = "#7FB2E5",  # light blue
  deepsynergy_preuer = "#E8A33D",  # orange
  matchmaker_brahim  = "#8E5FA8"   # purple
)
THRESHOLDS <- c(0, 5, 10, 20)

# 1b. Aesthetic constants (single source of truth for the look)
PT_BASE      <- 2.4    # baseline point size
PT_PAIRWISE  <- 3.3    # PAIRWISE point size (diamond)
STROKE_PW    <- 0.9    # PAIRWISE outline width
LINE_TRAJ    <- 0.45   # per-model trajectory line
EB_WIDTH     <- 0.16   # CI whisker width (x-units)
EB_LINEWIDTH <- 0.5
DODGE        <- 0.55   # horizontal separation between models at each threshold
LAB_SIZE     <- 2.5    # PAIRWISE value labels
BASE_SIZE    <- 12     # base font size
FONT         <- "sans"
DPI          <- 600    # line-art resolution for the raster copy

# ---- 2. Resolve input / output paths ----------------------------------------
args    <- commandArgs(trailingOnly = TRUE)
in_arg  <- if (length(args) >= 1) args[[1]] else NA_character_
out_arg <- if (length(args) >= 2) args[[2]] else NA_character_

find_input <- function(explicit) {
  if (!is.na(explicit) && file.exists(explicit)) return(normalizePath(explicit))
  candidates <- c(
    Sys.getenv("LOEWE_SENS_CSV", unset = NA_character_),
    "results/loewe_sensitivity_retrained.csv",
    "loewe_sensitivity_supplementary.csv",
    "/mnt/user-data/uploads/loewe_sensitivity_supplementary.csv"
  )
  hit <- candidates[!is.na(candidates) & file.exists(candidates)]
  if (length(hit) == 0)
    stop("Could not locate the results CSV. Pass it as the first argument, or ",
         "set $LOEWE_SENS_CSV.", call. = FALSE)
  normalizePath(hit[[1]])
}

IN_CSV  <- find_input(in_arg)
OUT_DIR <- if (!is.na(out_arg)) out_arg else file.path(dirname(IN_CSV), "figures")
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

message("Input : ", IN_CSV)
message("MD5   : ", unname(tools::md5sum(IN_CSV)))
message("Output: ", normalizePath(OUT_DIR))

# ---- 3. Read and validate ---------------------------------------------------
# Fail loudly on any schema/consistency problem: a figure script that silently
# skip incomplete rows rather than plotting NAs.
validate <- function(df) {
  required <- c("model", "threshold", "AUROC", "CI_lo_fold", "CI_hi_fold",
                "AUPRC", "AUPRC_lo", "AUPRC_hi", "n_pos", "n_total")
  missing <- setdiff(required, names(df))
  if (length(missing))
    stop("Missing required column(s): ", paste(missing, collapse = ", "),
         call. = FALSE)

  if (!setequal(unique(df$model), MODEL_LEVELS))
    stop("Model set mismatch.\n  expected: ", paste(MODEL_LEVELS, collapse = ", "),
         "\n  found   : ", paste(sort(unique(df$model)), collapse = ", "),
         call. = FALSE)

  if (!setequal(unique(df$threshold), THRESHOLDS))
    stop("Threshold set mismatch. expected {", paste(THRESHOLDS, collapse = ","),
         "}, found {", paste(sort(unique(df$threshold)), collapse = ","), "}",
         call. = FALSE)

  if (nrow(df) != length(MODEL_LEVELS) * length(THRESHOLDS))
    stop("Expected ", length(MODEL_LEVELS) * length(THRESHOLDS),
         " rows (models x thresholds), found ", nrow(df), ".", call. = FALSE)

  key_cols <- c("AUROC", "CI_lo_fold", "CI_hi_fold",
                "AUPRC", "AUPRC_lo", "AUPRC_hi", "n_pos", "n_total")
  if (any(is.na(df[key_cols])))
    stop("NA present in a plotted column; refuse to render.", call. = FALSE)

  if (length(unique(df$n_total)) != 1)
    stop("n_total is not constant across rows: ",
         paste(unique(df$n_total), collapse = ", "), call. = FALSE)

  bad_ci <- with(df, AUROC < CI_lo_fold - 1e-9 | AUROC > CI_hi_fold + 1e-9 |
                     AUPRC < AUPRC_lo - 1e-9 | AUPRC > AUPRC_hi + 1e-9)
  if (any(bad_ci))
    stop("Point estimate outside its CI on ", sum(bad_ci), " row(s).",
         call. = FALSE)

  rng <- range(c(df$AUROC, df$AUPRC, df$CI_lo_fold, df$CI_hi_fold,
                 df$AUPRC_lo, df$AUPRC_hi))
  if (rng[1] < 0 || rng[2] > 1)
    stop("A metric or CI bound falls outside [0, 1].", call. = FALSE)

  invisible(df)
}

raw <- read_csv(IN_CSV, show_col_types = FALSE) |>
  mutate(threshold = as.numeric(threshold))
validate(raw)

N_TOTAL <- unique(raw$n_total)

# Prevalence per threshold (n_pos is a property of the label rule on the full
# set, identical across models) — used for the x-axis annotation and Fig 2.
prevalence <- raw |>
  distinct(threshold, n_pos, n_total) |>
  mutate(prev_pct = 100 * n_pos / n_total) |>
  arrange(threshold)

# Threshold axis labels carry prevalence so the reader always sees that the
# strictest cut-off is a 1.1% positive class (where AUROC saturates).
thr_labels <- setNames(
  sprintf("%d\n(%.1f%%)", prevalence$threshold, prevalence$prev_pct),
  as.character(prevalence$threshold)
)

# ---- 4. Long ("tidy") table = Source Data -----------------------------------
# One row per (model, threshold, metric) with point estimate and 95% CI.
#   AUROC CI : test +/- 1.96 * SE_fold  (5-fold CV; matches Fig_model_comparison)
#   AUPRC CI : nonparametric bootstrap
metric_levels <- c("AUROC", "AUPRC")
plot_df <- bind_rows(
  raw |> transmute(model, threshold, metric = "AUROC",
                   point = AUROC, ci_lo = CI_lo_fold, ci_hi = CI_hi_fold),
  raw |> transmute(model, threshold, metric = "AUPRC",
                   point = AUPRC, ci_lo = AUPRC_lo,   ci_hi = AUPRC_hi)
) |>
  mutate(
    model     = factor(model, levels = MODEL_LEVELS),
    metric    = factor(metric, levels = metric_levels),
    threshold = factor(threshold, levels = THRESHOLDS)
  ) |>
  arrange(metric, model, threshold)

readr::write_csv(
  plot_df |>
    mutate(model_label = MODEL_LABELS[as.character(model)]) |>
    relocate(model_label, .after = model),
  file.path(OUT_DIR, "SourceData_Fig_loewe_sensitivity.csv")
)

# ---- 5. Machine-generated numerical summary ---------------------------------
# PAIRWISE vs the best BASELINE (non-PAIRWISE) at each threshold, for each
# metric, with a CI-separation flag (do the two 95% CIs overlap?). These are the
# numbers to quote in the response letter — generated here so the text can never
# drift from the figure.
summary_tbl <- plot_df |>
  group_by(metric, threshold) |>
  summarise(
    pairwise      = point[model == "pairwise"],
    pw_lo         = ci_lo[model == "pairwise"],
    pw_hi         = ci_hi[model == "pairwise"],
    best_base     = MODEL_LABELS[as.character(
                      model[model != "pairwise"][which.max(point[model != "pairwise"])])],
    best_base_val = max(point[model != "pairwise"]),
    bb_lo         = ci_lo[model != "pairwise"][which.max(point[model != "pairwise"])],
    bb_hi         = ci_hi[model != "pairwise"][which.max(point[model != "pairwise"])],
    .groups = "drop"
  ) |>
  mutate(
    delta         = pairwise - best_base_val,
    leader        = ifelse(delta >= 0, "PAIRWISE", best_base),
    ci_separated  = pw_lo > bb_hi | bb_lo > pw_hi
  ) |>
  arrange(metric, threshold)

readr::write_csv(summary_tbl, file.path(OUT_DIR, "Summary_loewe_sensitivity.csv"))

cat("\n================ PAIRWISE vs best baseline (auto-generated) ============\n")
summary_tbl |>
  mutate(across(c(pairwise, best_base_val, delta), ~sprintf("%+.4f", .x))) |>
  transmute(metric, threshold,
            PAIRWISE = pairwise, best_baseline = best_base,
            baseline_val = best_base_val, delta,
            leader, CI = ifelse(ci_separated, "separated", "overlap")) |>
  as.data.frame() |>
  print(row.names = FALSE)
cat("=======================================================================\n\n")

# ---- 6. Figure 1: AUROC & AUPRC vs Loewe threshold --------------------------
pd  <- position_dodge(width = DODGE)
pw  <- plot_df |> filter(model == "pairwise")
bl  <- plot_df |> filter(model != "pairwise")

p_main <- ggplot(plot_df,
                 aes(threshold, point, colour = model, group = model)) +
  # per-model trajectory (thin; makes the cross-threshold trend legible)
  geom_line(position = pd, linewidth = LINE_TRAJ, alpha = 0.55) +
  # 95% CI
  geom_errorbar(aes(ymin = ci_lo, ymax = ci_hi),
                position = pd, width = EB_WIDTH, linewidth = EB_LINEWIDTH) +
  # baselines: solid circles
  geom_point(data = bl, position = pd, size = PT_BASE, shape = 19) +
  # PAIRWISE: red diamond, black outline, larger (emphasis)
  geom_point(data = pw, position = pd, size = PT_PAIRWISE, shape = 23,
             fill = MODEL_COLORS[["pairwise"]], colour = "black",
             stroke = STROKE_PW) +
  # value labels for PAIRWISE only (keeps panels uncluttered)
  geom_text(data = pw, aes(y = ci_hi, label = sprintf("%.3f", point)),
            position = pd, vjust = -0.7, size = LAB_SIZE, colour = "black",
            show.legend = FALSE) +
  facet_wrap(~ metric, ncol = 2, scales = "free_y") +
  scale_x_discrete(labels = thr_labels) +
  scale_colour_manual(values = MODEL_COLORS, labels = MODEL_LABELS,
                      breaks = MODEL_LEVELS, name = NULL) +
  scale_y_continuous(expand = expansion(mult = c(0.06, 0.16))) +
  labs(
    title    = "Loewe-threshold sensitivity of drug-synergy classification",
    subtitle = paste0(
      "Points, held-out test metric after retraining at each cut-off. ",
      "AUROC CIs, test \u00B1 1.96\u00B7SE(5-fold CV); AUPRC CIs, bootstrap. ",
      "x-axis shows positive-class prevalence."),
    x = "Loewe synergy threshold  (positive if score > t)",
    y = NULL
  ) +
  theme_bw(base_size = BASE_SIZE, base_family = FONT) +
  theme(
    axis.text        = element_text(colour = "black"),
    panel.grid.major.x = element_blank(),
    panel.grid.minor   = element_blank(),
    panel.grid.major.y = element_line(colour = "grey90", linewidth = 0.3),
    strip.background = element_rect(fill = "grey92", colour = "grey70"),
    strip.text       = element_text(face = "bold", size = BASE_SIZE),
    legend.position  = "bottom",
    legend.key.width = unit(1.1, "lines"),
    plot.title       = element_text(face = "bold", hjust = 0.5),
    plot.subtitle    = element_text(hjust = 0.5, size = rel(0.72),
                                    colour = "grey30", lineheight = 1.05),
    plot.margin      = margin(10, 14, 6, 10)
  ) +
  guides(colour = guide_legend(nrow = 1, override.aes = list(
    linewidth = 0, size = 2.6, shape = 15)))

ggsave(file.path(OUT_DIR, "Fig_loewe_sensitivity.pdf"), p_main,
       width = 10, height = 5.2, device = cairo_pdf)
ggsave(file.path(OUT_DIR, "Fig_loewe_sensitivity.png"), p_main,
       width = 10, height = 5.2, dpi = DPI)

# ---- 7. Figure 2: positive-class prevalence ---------------------------------
prev_plot <- prevalence |>
  mutate(threshold = factor(threshold, levels = THRESHOLDS))

p_prev <- ggplot(prev_plot, aes(threshold, n_pos)) +
  geom_col(fill = "grey78", colour = "black", linewidth = 0.3, width = 0.62) +
  geom_text(aes(label = sprintf("%s\n(%.1f%%)",
                                 format(n_pos, big.mark = ","), prev_pct)),
            vjust = -0.35, size = 3.1, lineheight = 0.9) +
  scale_y_continuous(
    expand = expansion(mult = c(0, 0.18)),
    labels = label_number(scale_cut = cut_short_scale())
  ) +
  labs(
    title    = "Positive-class prevalence per Loewe threshold",
    subtitle = sprintf("Labels re-derived on the full dataset (n = %s pairs)",
                       format(N_TOTAL, big.mark = ",")),
    x = "Loewe synergy threshold",
    y = "Positive samples (n)"
  ) +
  theme_bw(base_size = BASE_SIZE, base_family = FONT) +
  theme(
    axis.text          = element_text(colour = "black"),
    panel.grid.major.x = element_blank(),
    panel.grid.minor   = element_blank(),
    plot.title         = element_text(face = "bold", hjust = 0.5),
    plot.subtitle      = element_text(hjust = 0.5, size = rel(0.75),
                                      colour = "grey30"),
    plot.margin        = margin(10, 14, 8, 10)
  )

ggsave(file.path(OUT_DIR, "Fig_loewe_sensitivity_prevalence.pdf"), p_prev,
       width = 7.2, height = 4.6, device = cairo_pdf)
ggsave(file.path(OUT_DIR, "Fig_loewe_sensitivity_prevalence.png"), p_prev,
       width = 7.2, height = 4.6, dpi = DPI)

# ---- 8. Provenance ----------------------------------------------------------
writeLines(
  c(paste0("Generated : ", format(Sys.time(), tz = "UTC", usetz = TRUE)),
    paste0("Input CSV : ", IN_CSV),
    paste0("Input MD5 : ", unname(tools::md5sum(IN_CSV))),
    "", capture.output(sessionInfo())),
  file.path(OUT_DIR, "sessionInfo_loewe_sensitivity.txt")
)

message("Done. Wrote figures, Source Data, summary, and sessionInfo to:\n  ",
        normalizePath(OUT_DIR))