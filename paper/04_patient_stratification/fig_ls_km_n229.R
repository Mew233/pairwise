#!/usr/bin/env Rscript
# LS cohort — KM (PFS) on the FULL leakage-free held-out set (n=229).
#
# Labels: 5-fold CV OOF 50-gene LPS (run_heldout_survival.py).
# Each of the 5 seeds drawn as a thin dashed KM; the consensus (majority vote
# across 5 seeds) drawn as the bold solid KM on top. Consensus log-rank p
# annotated on the plot.
#
# Usage:
#   Rscript fig_ls_km_n229.R

suppressPackageStartupMessages({
  library(dplyr)
  library(survival)
  library(survminer)
  library(ggplot2)
  library(gridExtra)
  library(grid)
})

ROOT <- if (dir.exists("results")) "." else "."
SURV <- file.path(ROOT, "results", "survival")
OUT  <- file.path(SURV, "figures")
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)

SEEDS <- paste0("seed", 1:5)
POS <- "Eff"
NEG <- "Noneff"
POS_LAB <- "BTKiCombo(+)"
NEG_LAB <- "BTKiCombo(-)"

build_meta <- function() {
  meta <- read.csv(file.path(SURV, "heldout_patient_metadata.csv"),
                   check.names = FALSE, stringsAsFactors = FALSE)
  lab  <- read.csv(file.path(SURV, "heldout_oof_labels.csv"),
                   check.names = FALSE, stringsAsFactors = FALSE)
  meta <- meta %>%
    inner_join(lab[, c("sample", SEEDS)], by = c("subject" = "sample")) %>%
    filter(!is.na(predicted_value),
           !is.na(pfs_time), pfs_time > 0, !is.na(pfs_event))
  meta
}

km_step_df <- function(meta, label_col, group) {
  sub <- meta[!is.na(meta[[label_col]]) & meta[[label_col]] == group, ]
  fit <- survfit(Surv(pfs_time, pfs_event) ~ 1, data = sub)
  t <- c(0, fit$time)
  sf <- summary(fit, times = t, extend = TRUE)$surv
  nr <- summary(fit, times = t, extend = TRUE)$n.risk
  data.frame(
    label_col = label_col,
    group     = group,
    time      = t,
    survival  = sf,
    at_risk   = nr,
    stringsAsFactors = FALSE
  )
}

collect_step_df <- function(meta) {
  rows <- list()
  # Per-seed (5 dashed KMs per group = 10 lines)
  for (s in SEEDS) {
    for (g in c(POS, NEG)) {
      rows[[length(rows) + 1]] <- km_step_df(meta, s, g)
    }
  }
  # Consensus (solid bold)
  for (g in c(POS, NEG)) {
    rows[[length(rows) + 1]] <- km_step_df(meta, "predicted_value", g)
  }
  d <- do.call(rbind, rows)
  # Pretty labels
  d$group_lab <- ifelse(d$group == POS, POS_LAB, NEG_LAB)
  d$source <- ifelse(d$label_col == "predicted_value", "Consensus",
                     paste0("Seed ", sub("seed", "", d$label_col)))
  d$linetype <- ifelse(d$source == "Consensus", "solid", "dashed")
  d
}

# Number-at-risk table (consensus labels) at fixed time points
risk_table_df <- function(meta, time_points) {
  out <- list()
  for (g in c(POS, NEG)) {
    sub <- meta[meta$predicted_value == g, ]
    fit <- survfit(Surv(pfs_time, pfs_event) ~ 1, data = sub)
    nr  <- summary(fit, times = time_points, extend = TRUE)$n.risk
    out[[length(out) + 1]] <- data.frame(
      group_lab = if (g == POS) POS_LAB else NEG_LAB,
      time = time_points,
      n.risk = nr,
      stringsAsFactors = FALSE
    )
  }
  do.call(rbind, out)
}

main <- function() {
  meta <- build_meta()
  cat(sprintf(
    "consensus n=%d  (+)=%d  (-)=%d  events=%d  (+events)=%d  (-events)=%d\n",
    nrow(meta),
    sum(meta$predicted_value == POS), sum(meta$predicted_value == NEG),
    sum(meta$pfs_event),
    sum(meta$pfs_event[meta$predicted_value == POS]),
    sum(meta$pfs_event[meta$predicted_value == NEG])
  ))

  fit <- survfit(Surv(pfs_time, pfs_event) ~ predicted_value, data = meta)
  lr  <- survdiff(Surv(pfs_time, pfs_event) ~ predicted_value, data = meta)
  p_lr <- 1 - pchisq(lr$chisq, length(lr$n) - 1)
  med  <- quantile(fit, probs = 0.5)$quantile
  cat(sprintf("consensus log-rank p = %.4g  (chi-sq = %.3f)\n", p_lr, lr$chisq))
  cat("Median PFS (years):\n"); print(med)

  cat("\nper-seed log-rank p:\n")
  for (s in SEEDS) {
    sub <- meta[!is.na(meta[[s]]), ]
    lr1 <- survdiff(Surv(pfs_time, pfs_event) ~ meta[[s]], data = sub)
    cat(sprintf("  %s: n=%d  (+)=%d  (-)=%d  p=%.4g\n", s, nrow(sub),
                sum(sub[[s]] == POS), sum(sub[[s]] == NEG),
                1 - pchisq(lr1$chisq, 1)))
  }

  # Export consensus KM CSV + logrank CSV
  km_consensus <- collect_step_df(meta)
  km_consensus_cons <- km_consensus[km_consensus$source == "Consensus", ]
  write.csv(km_consensus_cons[, c("time", "group", "survival", "at_risk")],
            file.path(SURV, "ls_lps_cv_km_n229.csv"), row.names = FALSE)
  write.csv(km_consensus,
            file.path(SURV, "ls_lps_cv_km_n229_allseeds.csv"),
            row.names = FALSE)
  write.csv(
    data.frame(
      logrank_p = p_lr,
      n = nrow(meta),
      n_pos = sum(meta$predicted_value == POS),
      n_neg = sum(meta$predicted_value == NEG),
      n_events = sum(meta$pfs_event),
      method = "5fold_cv_oof_consensus_full_set",
      subset = "all_patients_with_PFS"
    ),
    file.path(SURV, "ls_lps_cv_logrank_n229.csv"), row.names = FALSE
  )

  # Plot: ggsurvplot on consensus (with native risk table) + overlay 5 dashed seed KMs
  pval_lab <- if (p_lr < 1e-3) {
    e <- floor(log10(p_lr)); mm <- p_lr / 10^e
    sprintf("Log-rank p = %.1f \u00d7 10^{%d}", mm, e)
  } else {
    sprintf("Log-rank p = %.3f", p_lr)
  }

  # Seed step-line data for overlay (thin dashed)
  seed_overlay <- do.call(rbind, lapply(SEEDS, function(s) {
    do.call(rbind, lapply(c(POS, NEG), function(g) {
      d <- km_step_df(meta, s, g)
      d$group_lab <- if (d$group[1] == POS) POS_LAB else NEG_LAB
      d$seed <- s
      d
    }))
  }))

  gp <- ggsurvplot(
    fit, data = meta,
    pval = FALSE,
    conf.int = FALSE,
    risk.table = TRUE,
    risk.table.col = "strata",
    risk.table.y.text = FALSE,
    palette = c("#E8A33D", "#8C8C8C"),
    size = 1.2,
    legend.title = "",
    legend.labs = c("BTKiCombo(+)", "BTKiCombo(-)"),
    legend = c(0.78, 0.88),
    xlab = "Progression-free survival (years)",
    ylab = "PFS probability",
    title = NULL,
    break.time.by = 2,
    ggtheme = theme_bw(base_size = 10) +
      theme(
        plot.title = element_text(face = "bold", hjust = 0),
        panel.grid.minor = element_blank()
      )
  )

  # Overlay 5 dashed seed KMs on top of the consensus solid line
  gp$plot <- gp$plot +
    geom_step(
      data = seed_overlay,
      aes(x = time, y = survival,
          color = group_lab, group = interaction(seed, group_lab)),
      linetype = "dashed", linewidth = 0.4, alpha = 0.45,
      inherit.aes = FALSE
    ) +
    labs(
      title = "NIH Lymphoma/Leukemia cohort",
      subtitle = paste0(
        "5-fold CV OOF labels; thin dashed = each seed, bold solid = consensus; n = ",
        nrow(meta)
      )
    ) +
    annotate(
      "text", x = max(meta$pfs_time) * 0.02, y = 0.18,
      label = pval_lab,
      hjust = 0, vjust = 0, size = 3.8, fontface = "italic"
    )

  out_base <- file.path(OUT, "Fig_LS_n229_5fold")
  km_grob <- arrangeGrob(gp$plot, gp$table,
                         heights = unit(c(0.75, 0.25), "npc"))
  ggsave(paste0(out_base, ".png"), km_grob, width = 6, height = 5,
         dpi = 300, bg = "white")
  ggsave(paste0(out_base, ".pdf"), km_grob, width = 6, height = 5,
         bg = "white")

  cat(sprintf(
    "\nWrote:\n  %s.png\n  %s.pdf\n  %s\n  %s\n  %s\n",
    out_base, out_base,
    file.path(SURV, "ls_lps_cv_km_n229.csv"),
    file.path(SURV, "ls_lps_cv_km_n229_allseeds.csv"),
    file.path(SURV, "ls_lps_cv_logrank_n229.csv")
  ))
}

main()
