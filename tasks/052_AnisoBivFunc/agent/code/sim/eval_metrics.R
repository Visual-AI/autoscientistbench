## Headline metric of the reproduction.
##
## Metric (paper eq. (36), Figure 3):
##   R_alpha = |alpha_adj - alpha|,  alpha_adj = corrected angle estimate
##   (estimate_angle -> identify_angle -> angle_correct, i.e. Algorithms 1 and 2
##    of the paper); lower is better.
## Reported by the paper as boxplots of R_alpha per configuration (60
## configurations).  The paper's text states the risk is "below 0.1 for all
## different configurations, even in the extremely high noise setting sigma = 1".
##
## Summaries compared here (paper values are read programmatically from the
## vector PDFs of Figure 3, see code/sim/read_figure3.py):
##   * mean over the 60 configurations of the per-configuration median R_alpha
##     (primary), and the same for the mean;
##   * the largest per-configuration median, and the share of configurations
##     with median < 0.1;
##   * secondary: error of the estimated regularities (H_max ~ H1 = 0.8,
##     H_min ~ H2 = 0.5) and of the noise level.
## Usage: Rscript code/sim/eval_metrics.R [rep_csv] [paper_csv]

suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly = TRUE)
rep_csv <- if (length(args) >= 1) args[1] else "results/raw/replications_final_fast.csv"
paper_csv <- if (length(args) >= 2) args[2] else "results/paper_fig/figure3_values.csv"
stopifnot(file.exists(rep_csv))
if (!file.exists(paper_csv)) {
  ## Only the paper-side comparison is affected; the metric itself is always
  ## computed from the replications produced by run_train.sh.
  warning("paper value table missing; using the values embedded in the paper's ",
          "Figure 3 description. Regenerate it with code/sim/read_figure3.py.")
  paper_csv <- NULL
}

d <- read.csv(rep_csv)
key <- function(N, side, sigma, alpha) sprintf("N%d|%d|%.1f|%.4f", N, side, sigma, round(alpha, 4))
d$key <- key(d$N, d$side, d$sigma, d$alpha)

summ <- do.call(rbind, lapply(split(d, d$key), function(x) {
  data.frame(key = x$key[1], N = x$N[1], side = x$side[1], sigma = x$sigma[1],
             alpha = x$alpha[1], reps = nrow(x),
             median = median(x$risk_adj), mean = mean(x$risk_adj),
             q1 = unname(quantile(x$risk_adj, 0.25)), q3 = unname(quantile(x$risk_adj, 0.75)),
             p05 = unname(quantile(x$risk_adj, 0.05)), p95 = unname(quantile(x$risk_adj, 0.95)),
             max = max(x$risk_adj),
             median_H_max_err = median(x$H_max_err), median_H_min_err = median(x$H_min_err),
             median_sigma_hat = median(x$sigma_hat), stringsAsFactors = FALSE)
}))
summ <- summ[order(summ$N, summ$side, summ$sigma, summ$alpha), ]
if (!is.null(paper_csv)) {
  p <- read.csv(paper_csv)
  p$key <- key(p$n_side, p$side, p$sigma, round(p$alpha, 4))
  cmp <- merge(summ, p[, c("key", "median", "q1", "q3", "whisker_lo", "whisker_hi")],
               by = "key", suffixes = c("_repro", "_paper"))
  stopifnot(nrow(cmp) == nrow(summ))
  cmp <- cmp[order(cmp$N, cmp$side, cmp$sigma, cmp$alpha), ]
  paper_summary <- list(mean_of_config_medians = mean(p$median),
                        max_of_config_medians = max(p$median),
                        max_upper_whisker = max(p$whisker_hi),
                        share_of_configs_median_below_0.1 = mean(p$median < 0.1))
} else {
  cmp <- summ
  paper_summary <- list(mean_of_config_medians = 0.02877, max_of_config_medians = 0.05770,
                        max_upper_whisker = 0.11145, share_of_configs_median_below_0.1 = 1)
}
if (nrow(summ) != 60) warning(sprintf("expected 60 configurations, found %d", nrow(summ)))

m <- list(
  metric = "risk of the estimated anisotropy angle R_alpha = |alpha_adj - alpha| (paper eq. 36)",
  higher_is_better = FALSE,
  n_configurations = nrow(summ),
  reps_per_configuration = as.integer(max(summ$reps)),
  reproduced = list(
    mean_of_config_medians = mean(summ$median),
    mean_of_config_means = mean(summ$mean),
    max_of_config_medians = max(summ$median),
    share_of_configs_median_below_0.1 = mean(summ$median < 0.1),
    median_H_max_err = median(summ$median_H_max_err),
    median_H_min_err = median(summ$median_H_min_err)
  ),
  paper = c(paper_summary, list(
    source = "Figure 3 (p. 16 of the Bernoulli preprint / arXiv:2409.00817), 60 boxplots read from the vector PDFs in paper/src/plots/alpha_adj_*.pdf"
  )),
  source = list(replications_csv = rep_csv, paper_values_csv = paper_csv)
)
m$relative_gap <- (m$reproduced$mean_of_config_medians - m$paper$mean_of_config_medians) /
  m$paper$mean_of_config_medians
m$per_configuration <- cmp

cat("\n================ HEADLINE METRIC (lower is better) ================\n")
cat(sprintf("mean over the 60 configurations of the median R_alpha\n"))
cat(sprintf("  paper (Figure 3, read from vector PDFs) : %.5f\n", m$paper$mean_of_config_medians))
cat(sprintf("  reproduced (this run, %d reps/config)   : %.5f\n",
            m$reps_per_configuration, m$reproduced$mean_of_config_medians))
cat(sprintf("  relative gap                            : %+.2f%%\n", 100 * m$relative_gap))
cat(sprintf("largest per-configuration median R_alpha\n"))
cat(sprintf("  paper %.5f | reproduced %.5f\n",
            m$paper$max_of_config_medians, m$reproduced$max_of_config_medians))
cat(sprintf("share of configurations with median < 0.1: paper %.0f%% | reproduced %.0f%%\n",
            100 * m$paper$share_of_configs_median_below_0.1,
            100 * m$reproduced$share_of_configs_median_below_0.1))
if (is.null(paper_csv)) cat("(paper values: embedded fallback; run code/sim/read_figure3.py for the full table)\n")
cat(sprintf("secondary: median |H_max_hat - 0.8| = %.4f, median |H_min_hat - 0.5| = %.4f\n",
            m$reproduced$median_H_max_err, m$reproduced$median_H_min_err))
cat("===================================================================\n\n")

dir.create("results", showWarnings = FALSE)
write.csv(cmp, "results/per_config_metrics.csv", row.names = FALSE)
jsonlite::write_json(m, "results/metrics.json", pretty = TRUE, digits = 8,
                     auto_unbox = TRUE)
cat("wrote results/metrics.json and results/per_config_metrics.csv\n")
