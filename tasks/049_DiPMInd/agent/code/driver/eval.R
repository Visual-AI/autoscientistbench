#!/usr/bin/env Rscript
# DiPMInd (Sta-19) -- evaluation of the main experiment:
# empirical size (rho = 0) and power of the six proposed tests, from the
# per-replicate permutation p-values written by run_train.sh, next to the values
# plotted in Figure 1 of the paper (read off the vector PDF).
ARGS <- commandArgs(trailingOnly = TRUE)
root <- if (length(ARGS)) ARGS[1] else "."
setwd(root)

VARIANTS <- c("jt", "jt-AD", "jt-F", "prod", "prod-AD", "prod-F")
ALPHA <- 0.05

fix_names <- function(x) {
  nm <- names(x)
  nm[nm == "jt.AD"] <- "jt-AD"; nm[nm == "jt.F"] <- "jt-F"
  nm[nm == "prod.AD"] <- "prod-AD"; nm[nm == "prod.F"] <- "prod-F"
  names(x) <- nm
  x
}
d <- fix_names(read.csv("results/all_pvalues.csv", check.names = FALSE))
paper <- fix_names(read.csv("results/paper_figure1.csv", check.names = FALSE))

keys <- unique(sprintf("model%d_rho%.2f", d$model, d$rho))
keys <- keys[order(keys)]
sizepow <- function(df) sapply(VARIANTS, function(v) mean(df[[v]] <= ALPHA))

mat <- t(sapply(keys, function(k) {
  m <- as.integer(sub("model([0-9]+)_.*", "\\1", k)); r <- as.numeric(sub(".*_rho", "", k))
  sizepow(d[d$model == m & abs(d$rho - r) < 1e-9, ]) }))
pmat <- t(sapply(keys, function(k) {
  m <- as.integer(sub("model([0-9]+)_.*", "\\1", k)); r <- as.numeric(sub(".*_rho", "", k))
  sapply(VARIANTS, function(v) {
    s <- paper[paper$model == m & paper$method == v & abs(paper$rho - r) < 1e-6, "power"]
    if (!length(s)) NA_real_ else s }) }))
rownames(mat) <- rownames(pmat) <- keys
nrep <- nrow(d) / length(keys)

cat(sprintf("\nDiPMInd tests, n = 100, B = 500 permutations, alpha = 0.05, %d Monte-Carlo runs per setting\n", nrep))
cat(sprintf("%-14s %s   |  paper Figure 1\n", "setting", paste(sprintf("%8s", VARIANTS), collapse = "")))
for (k in keys) {
  cat(sprintf("%-14s %s   | %s\n", k,
              paste(sprintf("%8.3f", mat[k, ]), collapse = ""),
              paste(sprintf("%8.3f", pmat[k, ]), collapse = "")))
}
cat(sprintf("\nmean |reproduced - paper| over the %d table entries: %.4f\n",
            length(mat), mean(abs(mat - pmat), na.rm = TRUE)))

out <- data.frame(setting = keys, round(mat, 4), check.names = FALSE)
write.csv(out, "results/size_power_reproduced.csv", row.names = FALSE)
write.csv(data.frame(setting = keys, round(pmat, 4), check.names = FALSE),
          "results/size_power_paper.csv", row.names = FALSE)

# headline metric (printed here, saved to results/metrics.json by run_eval.sh):
hl_key <- "model1_rho0.80"; hl_var <- "prod-AD"
cat(sprintf("\nheadline metric: power of %s at rho = 0.8, Example 1 Model 1\n", hl_var))
cat(sprintf("  paper Figure 1 (p.15, left panel): %.3f\n", pmat[hl_key, hl_var]))
cat(sprintf("  reproduced here                 : %.3f  (%d Monte-Carlo runs, SE %.3f)\n",
            mat[hl_key, hl_var], nrep, sqrt(mat[hl_key, hl_var] * (1 - mat[hl_key, hl_var]) / nrep)))
cat(sprintf("  relative gap                    : %+.1f%%\n",
            100 * (mat[hl_key, hl_var] - pmat[hl_key, hl_var]) / pmat[hl_key, hl_var]))
cat("\nempirical size at rho = 0 (mean over the six tests)\n")
cat(sprintf("  Model 1: reproduced %.3f  paper %.3f\n",
            mean(mat["model1_rho0.00", ]), mean(pmat["model1_rho0.00", ])))
cat(sprintf("  Model 2: reproduced %.3f  paper %.3f\n",
            mean(mat["model2_rho0.00", ]), mean(pmat["model2_rho0.00", ])))
