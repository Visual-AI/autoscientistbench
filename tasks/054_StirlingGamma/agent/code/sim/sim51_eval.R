#!/usr/bin/env Rscript
## Headline metric of the paper's Section 5.1 experiment: the average posterior
## effective sample size (ESS) of the Dirichlet process precision alpha, across
## the 20 replicates of each (prior, k*) setting  -- Table 2 of the paper.
## Secondary metric: posterior mass on the true number of clusters k* (Fig. 3).
args <- commandArgs(trailingOnly = FALSE)
fa <- sub("^--file=", "", args[grep("^--file=", args)])
root <- if (length(fa)) normalizePath(file.path(dirname(fa[1]), "..", "..")) else getwd()
Sys.setenv(REPRO_ROOT = root)
source(file.path(root, "code", "sim", "common.R"))

## values printed in Table 2 of the paper (mean, sd) -- page 16 of the
## Bayesian Analysis advance publication / p. 16-17 of the arXiv v2 preprint
paper <- list(
  list(tag = "gamma24",  label = "alpha ~ Ga(2, 4)",     ess = c(8066.35, 7212.07, 7537.74),
       sd = c(581.56, 521.74, 394.80)),
  list(tag = "sg_4_1",   label = "alpha ~ Sg(4, 1, n)",  ess = c(8536.85, 8107.74, 8354.43),
       sd = c(602.03, 764.88, 399.28)),
  list(tag = "sg_1_.25", label = "alpha ~ Sg(1, 0.25, n)", ess = c(7826.41, 6974.12, 7449.69),
       sd = c(812.74, 582.84, 411.10)))

files <- list.files(TRACE_DIR, pattern = "^.*_k[0-9]+_rep[0-9]+\\.rds$", full.names = TRUE)
stopifnot(length(files) > 0)
rows <- do.call(rbind, lapply(files, function(f) {
  tr <- readRDS(f)
  ess  <- as.numeric(coda::effectiveSize(tr$alpha))                       # initial positive sequence
  sp   <- coda::spectrum0.ar(tr$alpha)$spec                      # AR(p) spectral density at 0
  essm <- if (sp > 0) length(tr$alpha) * stats::var(tr$alpha) / sp else NA_real_
  rho  <- stats::acf(tr$alpha, lag.max = 1, plot = FALSE)$acf[2]
  ess_ar1 <- length(tr$alpha) * (1 - rho) / (1 + rho)
  ks  <- tr$kstar
  data.frame(prior = tr$prior, kstar = ks, rep = tr$rep, ess = ess, ess_monotone = essm, ess_ar1 = ess_ar1,
             mean_alpha = mean(tr$alpha), mean_k = mean(tr$k), mode_k = as.integer(names(which.max(table(tr$k)))),
             p_true_k = mean(tr$k == ks), n = length(tr$alpha))
}))
write.csv(rows, file.path(RES_DIR, "per_replicate.csv"), row.names = FALSE)

agg <- lapply(paper, function(p) {
  for (j in seq_along(CFG$kstars)) {
    ks <- CFG$kstars[j]
    s <- rows[rows$prior == p$tag & rows$kstar == ks, ]
    stopifnot(nrow(s) > 0)
    cat(sprintf("%-20s k*=%d  ESS mean %8.2f (sd %7.2f)  N=%2d | paper %8.2f (sd %7.2f)  rel.gap %+6.2f%%\n",
                p$label, ks, mean(s$ess), sd(s$ess), nrow(s), p$ess[j], p$sd[j],
                100 * (mean(s$ess) - p$ess[j]) / p$ess[j]))
  }
  list(tag = p$tag, label = p$label,
       ess = vapply(CFG$kstars, function(ks) mean(rows$ess[rows$prior == p$tag & rows$kstar == ks]), 0),
       sd  = vapply(CFG$kstars, function(ks) sd(rows$ess[rows$prior == p$tag & rows$kstar == ks]), 0),
       p_true_k = vapply(CFG$kstars, function(ks) mean(rows$p_true_k[rows$prior == p$tag & rows$kstar == ks]), 0),
       mean_k   = vapply(CFG$kstars, function(ks) mean(rows$mean_k[rows$prior == p$tag & rows$kstar == ks]), 0),
       frac_mode_correct = vapply(CFG$kstars, function(ks)
           mean(rows$mode_k[rows$prior == p$tag & rows$kstar == ks] == ks), 0),
       ess_monotone = vapply(CFG$kstars, function(ks) mean(rows$ess_monotone[rows$prior == p$tag & rows$kstar == ks]), 0),
       ess_ar1      = vapply(CFG$kstars, function(ks) mean(rows$ess_ar1[rows$prior == p$tag & rows$kstar == ks]), 0))
})
paper_mean  <- mean(unlist(lapply(paper, function(p) p$ess)))
reprod_mean <- mean(unlist(lapply(agg, function(x) x$ess)))
rel_gap     <- (reprod_mean - paper_mean) / paper_mean

cat(sprintf("\nHEADLINE   mean posterior ESS of alpha over the 9 (prior x k*) cells\n"))
cat(sprintf("           paper %.2f  vs  reproduced %.2f   relative gap %+.2f%%\n",
            paper_mean, reprod_mean, 100 * rel_gap))
cat("\nSensitivity of the headline metric to the ESS estimator\n")
est <- do.call(rbind, lapply(files, function(f) { tr <- readRDS(f)
  a <- tr$alpha; n <- length(a)
  c(prior = tr$prior, kstar = tr$kstar,
    coda = as.numeric(coda::effectiveSize(a)),
    geyer_ips = n * stats::var(a) / coda::spectrum0(coda::mcmc(a))$spec,
    ar1 = (function(r) n * (1 - r) / (1 + r))(stats::acf(a, lag.max = 1, plot = FALSE)$acf[2])) }))
est <- as.data.frame(est); for (v in c("coda", "geyer_ips", "ar1")) est[[v]] <- as.numeric(est[[v]])
est_mean <- sapply(c("coda", "geyer_ips", "ar1"), function(v)
    mean(tapply(est[[v]], paste(est$prior, est$kstar), mean)))
for (v in names(est_mean)) cat(sprintf("  %-10s overall mean %8.2f  (%+.2f%% vs paper %.2f)\n",
    v, est_mean[[v]], 100 * (est_mean[[v]] - paper_mean) / paper_mean, paper_mean))

cat("\nSecondary metric (Figure 3): posterior probability of K_n = k*\n")
for (x in agg) cat(sprintf("  %-20s  P(K=k*) %s | mode=k* in %s of 20\n", x$label,
  paste(sprintf("k*=%d %.3f", CFG$kstars, x$p_true_k), collapse = "  "),
  paste(sprintf("%d", round(20 * x$frac_mode_correct)), collapse = "/")))

metrics <- list(
  id = "Sta-29",
  main_experiment = "Section 5.1: posterior comparison between Stirling-gamma and gamma priors in a Dirichlet process mixture of bivariate normals (n=500, k* in {2,4,6}, 20 replicates, 15,000 Gibbs iterations with 5,000 burn-in)",
  primary_metric = list(
    name = "Average posterior effective sample size (ESS) of the DP precision alpha",
    higher_is_better = TRUE,
    setting = "mean over the 9 (prior x k*) cells; each cell averages 20 replicates of 10,000 post-burn-in draws; ESS = coda::effectiveSize",
    paper_value = paper_mean,
    paper_source = "Table 2, p. 16 (Bayesian Analysis advance publication) / p. 16-17 of arXiv:2306.02360v2",
    reproduced_value = reprod_mean,
    relative_gap = rel_gap),
  per_prior = lapply(agg, function(x) list(
    prior = x$label, kstars = CFG$kstars, ess_mean = x$ess, ess_sd = x$sd,
    paper_ess_mean = paper[[which(vapply(paper, function(p) p$tag == x$tag, TRUE))]]$ess,
    paper_ess_sd = paper[[which(vapply(paper, function(p) p$tag == x$tag, TRUE))]]$sd,
    prob_true_k = x$p_true_k, mean_k = x$mean_k,
    frac_mode_correct = x$frac_mode_correct)),
  secondary_metrics = c(lapply(agg, function(x) list(
    metric = sprintf("posterior P(K_n = k*) under %s", x$label), values = x$p_true_k)),
    lapply(agg, function(x) list(
    metric = sprintf("fraction of replicates whose posterior mode of K_n equals k*, under %s", x$label),
    values = x$frac_mode_correct))),
  ess_estimator_sensitivity = as.list(est_mean),
  validation = if (file.exists(file.path(RES_DIR, "validation.rds")))
      readRDS(file.path(RES_DIR, "validation.rds")) else NULL,
  n_replicates = vapply(CFG$kstars, function(ks) sum(rows$kstar == ks) / length(paper), 0),
  burnin = CFG$burnin, niter = CFG$niter, threads = as.integer(Sys.getenv("REPRO_THREADS", 12)))
jsonlite::write_json(metrics, file.path(RES_DIR, "metrics.json"), auto_unbox = TRUE,
                     pretty = TRUE, digits = 8, null = "null")
cat(sprintf("\nwrote %s\n", file.path(RES_DIR, "metrics.json")))
