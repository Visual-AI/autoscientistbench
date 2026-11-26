#!/usr/bin/env Rscript
## Correctness checks for the reproduction of Section 5.1 of the paper.
##   (1) the ratio-of-uniforms port reproduces the authors' ConjugateDP sampler
##   (2) dSg() integrates to one (Stirling-gamma normalizing constant)
##   (3) the collapsed Gibbs sampler reproduces the *exact* posterior of alpha and
##       K_n obtained by enumerating every set partition of a small dataset
args <- commandArgs(trailingOnly = FALSE)
fa <- sub("^--file=", "", args[grep("^--file=", args)])
root <- if (length(fa)) normalizePath(file.path(dirname(fa[1]), "..", "..")) else getwd()
Sys.setenv(REPRO_ROOT = root)
source(file.path(root, "code", "sim", "common.R"))
load_sampler(); library(ConjugateDP)
out <- list()

## ---------------- (1) ratio-of-uniforms port --------------------------------
a <- 4; b <- 1; m <- 500; k <- 4; n <- 500
bd <- ConjugateDP:::getBounds_MuMv_posterior(a, b, m, k, n)
set.seed(1); ref  <- ConjugateDP:::rSg_posterior_ratio_uniforms(50000, a, b, m, k, n, bd$Mu, bd$Mv)
set.seed(1); mine <- rSg_post_ru_cpp(50000, a, b, m, k, n, bd$Mu, bd$Mv)
ks <- suppressWarnings(ks.test(ref, mine))
out$ru_port <- list(ks_statistic = unname(ks$statistic), p_value = unname(ks$p.value),
                    max_abs_diff = max(abs(sort(ref) - sort(mine))),
                    mean_ref = mean(ref), mean_mine = mean(mine))
cat(sprintf("(1) RU port vs authors' sampler rSg_posterior: KS D=%.3g, p=%.3f, mean %.6f vs %.6f\n",
            ks$statistic, ks$p.value, mean(ref), mean(mine)))

## ---------------- (2) dSg normalizing constant ------------------------------
x  <- seq(1e-4, 60, length.out = 6001); dx <- diff(x); xm <- x[-1] - dx/2
integral <- sum(dSg(xm, a = 5, b = 1, m = 100) * dx)
out$dsg_integral <- integral
cat(sprintf("(2) integral of dSg(x, a=5, b=1, m=100) over (0,60] = %.6f\n", integral))

## ---------------- (3) exact posterior by full partition enumeration ----------
nn <- 8L
set.seed(4242); y <- gen_dataset(2L, 4242L)[seq_len(nn), , drop = FALSE]
parts <- list()
rec <- function(cur, mx) {
  if (length(cur) == nn) { parts[[length(parts) + 1L]] <<- cur; return(invisible()) }
  for (v in 0:min(mx + 1L, nn - 1L)) rec(c(cur, v), max(mx, v))
}
rec(0L, 0L)
ll    <- vapply(parts, function(p) sum(sapply(unique(p), function(b)
             block_loglik(y[which(p == b), , drop = FALSE]))), 0)
kk    <- vapply(parts, function(p) length(unique(p)), 0L)
lfact <- vapply(parts, function(p) sum(lgamma(table(p))), 0)   ## sum_j log (n_j - 1)!
## log prior density on the whole grid, using dSg only once for the constant
prior_w <- function(al, prior) {
  if (prior$type == "gamma") return(dgamma(al, prior$a, prior$b, log = TRUE))
  lZ <- sg_log_normconst(prior$a, prior$b, CFG$m)
  (prior$a - 1) * log(al) - prior$b * (lgamma(al + CFG$m) - lgamma(al)) + lZ
}
for (prior in list(list(tag = "gamma24", type = "gamma", a = 2, b = 4),
                   list(tag = "sg_4_1",  type = "sg",    a = 4, b = 1))) {
  grid <- seq(1e-3, 8, length.out = 801)
  eppf <- function(al) kk * log(al) + lfact - (lgamma(al + nn) - lgamma(al))
  pw <- prior_w(grid, prior)
  lp <- vapply(seq_along(grid), function(gi) { lw <- eppf(grid[gi]) + ll; mx <- max(lw)
      pw[gi] + mx + log(sum(exp(lw - mx))) }, 0)
  ex <- exp(lp - max(lp)); ex <- ex / sum(ex)
  ## shift by the global maximum over all partitions to avoid underflow
  mxall <- vapply(seq_along(grid), function(gi) max(eppf(grid[gi]) + ll), 0)
  off <- max(pw + mxall)
  postK <- vapply(1:nn, function(k0) sum(vapply(seq_along(grid), function(gi) {
      lw <- (eppf(grid[gi]) + ll)[kk == k0] - mxall[gi]
      exp(pw[gi] + mxall[gi] - off + log(sum(exp(lw)))) }, 0)), 0)
  postK <- postK / sum(postK)
  mean_alpha_exact <- sum(ex * grid)
  g <- run_chain(y, prior, CFG$niter, 99L, nn = nn)          # 15,000 iterations
  sel <- (CFG$burnin + 1L):CFG$niter
  al <- g$alpha[sel]; kg <- g$k[sel]
  cdf_g <- ecdf(al)(grid); cdf_e <- cumsum(ex) / sum(ex)
  out[[paste0("exact_", prior$tag)]] <- list(
      max_abs_cdf_diff_alpha = max(abs(cdf_g - cdf_e)),
      mean_alpha_exact = mean_alpha_exact, mean_alpha_gibbs = mean(al),
      meanK_exact = sum((1:nn) * postK), meanK_gibbs = mean(kg),
      postK_exact = as.numeric(postK),
      postK_gibbs = as.numeric(table(factor(kg, 1:nn)) / length(kg)))
  cat(sprintf("(3) prior %-9s  KS(alpha) CDF diff %.4f | E[K] exact %.3f vs Gibbs %.3f\n",
              prior$tag, max(abs(cdf_g - cdf_e)), sum((1:nn) * postK), mean(kg)))
}
saveRDS(out, file.path(RES_DIR, "validation.rds"))
cat("validation saved to results/validation.rds\n")
