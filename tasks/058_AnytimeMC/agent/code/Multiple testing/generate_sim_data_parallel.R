#!/usr/bin/env Rscript
# ---------------------------------------------------------------------------
# Parallel driver for the simulation grids of Section 6.1 / Figure 2 of
# "Multiple testing with anytime-valid Monte Carlo p-values"
# (Fischer, Barry, Ramdas), i.e. the paper's main experiment.
#
# The authors' data_generator.R runs every simulation trial of a grid cell in a
# single loop inside one process.  This driver performs exactly the same
# computations -- sequential_BH.R is sourced unmodified and the body of the
# per-trial part of data_generator() is copied verbatim into trial_fun() below --
# but distributes the nsIm independent trials of a cell over worker processes.
#
# Exact reproducibility: the parent process advances the Mersenne-Twister
# stream in exactly the order of the sequential reference implementation
# (generate_sim_data.R calls set.seed(123) before every grid, and
# data_generator() draws, for every cell, first nsim x (rbinom(M,1,pi), rnorm(M))
# and then, for k = 1..nsim, rnorm(M*B_small) followed by rnorm(M*B_large)).
# The .Random.seed state at the beginning of every trial is recorded, so each
# worker regenerates bit-identical matrices and all reported quantities are
# identical to a sequential run of the authors' code.
#
# Usage: Rscript generate_sim_data_parallel.R <grid> <outfile.rda> <cores>
#        grids: pis | alphas | mus | Ms | standard
# ---------------------------------------------------------------------------
suppressPackageStartupMessages({ library(MASS); library(binom); library(parallel) })

args <- commandArgs(trailingOnly = TRUE)
grid <- args[1]
outfile <- args[2]
cores <- as.integer(args[3])
# optional 4th argument: comma separated cell numbers to evaluate (pilots only).
# All cells still advance the RNG exactly as in the sequential reference
# implementation, so a subset pilot returns the same numbers as the full run.
only <- if (length(args) >= 4 && nzchar(args[4])) as.integer(strsplit(args[4], ",")[[1]]) else NULL

## --- simulation grids of generate_sim_data.R (verbatim parameters) --------
grids <- list(
  pis      = list(nsim = 10, nhyp = 1000, mu_As = 2.5, mu_N = 0, pis = seq(0.1, 0.9, 0.1), alphas = 0.1),
  alphas   = list(nsim = 10, nhyp = 1000, mu_As = 2.5, mu_N = 0, pis = 0.4, alphas = c(0.01, 0.05, 0.1, 0.15, 0.2)),
  mus      = list(nsim = 10, nhyp = 1000, mu_As = c(1, 1.5, 2, 2.5, 3, 3.5, 4), mu_N = 0, pis = 0.4, alphas = 0.1),
  Ms       = list(nsim = 10, nhyp = c(100, 500, 1000, 3000, 5000, 10000), mu_As = 2.5, mu_N = 0, pis = 0.4, alphas = 0.1),
  standard = list(nsim = 1, nhyp = 1000, mu_As = 2.5, mu_N = 0, pis = 0.4, alphas = 0.1)
)
cfg <- grids[[grid]]
nsim <- cfg$nsim; nhyp <- cfg$nhyp; mu_As <- cfg$mu_As; mu_N <- cfg$mu_N
pis <- cfg$pis; alphas <- cfg$alphas
param_bm <- 0.9; param_bc <- 10; param_amt <- 0.01
peeking_times_AMT <- floor(c(cumsum(c(100, 100 * cumprod(rep(1.1, 24)))), 10000))
B_large <- 10000; B_small <- 200

source("sequential_BH.R")

## --- one simulation trial: verbatim from data_generator() ----------------
trial_fun <- function(state, Z_k, hypo_k, M, alpha) {
  assign(".Random.seed", state, envir = globalenv())
  # Generate simulated test statistics
  simZ <- matrix(rnorm(M * B_small), ncol = B_small)
  simZ_largeB <- matrix(rnorm(M * B_large), ncol = B_large)

  # Apply algorithms
  results_bm <- sequential_BH(Z_k, simZ_largeB, M, B_large, (1:B_large), alpha, param_bm, param_bc, param_amt, peeking_times_AMT, "bm")
  results_bc <- sequential_BH(Z_k, simZ_largeB, M, B_large, (1:B_large), alpha, param_bm, param_bc, param_amt, peeking_times_AMT, "bc")
  results_agg <- sequential_BH(Z_k, simZ_largeB, M, B_large, (1:B_large), alpha, param_bm, 1, param_amt, peeking_times_AMT, "bc")
  results_amt <- sequential_BH(Z_k, simZ_largeB, M, B_large, (1:B_large), (alpha - param_amt), param_bm, param_bc, param_amt, peeking_times_AMT, "AMT")

  # Calculate results
  out <- list()
  for (nm in c("bm", "bc", "agg", "amt")) {
    r <- get(paste0("results_", nm))
    out[[paste0("nperm_mean_", nm)]] <- mean(r$stop)
    out[[paste0("nperm_mean_rej_", nm)]] <- mean(r$stop[r$rejects])
    out[[paste0("nperm_mean_stop_", nm)]] <- mean(r$stop[setdiff(1:M, r$rejects)])
    out[[paste0("nperm_median_", nm)]] <- median(r$stop)
    out[[paste0("nperm_max_", nm)]] <- max(r$stop)
    out[[paste0("power_", nm)]] <- length(intersect(which(hypo_k == 1), r$rejects)) / max(1, sum(hypo_k))
  }
  out$FDR_bm <- length(intersect(which(hypo_k == 0), results_bm$rejects)) / max(1, max(1, length(results_bm$rejects)))
  out$FDR_bc <- length(intersect(which(hypo_k == 0), results_bc$rejects)) / max(1, length(results_bc$rejects))
  out$FDR_agg <- length(intersect(which(hypo_k == 0), results_agg$rejects)) / max(1, length(results_agg$rejects))
  out$FDR_amt <- length(intersect(which(hypo_k == 0), results_amt$rejects)) / max(1, length(results_amt$rejects))

  # Calculate classical permutation p-values
  ps_perm <- (rowSums(simZ_largeB >= Z_k) + 1) / (B_large + 1)
  ps_perm_BH <- p.adjust(ps_perm, "fdr")
  out$power_BH <- sum(hypo_k == 1 & ps_perm_BH <= alpha) / max(1, sum(hypo_k))
  out$FDR_BH <- sum(hypo_k == 0 & ps_perm_BH <= alpha) / max(1, sum(ps_perm_BH <= alpha))

  # Calculate classical permutation p-values with small B
  ps_perm_smallB <- (rowSums(simZ >= Z_k) + 1) / (B_small + 1)
  ps_perm_BH_smallB <- p.adjust(ps_perm_smallB, "fdr")
  out$power_BH_smallB <- sum(hypo_k == 1 & ps_perm_BH_smallB <= alpha) / max(1, sum(hypo_k))
  out$FDR_BH_smallB <- sum(hypo_k == 0 & ps_perm_BH_smallB <= alpha) / max(1, sum(ps_perm_BH_smallB <= alpha))

  # Calculate classical Besag-Clifford p-value
  time_fut <- apply(t(apply(simZ_largeB >= Z_k, 1, cumsum)), 1, function(row) {
    idx <- which(row >= param_bc)
    if (length(idx) == 0) (B_large + 1) else idx[1]
  })
  ps_bc_old <- (time_fut <= B_large) * param_bc / time_fut + (time_fut > B_large) * ps_perm
  time_fut <- pmin(time_fut, B_large)
  ps_bc_old_BH <- p.adjust(ps_bc_old, "fdr")
  out$power_bc_old <- sum(hypo_k == 1 & ps_bc_old_BH <= alpha) / max(1, sum(hypo_k))
  out$FDR_bc_old <- sum(hypo_k == 0 & ps_bc_old_BH <= alpha) / max(1, sum(ps_bc_old_BH <= alpha))
  out$nperm_mean_bc_old <- mean(time_fut)

  out$result_bm <- results_bm; out$result_bc <- results_bc
  out$result_agg <- results_agg; out$result_amt <- results_amt
  out
}

## --- allocate result matrices (names as in data_generator.R) -------------
ncells <- length(pis) * length(alphas) * length(mu_As) * length(nhyp)
scal <- c("power", "FDR", "nperm_mean", "nperm_mean_rej", "nperm_mean_stop", "nperm_median", "nperm_max")
meth <- c("bm", "bc", "agg", "amt")
mats <- list()
for (m in meth) for (s in scal) mats[[paste0(s, "_", m)]] <- matrix(0, nsim, ncells)
mats$power_BH <- matrix(0, nsim, ncells); mats$FDR_BH <- matrix(0, nsim, ncells)
mats$power_BH_smallB <- matrix(0, nsim, ncells); mats$FDR_BH_smallB <- matrix(0, nsim, ncells)
mats$power_bc_old <- matrix(0, nsim, ncells); mats$FDR_bc_old <- matrix(0, nsim, ncells)
mats$nperm_mean_bc_old <- matrix(0, nsim, ncells)
result_bm <- result_bc <- result_agg <- result_amt <- list()

# number of concurrent workers as a function of the matrix size (memory bound)
eff_cores <- function(M) max(1L, min(cores, cores))   # always up to `cores` workers (<= 12)

set.seed(123)   # same seed as generate_sim_data.R before every data_generator() call
count <- 1
for (M in nhyp) {
  for (alpha in alphas) {
    for (mu_A in mu_As) {
      for (pi_A in pis) {
        Z <- matrix(, nrow = M, ncol = nsim)
        hypo <- matrix(, nrow = M, ncol = nsim)
        for (j in 1:nsim) {
          hypo[, j] <- rbinom(M, 1, pi_A)
          X <- rnorm(M)
          Z[, j] <- mu_N * (hypo[, j] - 1) * (-1) + mu_A * hypo[, j] + X
        }
        # record the RNG state at the start of every trial and advance the
        # stream exactly as the sequential implementation would
        states <- vector("list", nsim)
        for (k in 1:nsim) {
          states[[k]] <- .Random.seed
          invisible(rnorm(M * B_small))
          invisible(rnorm(M * B_large))
        }
        end_state <- .Random.seed          # state after the whole cell, as in the sequential run
        if (!is.null(only) && !(count %in% only)) {   # pilot: skip computation, keep RNG flow
          assign(".Random.seed", end_state, envir = globalenv())
          count <- count + 1
          next
        }
        t0 <- Sys.time()
        res <- mclapply(1:nsim, function(k) trial_fun(states[[k]], Z[, k], hypo[, k], M, alpha),
                        mc.cores = eff_cores(M), mc.preschedule = TRUE)
        # mclapply may consume/alter the parent's RNG stream; restore the state
        # the sequential reference implementation would have after this cell.
        assign(".Random.seed", end_state, envir = globalenv())
        if (any(vapply(res, inherits, TRUE, "try-error"))) stop("worker failed")
        for (k in 1:nsim) {
          out <- res[[k]]
          for (nm in names(mats)) if (!is.null(out[[nm]])) mats[[nm]][k, count] <- out[[nm]]
          result_bm[[k]] <- out$result_bm; result_bc[[k]] <- out$result_bc
          result_agg[[k]] <- out$result_agg; result_amt[[k]] <- out$result_amt
        }
        cat(sprintf("[%s] cell %3d/%3d  M=%6d alpha=%.2f mu_A=%.1f pi_A=%.1f  nsim=%d  %6.1f s (cores %d)\n",
                    grid, count, ncells, M, alpha, mu_A, pi_A, nsim,
                    as.numeric(Sys.time() - t0, units = "secs"), eff_cores(M)), flush = TRUE)
        count <- count + 1
      }
    }
  }
}

# Average over different simulation runs (as in data_generator.R)
for (nm in names(mats)) assign(nm, colMeans(mats[[nm]]))

save(list = c(names(mats), "result_bm", "result_bc", "result_agg", "result_amt"), file = outfile)
cat("saved ", outfile, "\n", sep = "")
