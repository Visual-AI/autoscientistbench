## Equivalence checks: the vectorised kernels of fast_kernels.R must reproduce
## the authors' implementation exactly.
##
##   1. nn_index()            vs  which.min(abs(q - grid))   / pracma::interp2()
##   2. fbm_sheet_f()         vs  direg::fbm_sheet()          (identical seed)
##   3. whole pipeline        vs  the package functions       (identical seed)
##
## Usage: Rscript verify_equiv.R [--config-ids 1] [extra config ids ...]

Sys.setenv(DIREG_NO_MAIN = "1")
suppressPackageStartupMessages({library(purrr); library(pracma)})
a <- commandArgs(trailingOnly = TRUE)
i <- match("--config-ids", a)
ids <- if (is.na(i)) c(1L) else as.integer(strsplit(a[i + 1], ",")[[1]])
src <- normalizePath(sub("^--file=", "", commandArgs(trailingOnly = FALSE)[grep("^--file=", commandArgs(trailingOnly = FALSE))]))
source(file.path(dirname(src), "driver.R"))

cat("== 1. nearest-neighbour lookups ==\n")
set.seed(1)
side <- 51
grid <- seq(0, 1, length.out = side)
q <- c(seq(0, 1, length.out = side),
       (grid[-side] + grid[-1]) / 2,          # exact midpoints: tie rule
       runif(5000, 0, 1),
       rep(0, 3), rep(1, 3), 1e-16, 1 - 1e-16)
ref_whichmin <- vapply(q, function(z) which.min(abs(z - grid)), integer(1))
ref_pracma <- pracma::interp2(x = grid, y = grid,
                              Z = matrix(grid, nrow = side, ncol = side),
                              xp = q, yp = q, method = "nearest")
mine <- nn_index(q, grid)
mine2 <- nn_index_interp2(q, grid)
cat(sprintf("  nn_index (which.min)  : %d mismatches out of %d\n",
            sum(mine != ref_whichmin), length(q)))
cat(sprintf("  nn_index_interp2      : %d value mismatches out of %d (max |d| %.3e)\n",
            sum(grid[mine2] != ref_pracma), length(q),
            max(abs(grid[mine2] - ref_pracma))))
stopifnot(sum(mine != ref_whichmin) == 0, sum(grid[mine2] != ref_pracma) == 0)

cat("== 2. simulator ==\n")
for (alpha in c(pi/30, pi/5, pi/4, pi/3, pi/2 - pi/30)) {
  set.seed(42); A <- fbm_sheet(t_n = side, e_n = side, alpha = alpha, H1 = 0.8, H2 = 0.5,
                               type = "sum", sigma = 0.5)
  set.seed(42); B <- fbm_sheet_f(t_n = side, e_n = side, alpha = alpha, H1 = 0.8, H2 = 0.5,
                                 type = "sum", sigma = 0.5)
  cat(sprintf("  alpha = %6.3f : max |fbm_sheet_f - fbm_sheet| = %.3e\n",
              alpha, max(abs(A - B))))
  stopifnot(identical(dim(A), dim(B)), max(abs(A - B)) == 0)
}

cat("== 3. full estimation pipeline (fast vs package functions) ==\n")
grid60 <- make_grid()
for (id in ids) {
  cfg <- grid60[grid60$id == id, , drop = FALSE]
  t0 <- Sys.time()
  r_fast <- run_rep(cfg, rep = 1L, seed = 777L + id, engine = "fast")
  t1 <- Sys.time()
  r_orig <- run_rep(cfg, rep = 1L, seed = 777L + id, engine = "orig")
  t2 <- Sys.time()
  cat(sprintf("  cfg %2d (N=%d, side=%d, sigma=%.1f, alpha=%.3f):\n",
              id, cfg$N, cfg$side, cfg$sigma, cfg$alpha))
  cat(sprintf("    alpha_hat_adj  fast %.12f | orig %.12f | diff %.2e\n",
              r_fast$alpha_hat_adj, r_orig$alpha_hat_adj,
              abs(r_fast$alpha_hat_adj - r_orig$alpha_hat_adj)))
  cat(sprintf("    H_max_hat      fast %.12f | orig %.12f | diff %.2e\n",
              r_fast$H_max_hat, r_orig$H_max_hat, abs(r_fast$H_max_hat - r_orig$H_max_hat)))
  cat(sprintf("    H_min_hat      fast %.12f | orig %.12f | diff %.2e\n",
              r_fast$H_min_hat, r_orig$H_min_hat, abs(r_fast$H_min_hat - r_orig$H_min_hat)))
  cat(sprintf("    sigma_hat      fast %.12f | orig %.12f | diff %.2e\n",
              r_fast$sigma_hat, r_orig$sigma_hat, abs(r_fast$sigma_hat - r_orig$sigma_hat)))
  cat(sprintf("    time: fast %.2f s | orig %.2f s | speedup %.0fx\n",
              as.numeric(difftime(t1, t0, units = "secs")),
              as.numeric(difftime(t2, t1, units = "secs")),
              as.numeric(difftime(t2, t1, units = "secs")) /
                max(as.numeric(difftime(t1, t0, units = "secs")), 1e-6)))
  stopifnot(abs(r_fast$alpha_hat_adj - r_orig$alpha_hat_adj) < 1e-9,
            abs(r_fast$H_max_hat - r_orig$H_max_hat) < 1e-9,
            abs(r_fast$H_min_hat - r_orig$H_min_hat) < 1e-9,
            abs(r_fast$sigma_hat - r_orig$sigma_hat) < 1e-9)
}
cat("ALL EQUIVALENCE CHECKS PASSED\n")
