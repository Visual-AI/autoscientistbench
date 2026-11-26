# Protocol adapter: original estimator and interval calculation on observed inputs.
source("functions.R")
library(jsonlite)
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args) == 6L)
opts <- setNames(as.list(args[c(2,4,6)]), args[c(1,3,5)])
stopifnot(all(c("--model", "--input", "--output") %in% names(opts)))
input <- opts[["--input"]]
manifest <- fromJSON(file.path(input, "manifest.json"), simplifyVector=FALSE)
seed <- as.integer(Sys.getenv("BENCH_SEED", "1"))
stopifnot(as.integer(readLines(file.path(opts[["--model"]], "seed.txt"), warn=FALSE)) == seed)
workers <- max(1L, min(as.integer(Sys.getenv("BENCH_CPUS", "1")), length(manifest$settings)))
run_setting <- function(i) {
  setting <- manifest$settings[[i]]
  n <- as.integer(setting$n); d <- as.integer(setting$d); S <- as.integer(setting$replications)
  stopifnot(n > 3L, d > 0L, S > 0L)
  count <- n * (d+1L)
  con <- file(file.path(input, setting$file), "rb")
  on.exit(close(con))
  observation <- function() {
    values <- readBin(con, "double", n=count, size=8L, endian="little")
    stopifnot(length(values) == count, all(is.finite(values)))
    list(X=matrix(values[seq_len(n*d)], nrow=n, ncol=d), Y=values[n*d+seq_len(n)])
  }
  # Preserve authors' one bootstrap sample per setting, 10000 subsamples,
  # m=floor(sqrt(n)), no replacement, degree-two basis and lambda=n^-0.85.
  set.seed((as.double(seed) + i * 104729) %% 2147483647)
  m <- floor(sqrt(n)); fml <- make_quad_formula(d); lambda <- n^(-0.85)
  sim0 <- observation(); X0 <- sim0$X; Y0 <- sim0$Y
  T_bootstrap <- replicate(10000L, {
    idx <- sample.int(n, m, replace=FALSE)
    codec(Y0[idx], X0[idx, , drop=FALSE])
  })
  var_bootstrap <- m * mean((T_bootstrap - mean(T_bootstrap))^2)
  sigma_n <- sqrt(var_bootstrap)
  T_n_bc <- numeric(S)
  for (s in seq_len(S)) {
    sim <- observation(); X <- sim$X; Y <- sim$Y
    T_n_orig <- codec(Y, X)
    Lhat <- compute_Lhat(X, Y, fml, lambda)
    T_n_bc[s] <- T_n_orig - 6 * Lhat
  }
  stopifnot(length(readBin(con, "raw", n=1L)) == 0L)
  data.frame(setting_id=setting$setting_id, replicate=seq_len(S), estimate=T_n_bc,
             lower=T_n_bc - 1.96*sigma_n/sqrt(n), upper=T_n_bc + 1.96*sigma_n/sqrt(n))
}
results <- parallel::mclapply(seq_along(manifest$settings), run_setting, mc.cores=workers,
                              mc.set.seed=FALSE, mc.preschedule=TRUE)
stopifnot(all(vapply(results, is.data.frame, logical(1))))
write.csv(do.call(rbind, results), opts[["--output"]], row.names=FALSE, quote=FALSE)
