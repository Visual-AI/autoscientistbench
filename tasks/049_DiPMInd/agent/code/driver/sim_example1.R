#!/usr/bin/env Rscript
# DiPMInd reproduction: Example 1 of Section 3 (Figure 1) of
# Chen & Dubey, "DiPMInd: Distance profile based mutual independence testing for
# random objects" (Annals of Statistics, arXiv:2412.06766v4).
#
# Joint independence test between
#   X_{i,1} : random networks with 200 nodes (graph Laplacian matrices, l1 metric)
#   X_{i,2} : Gaussian random vectors (l1 metric on R^l)
# for the two models of the example.  Glue code written for this reproduction:
# the authors' repository only ships the two permutation statistics.
#
# Usage:
#   Rscript code/driver/sim_example1.R --model 2 --rho 0.8 --nrep 200 --B 500 \
#       --cores 12 --out results/pvals/model2_rho0.80.csv
suppressPackageStartupMessages({
  library(Rcpp); library(parallel); library(igraph)
})

args <- commandArgs(trailingOnly = TRUE)
getarg <- function(name, default) {
  i <- match(paste0("--", name), args)
  if (is.na(i)) return(default)
  args[i + 1]
}
model      <- as.integer(getarg("model", 1))
rho        <- as.numeric(getarg("rho", 0))
nrep       <- as.integer(getarg("nrep", 200))
B          <- as.integer(getarg("B", 500))
n          <- as.integer(getarg("n", 100))
nodes      <- as.integer(getarg("nodes", 200))
cores      <- as.integer(getarg("cores", 12))
seed0      <- as.integer(getarg("seed", 20261028))
pinterp    <- getarg("power-interp", "first")   # Model 1 only: how the 10-vector gamma is mapped to a scalar
graph_type <- getarg("graph", "default")        # default | undirected | asundirected
medges     <- as.integer(getarg("m", 1))
lap_norm   <- getarg("lap", "none")             # none | normalized
out        <- getarg("out", "")

source(file.path(dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1])),
                 "common.R"))

load_stats()

make_network <- function(power_i, type, m = 1L, lap = "none") {
  g <- switch(type,
    default      = igraph::sample_pa(nodes, power = power_i, m = m),
    undirected   = igraph::sample_pa(nodes, power = power_i, m = m, directed = FALSE),
    asundirected = igraph::as_undirected(igraph::sample_pa(nodes, power = power_i, m = m)),
    stop("unknown graph type"))
  if (!igraph::is_simple(g)) g <- igraph::simplify(g)
  as.matrix(igraph::laplacian_matrix(g, normalized = (lap == "normalized"), sparse = FALSE))
}

# ---- one Monte Carlo replicate: returns the six permutation p-values ----------
one_rep <- function(r, model, rho, n, B, pinterp, graph_type, medges, lap_norm, seed) {
  set.seed(seed + r)
  if (model == 1L) {
    l <- 40L
    V <- matrix(rnorm(n * 50), n, 50) %*% chol(sigma_ml(rho, 20, 50))
    X2 <- V[, 11:50, drop = FALSE]
    U <- matrix(runif(n * 10), n, 10)
    Y <- U / rowSums(U)
    G <- plogis(V[, 1:10, drop = FALSE] * V[, 11:20, drop = FALSE]) * Y
    power_vec <- switch(pinterp,
      first = G[, 1],
      sum   = rowSums(G),
      mean  = rowMeans(G),
      stop("unknown power-interp"))
  } else {
    l <- 30L
    V <- matrix(rnorm(n * 50), n, 50) %*% chol(sigma_ml(rho, 40, 50))
    X2 <- V[, 21:50, drop = FALSE]
    W <- 2 * (V[, 21] < 1) - 1
    power_vec <- apply(plogis(W * V[, 1:20, drop = FALSE]), 1, median)
  }
  Lmats <- lapply(power_vec, make_network, type = graph_type, m = medges, lap = lap_norm)
  Dx <- l1mat_dist(Lmats)
  Dy <- as.matrix(dist(X2, method = "manhattan"))
  perm_pvalues(Dx, Dy, B = B)
}

t0 <- proc.time()[3]
seeds <- seed0 + model * 1e6 + round(rho * 1e4)
cl_cores <- min(cores, nrep)
res <- mclapply(seq_len(nrep), one_rep, model = model, rho = rho, n = n, B = B,
                pinterp = pinterp, graph_type = graph_type, medges = medges,
                lap_norm = lap_norm, seed = seeds,
                mc.cores = cl_cores, mc.preschedule = FALSE)
bad <- vapply(res, function(z) !is.numeric(z), logical(1))
if (any(bad)) stop("replicate(s) failed: ", paste(which(bad), collapse = ","),
                   " : ", paste(utils::head(as.character(res[bad]), 2), collapse = " | "))
df <- as.data.frame(do.call(rbind, res))
df <- data.frame(model = model, rho = rho, rep = seq_len(nrep), df, check.names = FALSE)
names(df) <- c("model", "rho", "rep", VARIANTS)
el <- proc.time()[3] - t0
cat(sprintf("model=%d rho=%.2f nrep=%d B=%d cores=%d wall=%.1fs\n", model, rho, nrep, B, cl_cores, el))
cat("size/power (proportion of p<=0.05):\n")
print(round(colMeans(df[, VARIANTS] <= 0.05), 4))
if (nzchar(out)) {
  dir.create(dirname(out), showWarnings = FALSE, recursive = TRUE)
  write.csv(df, out, row.names = FALSE)
  cat("wrote", out, "\n")
}
