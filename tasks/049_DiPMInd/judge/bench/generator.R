#!/usr/bin/env Rscript
# Frozen observed-distance materializer for the source Example 1 equations.
# Usage: generator.R PLAN.csv OUTPUT_INPUT CODE_MAKEVARS CORES
# PLAN is operator-owned: case_id,model,rho,rep,seed. It is never copied into input.
suppressPackageStartupMessages({library(Rcpp); library(igraph)})
args <- commandArgs(TRUE)
plan <- read.csv(args[1],stringsAsFactors=FALSE)
out <- args[2]; dir.create(out,recursive=TRUE,showWarnings=FALSE)
Sys.setenv(R_MAKEVARS_USER=normalizePath(args[3]))
script <- sub("^--file=","",grep("^--file=",commandArgs(FALSE),value=TRUE)[1])
Rcpp::sourceCpp(file.path(dirname(script),"generator_netdist.cpp"),cacheDir=file.path(tempdir(),"generator_cache"))
nodes <- 200L
sigma_ml <- function(rho, m, l) {
  S <- diag(l)
  S[1:m, 1:m] <- (1 - rho) * diag(m) + rho
  S
}

# sigma(rho)_l : all coordinates equicorrelated
sigma_l <- function(rho, l) (1 - rho) * diag(l) + rho

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
  list(Dx=Dx, Dy=Dy)
}

make_one <- function(i) {
 p <- plan[i,]
 obj <- one_rep(p$rep,p$model,p$rho,n=100L,B=500L,pinterp="sum",graph_type="undirected",medges=1L,lap_norm="none",seed=p$seed)
 saveRDS(obj,file.path(out,paste0(p$case_id,".rds")),compress=FALSE)
 TRUE
}
cores <- as.integer(args[4])
for (first in seq(1L,nrow(plan),by=40L)) {
 idx <- first:min(first+39L,nrow(plan))
 done <- parallel::mclapply(idx,make_one,mc.cores=cores,mc.preschedule=FALSE)
 if (!all(vapply(done,identical,logical(1),TRUE))) stop("generation failed")
 cat("generated",max(idx),"of",nrow(plan),"\n");flush.console()
}
manifest <- data.frame(case_id=plan$case_id,file=paste0(plan$case_id,".rds"),n=100L)
write.csv(manifest,file.path(out,"manifest.csv"),row.names=FALSE,quote=FALSE)
