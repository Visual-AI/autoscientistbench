#!/usr/bin/env Rscript
# Shared helpers for the DiPMInd reproduction (glue code written for this task).
# Compiles / loads the authors' two Rcpp statistics plus our distance helper.

suppressPackageStartupMessages(library(Rcpp))

load_stats <- function(root = NULL) {
  if (is.null(root)) root <- .driver_root()
  cache <- file.path(root, "env", "rcpp_cache")
  dir.create(cache, showWarnings = FALSE, recursive = TRUE)
  sourceCpp(file.path(root, "code", "src", "teststatRcppsorted_fastest.cpp"), cacheDir = cache)
  sourceCpp(file.path(root, "code", "src", "mutualTeststatRcppFast.cpp"), cacheDir = cache)
  sourceCpp(file.path(root, "code", "driver", "netdist.cpp"), cacheDir = cache)
  invisible(TRUE)
}

.driver_root <- function() {
  a <- commandArgs(trailingOnly = FALSE)
  m <- grep("^--file=", a, value = TRUE)
  if (length(m)) {
    return(normalizePath(file.path(dirname(sub("^--file=", "", m[1])), "..", "..")))
  }
  normalizePath(getwd())
}

# six variants reported in the paper's Section 3 out of the eight returned columns
VARIANTS <- c("jt", "jt-AD", "jt-F", "prod", "prod-AD", "prod-F")
variant_index <- c("jt" = 1L, "jt-AD" = 2L, "jt-F" = 3L,
                   "prod" = 5L, "prod-AD" = 6L, "prod-F" = 7L)

# permutation p-values (Eq. (18) of the paper) for the six reported variants.
# Dx, Dy: n x n distance matrices of the two variables; B: number of permutations.
perm_pvalues <- function(Dx, Dy, B = 500L, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  obs <- teststatv2Rcpp_sorted2D(Dx, Dy)
  n <- nrow(Dx)
  ge <- numeric(8)
  for (b in seq_len(B)) {
    perm <- sample.int(n)
    Dyp <- Dy[perm, perm, drop = FALSE]
    s <- teststatv2Rcpp_sorted2D(Dx, Dyp)
    ge <- ge + (s >= obs)
  }
  p <- (1 + ge) / (B + 1)
  setNames(p[unname(variant_index)], names(variant_index))
}

# sigma(rho)_{m,l} : first m coordinates equicorrelated, rest independent
sigma_ml <- function(rho, m, l) {
  S <- diag(l)
  S[1:m, 1:m] <- (1 - rho) * diag(m) + rho
  S
}

# sigma(rho)_l : all coordinates equicorrelated
sigma_l <- function(rho, l) (1 - rho) * diag(l) + rho
