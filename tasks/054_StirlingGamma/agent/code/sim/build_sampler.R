#!/usr/bin/env Rscript
## Compile the added Rcpp Gibbs sampler into the local sourceCpp cache.
args <- commandArgs(trailingOnly = FALSE)
fa <- sub("^--file=", "", args[grep("^--file=", args)])
root <- if (length(fa)) normalizePath(file.path(dirname(fa[1]), "..", "..")) else getwd()
Sys.setenv(REPRO_ROOT = root)
source(file.path(root, "code", "sim", "common.R"))
load_sampler()
cat("sampler compiled and loaded OK; cache:", file.path(BUILD_DIR, "rcpp-cache"), "\n")
