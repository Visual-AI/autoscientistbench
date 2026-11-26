#!/usr/bin/env Rscript
# Protocol adapter: original authors' statistic and source permutation routine.
args <- commandArgs(TRUE)
mode <- args[1]; model_dir <- normalizePath(args[2], mustWork=TRUE)
source("driver/common.R")
cache <- file.path(model_dir, "rcpp_cache")
dir.create(cache, showWarnings=FALSE, recursive=TRUE)
Rcpp::sourceCpp("src/teststatRcppsorted_fastest.cpp", cacheDir=cache)
if (mode == "train") {
  writeLines("DiPMInd Example 1: prod-AD, n=100, B=500, alpha=0.05; no global fit", file.path(model_dir,"protocol.txt"))
  quit(status=0)
}
stopifnot(mode == "predict", file.exists(file.path(model_dir,"protocol.txt")))
input <- args[3]; output <- args[4]
cases <- read.csv(file.path(input,"manifest.csv"), stringsAsFactors=FALSE)
stopifnot(identical(names(cases), c("case_id","file","n")), !anyDuplicated(cases$case_id), all(cases$n==100L))
seed <- as.integer(Sys.getenv("BENCH_SEED","1"))
cores <- max(1L, min(nrow(cases), as.integer(Sys.getenv("BENCH_CPUS","1"))))
# Per-case permutation RNGs depend only on the evaluator seed and row position,
# never on model/rho, data-generation seeds or hidden labels.
one <- function(i) {
  observed <- readRDS(file.path(input, cases$file[i]))
  stopifnot(identical(names(observed),c("Dx","Dy")))
  stopifnot(all(dim(observed$Dx)==c(100L,100L)), all(dim(observed$Dy)==c(100L,100L)),
            all(is.finite(observed$Dx)),all(is.finite(observed$Dy)))
  p <- perm_pvalues(observed$Dx, observed$Dy, B=500L,
                    seed=as.integer((as.double(seed)*1000003+i*104729) %% 2147483646 + 1))
  data.frame(case_id=cases$case_id[i],p_value=unname(p["prod-AD"]))
}
rows <- parallel::mclapply(seq_len(nrow(cases)), one, mc.cores=cores, mc.preschedule=FALSE)
if (any(vapply(rows,inherits,logical(1),"try-error"))) stop("DiPMInd case failed")
write.csv(do.call(rbind,rows),output,row.names=FALSE,quote=FALSE)
