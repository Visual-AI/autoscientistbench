#!/usr/bin/env Rscript
args <- commandArgs(trailingOnly=TRUE)
if (length(args) != 6L || !identical(args[c(1,3,5)], c("--model","--input","--output"))) stop("invalid arguments")
model <- normalizePath(args[2], mustWork=TRUE)
.libPaths(c(file.path(model, "Rlib"), .libPaths()))
library(ConjugateDP)
stopifnot(normalizePath(find.package("ConjugateDP")) == normalizePath(file.path(model,"Rlib","ConjugateDP")))
cat("candidate package:", find.package("ConjugateDP"), "\n")
Sys.setenv(REPRO_ROOT=getwd())
source("sim/common.R")
BUILD_DIR <- file.path(tempdir(), "candidate-build")
load_sampler(rebuild=TRUE)
input <- jsonlite::fromJSON(args[4], simplifyVector=FALSE)
stopifnot(identical(names(input),c("version","datasets")), input$version == 1L)
niter <- CFG$niter; burnin <- CFG$burnin
if (identical(Sys.getenv("BENCH_SMOKE"),"1")) { niter <- 150L; burnin <- 50L }
prior <- CFG$priors[[3L]]
bounds <- sg_bounds(prior$a, prior$b, CFG$m, CFG$n)
seed <- as.integer(Sys.getenv("BENCH_SEED", "1"))
cpus <- max(1L, as.integer(Sys.getenv("BENCH_CPUS", "1")))
cat(sprintf("actual Gibbs: %d datasets; n=%d; niter=%d; burnin=%d; prior=Sg(1,.25,n); cpus=%d\n",length(input$datasets),CFG$n,niter,burnin,cpus))
result <- parallel::mclapply(seq_along(input$datasets), function(i) {
 d <- input$datasets[[i]]
 stopifnot(identical(names(d),c("id","y")))
 y <- do.call(rbind,lapply(d$y,unlist))
 stopifnot(identical(dim(y),c(CFG$n,2L)), all(is.finite(y)))
 chain_seed <- as.integer((as.double(seed) * 100003 + i) %% 2147483646 + 1)
 g <- run_chain(y, prior, niter=niter, seed=chain_seed, nn=nrow(y), bounds=bounds)
 keep <- seq.int(burnin+1L,niter)
 list(id=d$id,p_kn=as.numeric(tabulate(g$k[keep],nbins=nrow(y)))/length(keep))
}, mc.cores=min(cpus,length(input$datasets)),mc.preschedule=FALSE,mc.set.seed=FALSE)
if (any(vapply(result,inherits,logical(1),"try-error"))) stop("a Gibbs chain failed")
jsonlite::write_json(list(version=1L,predictions=result),args[6],auto_unbox=TRUE,digits=16)
