args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args)==6L, identical(args[c(1,3,5)], c("--model","--input","--output")))
model <- normalizePath(args[2], mustWork=TRUE)
.libPaths(c(file.path(model,"Rlib"), .libPaths()))
suppressPackageStartupMessages(library(BTSBM))
stopifnot(normalizePath(find.package("BTSBM")) == normalizePath(file.path(model,"Rlib","BTSBM")))
source("fit_helpers.R")
x <- jsonlite::fromJSON(args[4], simplifyVector=FALSE)
stopifnot(identical(names(x), "datasets"), length(x$datasets)>0)
seed <- as.integer(Sys.getenv("BENCH_SEED", "1"))
workers <- max(1L, as.integer(Sys.getenv("BENCH_CPUS", "1")))
smoke <- identical(Sys.getenv("BENCH_SMOKE"), "1")
T_iter <- if (smoke) 400L else 10000L
T_burn <- if (smoke) 100L else 2000L
run_one_observed <- function(i) {
  W <- do.call(rbind, lapply(x$datasets[[i]]$W, unlist))
  storage.mode(W) <- "integer"
  stopifnot(nrow(W)==ncol(W), all(W>=0L), all(diag(W)==0L))
  # Inference randomness is independent of the private data-generation seed.
  set.seed(as.integer((as.double(seed)*1000003 + i*9176) %% 2147483646 + 1))
  elapsed <- system.time({
    fit <- BTSBM::gibbs_bt_sbm(w_ij=W, T_iter=T_iter, T_burn=T_burn,
                             a=2, gamma=.71, prior="GN", verbose=FALSE)
  })[["elapsed"]]
  rel <- relabel_by_lambda_draws(fit$x_samples, fit$lambda_samples)
  psm <- mcclust::comp.psm(rel$x_samples_relabel)
  partition <- as.integer(mcclust.ext::minVI(psm)$cl)
  strength <- extract_btsbm_lambda_hat(fit)
  stopifnot(length(partition)==nrow(W), all(is.finite(strength)), elapsed>0)
  list(partition=partition, strength=as.numeric(strength), fit_time_s=unname(elapsed))
}
ans <- parallel::mclapply(seq_along(x$datasets), run_one_observed,
                         mc.cores=min(workers,length(x$datasets)), mc.set.seed=FALSE)
stopifnot(!any(vapply(ans,inherits,logical(1),"try-error")))
jsonlite::write_json(list(datasets=ans),args[6],auto_unbox=TRUE,digits=17,pretty=FALSE)
cat(sprintf("BT-SBM: rebuilt candidate package %s; %d datasets; %d iterations / %d burn-in; workers=%d\n",
            find.package("BTSBM"),length(ans),T_iter,T_burn,workers))
