args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args)==3, file.exists(file.path(args[1],"model.txt")))
suppressPackageStartupMessages(source("R/algorithms.R"))
source("R/modelling_utilities.R")
manifest <- jsonlite::read_json(file.path(args[2],"manifest.json"),simplifyVector=FALSE)
stopifnot(manifest$format == "alse-observations-v1")
observations <- readRDS(file.path(args[2],"observations.rds"))
seed <- as.integer(Sys.getenv("BENCH_SEED","1"))
workers <- max(1L,as.integer(Sys.getenv("BENCH_CPUS","1")))
run_case <- function(i) {
  info <- manifest$cases[[i]]
  df <- observations[[info$case_id]]
  stopifnot(identical(names(df),c("X1","X2","X3","A","Y")),nrow(df)==info$n,all(is.finite(as.matrix(df))))
  set.seed(as.integer((as.double(seed)+104729*i) %% 2147483646)+1L)
  folds <- rep(1:5,length.out=nrow(df))
  # Keep the complete original estimator computation, including both algorithms.
  fits <- list(gam=alse_analysis(df$Y,df$A,df[c("X1","X2","X3")],folds,fitfunc_gam),
               ranger=alse_analysis(df$Y,df$A,df[c("X1","X2","X3")],folds,fitfunc_ranger))
  do.call(rbind,lapply(names(fits),function(learner) {
    fit <- fits[[learner]]
    data.frame(case_id=info$case_id,n=info$n,learner=learner,
               estimate=fit$psi_0_1,std_error=fit$std_err_0_1)
  }))
}
results <- parallel::mclapply(seq_along(manifest$cases),run_case,mc.cores=min(workers,length(manifest$cases)),mc.set.seed=FALSE)
if (any(vapply(results,inherits,logical(1),"try-error"))) stop("case fitting failed")
write.csv(do.call(rbind,results),args[3],row.names=FALSE,quote=FALSE)
