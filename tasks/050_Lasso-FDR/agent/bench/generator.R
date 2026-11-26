# Frozen offline observation generator. Never called by candidate train/predict.
# Usage: Rscript generator.R PLAN_JSON OUTPUT_INPUT_DIR OUTPUT_LABEL_DIR
# PLAN_JSON holds operator-only opaque IDs and generation seeds. It is not input.
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(TRUE)
script <- sub("^--file=", "", commandArgs()[grepl("^--file=",commandArgs())][1])
here <- dirname(normalizePath(script))
source(file.path(here,"original_utils.R"))
parameters <- fromJSON(file.path(here,"calibration.json"),simplifyVector=FALSE)
plan <- fromJSON(args[1],simplifyVector=FALSE)
dir.create(args[2],recursive=TRUE,showWarnings=FALSE)
dir.create(args[3],recursive=TRUE,showWarnings=FALSE)
create <- function(item){
 param <- parameters[[match(item$design,vapply(parameters,`[[`,character(1),"design"))]]
 n<-as.integer(item$n);p<-as.integer(item$p)
 set.seed(as.integer(item$response_seed))
 X <- gene_X(item$design,n,p,as.integer(item$design_seed),scaling=FALSE,model_X=TRUE,signal=param$mu1,nonnulls=30,cov_seed=-2023,mis_model=FALSE)$X
 beta <- genmu(p,30/p,param$mu1,"random",1)
 y <- as.vector(X %*% beta+rnorm(n))
 lambda <- unlist(param$lambda)
 obs <- list(X=X,y=y,lambda=lambda)
 saveRDS(obs,file.path(args[2],item$file),compress=FALSE)
 saveRDS(c(obs,list(beta=beta)),file.path(args[3],item$file),compress=FALSE)
 list(id=item$id,file=item$file,design=item$design,n=n,p=p,n_lambda=length(lambda))
}
workers <- as.integer(Sys.getenv("MATERIALIZE_CPUS","2"))
instances <- parallel::mclapply(plan$instances,create,mc.cores=workers,mc.preschedule=FALSE,mc.set.seed=FALSE)
stopifnot(!any(vapply(instances,inherits,logical(1),"try-error")))
manifest <- list(version=1L,instances=instances)
write_json(manifest,file.path(args[2],"manifest.json"),auto_unbox=TRUE,pretty=TRUE,digits=NA)
write_json(manifest,file.path(args[3],"manifest.json"),auto_unbox=TRUE,pretty=TRUE,digits=NA)
cat("materialized",length(instances),"observations and separated labels\n")
