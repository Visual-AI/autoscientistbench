suppressPackageStartupMessages(library(hFDR))
suppressPackageStartupMessages(library(jsonlite))
source(file.path("R", "methods.R"))
args <- commandArgs(TRUE)
stopifnot(length(args)==6L, identical(args[c(1,3,5)], c("--model","--input","--output")))
model <- fromJSON(file.path(args[2], "protocol.json"))
manifest <- fromJSON(file.path(args[4], "manifest.json"), simplifyVector=FALSE)
cpus <- max(1L, as.integer(Sys.getenv("BENCH_CPUS", "1")))
seed <- as.integer(Sys.getenv("BENCH_SEED", as.character(model$seed)))
settings <- list(method_names=c("hFDR","std"),sel_method="lasso",model="gausslinear")
compute <- function(index) {
 item <- manifest$instances[[index]]
 obs <- readRDS(file.path(args[4], item$file))
 stopifnot(identical(names(obs),c("X","y","lambda")), is.matrix(obs$X), length(obs$y)==nrow(obs$X))
 # Method randomness is independent of the generator's private randomness.
 set.seed(as.integer((as.double(seed) + index*104729) %% 2147483646 + 1))
 result <- calc_hFDR(obs$X,obs$y,obs$lambda,settings)
 cat(sprintf("hFDR instance %d of %d completed\n",index,length(manifest$instances)),file=stderr())
 list(id=item$id,hFDR=unname(result$hFDR),hFDR_se=unname(result$std))
}
results <- parallel::mclapply(seq_along(manifest$instances),compute,mc.cores=cpus,mc.preschedule=FALSE,mc.set.seed=FALSE)
stopifnot(!any(vapply(results,inherits,logical(1),"try-error")))
write_json(list(version=1L,estimates=results),args[6],auto_unbox=TRUE,digits=NA,na="null")
