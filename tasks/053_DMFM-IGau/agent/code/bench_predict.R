suppressPackageStartupMessages({library(GIGrvg); library(extraDistr); library(jsonlite)})
# These files are editable candidate source, never a precompiled method package.
source("IG_functions_DMFM_network.R")
source("IG_functions_general.R")
author_sample_Alp_telescope <- sample_Alp_telescope
# Preserve the hard reproduction's Section 4.3.1 F(6,3) prior wrapper.
sample_Alp_telescope <- function(Alp, df1, df2, u, k, m, c, s_pro) {
  author_sample_Alp_telescope(Alp, df1=6, df2=3, u, k, m, c, s_pro)
}
args <- commandArgs(TRUE)
getarg <- function(key) { i <- match(key,args); stopifnot(!is.na(i), i < length(args)); args[i+1L] }
model <- fromJSON(file.path(getarg("--model"), "protocol.json"))
seed <- as.integer(Sys.getenv("BENCH_SEED"))
stopifnot(is.finite(seed), model$seed == seed)
cores <- as.integer(Sys.getenv("BENCH_CPUS", "1")); stopifnot(cores >= 1L)
data <- fromJSON(getarg("--input"), simplifyVector=FALSE)
smoke <- identical(Sys.getenv("BENCH_SMOKE"), "1")
if (smoke) {
  stopifnot(identical(data$schema,"sta26-smoke-v1"), length(data$networks)==2L)
  iters <- 200L; burn <- 100L
} else {
  stopifnot(identical(data$schema,"sta26-network-v1"), length(data$networks)==50L)
  iters <- 100000L; burn <- 50000L
}
cat("SAMPLER TS_IG_network_DMFM_BNB; candidate source:", normalizePath("IG_functions_DMFM_network.R"), "\n")
cat("iterations",iters,"burn",burn,"retained",iters-burn,"networks",length(data$networks),"smoke",smoke,"\n")
print(tools::md5sum(c("IG_functions_DMFM_network.R","IG_functions_general.R","bench_predict.R")))
work <- tempfile("sta26-draws-"); dir.create(work)
worker <- function(r) {
  compiler::enableJIT(3L)
  x <- data$networks[[r]]
  Ad <- matrix(as.integer(unlist(x$adjacency)),nrow=50L,ncol=50L,byrow=TRUE)
  stopifnot(identical(Ad,t(Ad)),all(diag(Ad)==0),all(Ad %in% 0:1))
  # BENCH_SEED=1 recovers source chain seeds 1001,...,1050.
  chain_seed <- as.integer((1000 + r + (as.double(seed)-1)*100000) %% 2147483647)
  set.seed(chain_seed)
  start <- proc.time()[["elapsed"]]
  History <- TS_IG_network_DMFM_BNB(Ad,iters,1,1,al_lam=1,a_pi=4,b_pi=3,M_max=50)
  keep <- seq.int(burn+1L,iters)
  labels <- t(vapply(History[keep], function(x) as.integer(x$c), integer(50)))
  k <- vapply(History[keep], function(x) as.integer(x$k), integer(1))
  M <- vapply(History[keep], function(x) as.integer(x$M), integer(1))
  M_na <- vapply(History[keep], function(x) as.integer(x$M_na), integer(1))
  rm(History); gc()
  stopifnot(nrow(labels)==iters-burn,all(M==k+M_na),all(M<=50L))
  out <- data.frame(id=rep(x$id,length(keep)),iteration=keep,k=k,M=M,M_na=M_na,labels,check.names=FALSE)
  names(out) <- c("id","iteration","k","M","M_na",paste0("c",1:50))
  file <- file.path(work,sprintf("%03d.csv",r))
  write.table(out,file,sep=",",quote=FALSE,row.names=FALSE,col.names=(r==1L))
  cat(sprintf("network=%d id=%s saved=%d elapsed=%.3f\n",r,x$id,nrow(labels),proc.time()[["elapsed"]]-start))
  file
}
files <- parallel::mclapply(seq_along(data$networks),worker,mc.cores=min(cores,length(data$networks)),mc.preschedule=FALSE,mc.set.seed=FALSE)
stopifnot(all(vapply(files,function(x) is.character(x)&&length(x)==1L&&file.exists(x),logical(1))))
output <- getarg("--output"); partial <- paste0(output,".tmp")
stopifnot(file.create(partial))
for (file in files) stopifnot(file.append(partial,file))
stopifnot(file.rename(partial,output))
unlink(work,recursive=TRUE)
