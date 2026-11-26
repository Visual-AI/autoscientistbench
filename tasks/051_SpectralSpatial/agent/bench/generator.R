# Frozen operator generator. Only observed X and Y are serialized into input/.
# Usage: Rscript generator.R BENCH_DIR OUTPUT_ROOT GRID_SIDE REPS SEED_OFFSET
args <- commandArgs(trailingOnly=TRUE)
bench <- args[1]; root <- args[2]; n1 <- as.integer(args[3])
reps <- as.integer(args[4]); offset <- as.integer(args[5])
src <- readLines(file.path(bench, "functions_sim.R"))
src[grep("^registerDoParallel", src)] <- "foreach::registerDoSEQ()"
suppressMessages(eval(parse(text=src), envir=globalenv()))
load(file.path(bench,"B1.RData")); load(file.path(bench,"B2.RData"))
beta <- matrix(c(3,-3,2,3,5,4,-2,4,5,7,2,-3,3,4,6,1,-4,2,3,4,5,1,3,4,6,
                 rep(0,25)), nrow=10,ncol=5,byrow=TRUE)/10
n <- n1*n1; Qmat <- makeQ(n1)
settings <- data.frame(setting=1:5,phi=c(1,1,1,2,2),bXZ=c(0,1,2,1,2))
dir.create(file.path(root,"input"),recursive=TRUE,showWarnings=FALSE)
manifest <- list(); labels <- list(); seeds <- list(); index <- 0L
for (s in seq_len(5)) for (rep in seq_len(reps)) {
  index <- index+1L; id <- sprintf("case%04d",index); seed <- offset+rep
  set.seed(seed)
  dat <- create_set7(n,10,10,5,Qmat,rep(3,9),rep(16,10),rep(2,5),rep(4,5),
                     .9999,.9,.9,beta,B1,B2,.9,.9,n1,settings$bXZ[s],settings$phi[s])
  observed <- list(X=dat$X,Y=dat$Y)
  stopifnot(identical(names(observed),c("X","Y")),all(dim(observed$X)==c(n,10)),
            all(dim(observed$Y)==c(n,5)),all(is.finite(observed$X)),all(is.finite(observed$Y)))
  saveRDS(observed,file.path(root,"input",paste0(id,".rds")),version=2)
  manifest[[index]] <- data.frame(id=id,n=n,grid_side=n1,observed_file=paste0(id,".rds"))
  for (e in 1:9) for (r in 1:5) labels[[length(labels)+1L]] <-
    data.frame(id=id,setting=s,exposure=e,outcome=r,truth=beta[e+1L,r])
  seeds[[index]] <- data.frame(id=id,setting=s,replicate=rep,generation_seed=seed)
  cat(sprintf("generated %s grid=%d setting=%d replicate=%d\n",id,n1,s,rep))
}
write.csv(do.call(rbind,manifest),file.path(root,"input","manifest.csv"),row.names=FALSE)
write.csv(do.call(rbind,labels),file.path(root,"labels.csv"),row.names=FALSE)
write.csv(do.call(rbind,seeds),file.path(root,"generation_seeds.csv"),row.names=FALSE)
