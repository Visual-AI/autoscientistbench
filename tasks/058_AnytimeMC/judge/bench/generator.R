# Offline materialization only: no sequential_BH invocation and no result reuse.
args <- commandArgs(trailingOnly=TRUE)
out <- args[1]; M <- as.integer(args[2]); B <- as.integer(args[3]); nsim <- as.integer(args[4])
seed <- as.integer(args[5]); cell <- as.integer(args[6]); ids <- readLines(args[7])
stopifnot(length(ids)==nsim,cell>=1,cell<=4)
set.seed(seed); options(digits=17)
for (c in seq_len(cell)) {
  pi_A <- if (cell==4) c/10 else 0.4
  Z <- matrix(NA_real_,M,nsim); hypo <- matrix(NA_integer_,M,nsim)
  for (j in seq_len(nsim)) {
    hypo[,j] <- rbinom(M,1,pi_A)
    X <- rnorm(M)
    Z[,j] <- 2.5*hypo[,j]+X
  }
  states <- list()
  for (k in seq_len(nsim)) {
    invisible(rnorm(M*200L))
    states[[ids[k]]] <- list(state=.Random.seed,M=M,B=B)
    invisible(rnorm(M*B))
  }
}
observed <- do.call(rbind,lapply(seq_len(nsim),function(k)
  data.frame(trial_id=ids[k],hypothesis=seq_len(M),observed=sprintf("%.17g",Z[,k]),B=B,alpha=0.1)))
truth <- do.call(rbind,lapply(seq_len(nsim),function(k)
  data.frame(trial_id=ids[k],hypothesis=seq_len(M),alternative=hypo[,k])))
write.table(observed,file.path(out,"input.csv"),sep=",",row.names=FALSE,quote=FALSE)
write.table(truth,file.path(out,"truth.csv"),sep=",",row.names=FALSE,quote=FALSE)
saveRDS(states,file.path(out,"states.rds"),compress=FALSE)
