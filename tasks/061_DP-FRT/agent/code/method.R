args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args)==3, file.exists(file.path(args[1], 'ready')))
source('DP-FRT.R')
input <- read.csv(args[2], stringsAsFactors=FALSE)
stopifnot(identical(names(input), c('id','n1','n0','epsilon','t11','t01')))
seed <- as.integer(Sys.getenv('BENCH_SEED','1'))
one <- function(i) {
  x <- input[i,]
  # Raw private counts are unavailable; placeholders are unused with noisy_counts.
  fit <- dp_frt(x$n1,x$n0,0,0,x$epsilon, prior=list(type='uniform'),
                method='sampling', R=10000, alpha_CI=0.05, seed=seed+i,
                clip=TRUE, noisy_counts=c(x$t11,x$t01))
  data.frame(id=x$id, lo=fit$summaries$credible_interval[1],
             hi=fit$summaries$credible_interval[2],
             credible_set=paste(sprintf('%.12f',fit$summaries$credible_set),collapse=';'))
}
out <- parallel::mclapply(seq_len(nrow(input)),one,mc.cores=max(1L,as.integer(Sys.getenv('BENCH_CPUS','1'))),mc.set.seed=FALSE)
if(any(vapply(out,inherits,FALSE,'try-error'))) stop('method worker failed')
write.csv(do.call(rbind,out),args[3],row.names=FALSE)
