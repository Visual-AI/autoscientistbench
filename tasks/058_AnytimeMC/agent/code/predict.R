args <- commandArgs(trailingOnly=TRUE)
if (length(args) != 6L || !setequal(args[c(1,3,5)], c("--model","--input","--output"))) stop("invalid arguments")
a <- setNames(args[c(2,4,6)],args[c(1,3,5)])
seed <- as.integer(Sys.getenv("BENCH_SEED", "1")); if (is.na(seed)) stop("invalid seed")
set.seed(seed)
source("Multiple testing/sequential_BH.R")
source(file.path(Sys.getenv("BENCH_BENCH_DIR"),"sampler.R"))
x <- read.csv(a[["--input"]], stringsAsFactors=FALSE)
outputs <- list()
for (trial in unique(x$trial_id)) {
  z <- x[x$trial_id==trial,]; M <- nrow(z); B <- unique(z$B); alpha <- unique(z$alpha)
  stopifnot(identical(as.integer(z$hypothesis),seq_len(M)),length(B)==1,length(alpha)==1)
  sampler <- open_sampler(trial, M, B)
  # Preserve original observed statistics and raw null values; the trusted sampler counts disclosures.
  fit <- sequential_BH(z$observed, sampler, M, B, 1:B, alpha,
                       param_bm=0.9, param_bc=10, method="bc")
  counts <- sampler$counts()
  if (!identical(as.integer(fit$stop),as.integer(counts))) stop("stopping times differ from sampler counts")
  outputs[[length(outputs)+1L]] <- data.frame(trial_id=trial,hypothesis=seq_len(M),
    rejected=as.integer(seq_len(M) %in% fit$rejects),draws=counts,sampler_only=1L)
}
write.table(do.call(rbind,outputs),a[["--output"]],sep=",",row.names=FALSE,quote=FALSE)
cat("Completed actual sequential_BH method bc; trials=",length(outputs),"\n",sep="")
