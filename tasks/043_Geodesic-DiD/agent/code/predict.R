args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 3L)
input <- args[1]; output <- args[2]; code <- args[3]
source(file.path(code, "R", "gdd.R"))
source(file.path(code, "R", "lcm.R"))
library(truncnorm)
idx <- read.csv(file.path(input, "index.csv"), stringsAsFactors = FALSE)
stopifnot(nrow(idx) > 0, !anyDuplicated(idx$id))
M <- 100L
one <- function(k) {
  x <- read.csv(file.path(input, idx$file[k]), stringsAsFactors = FALSE)
  stopifnot(nrow(x) == idx$n[k], all(x$group %in% c(0, 1)))
  pre <- lapply(seq_len(nrow(x)), function(i)
    c(x$lower[i], qtruncnorm(seq_len(M-1L)/M, a=x$lower[i], b=x$upper[i],
                           mean=x$pre_location[i], sd=x$pre_scale[i]), x$upper[i]))
  post <- lapply(seq_len(nrow(x)), function(i)
    c(x$lower[i], qtruncnorm(seq_len(M-1L)/M, a=x$lower[i], b=x$upper[i],
                           mean=x$post_location[i], sd=x$post_scale[i]), x$upper[i]))
  fit <- gdd(pre[x$group == 0], post[x$group == 0], pre[x$group == 1], post[x$group == 1],
             optns=list(type='measure', lower=-3, upper=3))
  c(fit$gdd$start, fit$gdd$end)
}
workers <- max(1L, min(as.integer(Sys.getenv("BENCH_CPUS", "1")), nrow(idx)))
res <- if(workers == 1L) lapply(seq_len(nrow(idx)), one) else
  parallel::mclapply(seq_len(nrow(idx)), one, mc.cores=workers, mc.set.seed=FALSE)
if(any(vapply(res, inherits, logical(1), "try-error"))) stop("an inference worker failed")
mat <- do.call(rbind, res)
stopifnot(ncol(mat) == 202L, all(is.finite(mat)))
colnames(mat) <- c(sprintf("start_%03d", 0:M), sprintf("end_%03d", 0:M))
dir.create(dirname(output), recursive=TRUE, showWarnings=FALSE)
write.table(data.frame(id=idx$id, mat, check.names=FALSE), output, sep=",", row.names=FALSE,
            col.names=TRUE, quote=FALSE)
