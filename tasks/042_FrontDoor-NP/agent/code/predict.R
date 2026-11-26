# Protocol adaptation of the authors' sim2-weak-overlap/binaryM/main.R.
# The estimator call and Wald interval are unchanged; observations arrive as a file.
args <- commandArgs(trailingOnly=TRUE)
opt <- list()
if (length(args) %% 2L != 0L) stop("expected --name value arguments")
for (i in seq(1L, length(args), by=2L)) opt[[args[i]]] <- args[i+1L]
stopifnot(all(c("--input", "--output", "--model") %in% names(opt)))
stopifnot(file.exists(file.path(opt[["--model"]], "model.json")))
suppressPackageStartupMessages(library(fdcausal))
data_all <- read.csv(opt[["--input"]], check.names=FALSE)
stopifnot(all(c("case_id", "X", "A", "M", "Y") %in% names(data_all)))
ids <- unique(data_all$case_id)
datasets <- split(data_all, factor(data_all$case_id, levels=ids))
fit_one <- function(index) {
  data <- datasets[[index]]
  n <- nrow(data)
  stopifnot(n == 500L)
  data$case_id <- NULL
  # Cross-fitting and SuperLearner remain disabled, as in the published driver.
  set.seed(as.integer(Sys.getenv("BENCH_SEED", "1")) + index)
  fit <- function(a) estfd(a=a, data=data, treatment="A", mediators=c("M"),
    outcome="Y", covariates=c("X"), onestep=TRUE, mediator.method="bayes",
    superlearner=FALSE, crossfit=FALSE, K=5, lib=c("SL.ranger"),
    n.iter=15000, eps=TRUE, cvg.criteria=0.001, linkA="identity",
    truncate_lower=0.001, truncate_upper=0.999, ATT=FALSE)
  y1 <- fit(1)$TMLE
  y0 <- fit(0)$TMLE
  estimate <- y1$estimated_psi - y0$estimated_psi
  half <- 1.96 * sqrt(mean((y1$EIF-y0$EIF)^2)/n)
  data.frame(case_id=ids[index], estimate=estimate, lower=estimate-half,
             upper=estimate+half)
}
cores <- max(1L, min(length(ids), as.integer(Sys.getenv("BENCH_CPUS", "1"))))
fits <- parallel::mclapply(seq_along(ids), fit_one, mc.cores=cores,
                          mc.preschedule=FALSE, mc.set.seed=FALSE)
if (any(vapply(fits, inherits, logical(1), "try-error"))) stop("at least one estimator failed")
result <- do.call(rbind, fits)
if (any(!is.finite(as.matrix(result[,c("estimate","lower","upper")])))) stop("nonfinite estimate")
write.table(result, file=opt[["--output"]], sep=",", row.names=FALSE, col.names=TRUE,
            quote=FALSE)
