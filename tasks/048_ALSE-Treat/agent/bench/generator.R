# Operator-only data materialization: no estimator fitting and no result replay.
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args)==5)
out <- args[1]; seed <- as.integer(args[2]); prefix <- args[3]
reps <- as.integer(args[4]); sizes <- as.integer(strsplit(args[5], ",", fixed=TRUE)[[1]])
dir.create(out,recursive=TRUE,showWarnings=FALSE)
set.seed(seed)
generate_data_simple <- function(n) {
  tibble::tibble(
    X1 = runif(n, -1, 1),
    X2 = runif(n, -1, 1),
    X3 = runif(n, -1, 1),
    A = rnorm(n, mean = X1 + 0.5 * X1^3 - 2 * X2^2 + X1^2 * X2, sd = 1 + X1^2),
    Y = rnorm(n, mean = A * (1 + X1 - X1^2 - 0.5 * X2^2) + X2 * (X3 - X1^2))
  )
}
cases <- list(); observations <- list(); index <- 0L
for (replicate in seq_len(reps)) for (n in sizes) {
  index <- index+1L
  id <- sprintf("%s_%04d", prefix, index)
  cases[[index]] <- list(case_id=id,n=n)
  observations[[id]] <- generate_data_simple(n)
}
saveRDS(observations, file.path(out,"observations.rds"),compress=FALSE)
jsonlite::write_json(list(format="alse-observations-v1",cases=cases),file.path(out,"manifest.json"),auto_unbox=TRUE,pretty=TRUE)
