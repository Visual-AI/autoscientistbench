# Frozen execution interface. Counters are owned here; candidate code receives only callbacks.
# Counter/interface bypass still requires patch review: R and the output share a candidate process.
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 4L)
code_dir <- args[1]; model_dir <- args[2]; input_path <- args[3]; output_path <- args[4]
stopifnot(file.exists(file.path(model_dir, "model.txt")))
source(file.path(Sys.getenv("BENCH_BENCH_DIR"), "rnorm_reflectionmax.R"))
source(file.path(code_dir, "method.R"))
x <- read.csv(input_path, stringsAsFactors = FALSE, check.names = FALSE)
stopifnot(identical(names(x), c("case_id", "initial_1", "initial_2", "initial_3", "initial_4")))
stopifnot(nrow(x) > 1L, !anyDuplicated(x$case_id), all(is.finite(as.matrix(x[-1]))))
seed <- as.integer(Sys.getenv("BENCH_SEED", "1"))
cpus <- max(1L, as.integer(Sys.getenv("BENCH_CPUS", "1")))
run_case <- function(i) {
  set.seed(as.integer((as.double(seed) * 100003 + i * 1009) %% 2147483646 + 1))
  single <- 0L; coupled <- 0L; initialized <- 0L
  single_kernel <- function(state) {
    single <<- single + 1L
    list(position = 0.99 * state$position + rnorm(1, 0, 1))
  }
  coupled_kernel <- function(state1, state2) {
    coupled <<- coupled + 1L
    nextvalues <- rnorm_reflectionmax(mu1 = 0.99 * state1$position,
                                     mu2 = 0.99 * state2$position, sigma = 1)
    list(state1 = list(position = nextvalues$xy[1]),
         state2 = list(position = nextvalues$xy[2]), identical = nextvalues$identical)
  }
  rinit <- function(position) {
    if (missing(position)) {
      initialized <<- initialized + 1L
      stopifnot(initialized <= 4L)
      position <- as.numeric(x[i, initialized + 1L])
    }
    list(position = position)
  }
  result <- estimate_one(single_kernel, coupled_kernel, rinit, function(x) x)
  counted_cost <- single + 2 * coupled
  stopifnot(length(result$estimator) == 1L, is.finite(result$estimator),
            length(result$cost) == 1L, result$cost == counted_cost, counted_cost > 0)
  list(case_id = x$case_id[i], estimate = as.numeric(result$estimator),
       reported_cost = as.numeric(result$cost), single = single, coupled = coupled)
}
rows <- parallel::mclapply(seq_len(nrow(x)), run_case, mc.cores = min(cpus, nrow(x)), mc.set.seed = FALSE)
if (any(vapply(rows, inherits, logical(1), "try-error"))) stop("estimator worker failed")
# CSV carries separate result and frozen-callback ledger columns; scorer ignores the estimator's cost for arithmetic.
out <- do.call(rbind, lapply(rows, as.data.frame, stringsAsFactors = FALSE))
write.csv(out, output_path, row.names = FALSE, quote = TRUE)
cat("Actual reservoir UPAVE:", nrow(out), "cases; transitions:", sum(out$single + 2*out$coupled), "\n")
