## Monte-Carlo driver for the main simulation study of
##   Kassi & Wang, "Structural adaptation and rate accelerated estimation in
##   bivariate functional data" (Bernoulli), Section 4.2 / Figure 3.
##
## The authors' repository ships only the R package (no simulation script); this
## driver reconstructs the study from the paper's description, following the
## structure used by the authors in code/.Rhistory for the companion smoothing
## study (same simulator, same delta rules, same estimatation pipeline):
##
##   Y^(i)(t_m) = X^(i)(t_m) + sigma * eps,  i = 1..N,  m = 1..M0,
##   X = sum of two fBms of regularities H1 = 0.8 (along u1) and H2 = 0.5
##       (along u2), u1 = (cos a, sin a), u2 = (-sin a, cos a);
##   grid: M0 = side x side equally spaced points on [0,1]^2;
##   delta_learn = M0^(-1/4) = 1/sqrt(side);
##   spacing grid for identification: Delta = seq(1/sqrt(side), 0.4, length.out = 15);
##   estimate_angle -> identify_angle -> angle_correct  (Algorithms 1 & 2 of the paper);
##   risk  R_alpha = |alpha_adj - alpha|   (eq. (36)).
##
## 60 configurations: N in {100,150} x side in {51,101} x sigma in {0.1,0.5,1}
##                    x alpha in {pi/30, pi/5, pi/4, pi/3, pi/2 - pi/30}.
##
## Usage:
##   Rscript driver.R --out <dir> --nrep <R> --cores <k> [--engine fast|orig]
##                    [--config-ids 1,2,3] [--tag label]
## Each replication = one independent learning set of N surfaces; one risk value.

suppressPackageStartupMessages({
  library(purrr)
  library(pracma)
})

script_dir <- function() {
  a <- commandArgs(trailingOnly = FALSE)
  f <- sub("^--file=", "", a[grep("^--file=", a)])
  if (length(f) == 0) return(getwd())
  dirname(normalizePath(f))
}
SIM_DIR <- script_dir()
CODE_ROOT <- normalizePath(file.path(SIM_DIR, ".."))

source(file.path(SIM_DIR, "fast_kernels.R"))
local({
  for (f in c("utils.R", "dgp.R", "smoothing.R", "parameters.R", "angle_estim.R", "ani_detect.R")) {
    source(file.path(CODE_ROOT, "R", f))
  }
}, envir = globalenv())

## ---------------------------- CLI ------------------------------------------
parse_args <- function(args) {
  get <- function(name, default = NULL) {
    i <- match(name, args)
    if (is.na(i) || i == length(args)) return(default)
    args[i + 1]
  }
  list(out = get("--out", file.path("results", "raw")),
       nrep = as.integer(get("--nrep", "10")),
       cores = as.integer(get("--cores", "12")),
       engine = get("--engine", "fast"),
       config_ids = get("--config-ids", NULL),
       tag = get("--tag", "run"),
       seed0 = as.integer(get("--seed0", "2026")))
}

## ------------------------- configuration grid ------------------------------
make_grid <- function() {
  g <- expand.grid(N = c(100, 150),
                   side = c(51, 101),
                   sigma = c(0.1, 0.5, 1),
                   alpha = c(pi/30, pi/5, pi/4, pi/3, pi/2 - pi/30),
                   KEEP.OUT.ATTRS = FALSE, stringsAsFactors = FALSE)
  g$id <- seq_len(nrow(g))
  g$M0 <- g$side^2
  g$delta_learn <- 1 / sqrt(g$side)
  g$cost <- g$N * g$M0
  g[order(g$id), ]
}

## --------------------------- one replication -------------------------------
run_rep <- function(cfg, rep, seed, engine) {
  set.seed(seed)
  side <- cfg$side
  xout <- seq(0, 1, length.out = side)
  H1 <- 0.8; H2 <- 0.5
  sim <- if (identical(engine, "fast")) fbm_sheet_f else fbm_sheet
  est_angle <- if (identical(engine, "fast")) estimate_angle_f else estimate_angle
  id_angle <- if (identical(engine, "fast")) identify_angle_f else identify_angle
  correct <- if (identical(engine, "fast")) angle_correct_f else angle_correct
  X_list <- lapply(seq_len(cfg$N), function(i) {
    list(t = xout,
         X = sim(t_n = side, e_n = side, alpha = cfg$alpha, H1 = H1, H2 = H2,
                 type = "sum", sigma = cfg$sigma))
  })
  delta_learn <- 1 / sqrt(side)
  delta_grid <- seq(1 / sqrt(side), 0.4, length.out = 15)
  g_hat <- est_angle(X_list = X_list, xout = xout, delta = delta_learn)
  alpha_hat <- id_angle(angles = g_hat, X_list = X_list, dout = delta_grid,
                        xout = xout, sigma = g_hat$sigma_hat)
  adj <- correct(X_list = X_list, xout = xout, g_hat = g_hat$g_hat,
                 alpha_hat = alpha_hat$alpha, delta = delta_learn,
                 Hmax = alpha_hat$H_max, Hmin = g_hat$H_min, sigma = g_hat$sigma_hat)
  data.frame(cfg_id = cfg$id, N = cfg$N, side = side, M0 = side^2,
             sigma = cfg$sigma, alpha = cfg$alpha, rep = rep,
             alpha_hat_raw = as.numeric(alpha_hat$alpha),
             alpha_hat_adj = as.numeric(adj$alpha_adj),
             raw_kind = names(alpha_hat$alpha),
             H_max_hat = alpha_hat$H_max, H_min_hat = g_hat$H_min,
             sigma_hat = g_hat$sigma_hat,
             risk_raw = abs(as.numeric(alpha_hat$alpha) - cfg$alpha),
             risk_adj = abs(as.numeric(adj$alpha_adj) - cfg$alpha),
             H_max_err = abs(alpha_hat$H_max - H1),
             H_min_err = abs(g_hat$H_min - H2),
             stringsAsFactors = FALSE)
}

## ------------------------------- main ---------------------------------------
main <- function() {
  a <- parse_args(commandArgs(trailingOnly = TRUE))
  grid <- make_grid()
  if (!is.null(a$config_ids)) {
    ids <- as.integer(strsplit(a$config_ids, ",")[[1]])
    grid <- grid[grid$id %in% ids, , drop = FALSE]
  }
  tasks <- do.call(rbind, lapply(seq_len(nrow(grid)), function(k) {
    data.frame(cfg_row = k, rep = seq_len(a$nrep), stringsAsFactors = FALSE)
  }))
  tasks$cost <- grid$cost[tasks$cfg_row]
  tasks <- tasks[order(-tasks$cost), , drop = FALSE]

  dir.create(a$out, recursive = TRUE, showWarnings = FALSE)
  t0 <- Sys.time()
  cat(sprintf("[driver] engine=%s cores=%d configs=%d reps/config=%d tasks=%d\n",
              a$engine, a$cores, nrow(grid), a$nrep, nrow(tasks)))
  flush.console()

  res <- parallel::mclapply(seq_len(nrow(tasks)), function(i) {
    cfg <- grid[tasks$cfg_row[i], , drop = FALSE]
    seed <- a$seed0 + 1000L * cfg$id + tasks$rep[i]
    out <- run_rep(cfg, rep = tasks$rep[i], seed = seed, engine = a$engine)
    if (tasks$rep[i] %% 5 == 0) {
      cat(sprintf("[driver] %s rep %d/%d done (%.0f s elapsed)\n",
                  a$tag, tasks$rep[i], a$nrep,
                  as.numeric(difftime(Sys.time(), t0, units = "secs"))))
      flush.console()
    }
    out
  }, mc.cores = a$cores, mc.preschedule = FALSE)

  ok <- vapply(res, is.data.frame, logical(1))
  if (!all(ok)) {
    for (i in which(!ok)) print(res[[i]])
    stop(sprintf("%d of %d tasks failed", sum(!ok), length(ok)))
  }
  out <- do.call(rbind, res)
  out <- out[order(out$cfg_id, out$rep), , drop = FALSE]
  out$engine <- a$engine
  out$tag <- a$tag
  out$seed <- a$seed0 + 1000L * out$cfg_id + out$rep
  fn <- file.path(a$out, sprintf("replications_%s_%s.csv", a$tag, a$engine))
  utils::write.csv(out, fn, row.names = FALSE)
  cat(sprintf("[driver] wrote %s (%d rows) in %.1f s\n", fn, nrow(out),
              as.numeric(difftime(Sys.time(), t0, units = "secs"))))
  flush.console()
}

if (!nzchar(Sys.getenv("DIREG_NO_MAIN"))) main()
