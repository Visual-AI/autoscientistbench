## Shared helpers for the reproduction of Section 5.1 of
##   Zito, Rigon & Dunson (2024), "Bayesian nonparametric modeling of latent
##   partitions via Stirling-gamma priors", Bayesian Analysis.
## (added by the reproduction; the upstream package code under code/R, code/src
##  is untouched.)

## ---- experiment configuration (paper, Section 5.1) -------------------------
CFG <- list(
  n        = 500L,                       # n = 500 observations
  kstars   = c(2L, 4L, 6L),              # k* = 2, 4, 6 equally weighted components
  means    = list(`2` = rbind(c(-1, -1), c(1, 1)),
                  `4` = rbind(c(-1, -1), c(1, -1), c(-1, 1), c(1, 1)),
                  `6` = rbind(c(-1, -1), c(1, -1), c(-1, 1), c(1, 1),
                              c(3, 0), c(0, 3))),
  sigma    = diag(0.15, 2),              # variance-covariance diag{0.15, 0.15}
  nrep     = 20L,                        # 20 replicates per k*
  niter    = 15000L,                     # 15,000 Gibbs iterations
  burnin   = 5000L,                      # first 5,000 discarded as burn-in
  kappa0   = 2, nu0 = 2,                 # N(mu; 0, Sigma/kappa0) IW(Sigma; nu0, I)
  mu0      = c(0, 0), Psi0 = diag(2),
  kmax     = 80L,                        # k range pre-tabulated for the RU bounds
  data_seed = 20240601L,
  chain_seed = 771000L,
  init_type = 1L,                        # DP-prior draw with alpha = 1 (paper does not state an init)
  priors   = list(
    list(tag = "gamma24",  type = "gamma", a = 2, b = 4,
         label = "Gamma(2, 4)"),
    list(tag = "sg_4_1",   type = "sg",    a = 4, b = 1,
         label = "Sg(4, 1, n)"),
    list(tag = "sg_1_.25", type = "sg",    a = 1, b = 0.25,
         label = "Sg(1, 0.25, n)"))
)
CFG$m <- CFG$n                           # Stirling-gamma reference size m = n

## ---- paths (all inside this folder) ---------------------------------------
ROOT      <- Sys.getenv("REPRO_ROOT", unset = getwd())
RES_DIR   <- file.path(ROOT, "results")
LOG_DIR   <- file.path(ROOT, "logs")
DATA_DIR  <- file.path(ROOT, "data")
TRACE_DIR <- file.path(RES_DIR, "traces")
BUILD_DIR <- file.path(ROOT, ".build")

## ---- compile / load the added Rcpp sampler --------------------------------
load_sampler <- function(rebuild = FALSE) {
  Rcpp::sourceCpp(file.path(ROOT, "sim", "dpm_gibbs.cpp"),
                  cacheDir = file.path(BUILD_DIR, "rcpp-cache"),
                  rebuild = rebuild, verbose = FALSE)
  invisible(TRUE)
}

## ---- synthetic data of Section 5.1 ----------------------------------------
gen_dataset <- function(kstar, seed) {
  set.seed(seed)
  mu  <- CFG$means[[as.character(kstar)]]
  lab <- sample.int(kstar, CFG$n, replace = TRUE)          # equal weights 1/k*
  L   <- t(chol(CFG$sigma))
  z   <- matrix(rnorm(2L * CFG$n), CFG$n, 2) %*% t(L)
  z + mu[lab, ]
}

## ---- normal-inverse-Wishart log predictive (p = 2) ------------------------
lpred <- function(y1, y2, m, s1, s2, S11, S12, S22,
                  kappa0 = CFG$kappa0, nu0 = CFG$nu0) {
  kappa_m <- kappa0 + m; nu_m <- nu0 + m
  mu1 <- s1 / kappa_m; mu2 <- s2 / kappa_m
  c0 <- if (m > 0) kappa0 * m / kappa_m else 0
  d1 <- if (m > 0) s1 / m else 0; d2 <- if (m > 0) s2 / m else 0
  P11 <- 1 + S11 + c0 * d1 * d1; P12 <- S12 + c0 * d1 * d2; P22 <- 1 + S22 + c0 * d2 * d2
  df <- nu_m - 1; sc <- (kappa_m + 1) / (kappa_m * df)
  A11 <- P11 * sc; A12 <- P12 * sc; A22 <- P22 * sc; detA <- A11 * A22 - A12 * A12
  e1 <- y1 - mu1; e2 <- y2 - mu2
  m2 <- (A22 * e1 * e1 - 2 * A12 * e1 * e2 + A11 * e2 * e2) / detA
  lgamma((df + 2) / 2) - lgamma(df / 2) - log(df * pi) - 0.5 * log(detA) -
    ((df + 2) / 2) * log1p(m2 / df)
}

## log marginal likelihood of a block (joint under the NIW base measure),
## computed as the sequential product of the predictive densities
block_loglik <- function(blk) {
  m <- 0; s1 <- 0; s2 <- 0; S11 <- 0; S12 <- 0; S22 <- 0; tot <- 0
  for (k in seq_len(nrow(blk))) {
    tot <- tot + lpred(blk[k, 1], blk[k, 2], m, s1, s2, S11, S12, S22)
    mm <- m + 1
    if (m > 0) { f <- m / mm; e1 <- blk[k, 1] - s1 / m; e2 <- blk[k, 2] - s2 / m
      S11 <- S11 + f * e1 * e1; S12 <- S12 + f * e1 * e2; S22 <- S22 + f * e2 * e2 }
    s1 <- s1 + blk[k, 1]; s2 <- s2 + blk[k, 2]; m <- mm
  }
  tot
}

## ---- ratio-of-uniforms bounds, from the authors' package -------------------
## Mu[k], Mv[k] for the full conditional alpha | k ~ Sg(a + k, b + 1, m)
sg_bounds <- function(a, b, m, n, kmax = CFG$kmax) {
  Mu <- Mv <- numeric(kmax + 1)
  for (k in seq_len(kmax)) {
    bd <- ConjugateDP:::getBounds_MuMv_posterior(a, b, m, k, n)
    Mu[k + 1] <- bd$Mu; Mv[k + 1] <- bd$Mv
  }
  list(Mu = Mu, Mv = Mv)
}

## ---- one Gibbs chain -------------------------------------------------------
run_chain <- function(y, prior, niter = CFG$niter, seed, nn = nrow(y),
                      bounds = NULL, init_type = CFG$init_type) {
  set.seed(seed)
  if (prior$type == "sg") {
    bd <- if (is.null(bounds)) sg_bounds(prior$a, prior$b, CFG$m, nn) else bounds
    dpm_chain_cpp(y, as.integer(niter), 0L, 1L, prior$a, prior$b, CFG$m,
                  bd$Mu, bd$Mv, CFG$kappa0, CFG$nu0, CFG$mu0, CFG$Psi0,
                  prior$a, prior$b, 1, as.integer(init_type))
  } else {
    dpm_chain_cpp(y, as.integer(niter), 0L, 0L, 1, 1, CFG$m,
                  numeric(CFG$kmax + 1), numeric(CFG$kmax + 1),
                  CFG$kappa0, CFG$nu0, CFG$mu0, CFG$Psi0,
                  prior$a, prior$b, 1, as.integer(init_type))
  }
}

## ---- Stirling-gamma log normalizing constant by numerical integration ------
## log Z for the density (a-1)log(x) - b(lgamma(x+m) - lgamma(x)); the finite
## bounds below are wide enough for every hyper-parameter used in this study.
sg_log_normconst <- function(a, b, m) {
  gx <- seq(-12, 8, length.out = 400001)        # log x
  x  <- exp(gx)
  lf <- (a - 1) * gx - b * (lgamma(x + m) - lgamma(x))
  mx <- max(lf); w <- exp(lf - mx)
  dx <- c(diff(gx), diff(gx)[length(diff(gx))])
  mx + log(sum(w * exp(gx) * dx))
}
