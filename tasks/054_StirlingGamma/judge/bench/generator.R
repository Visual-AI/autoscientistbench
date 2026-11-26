# Frozen original Section 5.1 CFG and generator, copied without arithmetic changes.
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


gen_dataset <- function(kstar, seed) {
  set.seed(seed)
  mu  <- CFG$means[[as.character(kstar)]]
  lab <- sample.int(kstar, CFG$n, replace = TRUE)          # equal weights 1/k*
  L   <- t(chol(CFG$sigma))
  z   <- matrix(rnorm(2L * CFG$n), CFG$n, 2) %*% t(L)
  z + mu[lab, ]
}

