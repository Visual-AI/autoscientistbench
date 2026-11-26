#!/usr/bin/env Rscript
## Main computation of the paper's Section 5.1 experiment: for every choice of
## prior on alpha, every k* and every replicate, run the collapsed Gibbs
## sampler (Neal 2000, Algorithm 3) for 15,000 iterations and store the
## post-burn-in trajectories of alpha and of the number of clusters K_n.
suppressWarnings(suppressMessages(library(parallel)))
args <- commandArgs(trailingOnly = FALSE)
fa <- sub("^--file=", "", args[grep("^--file=", args)])
root <- if (length(fa)) normalizePath(file.path(dirname(fa[1]), "..", "..")) else getwd()
Sys.setenv(REPRO_ROOT = root)
source(file.path(root, "code", "sim", "common.R"))
load_sampler()

opts <- list(niter = CFG$niter, reps = CFG$nrep, threads = 12L, priors = CFG$priors)
a <- commandArgs(trailingOnly = TRUE)
for (i in seq_along(a)) {
  if (a[i] == "--niter")   opts$niter   <- as.integer(a[i + 1])
  if (a[i] == "--reps")    opts$reps    <- as.integer(a[i + 1])
  if (a[i] == "--threads") opts$threads <- as.integer(a[i + 1])
  if (a[i] == "--prior")   opts$priors  <- Filter(function(p) p$tag == a[i + 1], CFG$priors)
}
niter <- opts$niter
dir.create(TRACE_DIR, recursive = TRUE, showWarnings = FALSE)

## ratio-of-uniforms bounds for the Stirling-gamma full conditionals (built
## once per prior with the authors' ConjugateDP:::getBounds_MuMv_posterior)
bounds <- lapply(opts$priors, function(p)
  if (p$type == "sg") sg_bounds(p$a, p$b, CFG$m, CFG$n) else NULL)

tasks <- list()
for (pi in seq_along(opts$priors)) for (ks in CFG$kstars) for (r in seq_len(opts$reps))
  tasks[[length(tasks) + 1L]] <- list(pi = pi, kstar = ks, rep = r)

t0 <- Sys.time()
cat(sprintf("[train] %d chains x %d iterations on %d threads\n", length(tasks), niter, opts$threads))
res <- mclapply(tasks, function(tk) {
  prior <- opts$priors[[tk$pi]]
  d <- readRDS(file.path(DATA_DIR, "sim51", sprintf("kstar%d_rep%02d.rds", tk$kstar, tk$rep)))
  seed <- CFG$chain_seed + 1000L * tk$pi + 10L * tk$kstar + tk$rep
  g <- run_chain(d$y, prior, niter, seed, nn = CFG$n, bounds = bounds[[tk$pi]])
  sel <- (CFG$burnin + 1L):niter
  if (niter <= CFG$burnin) sel <- seq_len(niter)
  obj <- list(prior = prior$tag, prior_label = prior$label, kstar = tk$kstar,
              rep = tk$rep, seed = seed, niter = niter, burnin = CFG$burnin,
              alpha = g$alpha[sel], k = g$k[sel])
  saveRDS(obj, file.path(TRACE_DIR, sprintf("%s_k%d_rep%02d.rds", prior$tag, tk$kstar, tk$rep)))
  data.frame(prior = prior$tag, kstar = tk$kstar, rep = tk$rep,
             mean_alpha = mean(g$alpha[sel]), mean_k = mean(g$k[sel]),
             file = sprintf("%s_k%d_rep%02d.rds", prior$tag, tk$kstar, tk$rep))
}, mc.cores = opts$threads, mc.preschedule = FALSE)
idx <- do.call(rbind, res)
saveRDS(idx, file.path(RES_DIR, "train_index.rds"))
cat(sprintf("[train] done in %.1f s; %d traces in %s\n",
            as.numeric(difftime(Sys.time(), t0, units = "secs")), nrow(idx), TRACE_DIR))
