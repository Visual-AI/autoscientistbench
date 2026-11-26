#!/usr/bin/env Rscript
## Build the synthetic data of the paper's main experiment (Section 5.1):
## 3 choices of k*, 20 datasets each, n = 500 observations from an equally
## weighted mixture of k* bivariate normals with Sigma = diag{0.15, 0.15}.
args <- commandArgs(trailingOnly = FALSE)
fa <- sub("^--file=", "", args[grep("^--file=", args)])
root <- if (length(fa)) normalizePath(file.path(dirname(fa[1]), "..", "..")) else getwd()
Sys.setenv(REPRO_ROOT = root)
source(file.path(root, "code", "sim", "common.R"))
dir.create(file.path(DATA_DIR, "sim51"), recursive = TRUE, showWarnings = FALSE)

idx <- list()
for (ks in CFG$kstars) for (r in seq_len(CFG$nrep)) {
  seed <- CFG$data_seed + 1000L * ks + r
  notice <- sprintf("kstar%d_rep%02d", ks, r)
  y <- gen_dataset(ks, seed)
  saveRDS(list(y = y, kstar = ks, rep = r, seed = seed,
               means = CFG$means[[as.character(ks)]], sigma = CFG$sigma),
          file.path(DATA_DIR, "sim51", paste0(notice, ".rds")))
  idx[[notice]] <- list(notice = notice, kstar = ks, rep = r, seed = seed)
}
saveRDS(idx, file.path(DATA_DIR, "sim51", "index.rds"))
cat(sprintf("wrote %d datasets to %s (n = %d, k* in {%s}, Sigma = diag(0.15, 0.15))\n",
            length(idx), file.path(DATA_DIR, "sim51"), CFG$n,
            paste(CFG$kstars, collapse = ", ")))
