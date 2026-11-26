## Smoke test used by prepare_data.sh: the authors' simulator must be importable
## and must produce the design used by the main experiment.
suppressPackageStartupMessages({library(purrr); library(pracma)})
a <- commandArgs(trailingOnly = FALSE)
f <- sub("^--file=", "", a[grep("^--file=", a)])
root <- normalizePath(file.path(dirname(normalizePath(f)), "..", ".."))
for (src in c("utils.R", "dgp.R", "smoothing.R", "parameters.R", "angle_estim.R", "ani_detect.R")) {
  source(file.path(root, "code", "R", src))
}
set.seed(1)
for (side in c(51, 101)) {
  X <- fbm_sheet(t_n = side, e_n = side, alpha = pi/3, H1 = 0.8, H2 = 0.5,
                 type = "sum", sigma = 0.5)
  stopifnot(all(dim(X) == c(side, side)), all(is.finite(X)),
            stats::sd(X) > 0)
  cat(sprintf("simulator OK: %d x %d surface, sd = %.4f\n", side, side, stats::sd(X)))
}
## the estimator pipeline must run on one small surface list
xout <- seq(0, 1, length.out = 51)
set.seed(2)
X_list <- lapply(1:20, function(i) list(t = xout,
  X = fbm_sheet(t_n = 51, e_n = 51, alpha = pi/4, H1 = 0.8, H2 = 0.5, type = "sum", sigma = 0.1)))
g <- estimate_angle(X_list, xout = xout, delta = 1/sqrt(51))
a <- identify_angle(g, X_list, dout = seq(1/sqrt(51), 0.4, length.out = 15), xout = xout,
                    sigma = g$sigma_hat)
cat(sprintf("estimator OK: alpha_hat = %.4f (true %.4f), sigma_hat = %.4f\n",
            as.numeric(a$alpha), pi/4, g$sigma_hat))
cat("PREPARE CHECK PASSED\n")
