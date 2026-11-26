# Sta-20 reproduction: the paper's headline simulation experiment (Figure 5,
# "Performance of hFDR and s.e. in the Gaussian linear model").
#
# This file is a copy of the authors' R/settings/fixedX-lasso.R with exactly two
# changes, both documented in REPRODUCTION.md:
#   1. X_types keeps only the four primary designs of Figure 5 (the authors' file
#      also runs the "MCC" design, which is the separate Figure 9 "unfavorable
#      case" and is out of scope here).
#   2. sample_size / n_cores can be overridden through the environment variables
#      HFDR_SAMPLE_SIZE / HFDR_N_CORES so that run_train.sh can size the run to
#      the available CPU budget (defaults are the paper's values: 200 / 12).
library(here)
library(hFDR)

source(here("R", "utils.R"))
source(here("R", "methods.R"))

experiment <- "fixedX-lasso-main"
sel_method <- "lasso"
model <- "gausslinear"

p_seq <- 500
n_seq <- 3*p_seq
pi1_seq <- 30 / p_seq

X_seed <- 2023
cov_seed <- -2023

X_types <- c("IID_Normal", "X_AR", "Coef_AR", "Sparse")
posit_types <- rep("random", length(X_types))
model_Xs <- !(X_types %in% c("MCC"))
random_Xs <- !(X_types %in% c("MCC"))
random_covs <- c(F)
scaling_Xs <- c(F)

X_mismodel <- F
y_mismodel <- F

target_at_alpha <- 0.2
targets <- c(0.8)
calib_methods <- c("lasso")

noises <- c(quote(rnorm(n)))

fig_var <- list(name = "Design Matrix", value = X_types)

makeup_vectors(p_seq = p_seq, n_seq = n_seq,
               X_types = X_types, posit_types = posit_types,
               model_Xs = model_Xs, random_Xs = random_Xs,
               random_covs = random_covs, scaling_Xs = scaling_Xs,
               pi1_seq = pi1_seq, targets = targets, calib_methods = calib_methods,
               noises = noises)

FDR_range <- c(0.01, 0.75)
n_tune <- 10
sample_size <- as.integer(Sys.getenv("HFDR_SAMPLE_SIZE", "200"))

n_cores <- as.integer(Sys.getenv("HFDR_N_CORES", "12"))

method_names <- c("hFDR", "std")

fig_titles <- paste0("fig: ", fig_var$value)
