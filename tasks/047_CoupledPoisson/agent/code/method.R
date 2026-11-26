# Original reservoir UPAVE estimator, sourced from candidate-controlled R files.
for (name in c("sample_unbiasedfishy", "sample_coupled_chains_and_fish", "sample_unbiasedvar_reservoir")) {
  source(file.path(code_dir, "R", paste0(name, ".R")))
}
estimate_one <- function(single_kernel, coupled_kernel, rinit, h) {
  sample_unbiasedvar_reservoir(single_kernel, coupled_kernel, rinit, h,
                              k = 500, m = 2500, lag = 500, x_0 = 0, natoms = 50)
}
