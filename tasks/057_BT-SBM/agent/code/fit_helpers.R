norm_geo1 <- function(v) {
  v <- as.numeric(v)
  v <- pmax(v, .Machine$double.xmin)
  v / exp(mean(log(v)))
}

relabel_by_lambda_draws <- function(x_samples, lambda_samples) {
  BTSBM::relabel_by_lambda(x_samples, lambda_samples)
}

extract_btsbm_lambda_hat <- function(fit_btsbm) {
  if (is.null(fit_btsbm$lambda_samples)) stop("BT-SBM fit missing lambda_samples.")
  inf_i <- relabel_by_lambda_draws(fit_btsbm$x_samples, fit_btsbm$lambda_samples)
  fit_btsbm$lambda_samples <- inf_i$lambda_samples_relabel
  norm_geo1(colMeans(fit_btsbm$lambda_samples))
}
