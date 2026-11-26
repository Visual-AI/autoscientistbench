### Code for simulation study for Poisson regression ###
### with increasing n and d ############################

# Model: y ~ Poisson(eta)
# eta <- x^T beta
# beta ~ student_t(nu)
# Approximations considered: Laplace, GVB and GEP
# D (a vector with different parameter dimensions) MUST have 5 different values

poisson_simulation <- function(X, N , D, beta_0, stan.model)
{
  source("Functions_approximations_and_godfit.R")
  source("Gauss_EP_Poisson.R")
  library(rstan)
  library(rstudioapi)
  library(dplyr)
  library(bpr)
  
  # log-posterior of the model
  
  log_post <- function(beta, x, y, nu)
  {
    # linear predictors
    eta <- x %*% beta
    # prior
    prior <- sum(dt(beta, df = nu, log = TRUE))
    # likelihood
    lik <- sum(dpois(y, exp(eta), log =  TRUE))
    return(prior + lik)
  }

  #############################################################
  #### Setting 1 ##############################################
  #############################################################

  y_1 <- rpois(N[1], exp(X[1:N[1], 1:D[1]] %*% beta_0[1:D[1]]))
  data1 <- list(
    N = N[1] ,
    D = D[1] ,
    X = X[1:N[1], 1:D[1]] ,
    y = y_1 ,
    nu = 3
  )

  #------------------------------------------------------------
  # HMC fit
  options(mc.cores = parallel::detectCores())
  rstan_options(auto_write = TRUE)
  fit1_hmc <- sampling(
    poisson_model_stan,
    data = data1,
    warmup = 1000,
    iter = 10^4 + 10^3,
    chains = 4
  )
  sims_fit1_hmc <- extract(fit1_hmc)
  
  #------------------------------------------------------------
  # Laplace approximation
  fit1_lap <- rstan::optimizing(poisson_model_stan, data = data1, hessian = TRUE)
  
  m_fit1_lap <- fit1_lap$theta_tilde[1:D[1]]
  cov_fit1_lap <- -solve(fit1_lap$hessian)
  
  # Performance Laplace 1st setting
  sim_fit1_lap <- mvtnorm::rmvnorm(10^4, mean = m_fit1_lap, sigma = cov_fit1_lap)
  
  # Joint skew-symmetric approximation
  w_fit1_la <- apply(sim_fit1_lap, 1, function(k)
    w_joint(
      k,
      xi =  m_fit1_lap,
      log_post =  log_post,
      x = X[1:N[1], 1:D[1]],
      y = y_1,
      nu = 3
    ))
  weighted_fit1_la <-  rbind(sim_fit1_lap, t(2 * m_fit1_lap - t(sim_fit1_lap)))
  weights_fit1_la <- c(w_fit1_la, 1 - w_fit1_la)
  
  # summaries
  kl_lap_fit1 <- kl_skew_lap_fit1 <-  rep(NA, length(m_fit1_lap))
  rv_kl_lap_fit1 <- rv_kl_skew_lap_fit1 <- rep(NA, length(m_fit1_lap))
  tv_lap_fit1 <- tv_skew_lap_fit1 <- rep(NA, length(m_fit1_lap))
  bias_lap_fit1 <- bias_skew_lap_fit1 <- rep(NA, length(m_fit1_lap))
  
  # Divergence Laplace
  for (i in 1:length(m_fit1_lap))
  {
    kls <- compute_kl_tv(sim_fit1_lap[, i], sims_fit1_hmc$beta[, i])
    kl_lap_fit1[i] <- kls$kl_divergence
    rv_kl_lap_fit1[i] <- kls$rev_kl_divergence
    tv_lap_fit1[i] <- kls$tv_distance
    bias_lap_fit1[i] <- kls$bias
  }
  
  # Divergence skew-Laplace
  for (i in 1:length(m_fit1_lap))
  {
    kls <- compute_kl_tv(weighted_fit1_la[, i], sims_fit1_hmc$beta[, i], weights = weights_fit1_la)
    kl_skew_lap_fit1[i] <- kls$kl_divergence
    rv_kl_skew_lap_fit1[i] <- kls$rev_kl_divergence
    tv_skew_lap_fit1[i] <- kls$tv_distance
    bias_skew_lap_fit1[i] <- kls$bias
  }
    
  fit_1_results_lap <- data.frame(
    c(kl_lap_fit1, kl_skew_lap_fit1),
    c(rv_kl_lap_fit1, rv_kl_skew_lap_fit1),
    c(tv_lap_fit1, tv_skew_lap_fit1),
    c(bias_lap_fit1, bias_skew_lap_fit1),
    c(rep("LAP", D[1]), rep("S-LAP", D[1])),
    rep("n = 200", 2 * D[1])
  )
  
  colnames(fit_1_results_lap) <-   c("KL", "revKL", "TV", "Bias", "Method", "N")
  
  #------------------------------------------------------------
  # GVB approximation
  fit1_gvb <- tryCatch(
    rstan::vb(
      poisson_model_stan,
      data = data1,
      iter = 250000,
      algorithm =  "fullrank",
      tol_rel_obj = 1e-8,
      seed = 1,
      output_samples = 10^4,
      adapt_iter = 400
    ) ,
    error = function(e)
      rstan::vb(
        poisson_model_stan,
        data = data1,
        iter = 250000,
        algorithm =  "fullrank",
        tol_rel_obj = 1e-8,
        seed = 123,
        output_samples = 10^4,
        grad_samples = 10,
        elbo_samples  = 500,
        adapt_iter = 1000
      )
  )
  
  sims_fit1_gvb <- extract(fit1_gvb)$beta
  m_fit1_gvb <- apply(sims_fit1_gvb, 2 , mean)
  cov_fit1_gvb  <- cov(sims_fit1_gvb)
  
  # Performance GVB 1st setting
  sim_fit1_gvb <- mvtnorm::rmvnorm(10^4, mean = m_fit1_gvb, sigma = cov_fit1_gvb)
  
  # Joint skew-symmetric approximation
  w_fit1_gvb <- apply(sim_fit1_gvb, 1, function(k)
    w_joint(
      k,
      xi =  m_fit1_gvb,
      log_post =  log_post,
      x = X[1:N[1], 1:D[1]],
      y = y_1,
      nu = 3
    ))
  weighted_fit1_gvb <-  rbind(sim_fit1_gvb, t(2 * m_fit1_gvb - t(sim_fit1_gvb)))
  weights_fit1_gvb <- c(w_fit1_gvb, 1 - w_fit1_gvb)
  
  # summaries 
  kl_gvb_fit1 <- kl_skew_gvb_fit1 <-  rep(NA, length(m_fit1_gvb))
  rv_kl_gvb_fit1 <- rv_kl_skew_gvb_fit1 <- rep(NA, length(m_fit1_gvb))
  tv_gvb_fit1 <- tv_skew_gvb_fit1 <- rep(NA, length(m_fit1_gvb))
  bias_gvb_fit1 <- bias_skew_gvb_fit1 <- rep(NA, length(m_fit1_gvb))
  
  # Divergence GVB  
  for (i in 1:length(m_fit1_gvb))
  {
    kls <- compute_kl_tv(sim_fit1_gvb[, i], sims_fit1_hmc$beta[, i])
    kl_gvb_fit1[i] <- kls$kl_divergence
    rv_kl_gvb_fit1[i] <- kls$rev_kl_divergence
    tv_gvb_fit1[i] <- kls$tv_distance
    bias_gvb_fit1[i] <- kls$bias
    
  }
  
  # Divergence skew-GVB  
  for (i in 1:length(m_fit1_gvb))
  {
    kls <- compute_kl_tv(weighted_fit1_gvb[, i], sims_fit1_hmc$beta[, i], weights = weights_fit1_gvb)
    kl_skew_gvb_fit1[i] <- kls$kl_divergence
    rv_kl_skew_gvb_fit1[i] <- kls$rev_kl_divergence
    tv_skew_gvb_fit1[i] <- kls$tv_distance
    bias_skew_gvb_fit1[i] <- kls$bias
  }
  
  fit_1_results_gvb <- data.frame(
    c(kl_gvb_fit1, kl_skew_gvb_fit1),
    c(rv_kl_gvb_fit1, rv_kl_skew_gvb_fit1),
    c(tv_gvb_fit1, tv_skew_gvb_fit1),
    c(bias_gvb_fit1, bias_skew_gvb_fit1),
    c(rep("GVB", D[1]), rep("S-GVB", D[1])),
    rep("n = 200", 2 * D[1])
  )
  colnames(fit_1_results_gvb) <-   c("KL", "revKL", "TV", "Bias", "Method", "N")

  #------------------------------------------------------------
  # GEP approximation
  # set hyper-param
  #om2p     =  1  # marginal prior variances
  #Omega0   = diag(om2p,D[1],D[1])
  #beta0    = rep(0.,D[1])
  fit1_gep <- ep_poisson_t(
    X[1:N[1], 1:D[1]],
    y_1,
    nu = 3,
    max_iter = 100,
    tol = 1e-4,
    damping = 0.6
  )
  
  m_fit1_ep <- c(fit1_gep$m)
  cov_fit1_ep  <- fit1_gep$Sigma
  
  # Performance EP 1st setting
  sim_fit1_ep <- mvtnorm::rmvnorm(10^4, mean = m_fit1_ep, sigma = cov_fit1_ep)
  
  # Joint skew-symmetric approximation
  w_fit1_ep <- apply(sim_fit1_ep, 1, function(k)
    w_joint(
      k,
      xi =  m_fit1_ep,
      log_post =  log_post,
      x = X[1:N[1], 1:D[1]],
      y = y_1,
      nu = 3
    ))
  weighted_fit1_ep <-  rbind(sim_fit1_ep, t(2 * m_fit1_ep - t(sim_fit1_ep)))
  weights_fit1_ep <- c(w_fit1_ep, 1 - w_fit1_ep)
  
  # summaries  
  kl_ep_fit1 <- kl_skew_ep_fit1 <-  rep(NA, length(m_fit1_ep))
  rv_kl_ep_fit1 <- rv_kl_skew_ep_fit1 <- rep(NA, length(m_fit1_ep))
  tv_ep_fit1 <- tv_skew_ep_fit1 <- rep(NA, length(m_fit1_ep))
  bias_ep_fit1 <- bias_skew_ep_fit1 <- rep(NA, length(m_fit1_ep))
  
  # Divergence EP  
  for (i in 1:length(m_fit1_ep))
  {
    kls <- compute_kl_tv(sim_fit1_ep[, i], sims_fit1_hmc$beta[, i])
    kl_ep_fit1[i] <- kls$kl_divergence
    rv_kl_ep_fit1[i] <- kls$rev_kl_divergence
    tv_ep_fit1[i] <- kls$tv_distance
    bias_ep_fit1[i] <- kls$bias
  }
  
  # Divergence skew-EP  
  for (i in 1:length(m_fit1_ep))
  {
    kls <- compute_kl_tv(weighted_fit1_ep[, i], sims_fit1_hmc$beta[, i], weights = weights_fit1_ep)
    kl_skew_ep_fit1[i] <- kls$kl_divergence
    rv_kl_skew_ep_fit1[i] <- kls$rev_kl_divergence
    tv_skew_ep_fit1[i] <- kls$tv_distance
    bias_skew_ep_fit1[i] <- kls$bias ##
  }
  
  fit_1_results_ep <- data.frame(
    c(kl_ep_fit1, kl_skew_ep_fit1),
    c(rv_kl_ep_fit1, rv_kl_skew_ep_fit1),
    c(tv_ep_fit1, tv_skew_ep_fit1),
    c(bias_ep_fit1, bias_skew_ep_fit1),
    c(rep("GEP", D[1]), rep("S-GEP", D[1])),
    rep("n = 200", 2 * D[1])
  )
  colnames(fit_1_results_ep) <-   c("KL", "revKL", "TV", "Bias", "Method", "N")
  
  #############################################################
  #### Setting 2 ##############################################
  #############################################################
  
  y_2 <- rpois(N[2], exp(X[1:N[2], 1:D[2]] %*% beta_0[1:D[2]]))
  data2 <- list(
    N = N[2] ,
    D = D[2] ,
    X = X[1:N[2], 1:D[2]] ,
    y = y_2 ,
    nu = 3
  )

  #------------------------------------------------------------
  # HMC fit
  options(mc.cores = parallel::detectCores())
  rstan_options(auto_write = TRUE)
  fit2_hmc <- sampling(
    poisson_model_stan,
    data = data2,
    warmup = 1000,
    iter = 10^4 + 10^3,
    chains = 4
  )
  sims_fit2_hmc <- extract(fit2_hmc)

  #------------------------------------------------------------
  # Laplace approximation
  fit2_lap <- rstan::optimizing(poisson_model_stan, data = data2, hessian = TRUE)
  
  m_fit2_lap <- fit2_lap$theta_tilde[1:D[2]]
  cov_fit2_lap <- -solve(fit2_lap$hessian)
  
  # Performance Laplace 2nd setting
  sim_fit2_lap <- mvtnorm::rmvnorm(10^4, mean = m_fit2_lap, sigma = cov_fit2_lap)
  
  # Joint skew-symmetric approximation
  w_fit2_la <- apply(sim_fit2_lap, 1, function(k)
    w_joint(
      k,
      xi =  m_fit2_lap,
      log_post =  log_post,
      x = X[1:N[2], 1:D[2]],
      y = y_2,
      nu = 3
    ))
  weighted_fit2_la <-  rbind(sim_fit2_lap, t(2 * m_fit2_lap - t(sim_fit2_lap)))
  weights_fit2_la <- c(w_fit2_la, 1 - w_fit2_la)
  
  # summaries
  kl_lap_fit2 <- kl_skew_lap_fit2 <-  rep(NA, length(m_fit2_lap))
  rv_kl_lap_fit2 <- rv_kl_skew_lap_fit2 <- rep(NA, length(m_fit2_lap))
  tv_lap_fit2 <- tv_skew_lap_fit2 <- rep(NA, length(m_fit2_lap))
  bias_lap_fit2 <- bias_skew_lap_fit2 <- rep(NA, length(m_fit2_lap))
  
  # Divergence Laplace  
  for (i in 1:length(m_fit2_lap))
  {
    kls <- compute_kl_tv(sim_fit2_lap[, i], sims_fit2_hmc$beta[, i])
    kl_lap_fit2[i] <- kls$kl_divergence
    rv_kl_lap_fit2[i] <- kls$rev_kl_divergence
    tv_lap_fit2[i] <- kls$tv_distance
    bias_lap_fit2[i] <- kls$bias
  }
  
  # Divergence skew-Laplace  
  for (i in 1:length(m_fit2_lap))
  {
    kls <- compute_kl_tv(weighted_fit2_la[, i], sims_fit2_hmc$beta[, i], weights = weights_fit2_la)
    kl_skew_lap_fit2[i] <- kls$kl_divergence
    rv_kl_skew_lap_fit2[i] <- kls$rev_kl_divergence
    tv_skew_lap_fit2[i] <- kls$tv_distance
    bias_skew_lap_fit2[i] <- kls$bias
  }
  
  fit_2_results_lap <- data.frame(
    c(kl_lap_fit2, kl_skew_lap_fit2),
    c(rv_kl_lap_fit2, rv_kl_skew_lap_fit2),
    c(tv_lap_fit2, tv_skew_lap_fit2),
    c(bias_lap_fit2, bias_skew_lap_fit2),
    c(rep("LAP", D[2]), rep("S-LAP", D[2])),
    rep("n = 400", 2 * D[2])
  )
  colnames(fit_2_results_lap) <-   c("KL", "revKL", "TV", "Bias", "Method", "N")

  #------------------------------------------------------------  
  # GVB approximation
  fit2_gvb <- tryCatch(
    rstan::vb(
      poisson_model_stan,
      data = data2,
      iter = 250000,
      algorithm =  "fullrank",
      tol_rel_obj = 1e-8,
      seed = 1,
      output_samples = 10^4,
      adapt_iter = 400
    ),
    error = function(e)
      rstan::vb(
        poisson_model_stan,
        data = data2,
        iter = 250000,
        algorithm =  "fullrank",
        tol_rel_obj = 1e-8,
        seed = 123,
        grad_samples = 10,
        output_samples = 10^4,
        elbo_samples  = 500,
        adapt_iter = 1000
      )
  )
    
  sims_fit2_gvb <- extract(fit2_gvb)$beta
  m_fit2_gvb <- apply(sims_fit2_gvb, 2 , mean)
  cov_fit2_gvb  <- cov(sims_fit2_gvb)
  
  # Performance GVB 2nd setting
  sim_fit2_gvb <- mvtnorm::rmvnorm(10^4, mean = m_fit2_gvb, sigma = cov_fit2_gvb)
  
  # Joint skew-symmetric approximation
  w_fit2_gvb <- apply(sim_fit2_gvb, 1, function(k)
    w_joint(
      k,
      xi =  m_fit2_gvb,
      log_post =  log_post,
      x = X[1:N[2], 1:D[2]],
      y = y_2,
      nu = 3
    ))
  weighted_fit2_gvb <-  rbind(sim_fit2_gvb, t(2 * m_fit2_gvb - t(sim_fit2_gvb)))
  weights_fit2_gvb <- c(w_fit2_gvb, 1 - w_fit2_gvb)
  
  # summaries  
  kl_gvb_fit2 <- kl_skew_gvb_fit2 <-  rep(NA, length(m_fit2_gvb))
  rv_kl_gvb_fit2 <- rv_kl_skew_gvb_fit2 <- rep(NA, length(m_fit2_gvb))
  tv_gvb_fit2 <- tv_skew_gvb_fit2 <- rep(NA, length(m_fit2_gvb))
  bias_gvb_fit2 <- bias_skew_gvb_fit2 <- rep(NA, length(m_fit2_gvb))
  
  # Divergence GVB  
  for (i in 1:length(m_fit2_gvb))
  {
    kls <- compute_kl_tv(sim_fit2_gvb[, i], sims_fit2_hmc$beta[, i])
    kl_gvb_fit2[i] <- kls$kl_divergence
    rv_kl_gvb_fit2[i] <- kls$rev_kl_divergence
    tv_gvb_fit2[i] <- kls$tv_distance
    bias_gvb_fit2[i] <- kls$bias
  }
  
  # Divergence skew-GVB  
  for (i in 1:length(m_fit2_gvb))
  {
    kls <- compute_kl_tv(weighted_fit2_gvb[, i], sims_fit2_hmc$beta[, i], weights = weights_fit2_gvb)
    kl_skew_gvb_fit2[i] <- kls$kl_divergence
    rv_kl_skew_gvb_fit2[i] <- kls$rev_kl_divergence
    tv_skew_gvb_fit2[i] <- kls$tv_distance
    bias_skew_gvb_fit2[i] <- kls$bias
    
  }
  
  fit_2_results_gvb <- data.frame(
    c(kl_gvb_fit2, kl_skew_gvb_fit2),
    c(rv_kl_gvb_fit2, rv_kl_skew_gvb_fit2),
    c(tv_gvb_fit2, tv_skew_gvb_fit2),
    c(bias_gvb_fit2, bias_skew_gvb_fit2),
    c(rep("GVB", D[2]), rep("S-GVB", D[2])),
    rep("n = 400", 2 * D[2])
  )
  colnames(fit_2_results_gvb) <-   c("KL", "revKL", "TV", "Bias", "Method", "N")
  
  #------------------------------------------------------------
  # GEP approximation
  # set hyper-param
  fit2_gep <- ep_poisson_t(
    X[1:N[2], 1:D[2]],
    y_2,
    nu = 3,
    max_iter = 100,
    tol = 1e-4,
    damping = 0.6
  )
  
  m_fit2_ep <- c(fit2_gep$m)
  cov_fit2_ep  <- fit2_gep$Sigma
  
  # Performance EP 2nd setting
  sim_fit2_ep <- mvtnorm::rmvnorm(10^4, mean = m_fit2_ep, sigma = cov_fit2_ep)
  
  # Joint skew-symmetric approximation
  w_fit2_ep <- apply(sim_fit2_ep, 1, function(k)
    w_joint(
      k,
      xi =  m_fit2_ep,
      log_post =  log_post,
      x = X[1:N[2], 1:D[2]],
      y = y_2,
      nu = 3
    ))
  weighted_fit2_ep <-  rbind(sim_fit2_ep, t(2 * m_fit2_ep - t(sim_fit2_ep)))
  weights_fit2_ep <- c(w_fit2_ep, 1 - w_fit2_ep)
  
  # summaries
  kl_ep_fit2 <- kl_skew_ep_fit2 <-  rep(NA, length(m_fit2_ep))
  rv_kl_ep_fit2 <- rv_kl_skew_ep_fit2 <- rep(NA, length(m_fit2_ep))
  tv_ep_fit2 <- tv_skew_ep_fit2 <- rep(NA, length(m_fit2_ep))
  bias_ep_fit2 <- bias_skew_ep_fit2 <- rep(NA, length(m_fit2_ep))
  
  # Divergence EP  
  for (i in 1:length(m_fit2_ep))
  {
    kls <- compute_kl_tv(sim_fit2_ep[, i], sims_fit2_hmc$beta[, i])
    kl_ep_fit2[i] <- kls$kl_divergence
    rv_kl_ep_fit2[i] <- kls$rev_kl_divergence
    tv_ep_fit2[i] <- kls$tv_distance
    bias_ep_fit2[i] <- kls$bias
  }
  
  # Divergence skew-EP  
  for (i in 1:length(m_fit2_ep))
  {
    kls <- compute_kl_tv(weighted_fit2_ep[, i], sims_fit2_hmc$beta[, i], weights = weights_fit2_ep)
    kl_skew_ep_fit2[i] <- kls$kl_divergence
    rv_kl_skew_ep_fit2[i] <- kls$rev_kl_divergence
    tv_skew_ep_fit2[i] <- kls$tv_distance
    bias_skew_ep_fit2[i] <- kls$bias
    
  }
  
  fit_2_results_ep <- data.frame(
    c(kl_ep_fit2, kl_skew_ep_fit2),
    c(rv_kl_ep_fit2, rv_kl_skew_ep_fit2),
    c(tv_ep_fit2, tv_skew_ep_fit2),
    c(bias_ep_fit2, bias_skew_ep_fit2),
    c(rep("GEP", D[2]), rep("S-GEP", D[2])),
    rep("n = 400", 2 * D[2])
  )
  
  colnames(fit_2_results_ep) <-   c("KL", "revKL", "TV", "Bias", "Method", "N")
  
  
  #############################################################
  #### Setting 3 ##############################################
  #############################################################
  
  y_3 <- rpois(N[3], exp(X[1:N[3], 1:D[3]] %*% beta_0[1:D[3]]))
  data3 <- list(
    N = N[3] ,
    D = D[3] ,
    X = X[1:N[3], 1:D[3]] ,
    y = y_3 ,
    nu = 3
  )

  #------------------------------------------------------------
  # HMC fit
  options(mc.cores = parallel::detectCores())
  rstan_options(auto_write = TRUE)
  fit3_hmc <- sampling(
    poisson_model_stan,
    data = data3,
    warmup = 1000,
    iter = 10^4 + 10^3,
    chains = 4
  )
  sims_fit3_hmc <- extract(fit3_hmc)
  
  # Laplace approximation
  fit3_lap <- rstan::optimizing(poisson_model_stan, data = data3, hessian = TRUE)
  
  m_fit3_lap <- fit3_lap$theta_tilde[1:D[3]]
  cov_fit3_lap <- -solve(fit3_lap$hessian)
  
  # Performance Laplace 3rd setting
  sim_fit3_lap <- mvtnorm::rmvnorm(10^4, mean = m_fit3_lap, sigma = cov_fit3_lap)
  
  # Joint skew-symmetric approximation
  w_fit3_la <- apply(sim_fit3_lap, 1, function(k)
    w_joint(
      k,
      xi =  m_fit3_lap,
      log_post =  log_post,
      x = X[1:N[3], 1:D[3]],
      y = y_3,
      nu = 3
    ))
  weighted_fit3_la <-  rbind(sim_fit3_lap, t(2 * m_fit3_lap - t(sim_fit3_lap)))
  weights_fit3_la <- c(w_fit3_la, 1 - w_fit3_la)
  
  # summaries
  kl_lap_fit3 <- kl_skew_lap_fit3 <-  rep(NA, length(m_fit3_lap))
  rv_kl_lap_fit3 <- rv_kl_skew_lap_fit3 <- rep(NA, length(m_fit3_lap))
  tv_lap_fit3 <- tv_skew_lap_fit3 <- rep(NA, length(m_fit3_lap))
  bias_lap_fit3 <- bias_skew_lap_fit3 <- rep(NA, length(m_fit3_lap))
  
  # Divergence Laplace  
  for (i in 1:length(m_fit3_lap))
  {
    kls <- compute_kl_tv(sim_fit3_lap[, i], sims_fit3_hmc$beta[, i])
    kl_lap_fit3[i] <- kls$kl_divergence
    rv_kl_lap_fit3[i] <- kls$rev_kl_divergence
    tv_lap_fit3[i] <- kls$tv_distance
    bias_lap_fit3[i] <- kls$bias
    
  }
  
  # Divergence skew-Laplace
  for (i in 1:length(m_fit3_lap))
  {
    kls <- compute_kl_tv(weighted_fit3_la[, i], sims_fit3_hmc$beta[, i], weights = weights_fit3_la)
    kl_skew_lap_fit3[i] <- kls$kl_divergence
    rv_kl_skew_lap_fit3[i] <- kls$rev_kl_divergence
    tv_skew_lap_fit3[i] <- kls$tv_distance
    bias_skew_lap_fit3[i] <- kls$bias
    
  }
    
  fit_3_results_lap <- data.frame(
    c(kl_lap_fit3, kl_skew_lap_fit3),
    c(rv_kl_lap_fit3, rv_kl_skew_lap_fit3),
    c(tv_lap_fit3, tv_skew_lap_fit3),
    c(bias_lap_fit3, bias_skew_lap_fit3),
    c(rep("LAP", D[3]), rep("S-LAP", D[3])),
    rep("n = 600", 2 * D[3])
  )
  
  colnames(fit_3_results_lap) <-   c("KL", "revKL", "TV", "Bias", "Method", "N")
  
  #------------------------------------------------------------
  # GVB approximation
  fit3_gvb <- tryCatch(
    rstan::vb(
      poisson_model_stan,
      data = data3,
      iter = 250000,
      algorithm =  "fullrank",
      tol_rel_obj = 1e-8,
      seed = 1,
      output_samples = 10^4,
      adapt_iter = 400
    ),
    error =  function(e)
      rstan::vb(
        poisson_model_stan,
        data = data3,
        iter = 250000,
        algorithm =  "fullrank",
        tol_rel_obj = 1e-8,
        seed = 123,
        grad_samples = 10,
        output_samples = 10^4,
        elbo_samples  = 500,
        adapt_iter = 1000
      )
  )
  
  sims_fit3_gvb <- extract(fit3_gvb)$beta
  m_fit3_gvb <- apply(sims_fit3_gvb, 2 , mean)
  cov_fit3_gvb  <- cov(sims_fit3_gvb)
  
  # Performance GVB 3rd setting
  sim_fit3_gvb <- mvtnorm::rmvnorm(10^4, mean = m_fit3_gvb, sigma = cov_fit3_gvb)
  
  # Joint skew-symmetric approximation
  w_fit3_gvb <- apply(sim_fit3_gvb, 1, function(k)
    w_joint(
      k,
      xi =  m_fit3_gvb,
      log_post =  log_post,
      x = X[1:N[3], 1:D[3]],
      y = y_3,
      nu = 3
    ))
  weighted_fit3_gvb <-  rbind(sim_fit3_gvb, t(2 * m_fit3_gvb - t(sim_fit3_gvb)))
  weights_fit3_gvb <- c(w_fit3_gvb, 1 - w_fit3_gvb)
  
  # summaries
  kl_gvb_fit3 <- kl_skew_gvb_fit3 <-  rep(NA, length(m_fit3_gvb))
  rv_kl_gvb_fit3 <- rv_kl_skew_gvb_fit3 <- rep(NA, length(m_fit3_gvb))
  tv_gvb_fit3 <- tv_skew_gvb_fit3 <- rep(NA, length(m_fit3_gvb))
  bias_gvb_fit3 <- bias_skew_gvb_fit3 <- rep(NA, length(m_fit3_gvb))
  
  # Divergence GVB
  for (i in 1:length(m_fit3_gvb))
  {
    kls <- compute_kl_tv(sim_fit3_gvb[, i], sims_fit3_hmc$beta[, i])
    kl_gvb_fit3[i] <- kls$kl_divergence
    rv_kl_gvb_fit3[i] <- kls$rev_kl_divergence
    tv_gvb_fit3[i] <- kls$tv_distance
    bias_gvb_fit3[i] <- kls$bias
  }
  
  # Divergence skew-GVB
  for (i in 1:length(m_fit3_gvb))
  {
    kls <- compute_kl_tv(weighted_fit3_gvb[, i], sims_fit3_hmc$beta[, i], weights = weights_fit3_gvb)
    kl_skew_gvb_fit3[i] <- kls$kl_divergence
    rv_kl_skew_gvb_fit3[i] <- kls$rev_kl_divergence
    tv_skew_gvb_fit3[i] <- kls$tv_distance
    bias_skew_gvb_fit3[i] <- kls$bias
  }
  
  fit_3_results_gvb <- data.frame(
    c(kl_gvb_fit3, kl_skew_gvb_fit3),
    c(rv_kl_gvb_fit3, rv_kl_skew_gvb_fit3),
    c(tv_gvb_fit3, tv_skew_gvb_fit3),
    c(bias_gvb_fit3, bias_skew_gvb_fit3),
    c(rep("GVB", D[3]), rep("S-GVB", D[3])),
    rep("n = 600", 2 * D[3])
  )
  colnames(fit_3_results_gvb) <-   c("KL", "revKL", "TV", "Bias", "Method", "N")
  
  #------------------------------------------------------------
  # GEP approximation
  # set hyper-param
  fit3_gep <- ep_poisson_t(
    X[1:N[3], 1:D[3]],
    y_3,
    nu = 3,
    max_iter = 100,
    tol = 1e-4,
    damping = 0.6
  )
  
  m_fit3_ep <- c(fit3_gep$m)
  cov_fit3_ep  <- fit3_gep$Sigma
    
  # Performance EP 3rd setting
  sim_fit3_ep <- mvtnorm::rmvnorm(10^4, mean = m_fit3_ep, sigma = cov_fit3_ep)
  
  # Joint skew-symmetric approximation
  w_fit3_ep <- apply(sim_fit3_ep, 1, function(k)
    w_joint(
      k,
      xi =  m_fit3_ep,
      log_post =  log_post,
      x = X[1:N[3], 1:D[3]],
      y = y_3,
      nu = 3
    ))
  weighted_fit3_ep <-  rbind(sim_fit3_ep, t(2 * m_fit3_ep - t(sim_fit3_ep)))
  weights_fit3_ep <- c(w_fit3_ep, 1 - w_fit3_ep)
  
  # summaries  
  kl_ep_fit3 <- kl_skew_ep_fit3 <-  rep(NA, length(m_fit3_ep))
  rv_kl_ep_fit3 <- rv_kl_skew_ep_fit3 <- rep(NA, length(m_fit3_ep))
  tv_ep_fit3 <- tv_skew_ep_fit3 <- rep(NA, length(m_fit3_ep))
  bias_ep_fit3 <- bias_skew_ep_fit3 <- rep(NA, length(m_fit3_ep))
  
  # Divergence EP  
  for (i in 1:length(m_fit3_ep))
  {
    kls <- compute_kl_tv(sim_fit3_ep[, i], sims_fit3_hmc$beta[, i])
    kl_ep_fit3[i] <- kls$kl_divergence
    rv_kl_ep_fit3[i] <- kls$rev_kl_divergence
    tv_ep_fit3[i] <- kls$tv_distance
    bias_ep_fit3[i] <- kls$bias
    
  }
  
  # Divergence skew-EP
  for (i in 1:length(m_fit3_ep))
  {
    kls <- compute_kl_tv(weighted_fit3_ep[, i], sims_fit3_hmc$beta[, i], weights = weights_fit3_ep)
    kl_skew_ep_fit3[i] <- kls$kl_divergence
    rv_kl_skew_ep_fit3[i] <- kls$rev_kl_divergence
    tv_skew_ep_fit3[i] <- kls$tv_distance
    bias_skew_ep_fit3[i] <- kls$bias
  }
  
  fit_3_results_ep <- data.frame(
    c(kl_ep_fit3, kl_skew_ep_fit3),
    c(rv_kl_ep_fit3, rv_kl_skew_ep_fit3),
    c(tv_ep_fit3, tv_skew_ep_fit3),
    c(bias_ep_fit3, bias_skew_ep_fit3),
    c(rep("GEP", D[3]), rep("S-GEP", D[3])),
    rep("n = 600", 2 * D[3])
  )
  colnames(fit_3_results_ep) <-   c("KL", "revKL", "TV", "Bias", "Method", "N")
  
  
  #############################################################
  #### Setting 4 ##############################################
  #############################################################
  
  y_4 <- rpois(N[4], exp(X[1:N[4], 1:D[4]] %*% beta_0[1:D[4]]))
  data4 <- list(
    N = N[4] ,
    D = D[4] ,
    X = X[1:N[4], 1:D[4]] ,
    y = y_4 ,
    nu = 3
  )

  #------------------------------------------------------------
  # HMC fit
  options(mc.cores = parallel::detectCores())
  rstan_options(auto_write = TRUE)
  fit4_hmc <- sampling(
    poisson_model_stan,
    data = data4,
    warmup = 1000,
    iter = 10^4 + 10^3,
    chains = 4
  )
  sims_fit4_hmc <- extract(fit4_hmc)

  #------------------------------------------------------------
  # Laplace approximation
  fit4_lap <- rstan::optimizing(poisson_model_stan, data = data4, hessian = TRUE)
  
  m_fit4_lap <- fit4_lap$theta_tilde[1:D[4]]
  cov_fit4_lap <- -solve(fit4_lap$hessian)
  
  # Performance Laplace 4th setting
  sim_fit4_lap <- mvtnorm::rmvnorm(10^4, mean = m_fit4_lap, sigma = cov_fit4_lap)
  
  # Joint skew-symmetric approximation
  w_fit4_la <- apply(sim_fit4_lap, 1, function(k)
    w_joint(
      k,
      xi =  m_fit4_lap,
      log_post =  log_post,
      x = X[1:N[4], 1:D[4]],
      y = y_4,
      nu = 3
    ))
  weighted_fit4_la <-  rbind(sim_fit4_lap, t(2 * m_fit4_lap - t(sim_fit4_lap)))
  weights_fit4_la <- c(w_fit4_la, 1 - w_fit4_la)
  
  # summaries
  kl_lap_fit4 <- kl_skew_lap_fit4 <-  rep(NA, length(m_fit4_lap))
  rv_kl_lap_fit4 <- rv_kl_skew_lap_fit4 <- rep(NA, length(m_fit4_lap))
  tv_lap_fit4 <- tv_skew_lap_fit4 <- rep(NA, length(m_fit4_lap))
  bias_lap_fit4 <- bias_skew_lap_fit4 <- rep(NA, length(m_fit4_lap))
  
  # Divergence Laplace 
  for (i in 1:length(m_fit4_lap))
  {
    kls <- compute_kl_tv(sim_fit4_lap[, i], sims_fit4_hmc$beta[, i])
    kl_lap_fit4[i] <- kls$kl_divergence
    rv_kl_lap_fit4[i] <- kls$rev_kl_divergence
    tv_lap_fit4[i] <- kls$tv_distance
    bias_lap_fit4[i] <- kls$bias
    
  }
  
  # Divergence skew-Laplace
  for (i in 1:length(m_fit4_lap))
  {
    kls <- compute_kl_tv(weighted_fit4_la[, i], sims_fit4_hmc$beta[, i], weights = weights_fit4_la)
    kl_skew_lap_fit4[i] <- kls$kl_divergence
    rv_kl_skew_lap_fit4[i] <- kls$rev_kl_divergence
    tv_skew_lap_fit4[i] <- kls$tv_distance
    bias_skew_lap_fit4[i] <- kls$bias
  }
    
  fit_4_results_lap <- data.frame(
    c(kl_lap_fit4, kl_skew_lap_fit4),
    c(rv_kl_lap_fit4, rv_kl_skew_lap_fit4),
    c(tv_lap_fit4, tv_skew_lap_fit4),
    c(bias_lap_fit4, bias_skew_lap_fit4),
    c(rep("LAP", D[4]), rep("S-LAP", D[4])),
    rep("n = 800", 2 * D[4])
  )
  
  colnames(fit_4_results_lap) <-   c("KL", "revKL", "TV", "Bias", "Method", "N")

  #------------------------------------------------------------
  # GVB approximation
  fit4_gvb <- tryCatch(
    rstan::vb(
      poisson_model_stan,
      data = data4,
      iter = 250000,
      algorithm =  "fullrank",
      tol_rel_obj = 1e-8,
      seed = 1,
      output_samples = 10^4,
      adapt_iter = 400
    ),
    error = function(e)
      rstan::vb(
        poisson_model_stan,
        data = data4,
        iter = 250000,
        algorithm =  "fullrank",
        tol_rel_obj = 1e-8,
        grad_samples = 10,
        seed = 123,
        output_samples = 10^4,
        elbo_samples  = 500,
        adapt_iter = 1000
      )
  )
  
  sims_fit4_gvb <- extract(fit4_gvb)$beta
  m_fit4_gvb <- apply(sims_fit4_gvb, 2 , mean)
  cov_fit4_gvb  <- cov(sims_fit4_gvb)
  
  # Performance GVB 4th setting
  sim_fit4_gvb <- mvtnorm::rmvnorm(10^4, mean = m_fit4_gvb, sigma = cov_fit4_gvb)
  
  # Joint skew-symmetric approximation
  w_fit4_gvb <- apply(sim_fit4_gvb, 1, function(k)
    w_joint(
      k,
      xi =  m_fit4_gvb,
      log_post =  log_post,
      x = X[1:N[4], 1:D[4]],
      y = y_4,
      nu = 3
    ))
  weighted_fit4_gvb <-  rbind(sim_fit4_gvb, t(2 * m_fit4_gvb - t(sim_fit4_gvb)))
  weights_fit4_gvb <- c(w_fit4_gvb, 1 - w_fit4_gvb)
  
  # summaries
  kl_gvb_fit4 <- kl_skew_gvb_fit4 <-  rep(NA, length(m_fit4_gvb))
  rv_kl_gvb_fit4 <- rv_kl_skew_gvb_fit4 <- rep(NA, length(m_fit4_gvb))
  tv_gvb_fit4 <- tv_skew_gvb_fit4 <- rep(NA, length(m_fit4_gvb))
  bias_gvb_fit4 <- bias_skew_gvb_fit4 <- rep(NA, length(m_fit4_gvb))
  
  # Divergence GVB  
  for (i in 1:length(m_fit4_gvb))
  {
    kls <- compute_kl_tv(sim_fit4_gvb[, i], sims_fit4_hmc$beta[, i])
    kl_gvb_fit4[i] <- kls$kl_divergence
    rv_kl_gvb_fit4[i] <- kls$rev_kl_divergence
    tv_gvb_fit4[i] <- kls$tv_distance
    bias_gvb_fit4[i] <- kls$bias
  }
  
  # Divergence skew-GVB  
  for (i in 1:length(m_fit4_gvb))
  {
    kls <- compute_kl_tv(weighted_fit4_gvb[, i], sims_fit4_hmc$beta[, i], weights = weights_fit4_gvb)
    kl_skew_gvb_fit4[i] <- kls$kl_divergence
    rv_kl_skew_gvb_fit4[i] <- kls$rev_kl_divergence
    tv_skew_gvb_fit4[i] <- kls$tv_distance
    bias_skew_gvb_fit4[i] <- kls$bias
    
  }
  
  fit_4_results_gvb <- data.frame(
    c(kl_gvb_fit4, kl_skew_gvb_fit4),
    c(rv_kl_gvb_fit4, rv_kl_skew_gvb_fit4),
    c(tv_gvb_fit4, tv_skew_gvb_fit4),
    c(bias_gvb_fit4, bias_skew_gvb_fit4),
    c(rep("GVB", D[4]), rep("S-GVB", D[4])),
    rep("n = 800", 2 * D[4])
  )
  
  colnames(fit_4_results_gvb) <-   c("KL", "revKL", "TV", "Bias", "Method", "N")

  #------------------------------------------------------------
  # GEP approximation
  # set hyper-param
  
  fit4_gep <- ep_poisson_t(
    X[1:N[4], 1:D[4]],
    y_4,
    nu = 3 ,
    max_iter = 100,
    tol = 1e-4,
    damping = 0.6
  )
  
  m_fit4_ep <- c(fit4_gep$m)
  cov_fit4_ep  <- fit4_gep$Sigma  
  
  # Performance EP 4th setting
  sim_fit4_ep <- mvtnorm::rmvnorm(10^4, mean = m_fit4_ep, sigma = cov_fit4_ep)
  
  # Joint skew-symmetric approximation
  w_fit4_ep <- apply(sim_fit4_ep, 1, function(k)
    w_joint(
      k,
      xi =  m_fit4_ep,
      log_post =  log_post,
      x = X[1:N[4], 1:D[4]],
      y = y_4,
      nu = 3
    ))
  weighted_fit4_ep <-  rbind(sim_fit4_ep, t(2 * m_fit4_ep - t(sim_fit4_ep)))
  weights_fit4_ep <- c(w_fit4_ep, 1 - w_fit4_ep)
  
  # summaries  
  kl_ep_fit4 <- kl_skew_ep_fit4 <-  rep(NA, length(m_fit4_ep))
  rv_kl_ep_fit4 <- rv_kl_skew_ep_fit4 <- rep(NA, length(m_fit4_ep))
  tv_ep_fit4 <- tv_skew_ep_fit4 <- rep(NA, length(m_fit4_ep))
  bias_ep_fit4 <- bias_skew_ep_fit4 <- rep(NA, length(m_fit4_ep))
  
  # Divergence EP
  for (i in 1:length(m_fit4_ep))
  {
    kls <- compute_kl_tv(sim_fit4_ep[, i], sims_fit4_hmc$beta[, i])
    kl_ep_fit4[i] <- kls$kl_divergence
    rv_kl_ep_fit4[i] <- kls$rev_kl_divergence
    tv_ep_fit4[i] <- kls$tv_distance
    bias_ep_fit4[i] <- kls$bias
    
  }
  
  # Divergence skew-EP
  for (i in 1:length(m_fit4_ep))
  {
    kls <- compute_kl_tv(weighted_fit4_ep[, i], sims_fit4_hmc$beta[, i], weights = weights_fit4_ep)
    kl_skew_ep_fit4[i] <- kls$kl_divergence
    rv_kl_skew_ep_fit4[i] <- kls$rev_kl_divergence
    tv_skew_ep_fit4[i] <- kls$tv_distance
    bias_skew_ep_fit4[i] <- kls$bias
  }
  
  fit_4_results_ep <- data.frame(
    c(kl_ep_fit4, kl_skew_ep_fit4),
    c(rv_kl_ep_fit4, rv_kl_skew_ep_fit4),
    c(tv_ep_fit4, tv_skew_ep_fit4),
    c(bias_ep_fit4, bias_skew_ep_fit4),
    c(rep("GEP", D[4]), rep("S-GEP", D[4])),
    rep("n = 800", 2 * D[4])
  )
  
  colnames(fit_4_results_ep) <-   c("KL", "revKL", "TV", "Bias", "Method", "N")

                     
  #############################################################
  #### Setting 5 ##############################################
  #############################################################
  
  y_5 <- rpois(N[5], exp(X[1:N[5], 1:D[5]] %*% beta_0[1:D[5]]))
  data5 <- list(
    N = N[5] ,
    D = D[5] ,
    X = X[1:N[5], 1:D[5]] ,
    y = y_5 ,
    nu = 3
  )

  #------------------------------------------------------------
  # HMC fit
  options(mc.cores = parallel::detectCores())
  rstan_options(auto_write = TRUE)
  fit5_hmc <- sampling(
    poisson_model_stan,
    data = data5,
    warmup = 1000,
    iter = 10^4 + 10^3,
    chains = 4
  )
  sims_fit5_hmc <- extract(fit5_hmc)

  #------------------------------------------------------------
  # Laplace approximation
  fit5_lap <- rstan::optimizing(poisson_model_stan, data = data5, hessian = TRUE)
  
  m_fit5_lap <- fit5_lap$theta_tilde[1:D[5]]
  cov_fit5_lap <- -solve(fit5_lap$hessian)
  
  # Performance Laplace 5th setting
  sim_fit5_lap <- mvtnorm::rmvnorm(10^4, mean = m_fit5_lap, sigma = cov_fit5_lap)
  
  # Joint skew-symmetric approximation
  w_fit5_la <- apply(sim_fit5_lap, 1, function(k)
    w_joint(
      k,
      xi =  m_fit5_lap,
      log_post =  log_post,
      x = X[1:N[5], 1:D[5]],
      y = y_5,
      nu = 3
    ))
  weighted_fit5_la <-  rbind(sim_fit5_lap, t(2 * m_fit5_lap - t(sim_fit5_lap)))
  weights_fit5_la <- c(w_fit5_la, 1 - w_fit5_la)
  
  # summaries
  kl_lap_fit5 <- kl_skew_lap_fit5 <-  rep(NA, length(m_fit5_lap))
  rv_kl_lap_fit5 <- rv_kl_skew_lap_fit5 <- rep(NA, length(m_fit5_lap))
  tv_lap_fit5 <- tv_skew_lap_fit5 <- rep(NA, length(m_fit5_lap))
  bias_lap_fit5 <- bias_skew_lap_fit5 <- rep(NA, length(m_fit5_lap))
  
  # Divergence Laplace  
  for (i in 1:length(m_fit5_lap))
  {
    kls <- compute_kl_tv(sim_fit5_lap[, i], sims_fit5_hmc$beta[, i])
    kl_lap_fit5[i] <- kls$kl_divergence
    rv_kl_lap_fit5[i] <- kls$rev_kl_divergence
    tv_lap_fit5[i] <- kls$tv_distance
    bias_lap_fit5[i] <- kls$bias
    
  }
  
  # Divergence skew-Laplace 
  for (i in 1:length(m_fit5_lap))
  {
    kls <- compute_kl_tv(weighted_fit5_la[, i], sims_fit5_hmc$beta[, i], weights = weights_fit5_la)
    kl_skew_lap_fit5[i] <- kls$kl_divergence
    rv_kl_skew_lap_fit5[i] <- kls$rev_kl_divergence
    tv_skew_lap_fit5[i] <- kls$tv_distance
    bias_skew_lap_fit5[i] <- kls$bias
    
  }
  
  
  fit_5_results_lap <- data.frame(
    c(kl_lap_fit5, kl_skew_lap_fit5),
    c(rv_kl_lap_fit5, rv_kl_skew_lap_fit5),
    c(tv_lap_fit5, tv_skew_lap_fit5),
    c(bias_lap_fit5, bias_skew_lap_fit5),
    c(rep("LAP", D[5]), rep("S-LAP", D[5])),
    rep("n = 1000", 2 * D[5])
  )
  
  colnames(fit_5_results_lap) <-   c("KL", "revKL", "TV", "Bias", "Method", "N")
  
  #------------------------------------------------------------
  # GVB approximation
  fit5_gvb <- tryCatch(
    rstan::vb(
      poisson_model_stan,
      data = data5,
      iter = 250000,
      algorithm =  "fullrank",
      tol_rel_obj = 1e-8,
      seed = 1,
      output_samples = 10^4,
      adapt_iter = 400
    ),
    error = function(e)
      rstan::vb(
        poisson_model_stan,
        data = data5,
        iter = 250000,
        algorithm =  "fullrank",
        tol_rel_obj = 1e-8,
        grad_samples = 10,
        seed = 123,
        output_samples = 10^4,
        elbo_samples  = 500,
        adapt_iter = 1000
      )
  )
  
  sims_fit5_gvb <- extract(fit5_gvb)$beta
  m_fit5_gvb <- apply(sims_fit5_gvb, 2 , mean)
  cov_fit5_gvb  <- cov(sims_fit5_gvb)
  
  # Performance GVB 5th setting
  sim_fit5_gvb <- mvtnorm::rmvnorm(10^4, mean = m_fit5_gvb, sigma = cov_fit5_gvb)
  
  # Joint skew-symmetric approximation
  w_fit5_gvb <- apply(sim_fit5_gvb, 1, function(k)
    w_joint(
      k,
      xi =  m_fit5_gvb,
      log_post =  log_post,
      x = X[1:N[5], 1:D[5]],
      y = y_5,
      nu = 3
    ))
  weighted_fit5_gvb <-  rbind(sim_fit5_gvb, t(2 * m_fit5_gvb - t(sim_fit5_gvb)))
  weights_fit5_gvb <- c(w_fit5_gvb, 1 - w_fit5_gvb)
  
  # summaries  
  kl_gvb_fit5 <- kl_skew_gvb_fit5 <-  rep(NA, length(m_fit5_gvb))
  rv_kl_gvb_fit5 <- rv_kl_skew_gvb_fit5 <- rep(NA, length(m_fit5_gvb))
  tv_gvb_fit5 <- tv_skew_gvb_fit5 <- rep(NA, length(m_fit5_gvb))
  bias_gvb_fit5 <- bias_skew_gvb_fit5 <- rep(NA, length(m_fit5_gvb))
  
  # Divergence GVB 
  for (i in 1:length(m_fit5_gvb))
  {
    kls <- compute_kl_tv(sim_fit5_gvb[, i], sims_fit5_hmc$beta[, i])
    kl_gvb_fit5[i] <- kls$kl_divergence
    rv_kl_gvb_fit5[i] <- kls$rev_kl_divergence
    tv_gvb_fit5[i] <- kls$tv_distance
    bias_gvb_fit5[i] <- kls$bias
    
  }
  
  # Divergence skew-GVB  
  for (i in 1:length(m_fit5_gvb))
  {
    kls <- compute_kl_tv(weighted_fit5_gvb[, i], sims_fit5_hmc$beta[, i], weights = weights_fit5_gvb)
    kl_skew_gvb_fit5[i] <- kls$kl_divergence
    rv_kl_skew_gvb_fit5[i] <- kls$rev_kl_divergence
    tv_skew_gvb_fit5[i] <- kls$tv_distance
    bias_skew_gvb_fit5[i] <- kls$bias
    
  }
  
  fit_5_results_gvb <- data.frame(
    c(kl_gvb_fit5, kl_skew_gvb_fit5),
    c(rv_kl_gvb_fit5, rv_kl_skew_gvb_fit5),
    c(tv_gvb_fit5, tv_skew_gvb_fit5),
    c(bias_gvb_fit5, bias_skew_gvb_fit5),
    c(rep("GVB", D[5]), rep("S-GVB", D[5])),
    rep("n = 1000", 2 * D[5])
  )
  colnames(fit_5_results_gvb) <-   c("KL", "revKL", "TV", "Bias", "Method", "N")

  #------------------------------------------------------------                    
  # GEP approximation
  # set hyper-param
  fit5_gep <- ep_poisson_t(
    X[1:N[5], 1:D[5]],
    y_5,
    nu = 3 ,
    max_iter = 100,
    tol = 1e-4,
    damping = 0.5,
    reltol_integ = 1e-8
  )
  
  m_fit5_ep <- c(fit5_gep$m)
  cov_fit5_ep  <- fit5_gep$Sigma
  
  # Performance EP 5th setting
  sim_fit5_ep <- mvtnorm::rmvnorm(10^4, mean = m_fit5_ep, sigma = cov_fit5_ep)
  
  # approximation with joint skew-symmetric approximation
  w_fit5_ep <- apply(sim_fit5_ep, 1, function(k)
    w_joint(
      k,
      xi =  m_fit5_ep,
      log_post =  log_post,
      x = X[1:N[5], 1:D[5]],
      y = y_5,
      nu = 3
    ))
  weighted_fit5_ep <-  rbind(sim_fit5_ep, t(2 * m_fit5_ep - t(sim_fit5_ep)))
  weights_fit5_ep <- c(w_fit5_ep, 1 - w_fit5_ep)
  
  # summaries
  kl_ep_fit5 <- kl_skew_ep_fit5 <-  rep(NA, length(m_fit5_ep))
  rv_kl_ep_fit5 <- rv_kl_skew_ep_fit5 <- rep(NA, length(m_fit5_ep))
  tv_ep_fit5 <- tv_skew_ep_fit5 <- rep(NA, length(m_fit5_ep))
  bias_ep_fit5 <- bias_skew_ep_fit5 <- rep(NA, length(m_fit5_ep))
  
  # Divergence EP
  for (i in 1:length(m_fit5_ep))
  {
    kls <- compute_kl_tv(sim_fit5_ep[, i], sims_fit5_hmc$beta[, i])
    kl_ep_fit5[i] <- kls$kl_divergence
    rv_kl_ep_fit5[i] <- kls$rev_kl_divergence
    tv_ep_fit5[i] <- kls$tv_distance
    bias_ep_fit5[i] <- kls$bias
    
  }
  
  # Divergence skew-EP
  for (i in 1:length(m_fit5_ep))
  {
    kls <- compute_kl_tv(weighted_fit5_ep[, i], sims_fit5_hmc$beta[, i], weights = weights_fit5_ep)
    kl_skew_ep_fit5[i] <- kls$kl_divergence
    rv_kl_skew_ep_fit5[i] <- kls$rev_kl_divergence
    tv_skew_ep_fit5[i] <- kls$tv_distance
    bias_skew_ep_fit5[i] <- kls$bias
    
  }
  
  fit_5_results_ep <- data.frame(
    c(kl_ep_fit5, kl_skew_ep_fit5),
    c(rv_kl_ep_fit5, rv_kl_skew_ep_fit5),
    c(tv_ep_fit5, tv_skew_ep_fit5),
    c(bias_ep_fit5, bias_skew_ep_fit5),
    c(rep("GEP", D[5]), rep("S-GEP", D[5])),
    rep("n = 1000", 2 * D[5])
  )
  colnames(fit_5_results_ep) <-   c("KL", "revKL", "TV", "Bias", "Method", "N")
  
  
  #######################################################
  ## TABLES RESULTS #####################################
  #######################################################

  results_lap <-  data.frame(
    rbind(
      fit_1_results_lap,
      fit_2_results_lap,
      fit_3_results_lap,
      fit_4_results_lap,
      fit_5_results_lap
    )
  )
  
  colnames(results_lap) <- c("KL", "revKL", "TV", "Bias", "Method", "N")
  results_lap[, 6] <- factor(results_lap[, 6])
  results_lap[, 5] <- factor(results_lap[, 5])
  
  
  results_gvb <-  data.frame(
    rbind(
      fit_1_results_gvb,
      fit_2_results_gvb,
      fit_3_results_gvb,
      fit_4_results_gvb,
      fit_5_results_gvb
    )
  )
  colnames(results_gvb) <- c("KL", "revKL", "TV", "Bias", "Method", "N")
  results_gvb[, 6] <- factor(results_gvb[, 6])
  results_gvb[, 5] <- factor(results_gvb[, 5])
  
  
  results_gep <-  rbind(
    fit_1_results_ep,
    fit_2_results_ep,
    fit_3_results_ep,
    fit_4_results_ep,
    fit_5_results_ep
  )
  
  colnames(results_gep) <- c("KL", "revKL", "TV", "Bias", "Method", "N")
  results_gep[, 6] <- factor(results_gep[, 6])
  results_gep[, 5] <- factor(results_gep[, 5])
  
  return(
    list(
      results_lap = results_lap,
      results_gvb = results_gvb,
      results_gep = results_gep
    )
  )
  
}
