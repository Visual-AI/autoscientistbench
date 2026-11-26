# Log-sum-exp
logSumExp <- function(values)
{
  m <- max(values)
  return(m + log( sum( exp(values - m) ) ))
}

# Skewing function joint distribution 
w_joint <- function(sim, xi, log_post, ...)
{
  w1 <- log_post(sim,...)
  w2 <- log_post(2*xi -sim,...)
  pr <- 1/(1 + exp( w2 - w1 ))
  pr
}

# Sim skew for importance sampling
sim_skew <- function(sim, m, weights)
{
  bern <- rbinom(dim(sim)[1], 1, weights  )
  sim[bern == 0,] <- t( 2 * m - t(sim[bern == 0, ] ) )
  weights[bern == 0] <- 1 - weights[bern == 0]
  return( list( sim = sim, weights = weights, bern = bern  ) )
}

# Test subset: compute with mcmc a proxy of the kl-divergence
kl_up_to_const <- function(sims, weights, m,... )
{
  sim <- rbind( sims, t(- t(sims) + 2 * m ) ) 
  w <- c(weights, 1 - weights )
  out1 <- rep(NA, length(w)) 
  out1[w>0] <- log(2 * w[(w > 0) ]) * ( 2 * w[(w > 0) ])
  out1[w == 0] <- 0
  out2 <- ( (apply(sim, 1, function(k) log_post(k,...)) * ( 2 * w - 1)) )
  est <-  2*mean(out1 - out2 )
  sd.est <- sqrt( sum( (out1 - out2 - est )^2 )/length(weights) )
  return(list( est = est , sd.est = sd.est ))
}

# KL divergence, TV, bias etc
# x: a sample from the approximation
# y : a sample from the target density

compute_kl_tv <- function(x, y, smoothing = 6e-12, n_bins = 300, weights = NULL) {
  require(weights)
  # Check input
  if (!is.numeric(x) || !is.numeric(y)) {
    stop("Both input must be numeric vectors.")
  }
  if (length(x) < 2 || length(y) < 2) {
    stop("Both vector must have at least to elements.")
  }
  # Brakes of the sample space
  combined_range <- range(c(x, y))
  breaks <- seq(combined_range[1], combined_range[2], length.out = n_bins + 1)
  # Empirical densities (normalized histograms)
  if(is.null(weights))
  {
    x_density <- hist(x, breaks = breaks, plot = FALSE)
    y_density <- hist(y, breaks = breaks, plot = FALSE)
    bias <- abs(mean(x) - mean(y))/sd(y)
  }
  else{
    x_density <- wtd.hist(x, breaks = breaks, plot = FALSE, weight = weights)
    y_density <- wtd.hist(y, breaks = breaks, plot = FALSE)
    bias <- abs(2*mean(x*weights)- mean(y))/sd(y)
  }
  # Normalied probabilities with smoothing
  p_x <- (x_density$counts + smoothing) / sum(x_density$counts + smoothing)
  p_y <- (y_density$counts + smoothing) / sum(y_density$counts + smoothing)
  #KL divergence
  kl_divergence <- sum(p_x * log(p_x / p_y))
  #Rev-KL divergence
  rev_kl_divergence <- sum(p_y * log(p_y / p_x))
  #TV distance
  tv_distance <- (1/2)*sum( abs( p_x - p_y ) )
  return(list(kl_divergence =kl_divergence, rev_kl_divergence = rev_kl_divergence,
              tv_distance = tv_distance, bias = bias  ) )
}

# Function computing the bias of an approximation
compute_bias <- function( x, y, weights = NULL)
{
  if(is.null(weights))
  {
    bias <- mean(abs( apply(x,2,mean) - apply(y,2,mean) ))
  }
  else{
    bias <- mean(abs( apply(x,2, function(k) sum( weights * k )/(length(k)/2) ) - apply(y,2,mean) ))
  }
  
  return(bias)
}

# Weights importance sample
# weights = weights skew-symmetric
weights_is <- function(sim, m, Cov, weights, log_post,...)
{
  d <- dim(sim)[2]
  log_nc <- - (d/2)*log(det(Cov))
  prec <- solve(Cov)
  log_p <- apply(sim, 1, function(k) log_post(k,...) )
  log_gaus <-  apply(sim, 1, function(k) - 0.5 * t(k - m)%*%prec%*%(k - m) )
  log_w <- log_p - log_gaus - log(2 * weights ) + log_nc
  max_log <- max(log_w)
  exp( log_w - max_log )
}
# Function to compute ESS from a vector of un-normalized weights
ess <- function(w_is) {
  sum_w <- sum(w_is)
  return((sum_w)^2 / sum(w_is^2) * 1 / length(w_is))
}


# Function multiple ESS
multiple_ess <- function(nsim, m, Cov,log_post, N = 10^4,...)
{
  ess_rep <- ess_is_s <-ess_is<- rep(NA, nsim)
  for(k in 1:nsim)
  {
    sim <- mvtnorm::rmvnorm(N, mean = m, sigma = Cov  )
    w_skew <- apply(sim, 1, function(r) w_joint(r, m, log_post))
    sim_s <- sim_skew(sim,m,w_skew)
    weights_sim_s <- sim_s$weights
    sim_s <- sim_s$sim
    
    # Weights is
    weights_is <- weights_is(sim,
                             m,
                             Cov,
                             rep(0.5,N),
                             log_post)
    weights_is_s <- weights_is(sim_s,
                               m,
                               Cov,
                               weights_sim_s,
                               log_post)
    ess_is_s[k] <- ess(weights_is_s)
    ess_is[k] <- ess(weights_is)
  }
  
  ess_rep <- 100*ess_is_s/ess_is - 100
  
  return(list(ess_rep= ess_rep,
              ess_is_s = N * ess_is_s,
              ess_is = N * ess_is ))
}
