# Gaussian expectation propagation for Poisson regression

ep_poisson_t <- function(X, y, nu,
                         max_iter = 100,
                         tol = 1e-4,
                         damping = 0.5,
                         reltol_integ = 1e-6,
                         verbose = TRUE) {
  # X: design matrix (n x p)
  # y: count responses (nonnegative integers)
  # nu: degrees of freedom for the Student-t prior on each beta
  # max_iter: maximum number of EP sweeps
  # tol: convergence tolerance (max change in posterior mean)
  # damping: damping factor (between 0 and 1)
  # reltol_integ: relative tolerance for numerical integration
  #
  # The likelihood for observation i is:
  #   p(y_i | beta) = exp(y_i * f - exp(f)) / y_i!   where f = x_i^T beta.
  # In EP, we approximate each factor by a Gaussian site in f.
  
  n <- nrow(X)
  p <- ncol(X)
  
  # Initialize likelihood sites -
  # Each likelihood factor is approximated by a Gaussian in f = x_i^T beta.
  tau_lik <- rep(0, n)  # precision contributions from likelihood factors
  eta_lik <- rep(0, n)  # linear contributions from likelihood factors
  
  # Initialize prior sites (one per beta_j) -
  tau_prior <- rep(0, p)
  eta_prior <- rep(0, p)
  
  # Global posterior -
  # Global natural parameters:
  #   Q = X^T diag(tau_lik) X + diag(tau_prior)
  #   r = X^T eta_lik + eta_prior
  # We initialize with a small diagonal so Q is invertible.
  Q_global <- diag(1e-6, p)
  r_global <- rep(0, p)
  
  Sigma <- tryCatch(solve(Q_global), error = function(e) diag(1e6, p))
  m <- as.vector(Sigma %*% r_global)
  
  # Helper function for global update
  update_global <- function(Q, r, vec, dtau, deta) {
    # vec is either x_i (for a likelihood factor) or the unit vector for beta_j (for a prior factor).
    Q_new <- Q + dtau * (vec %o% vec)
    r_new <- r + deta * vec
    list(Q = Q_new, r = r_new)
  }
  
  # EP Iterations
  for (iter in 1:max_iter) {
    m_old <- m  # to check convergence later
    
    # Update likelihood sites (for Poisson regression)
    for (i in 1:n) {
      xi <- as.vector(X[i, ])
      
      # Current marginal for f_i = x_i^T beta:
      m_i <- sum(xi * m)
      v_i <- as.numeric(t(xi) %*% Sigma %*% xi)
      
      # Compute cavity for likelihood site i:
      tau_cav <- 1/v_i - tau_lik[i]
      if (tau_cav <= 0) tau_cav <- 2e-16
      v_cav <- 1 / tau_cav
      m_cav <- v_cav * (m_i/v_i - eta_lik[i])
      
      # The tilted distribution is proportional to:
      #   N(f; m_cav, v_cav) * exp(y_i * f - exp(f))
      # (the factor 1/y_i! is omitted as it cancels in moment matching)
      
      # Define integrands for the normalization constant and moments.
      integrand_Z <- function(f) {
         exp( -1/(2*v_cav)*( f -  m_cav )^2 +  y[i] * f - exp(f))
      }
      Z <- tryCatch({
        integrate(integrand_Z, lower = -Inf, upper = Inf, rel.tol = reltol_integ,subdivisions = 200L)$value
      }, error = function(e) {
        warning(paste("Integration error in Z for observation", i, "; using small Z"))
        2e-16
      })
      if (Z < 2e-16) Z <- 2e-16
      
      integrand_mean <- function(f) {
        f *  exp( -1/(2*v_cav)*( f -  m_cav )^2 +  y[i] * f - exp(f))
      }
      m_tilt <- tryCatch({
        integrate(integrand_mean, lower = -Inf, upper = Inf, rel.tol = reltol_integ,subdivisions = 200L)$value / Z
      }, error = function(e) {
        warning(paste("Integration error in mean for observation", i, "; using cavity mean"))
        m_cav
      })
      
      integrand_second <- function(f) {
        f^2 *  exp( -1/(2*v_cav)*( f -  m_cav )^2 +  y[i] * f - exp(f))
      }
      m2_tilt <- tryCatch({
        integrate(integrand_second, lower = -Inf, upper = Inf, rel.tol = reltol_integ,subdivisions = 200L)$value / Z
      }, error = function(e) {
        warning(paste("Integration error in second moment for observation", i, "; using cavity variance"))
        m_cav^2 + v_cav
      })
      v_tilt <- m2_tilt - m_tilt^2
      if (v_tilt <= 0) v_tilt <- 2e-16  # safeguard
      
      # Moment matching: update the likelihood site parameters.
      tau_new <- max(0, 1/v_tilt - 1/v_cav)
      eta_new <- m_tilt/v_tilt - m_cav/v_cav
      
      # Damping:
      tau_update <- (1 - damping) * tau_lik[i] + damping * tau_new
      eta_update <- (1 - damping) * eta_lik[i] + damping * eta_new
      dtau <- tau_update - tau_lik[i]
      deta <- eta_update - eta_lik[i]
      
      tau_lik[i] <- tau_update
      eta_lik[i] <- eta_update
      
      # Update global natural parameters.
      out <- update_global(Q_global, r_global, xi, dtau, deta)
      Q_global <- out$Q
      r_global <- out$r
      
      Sigma <- tryCatch(solve(Q_global), error = function(e) diag(1e6, p))
      m <- as.vector(Sigma %*% r_global)
    } # end likelihood loop
    
    # Update prior sites (Student-t priors for each beta_j) -
    for (j in 1:p) {
      # The prior for beta_j is:
      #   p(beta_j) ∝ dt(beta_j; df = nu)
      # Its EP site is univariate.
      m_j <- m[j]
      v_j <- Sigma[j, j]
      
      # Compute cavity for beta_j:
      tau_cav <- 1/v_j - tau_prior[j]
      if (tau_cav <= 0) tau_cav <- 2e-16
      v_cav <- 1/tau_cav
      m_cav <- v_cav * (m_j/v_j - eta_prior[j])
      
      # The tilted distribution is proportional to:
      #   N(x; m_cav, v_cav) * dt(x; df = nu)
      integrand <- function(x) {
        dnorm(x, mean = m_cav, sd = sqrt(v_cav)) * dt(x, df = nu)
      }
      Z <- tryCatch({
        integrate(integrand, lower = -Inf, upper = Inf, rel.tol = reltol_integ)$value
      }, error = function(e) {
        warning(paste("Integration error in prior Z for beta", j, "; using small Z"))
        2e-16
      })
      if (Z < 2e-16) Z <- 2e-16
      
      integrand_mean <- function(x) {
        x * dnorm(x, mean = m_cav, sd = sqrt(v_cav)) * dt(x, df = nu)
      }
      m_tilt <- tryCatch({
        integrate(integrand_mean, lower = -Inf, upper = Inf, rel.tol = reltol_integ)$value / Z
      }, error = function(e) {
        warning(paste("Integration error in prior mean for beta", j, "; using cavity mean"))
        m_cav
      })
      
      integrand_second <- function(x) {
        x^2 * dnorm(x, mean = m_cav, sd = sqrt(v_cav)) * dt(x, df = nu)
      }
      m2_tilt <- tryCatch({
        integrate(integrand_second, lower = -Inf, upper = Inf, rel.tol = reltol_integ)$value / Z
      }, error = function(e) {
        warning(paste("Integration error in prior second moment for beta", j, "; using cavity variance"))
        m_cav^2 + v_cav
      })
      v_tilt <- m2_tilt - m_tilt^2
      if (v_tilt <= 0) v_tilt <- 2e-16
      
      # New site parameters for beta_j:
      tau_new <- max(0, 1/v_tilt - 1/v_cav)
      eta_new <- m_tilt/v_tilt - m_cav/v_cav
      
      # Damping:
      tau_update <- (1 - damping) * tau_prior[j] + damping * tau_new
      eta_update <- (1 - damping) * eta_prior[j] + damping * eta_new
      dtau <- tau_update - tau_prior[j]
      deta <- eta_update - eta_prior[j]
      
      tau_prior[j] <- tau_update
      eta_prior[j] <- eta_update
      
      # Update global parameters.
      ej <- rep(0, p); ej[j] <- 1
      out <- update_global(Q_global, r_global, ej, dtau, deta)
      Q_global <- out$Q
      r_global <- out$r
      
      Sigma <- tryCatch(solve(Q_global), error = function(e) diag(1e6, p))
      m <- as.vector(Sigma %*% r_global)
    } # end prior loop
    
    # Check convergence (by max change in posterior mean).
    if (max(abs(m - m_old)) < tol) {
      if (verbose) cat("Converged in", iter, "iterations.\n")
      break
    }
    if (verbose) cat("Iteration", iter, "max change =", max(abs(m - m_old)), "\n")
  } # end EP iterations
  
  return(list(m = m,
              Sigma = Sigma,
              tau_lik = tau_lik,
              eta_lik = eta_lik,
              tau_prior = tau_prior,
              eta_prior = eta_prior,
              Q = Q_global,
              r = r_global,
              iterations = iter))
}

# === Example Usage ===
# # Simulate some data:
# set.seed(123)
# n <- 400
# p <- 5
# X <- matrix(rnorm(n * p), n, p)
# X[,1] <- rep(1,n)
# beta_true <- rnorm(p)
# # Poisson regression: lambda = exp(X beta)
# lambda <- exp(X %*% beta_true)
# y <- rpois(n, lambda)
# 
# # Run EP with Student-t priors (e.g., nu = 4 degrees of freedom):
# ep_result <- ep_poisson_t(X, y, nu = 1, max_iter = 100, tol = 1e-4, damping = 0.6)
# 
# # Print the approximate posterior mean and diagonal variances:
# print(ep_result$m)
# print(diag(ep_result$Sigma))
