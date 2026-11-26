
data {
  int<lower=0> N;          // Number of observations
  int<lower=0> D;          // Number of predictors
  matrix[N, D] X;          // Predictor matrix
  array[N] int<lower=0> y;       // Response variable 
  real<lower = 0> nu; // standard error prior
}

parameters {
  vector[D] beta; // Coefficients for predictors
}


model {
  // Gaussian priors on the coefficients
  beta ~ student_t(nu, 0, 1);  //normal(0, sd_p);
  
  // Response
  y ~ poisson(exp(X * beta ) );  // Likelihood 
}

