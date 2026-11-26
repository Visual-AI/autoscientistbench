### Simulation study on for Poisson regression                  ###
### These results are reported in Section 4.2 of the main paper ###

rm(list = ls())
set.seed(123)
# IMPORTANT: Set the working directory to your file source location
# Load packages 
library(rstan)
source("Functions_Poisson_HD.R")

# Define the stan model
poisson_model_code <- "
data {
  int<lower=0> N;          // Number of observations
  int<lower=0> D;          // Number of predictors
  matrix[N, D] X;          // Predictor matrix
  int<lower=0> y[N];       // Response variable 
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

"
# Compile stan model
poisson_model_stan <- stan_model( model_code = poisson_model_code)

# Simulation design matrix and regression coefficents
# The predictors are Bernoulli with different probabilities
prs <- runif(500,0.02,0.10)
X <- cbind( rep(1,10^3), matrix(rbinom(10^3*499,1,prs), ncol = 499, byrow = TRUE) )
beta_0 <- runif(500,-0.5,0.5)/sqrt(1:500)

# Set different sample sizes and parameter dimensions
N <- 200*seq(1:5)
D <- 6*floor( (N/2)^(1/2) )

# Number of experiments
nsim <- 10

# Run simulation study
set.seed(123)
counter <- 0
while(counter < nsim)
{
  sim_p <- tryCatch( poisson_simulation( X = X, N = N ,D = D, beta_0 =beta_0, 
                     stan.model = poisson_model_code ) , error = function(e) NA)
  if( ( !all(is.na(sim_p)) ) & all(sim_p$results_lap[,4]< 0.7) &
        all(sim_p$results_gep[,4]< 0.7) & all(sim_p$results_gvb[,4]< 0.7) )
  {
    counter <- counter + 1
    if(counter == 1)
    {  
    results_lap_sqrt <- sim_p$results_lap 
    results_gvb_sqrt <-  sim_p$results_gvb
    results_gep_sqrt <-  sim_p$results_gep
    }
    if(counter > 1)
    {
      results_lap_sqrt[,1:4] <- results_lap_sqrt[,1:4] + sim_p$results_lap[1:4] # for averaging between replications
      results_gvb_sqrt[,1:4] <- results_gvb_sqrt[,1:4] + sim_p$results_gvb[1:4]
      results_gep_sqrt[,1:4] <- results_gep_sqrt[,1:4] + sim_p$results_gep[1:4]
    } 
  }
}

# Taking the average over the different experiments
results_lap_sqrt[,1:4] <- results_lap_sqrt[,1:4]/counter
results_gvb_sqrt[,1:4] <- results_gvb_sqrt[,1:4]/counter
results_gep_sqrt[,1:4] <- results_gep_sqrt[,1:4]/counter

save(results_lap_sqrt, file = "results_lap_sqrt" )
save(results_gvb_sqrt, file = "results_gvb_sqrt" )
save(results_gep_sqrt, file = "results_gep_sqrt" )

# Load experiments
load( file = "results_lap_sqrt" )
load( file = "results_gvb_sqrt" )
load( file = "results_gep_sqrt" )

# Compute relative accuracy improvement (in percentage)
#-----------------------------------------------------------------
# Laplace                    
Results<-results_lap_sqrt[which(results_lap_sqrt$N=="n = 200"),]
as.matrix(apply((1-Results[which(Results$Method=="S-LAP"),c(3,1,2,4)]/Results[which(Results$Method=="LAP"),c(3,1,2,4)])*100,2,median),4,1)

Results<-results_lap_sqrt[which(results_lap_sqrt$N=="n = 400"),]
as.matrix(apply((1-Results[which(Results$Method=="S-LAP"),c(3,1,2,4)]/Results[which(Results$Method=="LAP"),c(3,1,2,4)])*100,2,median),4,1)

Results<-results_lap_sqrt[which(results_lap_sqrt$N=="n = 600"),]
as.matrix(apply((1-Results[which(Results$Method=="S-LAP"),c(3,1,2,4)]/Results[which(Results$Method=="LAP"),c(3,1,2,4)])*100,2,median),4,1)

Results<-results_lap_sqrt[which(results_lap_sqrt$N=="n = 800"),]
as.matrix(apply((1-Results[which(Results$Method=="S-LAP"),c(3,1,2,4)]/Results[which(Results$Method=="LAP"),c(3,1,2,4)])*100,2,median),4,1)

Results<-results_lap_sqrt[which(results_lap_sqrt$N=="n = 1000"),]
as.matrix(apply((1-Results[which(Results$Method=="S-LAP"),c(3,1,2,4)]/Results[which(Results$Method=="LAP"),c(3,1,2,4)])*100,2,median),4,1)

#-----------------------------------------------------------------
# VB
Results<-results_gvb_sqrt[which(results_gvb_sqrt$N=="n = 200"),]
as.matrix(apply((1-Results[which(Results$Method=="S-GVB"),c(3,1,2,4)]/Results[which(Results$Method=="GVB"),c(3,1,2,4)])*100,2,median),4,1)

Results<-results_gvb_sqrt[which(results_gvb_sqrt$N=="n = 400"),]
as.matrix(apply((1-Results[which(Results$Method=="S-GVB"),c(3,1,2,4)]/Results[which(Results$Method=="GVB"),c(3,1,2,4)])*100,2,median),4,1)

Results<-results_gvb_sqrt[which(results_gvb_sqrt$N=="n = 600"),]
as.matrix(apply((1-Results[which(Results$Method=="S-GVB"),c(3,1,2,4)]/Results[which(Results$Method=="GVB"),c(3,1,2,4)])*100,2,median),4,1)

Results<-results_gvb_sqrt[which(results_gvb_sqrt$N=="n = 800"),]
as.matrix(apply((1-Results[which(Results$Method=="S-GVB"),c(3,1,2,4)]/Results[which(Results$Method=="GVB"),c(3,1,2,4)])*100,2,median),4,1)

Results<-results_gvb_sqrt[which(results_gvb_sqrt$N=="n = 1000"),]
as.matrix(apply((1-Results[which(Results$Method=="S-GVB"),c(3,1,2,4)]/Results[which(Results$Method=="GVB"),c(3,1,2,4)])*100,2,median),4,1)

#-----------------------------------------------------------------
# EP
Results<-results_gep_sqrt[which(results_gep_sqrt$N=="n = 200"),]
as.matrix(apply((1-Results[which(Results$Method=="S-GEP"),c(3,1,2,4)]/Results[which(Results$Method=="GEP"),c(3,1,2,4)])*100,2,median),4,1)

Results<-results_gep_sqrt[which(results_gep_sqrt$N=="n = 400"),]
as.matrix(apply((1-Results[which(Results$Method=="S-GEP"),c(3,1,2,4)]/Results[which(Results$Method=="GEP"),c(3,1,2,4)])*100,2,median),4,1)

Results<-results_gep_sqrt[which(results_gep_sqrt$N=="n = 600"),]
as.matrix(apply((1-Results[which(Results$Method=="S-GEP"),c(3,1,2,4)]/Results[which(Results$Method=="GEP"),c(3,1,2,4)])*100,2,median),4,1)

Results<-results_gep_sqrt[which(results_gep_sqrt$N=="n = 800"),]
as.matrix(apply((1-Results[which(Results$Method=="S-GEP"),c(3,1,2,4)]/Results[which(Results$Method=="GEP"),c(3,1,2,4)])*100,2,median),4,1)

Results<-results_gep_sqrt[which(results_gep_sqrt$N=="n = 1000"),]
as.matrix(apply((1-Results[which(Results$Method=="S-GEP"),c(3,1,2,4)]/Results[which(Results$Method=="GEP"),c(3,1,2,4)])*100,2,median),4,1)
                    
