suppressPackageStartupMessages(library(rstan))
source('Functions_approximations_and_godfit.R')
args <- commandArgs(TRUE)
stopifnot(length(args)==6,identical(args[c(1,3,5)],c('--model','--input','--output')))
model <- readRDS(file.path(args[2],'model.rds'))
invisible(rstan:::grab_cxxfun(model@dso))
options(mc.cores=1); rstan_options(auto_write=FALSE,threads_per_chain=1)
seed <- as.integer(Sys.getenv('BENCH_SEED','1'))
stopifnot(!is.na(seed),seed>=0)
input <- jsonlite::read_json(args[4],simplifyVector=FALSE)
# Input contains observed X/y only. No reference summaries are opened here.
results <- lapply(input$cases,function(case) {
 dat <- list(N=as.integer(case$n),D=as.integer(case$d),nu=case$nu,
             X=do.call(rbind,lapply(case$X,unlist)),y=as.integer(unlist(case$y)))
 fit_seed <- as.integer((as.double(case$fit_seed)+(seed-1)*1000003) %% 2147483646 + 1 - 1)
 set.seed(fit_seed); .Call('metric_hd_reset_count'); t0 <- proc.time()[3]
 optlog <- capture.output(fit <- rstan::optimizing(model,data=dat,hessian=TRUE,seed=fit_seed,refresh=1))
 compiled_calls <- .Call('metric_hd_get_count')
 stopifnot(fit$return_code==0)
 m <- as.numeric(fit$theta_tilde[1:dat$D]); covar <- -solve(fit$hessian)
 sims <- mvtnorm::rmvnorm(10^4,mean=m,sigma=covar)
 calls <- 0L
 log_post <- function(beta,x,y,nu) {
   calls <<- calls+1L
   eta <- x %*% beta
   sum(dt(beta,df=nu,log=TRUE))+sum(dpois(y,exp(eta),log=TRUE))
 }
 w <- apply(sims,1,function(k) w_joint(k,xi=m,log_post=log_post,x=dat$X,y=dat$y,nu=dat$nu))
 sl_wall <- unname(proc.time()[3]-t0)
 stopifnot(calls==20000L,all(is.finite(w)))
 reflected <- t(2*m-t(sims)); weights <- c(w,1-w)
 mean_q <- 2*colMeans(rbind(sims,reflected)*weights)
 cat(sprintf('%s: d=%d, full-fit evaluations=%d, wall_s=%.6f\n',case$id,dat$D,calls+compiled_calls,sl_wall))
 list(id=case$id,mean=unname(mean_q),full_fit_evals=as.integer(calls+compiled_calls),
      wall_s=sl_wall,reference_accesses=0L)
})
jsonlite::write_json(list(version=1,cases=results),args[6],auto_unbox=TRUE,digits=NA,pretty=FALSE)
