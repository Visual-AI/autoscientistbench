suppressPackageStartupMessages(library(rstan))
root <- Sys.getenv('PAPER_DIR'); out <- commandArgs(TRUE)[1]
source(file.path(root,'.repro/metric/Functions_approximations_and_godfit.R'))
model <- readRDS(file.path(root,'.repro/metric/model.rds'))
invisible(rstan:::grab_cxxfun(model@dso))
options(mc.cores=1); rstan_options(auto_write=FALSE, threads_per_chain=1)
cpus <- as.integer(Sys.getenv('MERLIN_CPUS','16'))
workers <- min(cpus, as.integer(Sys.getenv('METRIC_WORKERS',as.character(cpus))))
N <- as.integer(strsplit(Sys.getenv('METRIC_N','200,400,600,800,1000'),',')[[1]])
reps <- as.integer(Sys.getenv('METRIC_REPS','10'))
set.seed(123)
prs <- runif(500,0.02,0.10)
X <- cbind(rep(1,10^3), matrix(rbinom(10^3*499,1,prs), ncol=499, byrow=TRUE))
beta_0 <- runif(500,-0.5,0.5)/sqrt(1:500)
saveRDS(list(seed=123,prs=prs,X=X,beta_0=beta_0),file.path(out,'design.rds'))
write_json <- function(x,p) jsonlite::write_json(x,p,auto_unbox=TRUE,pretty=TRUE,digits=16,null='null')
write_json(list(N=N,D=6*floor(sqrt(N/2)),reps=reps,workers=workers,chains=4,warmup=1000,iter=11000,draws=10000,nu=3),file.path(out,'config.json'))
# Fits receive only the dataset and independent seeds; no HMC reference is in scope.
approximate <- function(dat,seed,dir) {
  set.seed(seed); .Call('metric_hd_reset_count'); t0 <- proc.time()[3]
  optlog <- capture.output(fit <- rstan::optimizing(model,data=dat,hessian=TRUE,seed=seed,refresh=1))
  writeLines(optlog,file.path(dir,'optimizer.log'))
  compiled_calls <- .Call('metric_hd_get_count')
  stopifnot(fit$return_code==0)
  optlog <- sub('^Chain [0-9]+: ', '', optlog)
  rows <- grep('^\\s+[0-9]+\\s+[-0-9.e+]+\\s+',optlog,value=TRUE)
  native_evals <- if(length(rows)) as.numeric(strsplit(trimws(tail(rows,1)),'\\s+')[[1]][7]) else NA_real_
  m <- as.numeric(fit$theta_tilde[1:dat$D]); covar <- -solve(fit$hessian)
  sims <- mvtnorm::rmvnorm(10^4,mean=m,sigma=covar)
  lap_wall <- unname(proc.time()[3]-t0)
  calls <- 0L
  log_post <- function(beta,x,y,nu) {
    calls <<- calls+1L
    eta <- x %*% beta
    sum(dt(beta,df=nu,log=TRUE))+sum(dpois(y,exp(eta),log=TRUE))
  }
  w <- apply(sims,1,function(k) w_joint(k,xi=m,log_post=log_post,x=dat$X,y=dat$y,nu=dat$nu))
  sl_wall <- unname(proc.time()[3]-t0)
  stopifnot(calls==20000L,all(is.finite(w)))
  # Exact transcription of compute_kl_tv's unweighted/weighted mean.
  reflected <- t(2*m-t(sims)); weights <- c(w,1-w)
  ans <- list(lap_mean=colMeans(sims),slap_mean=2*colMeans(rbind(sims,reflected)*weights),
       mode=m,covariance=covar,draws=sims,skew_weights=w,seed=seed,
       lap_wall_s=lap_wall,slap_wall_s=sl_wall,skew_log_post_calls=calls,
       optimizer_gradient_evals=native_evals,hessian_gradient_evals=2*dat$D,
       lap_total_evals=compiled_calls,
       slap_total_evals=calls+compiled_calls,
       reference_accesses=0L)
  saveRDS(ans,file.path(dir,'approximation.rds'),compress=FALSE)
  ans
}
reference <- function(dat,seed,dir,prefix='reference') {
  t0 <- proc.time()[3]
  fit <- sampling(model,data=dat,warmup=1000,iter=11000,chains=4,cores=1,seed=seed,refresh=1000)
  arr <- as.array(fit,pars='beta'); draws <- matrix(arr,ncol=dat$D)
  summ <- summary(fit,pars='beta')$summary
  sp <- get_sampler_params(fit,inc_warmup=FALSE)
  ans <- list(mean=colMeans(draws),sd=apply(draws,2,sd),seed=seed,wall_s=unname(proc.time()[3]-t0),
       summary=summ,diagnostics=list(max_rhat=max(summ[,'Rhat']),min_ess=min(summ[,'n_eff']),
       divergences=sum(vapply(sp,function(z)sum(z[,'divergent__']),0)),
       max_treedepth_hits=sum(vapply(sp,function(z)sum(z[,'treedepth__']>=10),0))),
       chain_means=apply(arr,c(2,3),mean),chain_sds=apply(arr,c(2,3),sd))
  # Contiguous batches retain autocorrelation for the HMC-floor MC standard error.
  ans$batch_means <- do.call(rbind,lapply(1:4,function(ch) t(vapply(1:20,function(b) colMeans(arr[((b-1)*500+1):(b*500),ch,]),numeric(dat$D)))))
  saveRDS(ans,file.path(dir,paste0(prefix,'.rds')),compress=FALSE)
  ans
}
run_task <- function(task) {
  dir <- file.path(out,sprintf('candidate_%03d/n_%04d',task$candidate,task$dat$N)); dir.create(dir,recursive=TRUE)
  log <- file(file.path(root,'logs/metric',paste0(basename(out),'_c',task$candidate,'_n',task$dat$N,'.log')),'wt')
  sink(log); sink(log,type='message'); on.exit({sink(type='message');sink();close(log)})
  tryCatch({
    saveRDS(list(candidate=task$candidate,data_seed=task$data_seed,data=task$dat),file.path(dir,'dataset.rds'))
    a <- approximate(task$dat,task$fit_seed,dir)
    h <- reference(task$dat,task$hmc_seed,dir)
    bias <- cbind(LAP=abs(a$lap_mean-h$mean)/h$sd,SLAP=abs(a$slap_mean-h$mean)/h$sd)
    accepted <- all(is.finite(bias)) && all(bias<0.7)
    saveRDS(list(bias=bias,accepted=accepted),file.path(dir,'acceptance.rds'))
    list(candidate=task$candidate,n=task$dat$N,accepted=accepted,dir=dir)
  },error=function(e) {writeLines(conditionMessage(e),file.path(dir,'error.txt'));list(candidate=task$candidate,n=task$dat$N,accepted=FALSE,error=conditionMessage(e))})
}
accepted <- integer(); records <- list(); next_candidate <- 1L
while(length(accepted)<reps) {
  ids <- seq.int(next_candidate,length.out=reps-length(accepted))
  tasks <- unlist(lapply(ids,function(id) {
    data_seed <- 123L+(id-1L)*100003L; set.seed(data_seed)
    lapply(N,function(n) {d <- 6*floor(sqrt(n/2));dat <- list(N=n,D=d,X=X[1:n,1:d],y=rpois(n,exp(X[1:n,1:d]%*%beta_0[1:d])),nu=3)
      list(candidate=id,data_seed=data_seed,dat=dat,fit_seed=data_seed+10000L+n,hmc_seed=data_seed+20000L+n)})
  }),recursive=FALSE)
  cat('Starting candidates',paste(ids,collapse=','),'with',length(tasks),'fits on',workers,'workers\n');flush.console()
  res <- parallel::mclapply(tasks,run_task,mc.cores=workers,mc.preschedule=FALSE,mc.set.seed=FALSE)
  records <- c(records,res)
  if(any(vapply(res,function(z)!is.null(z$error),TRUE))) stop('Fit error; see candidate error.txt (do not reject implementation errors as data)')
  for(id in ids) {
    rr <- Filter(function(z)z$candidate==id,res)
    if(length(rr)==length(N)&&all(vapply(rr,function(z)isTRUE(z$accepted),TRUE))) accepted <- c(accepted,id)
  }
  write_json(list(accepted=accepted,attempts=records),file.path(out,'acceptance_manifest.json'))
  cat('Accepted',length(accepted),'of',reps,'replicates\n');flush.console()
  next_candidate <- max(ids)+1L
  if(next_candidate>100L) stop('100 candidate limit reached')
}
# A second independent four-chain run on the first accepted replicate only.
floors <- parallel::mclapply(N,function(n) {
  dir <- file.path(out,sprintf('candidate_%03d/n_%04d',accepted[1],n))
  dat <- readRDS(file.path(dir,'dataset.rds'))$data
  log <- file(file.path(root,'logs/metric',paste0(basename(out),'_floor_n',n,'.log')),'wt')
  sink(log);sink(log,type='message');on.exit({sink(type='message');sink();close(log)})
  reference(dat,900000L+n,dir,'reference_independent');TRUE
},mc.cores=min(workers,length(N)),mc.set.seed=FALSE)
stopifnot(all(vapply(floors,isTRUE,TRUE)))
write_json(list(accepted_candidates=accepted,N=N,complete=TRUE),file.path(out,'complete.json'))
cat('Simulation complete.\n')
