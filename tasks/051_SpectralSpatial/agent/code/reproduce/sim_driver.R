# Protocol adaptation: fit the original five models to supplied observed X/Y.
args <- commandArgs(trailingOnly=TRUE)
input <- args[1]; output <- args[2]; model_dir <- args[3]
src <- readLines("simulation-code/functions_sim.R")
src[grep("^registerDoParallel",src)] <- "foreach::registerDoSEQ()"
suppressMessages(eval(parse(text=src),envir=globalenv()))
smoke <- identical(Sys.getenv("BENCH_SMOKE"),"1")
iters <- if(smoke) 30L else 5000L
burn <- if(smoke) 10L else 1000L
L <- 10L; K <- 5L; E <- 10L; R <- 5L
seed <- as.integer(Sys.getenv("BENCH_SEED","1"))
cpus <- max(1L,as.integer(Sys.getenv("BENCH_CPUS","1")))
manifest <- read.csv(file.path(input,"manifest.csv"),stringsAsFactors=FALSE)
stopifnot(!anyDuplicated(manifest$id),nrow(manifest)>0L)
if (!smoke) stopifnot(nrow(manifest)==60L,all(manifest$n==400L),all(manifest$grid_side==20L))
post_summary <- function(draws) {
  d <- dim(draws); M <- matrix(draws,prod(d[-length(d)]),d[length(d)])
  s <- t(apply(M,1,function(v)c(mean(v),quantile(v,.025),median(v),quantile(v,.975),sd(v))))
  array(s,c(d[-length(d)],5))
}
fit_one <- function(j) {
  row <- manifest[j,]; n <- as.integer(row$n); Qmat <- makeQ(as.integer(row$grid_side))
  dat <- readRDS(file.path(input,row$observed_file))
  stopifnot(identical(names(dat),c("X","Y")),all(dim(dat$X)==c(n,E)),all(dim(dat$Y)==c(n,R)))
  # Fitting seeds are independent of hidden data-generation seeds.
  set.seed(as.integer((as.double(seed)*100003+j) %% 2147483646)+1L)
  EI <- eigen(Qmat); Gamma <- EI$vectors; W <- EI$values
  B <- bSpline(W,df=L,intercept=TRUE)
  Xstar <- t(Gamma)%*%dat$X; Ystar <- t(Gamma)%*%dat$Y
  cat(sprintf("fitting %s n=%d iters=%d burn=%d all five models\n",row$id,n,iters,burn))
  f <- MCMC_model4_8_v7(Ystar,Xstar,Qmat=Qmat,E=E,n=n,R=R,Q=R,L,K,
                        B=B,iters=iters,burn=burn,full_result=TRUE,W=W)
  msm <- post_summary(f$beta)[1,,,]
  comparisons <- list(MSM=msm)
  rm(f); gc(FALSE)
  for(m in 2:4) {
    num2 <- as.integer(c(n,.8*n,.4*n)[m-1L])
    f <- MCMC_model3_2(Ystar,Xstar,Q=Qmat,p=E,n=n,r=R,q=R,
                       iters=iters,burn=burn,1,num2)
    comparisons[[c("Naive","Spatial+80","Spatial+40")[m-1L]]] <- post_summary(f$beta)
  }
  usm <- array(NA_real_,c(E,R,5))
  for(r in 1:R) {
    f <- MCMC_model4_9_v2(Ystar[,r],Xstar,Qmat=Qmat,E=E,n=n,L,K,
                          B=B,iters=iters,burn=burn)
    usm[,r,] <- post_summary(f$beta)[1,,]
  }
  comparisons$USM <- usm
  saveRDS(comparisons,file.path(model_dir,paste0(row$id,"_comparisons.rds")))
  rows <- expand.grid(exposure=1:9,outcome=1:5)
  rows$id <- row$id
  idx <- cbind(rows$exposure+1L,rows$outcome)
  rows$mean <- msm[,,1][idx]; rows$lower <- msm[,,2][idx]; rows$upper <- msm[,,4][idx]
  rows[,c("id","exposure","outcome","mean","lower","upper")]
}
res <- parallel::mclapply(seq_len(nrow(manifest)),fit_one,mc.cores=min(cpus,nrow(manifest)),
                         mc.preschedule=FALSE,mc.set.seed=FALSE)
if(any(vapply(res,inherits,logical(1),"try-error"))) stop("a model fit failed")
write.csv(do.call(rbind,res),output,row.names=FALSE)
