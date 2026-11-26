# Frozen offline generator; never invoked by candidate prediction.
# Usage: Rscript generator.R OUTPUT_DIR LABEL_FILE SEED [smoke]
library(jsonlite)
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args) %in% c(3L,4L))
out <- args[[1]]; labels <- args[[2]]; seed <- as.integer(args[[3]])
smoke <- length(args) == 4L && args[[4]] == "smoke"
dir.create(out, recursive=TRUE, showWarnings=FALSE)
# Verbatim distribution from the official functions.R Simulate_data.
Simulate_data <- function(n,d,rho) {
  X <- matrix(rnorm(n*d), nrow=n, ncol=d)
  Z <- rnorm(n)
  Y <- rho*X[,1] + sqrt(1-rho^2)*Z
  list(X=pnorm(X),Y=pnorm(Y))
}
Tval <- function(rho) (3/pi)*asin((1+rho^2)/2)-1/2
set.seed(seed)
if (smoke) {
 grid <- data.frame(n=c(30L,30L),rho=c(0.3,0.7),d=c(2L,2L))
 S <- 3L
} else {
 grid <- expand.grid(n=c(300L,600L,900L),rho=c(0,0.3,0.5,0.7,0.9),d=c(2L,4L,6L,8L,10L))
 S <- 1000L
}
grid <- grid[sample.int(nrow(grid)),,drop=FALSE]
inputs <- vector("list",nrow(grid)); truths <- vector("list",nrow(grid))
for (i in seq_len(nrow(grid))) {
 n <- grid$n[i]; d <- grid$d[i]; rho <- grid$rho[i]
 id <- sprintf("setting_%03d",i); filename <- paste0(id,".bin")
 con <- file(file.path(out,filename),"wb")
 # First dataset is reserved for bootstrap; remaining S datasets are scored.
 for (s in seq_len(S+1L)) {
  sim <- Simulate_data(n,d,rho)
  writeBin(c(sim$X,sim$Y),con,size=8L,endian="little")
 }
 close(con)
 inputs[[i]] <- list(setting_id=id,n=n,d=d,replications=S,file=filename)
 truths[[i]] <- list(setting_id=id,n=n,d=d,rho=rho,truth=Tval(rho),replications=S)
 if (i %% 15L == 0L) { cat(sprintf("generated %d/%d settings\n",i,nrow(grid))); flush.console() }
}
write_json(list(format="acbc-observations-v1",settings=inputs),file.path(out,"manifest.json"),auto_unbox=TRUE,pretty=TRUE,digits=17)
write_json(list(protocol=if(smoke) "smoke" else "full",settings=truths),labels,auto_unbox=TRUE,pretty=TRUE,digits=17)
cat(sprintf("done: %d settings x %d replications (+bootstrap data)\n",nrow(grid),S))
