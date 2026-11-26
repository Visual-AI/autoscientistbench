# Offline observation generation only; no posterior fitting.
args <- commandArgs(trailingOnly=TRUE)
set.seed(as.integer(args[1])); reps <- as.integer(args[2])
rows <- list(); k <- 0L
for(n in c(100L,500L,1000L)) for(rate in c(.50,.55,.60,.65)) for(ep in c(.2,.5,1)) {
  n1 <- n/2; n0 <- n/2; n11 <- round(n1*rate); n01 <- round(n0*.5)
  p <- phyper(n11-1,n11+n01,n-n11-n01,n1,lower.tail=FALSE)
  for(i in seq_len(reps)) {
    rho <- exp(-ep)
    t11 <- max(0,min(n1,n11+rgeom(1,1-rho)-rgeom(1,1-rho)))
    t01 <- max(0,min(n0,n01+rgeom(1,1-rho)-rgeom(1,1-rho)))
    k <- k+1L
    rows[[k]] <- data.frame(n1=n1,n0=n0,epsilon=ep,t11=t11,t01=t01,
                          cell=paste(n,rate,ep,sep='_'),truth=p)
  }
}
write.csv(do.call(rbind,rows),args[3],row.names=FALSE)
