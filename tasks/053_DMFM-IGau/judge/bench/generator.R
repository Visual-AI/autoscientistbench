# Frozen materialization generator. Specs (IDs, seeds, permutations) are operator
# inputs and must never be passed to the candidate. No fitting occurs here.
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(TRUE)
spec <- fromJSON(args[1],simplifyVector=FALSE)
observed <- list(); truth <- list()
for (i in seq_along(spec$networks)) {
 x <- spec$networks[[i]]
 set.seed(as.integer(x$data_seed))
 true_labels <- seq_len(50L) %% 2L
 true_labels[true_labels == 0L] <- 2L
 true_Q <- matrix(0.1,2,2); diag(true_Q) <- 0.35
 Ad <- matrix(0,50,50)
 # Exact original commu_git.R/common.R loop, including diagonal RNG draws.
 for (a in 1:50) for (b in a:50) {
   Ad[a,b] <- rbinom(1,1,true_Q[true_labels[a],true_labels[b]])
   Ad[b,a] <- Ad[a,b]
 }
 diag(Ad) <- 0
 set.seed(as.integer(x$permutation_seed)); perm <- sample.int(50L)
 # Relabel nodes consistently to keep the original alternating truth private.
 observed[[i]] <- list(id=x$id,adjacency=Ad[perm,perm])
 truth[[i]] <- c(x,list(labels=true_labels[perm],permutation=perm))
}
write_json(list(schema=spec$schema,networks=observed),args[2],auto_unbox=TRUE,digits=16)
write_json(list(schema=spec$schema,iterations=spec$iterations,burn_in=spec$burn_in,
 parameters=list(n=50,communities=2,sizes=c(25,25),p=0.35,q=0.1),networks=truth),args[3],auto_unbox=TRUE,digits=16)
