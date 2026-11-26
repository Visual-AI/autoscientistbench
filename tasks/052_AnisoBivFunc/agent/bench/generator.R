# Frozen observation-only materializer. Exact original fbm_sheet_f RNG and doubles.
# Operator provides a private plan; candidate inputs never contain this plan.
a <- commandArgs(TRUE)
script <- sub('^--file=', '', commandArgs(FALSE)[grepl('^--file=', commandArgs(FALSE))][1])
base <- dirname(normalizePath(script))
source(file.path(base,'generator_source','dgp.R'))
source(file.path(base,'generator_source','fast_kernels.R'))
plan <- read.csv(a[1],stringsAsFactors=FALSE)
cores <- as.integer(a[2])
groups <- split(seq_len(nrow(plan)),plan$path)
t0 <- Sys.time()
res <- parallel::mclapply(groups,function(ix) {
 rows <- plan[ix,,drop=FALSE]; target<-rows$path[1]
 expected <- sum(rows$N*rows$side^2*8)
 if(file.exists(target) && file.info(target)$size==expected) return(TRUE)
 output <- paste0(target,'.partial')
 f<-file(output,'wb')
 tryCatch({
  for(k in seq_len(nrow(rows))) {
   row<-rows[k,];set.seed(row$seed)
   for(i in seq_len(row$N)) {
    X<-fbm_sheet_f(row$side,row$side,row$alpha,0.8,0.5,'sum',row$sigma)
    writeBin(as.vector(X),f,size=8,endian='little')
   }
  }
  close(f)
  stopifnot(file.info(output)$size==expected)
  stopifnot(file.rename(output,target))
  cat(sprintf('completed %s %d records %.0f bytes %.1fs\n',basename(target),nrow(rows),expected,as.numeric(difftime(Sys.time(),t0,units='secs'))));flush.console()
  TRUE
 },error=function(e){try(close(f),silent=TRUE);stop(e)})
},mc.cores=cores,mc.preschedule=FALSE)
stopifnot(all(vapply(res,isTRUE,logical(1))))
cat(sprintf('FULL_GENERATION_COMPLETE records=%d seconds=%.1f\n',nrow(plan),as.numeric(difftime(Sys.time(),t0,units='secs'))))
