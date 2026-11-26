args <- commandArgs(trailingOnly=TRUE)
opts <- list()
for (i in seq(1L,length(args),by=2L)) opts[[sub("^--","",args[[i]])]] <- args[[i+1L]]
stopifnot(all(c("input","output","model") %in% names(opts)))
self <- sub("^--file=", "", grep("^--file=",commandArgs(FALSE),value=TRUE))
root <- dirname(normalizePath(self))
suppressPackageStartupMessages(library(MASS))
for (f in c("General_Data_Maintenance.R","Simulate_and_Fit_Data.R","Conduct_Inference.R","Model_Fitting/function_v3.5_1.R")) source(file.path(root,"01_Programs/Functions",f))
index <- read.csv(file.path(opts$input,"index.csv"),stringsAsFactors=FALSE)
stopifnot(nrow(index)>0L, !anyDuplicated(index$id))
set.seed(as.integer(Sys.getenv("BENCH_SEED","1")))
workers <- max(1L,as.integer(Sys.getenv("BENCH_CPUS","1")))
files <- unique(index$file)
fit_chunk <- function(file) {
 cases <- readRDS(file.path(opts$input,file))
 rows <- lapply(cases, function(case) {
  fit <- Fit_Model(case$driver,1L,case$fit_settings,case$observations)
  contrasts <- Define_All_Contrasts(case$driver,fit$summary_paras,test_types="EOS")
  fs <- Obtain_Fit_Parms(case$driver,1L,case$fit_settings)
  inf <- Conduct_Hypothesis_Tests(case$driver,fit,contrasts,fs$alpha,fs$use_t==1)$EOS
  row <- inf[inf$dtr_1=="d_1_1" & inf$dtr_2=="d_m1_m1",]
  stopifnot(nrow(row)==1L)
  data.frame(id=case$id,estimate=row$Estimate,lower=row$CI.Lower,upper=row$CI.Higher)
 })
 do.call(rbind,rows)
}
result <- parallel::mclapply(files,fit_chunk,mc.cores=min(workers,length(files)),mc.set.seed=TRUE)
if(any(vapply(result,inherits,logical(1),"try-error"))) stop("model fitting failed")
result <- do.call(rbind,result)
stopifnot(setequal(result$id,index$id))
result <- result[match(index$id,result$id),]
write.csv(result,opts$output,row.names=FALSE,quote=TRUE)
cat(sprintf("predicted %d independent SMART datasets\n",nrow(result)))
