suppressPackageStartupMessages(library(jsonlite))
suppressPackageStartupMessages(library(glmnet))
args <- commandArgs(TRUE)
manifest <- fromJSON(file.path(args[1],"manifest.json"),simplifyVector=FALSE)
results <- lapply(manifest$instances,function(item){
 label <- readRDS(file.path(args[1],item$file))
 selected <- as.matrix(glmnet(label$X,label$y,lambda=label$lambda,intercept=TRUE,standardize=TRUE,family="gaussian")$beta != 0)
 H0 <- label$beta == 0
 FDP <- vapply(seq_len(ncol(selected)),function(k) sum(selected[,k]&H0)/max(1,sum(selected[,k])),numeric(1))
 list(id=item$id,FDP=unname(FDP))
})
cat(toJSON(results,auto_unbox=TRUE,digits=NA))
