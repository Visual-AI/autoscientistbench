# Trusted scorer-side R process. Candidate code never runs in this process.
args<-commandArgs(trailingOnly=TRUE)
registry<-readRDS(args[1]);item<-registry[[args[2]]]
if(is.null(item))stop("unregistered private sampler")
M<-item$M;B<-item$B;assign(".Random.seed",item$state,envir=.GlobalEnv)
cache<-matrix(NA_real_,M,B);generated<-0L;disclosed<-integer(M)
input<-file("stdin",open="r");cat("ready\n");flush(stdout())
repeat {
 line<-readLines(input,n=1,warn=FALSE);if(!length(line))break
 fields<-strsplit(line,"\t",fixed=TRUE)[[1]]
 indices<-as.integer(strsplit(fields[1],",",fixed=TRUE)[[1]]);end<-as.integer(fields[2])
 if(anyNA(indices)||!length(indices)||anyDuplicated(indices)||any(indices<1|indices>M)||is.na(end)||end<1||end>B)stop("invalid request")
 if(end>generated) {
   new<-seq.int(generated+1L,end)
   cache[,new]<-matrix(rnorm(M*length(new)),nrow=M)
   generated<-end
 }
 old<-disclosed[indices]
 chunks<-vapply(seq_along(indices),function(k) {
   if(end<=old[k])return("")
   paste(sprintf("%.17g",cache[indices[k],seq.int(old[k]+1L,end)]),collapse=",")
 },character(1))
 disclosed[indices]<-pmax(disclosed[indices],end)
 cat(paste(old,collapse=","),"\t",paste(chunks[nzchar(chunks)],collapse=","),"\n",sep="");flush(stdout())
}
