# Only originally authorized null values enter this worker; RNG state stays private.
open_sampler <- local({
  connection <- NULL
  function(trial_id,M,B) {
    if (is.null(connection)) {
      portfile <- tempfile("rpc-port-")
      system2("python3",c(shQuote(file.path(Sys.getenv("BENCH_BENCH_DIR"),"r_rpc_bridge.py")),shQuote(portfile)),
              wait=FALSE,stdout="/tmp/rpc-bridge.stdout",stderr="/tmp/rpc-bridge.stderr")
      for (i in seq_len(200)) {if(file.exists(portfile))break;Sys.sleep(0.025)}
      if (!file.exists(portfile))stop("sampler bridge unavailable")
      port <- as.integer(readLines(portfile,warn=FALSE))
      connection <<- socketConnection("127.0.0.1",port=port,blocking=TRUE,open="r+",timeout=120)
    }
    cache <- matrix(NA_real_,nrow=M,ncol=B)
    disclosed <- integer(M)
    sample_prefix <- function(indices,times,drop=TRUE) {
      if (!is.numeric(indices)||anyNA(indices)||any(indices!=as.integer(indices))||
          any(indices<1|indices>M)||anyDuplicated(indices))stop("invalid hypotheses")
      if (!is.numeric(times)||anyNA(times)||!length(times)||!identical(as.integer(times),seq_len(max(times)))||max(times)>B)stop("prefix required")
      end<-as.integer(max(times))
      if(!length(indices))return(cache[indices,times,drop=drop])
      # Large prefixes use multiple transport messages against one stream.
      start<-1L
      while(start<=length(indices)) {
        remaining<-indices[seq.int(start,length(indices))]
        size<-sum(cumsum(pmax(0L,end-disclosed[remaining]))<=16384L)
        if(size<1L)stop("prefix exceeds the original B limit")
        chosen<-remaining[seq_len(size)]
        writeLines(paste("sample",paste(chosen,collapse=","),end,sep="\t"),connection);flush(connection)
        line<-readLines(connection,n=1,warn=FALSE)
        if(length(line)!=1||!startsWith(line,"ok\t"))stop("private sampler rejected request")
        parts<-strsplit(substring(line,4),"\t",fixed=TRUE)[[1]]
        old<-as.integer(strsplit(parts[1],",",fixed=TRUE)[[1]])
        values<-if(length(parts)>1)as.numeric(strsplit(parts[2],",",fixed=TRUE)[[1]]) else numeric(0)
        if(length(old)!=length(chosen)||!identical(old,disclosed[chosen]))stop("sampler count mismatch")
        if(any(!is.finite(values)))stop("nonfinite sampler response")
        pos<-0L
        for(k in seq_along(chosen)) {
          if(end>old[k]) {
            n<-end-old[k];cache[chosen[k],seq.int(old[k]+1L,end)] <<- values[seq.int(pos+1L,pos+n)]
            pos<-pos+n
          }
        }
        if(pos!=length(values))stop(paste("malformed sampler value response: expected",pos,"received",length(values)))
        disclosed[chosen] <<- pmax(disclosed[chosen],end)
        start<-start+length(chosen)
      }
      cache[indices,times,drop=drop]
    }
    structure(list(sample=sample_prefix,counts=function()as.integer(disclosed)),class="bench_sampler")
  }
})
`[.bench_sampler`<-function(x,i,j,...,drop=TRUE)x$sample(i,j,drop=drop)
