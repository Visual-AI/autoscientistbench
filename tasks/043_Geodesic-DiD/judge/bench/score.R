args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args) == 3L)
source(file.path(args[3], 'gdd_reference.R'))
p <- read.csv(args[1], check.names=FALSE)
truth <- read.csv(args[2])
stopifnot(nrow(truth) == 101L, all(is.finite(as.matrix(truth))))
errors <- vapply(seq_len(nrow(p)), function(i) {
  start <- as.numeric(p[i, 2:102]); end <- as.numeric(p[i, 103:203])
  transported <- gtm(start, end, truth$reference,
                      optns=list(type='measure', lower=-3, upper=3))
  sqrt(sum((truth$target - transported)^2)/101)
}, numeric(1))
stopifnot(all(is.finite(errors)))
cat(sprintf('%.17g\n', mean(errors)))
