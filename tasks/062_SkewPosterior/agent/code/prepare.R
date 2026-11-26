suppressPackageStartupMessages(library(rstan))
out <- commandArgs(TRUE)[1]; base <- '.'
rstan_options(auto_write=FALSE)
ret <- stanc(file=file.path(base,'poisson_hd.stan'),model_name='poisson_hd_counted')
original <- ret$cppcode
# Count every compiled log-density invocation, including initialization, line
# search and numerical Hessian gradients. No arithmetic/model change.
header <- paste0('static unsigned long long metric_hd_counter = 0;\n',
  'extern "C" SEXP metric_hd_get_count() { return Rf_ScalarReal(static_cast<double>(metric_hd_counter)); }\n',
  'extern "C" SEXP metric_hd_reset_count() { metric_hd_counter = 0; return R_NilValue; }\n')
ret$cppcode <- paste0(header,gsub('(log_prob_impl\\([^{}]+\\{)', '\\1\n    ++metric_hd_counter;',ret$cppcode,perl=TRUE))
stopifnot(length(gregexpr('\\+\\+metric_hd_counter',ret$cppcode)[[1]])==2)
writeLines(original,file.path(out,'generated_original.hpp'))
writeLines(ret$cppcode,file.path(out,'generated_counted.hpp'))
model <- stan_model(stanc_ret=ret,auto_write=FALSE)
saveRDS(model,file.path(out,'model.rds'))
.Call('metric_hd_reset_count'); stopifnot(.Call('metric_hd_get_count')==0)
writeLines(capture.output(sessionInfo()),file.path(out,'sessionInfo.txt'))
