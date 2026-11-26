# combine the per-(model, rho) replicate files written by sim_example1.R
f <- list.files("results/pvals", pattern = "^model[12]_rho.*\\.csv$", full.names = TRUE)
if (!length(f)) stop("no p-value files found in results/pvals")
fix_names <- function(x) {
  nm <- names(x)
  nm[nm == "jt.AD"] <- "jt-AD"; nm[nm == "jt.F"] <- "jt-F"
  nm[nm == "prod.AD"] <- "prod-AD"; nm[nm == "prod.F"] <- "prod-F"
  names(x) <- nm
  x
}
d <- fix_names(do.call(rbind, lapply(f, function(z) read.csv(z, check.names = FALSE))))
d <- d[order(d$model, d$rho, d$rep), ]
write.csv(d, "results/all_pvalues.csv", row.names = FALSE)
if (!identical(names(d), c("model","rho","rep","jt","jt-AD","jt-F","prod","prod-AD","prod-F"))) stop("unexpected column names")
cat("combined", nrow(d), "replicates ->", "results/all_pvalues.csv", "\n")
