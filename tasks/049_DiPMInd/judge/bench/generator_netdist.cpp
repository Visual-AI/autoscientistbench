// Glue code written for this reproduction (not part of the authors' repository).
// L1 (Manhattan) distance matrix between m square matrices of equal size,
// i.e. d(A,B) = sum_{j,j'} |A_{j,j'} - B_{j,j'}| as used in Section 3 of the paper
// for graph Laplacian matrices.  Input: a list of m numeric matrices (m * k * k).
// [[Rcpp::depends(Rcpp)]]
#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
NumericMatrix l1mat_dist(const Rcpp::List& mats) {
  const int m = mats.size();
  NumericMatrix out(m, m);
  std::vector<const double*> p(m);
  std::vector<int> nrow(m), ncol(m);
  for (int i = 0; i < m; ++i) {
    NumericMatrix a = mats[i];
    if (a.nrow() != a.ncol()) Rcpp::stop("matrices must be square");
    p[i] = a.begin(); nrow[i] = a.nrow(); ncol[i] = a.ncol();
  }
  for (int i = 0; i < m; ++i) {
    out(i, i) = 0.0;
    for (int j = i + 1; j < m; ++j) {
      if (nrow[i] != nrow[j]) Rcpp::stop("matrices must have equal size");
      const double* a = p[i];
      const double* b = p[j];
      const int tot = nrow[i] * ncol[i];
      double s = 0.0;
      for (int t = 0; t < tot; ++t) s += std::fabs(a[t] - b[t]);
      out(i, j) = s; out(j, i) = s;
    }
  }
  return out;
}

// same but for matrices stored in a 3D numeric array (k x k x m)
// [[Rcpp::export]]
NumericMatrix l1mat_dist_arr(const Rcpp::NumericVector& arr, int k, int m) {
  NumericMatrix out(m, m);
  const double* p = arr.begin();
  const int tot = k * k;
  for (int i = 0; i < m; ++i) {
    out(i, i) = 0.0;
    for (int j = i + 1; j < m; ++j) {
      const double* a = p + (size_t)i * tot;
      const double* b = p + (size_t)j * tot;
      double s = 0.0;
      for (int t = 0; t < tot; ++t) s += std::fabs(a[t] - b[t]);
      out(i, j) = s; out(j, i) = s;
    }
  }
  return out;
}
