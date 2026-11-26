// ---------------------------------------------------------------------------
// Collapsed ("marginal") Gibbs sampler for a Dirichlet process mixture of
// bivariate normals -- Neal (2000), Algorithm 3 (state = allocation vector c).
//
// Added by the reproduction of Zito, Rigon & Dunson (2024), Section 5.1.
// The model, priors and MCMC scheme follow the paper:
//   y_i | theta_i ~ N_2(mu_i, Sigma_i),  theta_i ~ P,  P ~ DP(alpha, P0)
//   P0 = N(mu; 0, Sigma/kappa0) x IW(Sigma; nu0, I),  kappa0 = nu0 = 2
//   alpha ~ Sg(a, b, n)   or   alpha ~ Ga(2, 4)
// The conditional for c_i given c_{-i} is Eq. (3.7) of Neal (2000), with the
// normal-inverse-Wishart predictive density obtained analytically.
//
// The ratio-of-uniforms step for the Stirling-gamma full conditional
// alpha | k ~ Sg(a + k, b + 1, n) is a direct port of
//   code/src/rSg_posterior.cpp :: rSg_posterior_ratio_uniforms
// from the authors' package ConjugateDP (same acceptance test, same
// log_pdf_Sg_posterior, with (Mu, Mv) precomputed in R by the package's own
// ConjugateDP:::getBounds_MuMv_posterior).
// ---------------------------------------------------------------------------
#include <Rcpp.h>
#include <vector>
#include <cmath>
#include <algorithm>
using namespace Rcpp;

// log density of the Stirling-gamma posterior  (== ConjugateDP::log_pdf_Sg_posterior)
static inline double log_pdf_Sg_posterior(double x, double a, double b, double m,
                                          double k, double n) {
  return (a + k - 1.0) * std::log(x)
       - b * (std::lgamma(x + m) - std::lgamma(x))
       - (std::lgamma(x + n) - std::lgamma(x));
}

// log multivariate-t predictive (p = 2) of y under the NIW posterior of a
// cluster containing m observations with sum (s1,s2) and scatter (S11,S12,S22).
static inline double log_pred(double y1, double y2,
                              double m, double s1, double s2,
                              double S11, double S12, double S22,
                              double kappa0, double nu0,
                              const double* mu0, const double* Psi0) {
  double kappa_m = kappa0 + m;
  double nu_m    = nu0 + m;
  double ybar1 = (m > 0.0) ? s1 / m : 0.0;
  double ybar2 = (m > 0.0) ? s2 / m : 0.0;
  double mu_m1 = (kappa0 * mu0[0] + s1) / kappa_m;
  double mu_m2 = (kappa0 * mu0[1] + s2) / kappa_m;
  double c0 = (m > 0.0) ? kappa0 * m / kappa_m : 0.0;
  double d1 = ybar1 - mu0[0], d2 = ybar2 - mu0[1];
  double P11 = Psi0[0] + S11 + c0 * d1 * d1;
  double P12 = Psi0[1] + S12 + c0 * d1 * d2;
  double P22 = Psi0[3] + S22 + c0 * d2 * d2;
  double df = nu_m - 1.0;                      // nu_m - p + 1, p = 2
  double sc = (kappa_m + 1.0) / (kappa_m * df);
  double A11 = P11 * sc, A12 = P12 * sc, A22 = P22 * sc;
  double detA = A11 * A22 - A12 * A12;
  double e1 = y1 - mu_m1, e2 = y2 - mu_m2;
  double m2 = (A22 * e1 * e1 - 2.0 * A12 * e1 * e2 + A11 * e2 * e2) / detA;
  return std::lgamma((df + 2.0) / 2.0) - std::lgamma(df / 2.0)
       - std::log(df * M_PI) - 0.5 * std::log(detA)
       - ((df + 2.0) / 2.0) * std::log1p(m2 / df);
}

// [[Rcpp::export]]
List dpm_chain_cpp(NumericMatrix y, int niter, int burnin, int prior_type,
                   double a_sg, double b_sg, double m_sg,
                   NumericVector Mu, NumericVector Mv,
                   double kappa0, double nu0, NumericVector mu0, NumericMatrix Psi0,
                   double a_g, double b_g, double alpha_init,
                   int init_type = 1) {
  const int n = y.nrow();
  const double* Y = REAL(y);
  const double mu0v[2] = {mu0[0], mu0[1]};
  const double Psi0v[4] = {Psi0(0,0), Psi0(0,1), Psi0(1,0), Psi0(1,1)};

  std::vector<int> c(n, 0);
  std::vector<double> ncnt(n, 0.0), ssum(n * 2, 0.0), Sc(n * 3, 0.0);
  std::vector<int> active;
  active.reserve(64);

  // ---- initial allocation -------------------------------------------------
  // init_type 0: every observation in one cluster (the paper does not state
  //               its starting state; kept for reference)
  // init_type 1: a draw from the Dirichlet process prior with precision
  //               alpha_init (sequential Polya urn) -- neutral starting state
  if (init_type == 0) {
    for (int i = 0; i < n; i++) c[i] = 0;
  } else {
    int nact = 1; c[0] = 0;
    std::vector<int> cnt(n, 0); cnt[0] = 1;
    for (int i = 1; i < n; i++) {
      double tot = alpha_init, u = R::runif(0.0, 1.0) * (alpha_init + (double)i), acc = 0.0;
      int sel = -1;
      for (int j = 0; j < nact; j++) { acc += (double)cnt[j]; if (u <= acc) { sel = j; break; } }
      if (sel < 0) { sel = nact; cnt[nact] = 0; nact++; }
      c[i] = sel; cnt[sel]++;
    }
  }
  // ---- sufficient statistics of the initial allocation --------------------
  for (int i = 0; i < n; i++) {
    int j = c[i];
    if (ncnt[j] == 0.0) active.push_back(j);
    ncnt[j] += 1.0;
    ssum[j * 2] += Y[i]; ssum[j * 2 + 1] += Y[i + n];
  }
  for (size_t t = 0; t < active.size(); t++) {
    int j = active[t]; double m = ncnt[j];
    double m1 = ssum[j * 2] / m, m2 = ssum[j * 2 + 1] / m;
    double t11 = 0, t12 = 0, t22 = 0;
    for (int i = 0; i < n; i++) if (c[i] == j) {
      double e1 = Y[i] - m1, e2 = Y[i + n] - m2;
      t11 += e1 * e1; t12 += e1 * e2; t22 += e2 * e2;
    }
    Sc[j * 3] = t11; Sc[j * 3 + 1] = t12; Sc[j * 3 + 2] = t22;
  }

  double alpha = alpha_init;
  std::vector<double> alpha_trace(niter), k_trace(niter);
  std::vector<double> lw;
  lw.reserve(512);
  std::vector<int> chosen;
  chosen.reserve(512);
  const int Kmax = Mu.size() - 1;

  for (int it = 0; it < niter; it++) {
    for (int i = 0; i < n; i++) {
      double yi1 = Y[i], yi2 = Y[i + n];
      int old = c[i];
      double m_old = ncnt[old];
      // ---- remove observation i from its cluster --------------------------
      { double b1 = ssum[old * 2] / m_old, b2 = ssum[old * 2 + 1] / m_old;
        if (m_old > 1.0) {
          double f = m_old / (m_old - 1.0);
          double e1 = yi1 - b1, e2 = yi2 - b2;
          Sc[old * 3]     -= f * e1 * e1;
          Sc[old * 3 + 1] -= f * e1 * e2;
          Sc[old * 3 + 2] -= f * e2 * e2;
        } else {
          Sc[old * 3] = Sc[old * 3 + 1] = Sc[old * 3 + 2] = 0.0;
        }
        ssum[old * 2] -= yi1; ssum[old * 2 + 1] -= yi2;
        ncnt[old] = m_old - 1.0; }
      if (ncnt[old] == 0.0) {
        for (size_t j = 0; j < active.size(); j++)
          if (active[j] == old) { active[j] = active.back(); active.pop_back(); break; }
      }
      // ---- conditional (Neal 2000, Eq. 3.7) -------------------------------
      lw.clear(); chosen.clear();
      double mx = -R_PosInf;
      for (size_t j = 0; j < active.size(); j++) {
        int cl = active[j];
        double lp = std::log(ncnt[cl]) + log_pred(yi1, yi2, ncnt[cl],
                        ssum[cl*2], ssum[cl*2+1], Sc[cl*3], Sc[cl*3+1], Sc[cl*3+2],
                        kappa0, nu0, mu0v, Psi0v);
        lw.push_back(lp); chosen.push_back(cl);
        if (lp > mx) mx = lp;
      }
      double lp_new_log = std::log(alpha) + log_pred(yi1, yi2, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, kappa0, nu0, mu0v, Psi0v);
      lw.push_back(lp_new_log); chosen.push_back(-1);
      if (lp_new_log > mx) mx = lp_new_log;
      double tot = 0.0;
      for (size_t j = 0; j < lw.size(); j++) { lw[j] = std::exp(lw[j] - mx); tot += lw[j]; }
      double u = R::runif(0.0, 1.0) * tot, acc = 0.0;
      size_t sel = lw.size() - 1;
      for (size_t j = 0; j < lw.size(); j++) { acc += lw[j]; if (u <= acc) { sel = j; break; } }
      // ---- add observation i back -----------------------------------------
      int cl = chosen[sel];
      if (cl < 0) {                       // new cluster
        cl = -1;
        for (int j = 0; j < n; j++) if (ncnt[j] == 0.0) { cl = j; break; }
        ncnt[cl] = 0.0; ssum[cl*2] = 0.0; ssum[cl*2+1] = 0.0;
        Sc[cl*3] = Sc[cl*3+1] = Sc[cl*3+2] = 0.0;
        active.push_back(cl);
      } else {
        double m_cur = ncnt[cl];
        double b1 = ssum[cl*2] / m_cur, b2 = ssum[cl*2+1] / m_cur;
        double f = m_cur / (m_cur + 1.0);
        double e1 = yi1 - b1, e2 = yi2 - b2;
        Sc[cl*3]     += f * e1 * e1;
        Sc[cl*3+1]   += f * e1 * e2;
        Sc[cl*3+2]   += f * e2 * e2;
      }
      ssum[cl*2] += yi1; ssum[cl*2+1] += yi2; ncnt[cl] += 1.0;
      c[i] = cl;
    }

    double k = (double)active.size();
    // ---- update alpha given the current partition ------------------------
    if (prior_type == 2) {                       // alpha held fixed (test mode)
      alpha = alpha_init;
    } else if (prior_type == 1) {                       // alpha | k ~ Sg(a + k, b + 1, n)
      int kk = (int)k; if (kk > Kmax) kk = Kmax; if (kk < 1) kk = 1;
      double Mu_ = Mu[kk], Mv_ = Mv[kk];
      double s = std::exp(0.5 * (Mv_ - Mu_));
      double x = 0.0;
      while (true) {
        double u = R::runif(0.0, 1.0);
        double v = R::runif(0.0, 1.0);
        x = s * v / u;
        if (std::log(u) < 0.5 * (log_pdf_Sg_posterior(x, a_sg, b_sg, m_sg, k, (double)n) - Mu_))
          break;
      }
      alpha = x;
    } else {                                     // Escobar & West (1995) augmentation
      double eta  = R::rbeta(alpha + 1.0, (double)n);
      double rate = b_g - std::log(eta);
      double pi_w = (a_g + k - 1.0) / ((a_g + k - 1.0) + n * rate);
      if (R::runif(0.0, 1.0) < pi_w) alpha = R::rgamma(a_g + k, 1.0) / rate;
      else                           alpha = R::rgamma(a_g + k - 1.0, 1.0) / rate;
    }
    alpha_trace[it] = alpha;
    k_trace[it] = k;
  }

  IntegerVector c_out(n);
  for (int i = 0; i < n; i++) c_out[i] = c[i] + 1;
  return List::create(_["alpha"] = alpha_trace, _["k"] = k_trace,
                      _["c"] = c_out, _["n_active"] = (int)active.size());
}

// ---------------------------------------------------------------------------
// Stand-alone port of the authors' ratio-of-uniforms sampler, exposed so that
// the reproduction can check it against
// ConjugateDP:::rSg_posterior_ratio_uniforms (identical algorithm).
// ---------------------------------------------------------------------------
// [[Rcpp::export]]
NumericVector rSg_post_ru_cpp(int nsamples, double a, double b, double m,
                              double k, double n, double Mu, double Mv) {
  NumericVector out(nsamples);
  double s = std::exp(0.5 * (Mv - Mu));
  for (int i = 0; i < nsamples; i++) {
    while (true) {
      double u = R::runif(0.0, 1.0);
      double v = R::runif(0.0, 1.0);
      double x = s * v / u;
      if (std::log(u) < 0.5 * (log_pdf_Sg_posterior(x, a, b, m, k, n) - Mu)) {
        out[i] = x;
        break;
      }
    }
  }
  return out;
}
