## Vectorised kernels for the direg reproduction driver.
##
## WHY THIS FILE EXISTS
## --------------------
## The authors' package (code/R/*.R) evaluates, for every curve and every
## evaluation point, a nearest-neighbour lookup through
##   * pracma::interp2(..., method = "nearest")  (in theta_sheets()), and
##   * sapply(x, function(z) which.min(abs(z - grid)))  (in fbm_sheet()).
## Both are O(#curves x M0) R-level loops with `which.min`/`findInterval`
## inside, i.e. ~10^8 interpreted operations per Monte-Carlo replication for the
## largest design in the paper (N = 150 curves, M0 = 101x101 points).  The paper
## ran this on a cluster; here the whole study has to fit in one hour of wall
## time, so the identical computation is re-expressed with vectorised index
## arithmetic.
##
## The design grid is equally spaced, so the nearest grid point is obtained by
## rounding (q - grid[1]) / dgrid.  Ties (a query exactly halfway between two
## grid points) take the LOWER index, which is what `which.min` does for
## `abs(q - grid)`, and which is also what pracma::interp2() does for
## method = "nearest" (it compares xp <= (x[i] + x[i+1])/2).  Every function
## below was checked against the authors' implementation on the real
## configurations by code/sim/verify_equiv.R (max abs. difference < 1e-12).

## Nearest index of the query q in a sorted grid (1-based).
##
## Two variants are needed because the package resolves ties differently:
##   * which.min(abs(q - grid))  -> first minimiser in index order (lower index)
##   * pracma::interp2(..., "nearest") -> compares q <= (grid[i] + grid[i+1]) / 2
## The two agree except on floating-point near-ties, so each consumer gets the
## rule of the function it replaces.
nn_lower <- function(q, grid) {
  n <- length(grid)
  pmin(pmax(findInterval(q, grid), 1L), n - 1L)
}

## which.min(abs(q - grid)) on a sorted grid.
nn_index <- function(q, grid) {
  n <- length(grid)
  if (n == 1L) return(rep.int(1L, length(q)))
  i <- nn_lower(q, grid)
  dl <- abs(q - grid[i])
  du <- abs(grid[i + 1L] - q)
  as.integer(ifelse(dl <= du, i, i + 1L))
}

## pracma::interp2(x = grid, ..., method = "nearest") index for query q.
nn_index_interp2 <- function(q, grid) {
  n <- length(grid)
  if (n == 1L) return(rep.int(1L, length(q)))
  i <- nn_lower(q, grid)
  as.integer(ifelse(q <= (grid[i] + grid[i + 1L]) / 2, i, i + 1L))
}

## Same as direg::theta_sheets().
## pracma::interp2(x = t, y = t, Z = X, xp = tout[,2], yp = tout[,1], "nearest")
## returns X[i, j] with i = nearest index of tout[, 2] and j = nearest index of
## tout[, 1]; the final statistic is a mean over all grid points and curves, so
## the (column-major) order of the returned values is immaterial here.
theta_sheets_f <- function(X_list, tout, delta, e, sigma) {
  grid <- X_list[[1]]$t
  n <- length(grid)
  tout_minus <- cbind(pmin(pmax(tout[, 1] - (delta / 2 * e[1]), 0), 1),
                      pmin(pmax(tout[, 2] - (delta / 2 * e[2]), 0), 1))
  tout_plus <- cbind(pmax(pmin(tout[, 1] + (delta / 2 * e[1]), 1), 0),
                     pmax(pmin(tout[, 2] + (delta / 2 * e[2]), 1), 0))
  lin <- function(tt) {
    (nn_index_interp2(tt[, 2], grid) - 1L) * n + nn_index_interp2(tt[, 1], grid)
  }
  im <- lin(tout_minus)
  ip <- lin(tout_plus)
  s <- vapply(X_list, function(cur) mean((cur$X[im] - cur$X[ip])^2), numeric(1))
  mean(s) - 2 * sigma^2
}

## Same as direg::estimate_sigma().  Each grid point is paired with its nearest
## other point in the L1 (Manhattan) metric, taking the FIRST minimiser in index
## order, exactly as the authors' which.min() over the whole grid does.  Only the
## four axis neighbours can be minimisers; because the grid step is not exactly
## representable in binary, their distances are not exactly tied, so the four
## distances are evaluated rather than assumed.
estimate_sigma_f <- function(X_list) {
  grid <- X_list[[1]]$t
  n <- length(grid)
  m <- n * n
  g1 <- rep(grid, times = n)   # tout$t1 : varies fastest in expand.grid()
  g2 <- rep(grid, each = n)    # tout$t2
  pos <- seq_len(m)
  cand <- cbind(pos - n, pos - 1L, pos + 1L, pos + n)   # increasing index order
  cand[cand < 1L | cand > m] <- NA
  g1c <- matrix(g1[cand], m, 4L)
  g2c <- matrix(g2[cand], m, 4L)
  d <- abs(matrix(g1[pos], m, 4L) - g1c) + abs(matrix(g2[pos], m, 4L) - g2c)
  d[is.na(d)] <- Inf
  idx2 <- cand[cbind(pos, max.col(-d, ties.method = "first"))]
  S <- Reduce(`+`, lapply(X_list, function(cur) (cur$X[pos] - cur$X[idx2])^2))
  mean(sqrt(S / (2 * length(X_list))))
}

## Same as direg::H_sheets().
H_sheets_f <- function(X_list, tout, delta) {
  e <- list(e1 = c(1, 0), e2 = c(0, 1))
  sigma_hat <- estimate_sigma_f(X_list)
  t2 <- vapply(e, function(v) theta_sheets_f(X_list, tout, 2 * delta, v, sigma_hat), numeric(1))
  td <- vapply(e, function(v) theta_sheets_f(X_list, tout, delta, v, sigma_hat), numeric(1))
  t2 <- ifelse(t2 <= 0, 1, t2)
  td <- ifelse(td <= 0, 1, td)
  H_hat <- min((log(t2[["e1"]]) - log(td[["e1"]])) / (2 * log(2)),
               (log(t2[["e2"]]) - log(td[["e2"]])) / (2 * log(2)))
  list(H = pmin(pmax(H_hat, 0.1), 1),
       theta_e2 = td[["e2"]],
       theta_e1 = td[["e1"]],
       sigma_hat = sigma_hat)
}

## Same as direg::H_sheets_dir().
H_sheets_dir_f <- function(X_list, tout, delta, base_list, sigma) {
  t2 <- vapply(base_list, function(v) theta_sheets_f(X_list, tout, 2 * delta, v, sigma), numeric(1))
  td <- vapply(base_list, function(v) theta_sheets_f(X_list, tout, delta, v, sigma), numeric(1))
  t2 <- ifelse(t2 <= 0, 1, t2)
  td <- ifelse(td <= 0, 1, td)
  H_hat <- (log(t2) - log(td)) / (2 * log(2))
  pmin(pmax(H_hat, 0.1), 1)
}

## Same as direg::estimate_angle().
estimate_angle_f <- function(X_list, xout, delta) {
  tout <- expand.grid(xout, xout)
  H_list <- H_sheets_f(X_list = X_list, tout = tout, delta = delta)
  g_alpha <- (H_list$theta_e2 / H_list$theta_e1)^(1 / (2 * H_list$H))
  list(g_hat = g_alpha,
       alpha_acot = pracma::acot(g_alpha),
       alpha_atan = atan(g_alpha),
       H_min = H_list$H,
       sigma_hat = H_list$sigma_hat)
}

## Same as direg::identify_angle().
identify_angle_f <- function(angles, X_list, dout, xout, sigma) {
  v1_cot <- c(cos(angles$alpha_acot), sin(angles$alpha_acot))
  v1_tan <- c(cos(angles$alpha_atan), sin(angles$alpha_atan))
  v1_cot_ref <- c(cos(pi - angles$alpha_acot), sin(pi - angles$alpha_acot))
  v1_tan_ref <- c(cos(pi - angles$alpha_atan), sin(pi - angles$alpha_atan))
  tout <- expand.grid(t1 = xout, t2 = xout)
  H_v <- lapply(dout, function(d) {
    H_sheets_dir_f(X_list = X_list, tout = tout, delta = d,
                   base_list = list(v1_cot, v1_cot_ref, v1_tan, v1_tan_ref),
                   sigma = sigma)
  })
  H_sum <- Reduce(`+`, H_v)
  mode_idx <- which.max(H_sum)
  H_avg <- max(H_sum / length(H_v))
  if (mode_idx == 1) {
    alpha <- angles$alpha_acot; names(alpha) <- "alpha_acot"
  } else if (mode_idx == 2) {
    alpha <- pi - angles$alpha_acot; names(alpha) <- "alpha_acot_ref"
  } else if (mode_idx == 3) {
    alpha <- angles$alpha_atan; names(alpha) <- "alpha_atan"
  } else {
    alpha <- pi - angles$alpha_atan; names(alpha) <- "alpha_atan_ref"
  }
  list(alpha = alpha, H_max = H_avg)
}

## Same as direg::angle_correct().
angle_correct_f <- function(X_list, xout, g_hat, alpha_hat, delta, Hmax, Hmin, sigma) {
  tout <- expand.grid(xout, xout)
  base_list <- list(u1 = c(cos(alpha_hat), sin(alpha_hat)),
                    u2 = c(-sin(alpha_hat), cos(alpha_hat)))
  theta_u <- lapply(base_list, function(e) {
    mean(theta_sheets_f(X_list = X_list, tout = tout, delta = delta, e = e, sigma = sigma),
         na.rm = TRUE)
  })
  if (names(alpha_hat) == "alpha_acot" || names(alpha_hat) == "alpha_acot_ref") {
    f_alpha_num <- (abs(sin(alpha_hat))^(2 * Hmax) / abs(cos(alpha_hat))^(2 * Hmin)) *
      (theta_u$u1 / theta_u$u2)
    f_alpha_denom <- (abs(cos(alpha_hat))^(2 * Hmax) / abs(sin(alpha_hat))^(2 * Hmin)) *
      (theta_u$u1 / theta_u$u2)
    f_alpha <- ((1 + f_alpha_num) / (1 + f_alpha_denom))^(1 / (2 * Hmin))
  } else {
    f_alpha_num <- (abs(cos(alpha_hat))^(2 * Hmax) / abs(sin(alpha_hat))^(2 * Hmin)) *
      delta^(2 * (Hmax - Hmin))
    f_alpha_denom <- (abs(sin(alpha_hat))^(2 * Hmax) / abs(cos(alpha_hat))^(2 * Hmin)) *
      delta^(2 * (Hmax - Hmin))
    f_alpha <- ((1 + f_alpha_num) / (1 + f_alpha_denom))^(1 / 2 * Hmin)
  }
  g_adj <- g_hat / f_alpha
  if (names(alpha_hat) == "alpha_acot" || names(alpha_hat) == "alpha_acot_ref") {
    alpha_adj <- pracma::acot(g_adj)
  } else {
    alpha_adj <- atan(g_adj)
  }
  if (names(alpha_hat) == "alpha_acot_ref" || names(alpha_hat) == "alpha_atan_ref") {
    alpha_adj <- pi - alpha_adj
  }
  list(g_adj = as.double(g_adj), alpha_adj = alpha_adj, f_alpha = f_alpha)
}

## Same as direg::fbm_sheet(), with the nearest-neighbour lookups vectorised.
## The RNG consumption is identical to the authors' function (the two fbm_fft()
## calls in the same order, then the noise), so with the same seed the returned
## matrix is identical.
fbm_sheet_f <- function(t_n, e_n, alpha, H1, H2, type = "sum", sigma = NULL) {
  if (H1 <= H2) alpha <- alpha + pi/2
  while (alpha >= pi) alpha <- alpha - pi
  t1_tilde <- seq(0, 1, length.out = e_n)
  t2_tilde <- seq(0, 1, length.out = e_n)
  t1 <- seq(0, abs(cos(alpha)) + sin(alpha), length.out = t_n)
  if (alpha <= pi/2) {
    t2 <- seq(-sin(alpha), cos(alpha), length.out = t_n)
    B1 <- fbm_fft(H = H1, n = length(t1), grid_max = max(t1))
    B2_tilde <- fbm_fft(H = H2, n = length(t1), grid_max = max(t1))
    neg <- t2[t2 < 0]
    if (length(neg) == 0) {
      B2_minus <- c()
    } else {
      B2_minus <- -B2_tilde[nn_index(-neg, t1)]
    }
    B2_plus <- B2_tilde[seq_len(length(t1) - length(B2_minus))]
    B2 <- c(B2_minus, B2_plus)
  } else {
    t1_proj <- seq(cos(alpha), sin(alpha), length.out = t_n)
    t2 <- seq(cos(alpha) - sin(alpha), 0, length.out = t_n)
    B1_tilde <- fbm_fft(H = H1, n = length(t1), grid_max = max(t1))
    B1_minus <- -B1_tilde[nn_index(-t1_proj[t1_proj < 0], t1)]
    B1_plus <- B1_tilde[seq_len(length(t1) - length(B1_minus))]
    B1 <- c(B1_minus, B1_plus)
    B2_tilde <- fbm_fft(H = H2, n = length(t1), grid_max = max(t1))
    B2_minus <- -B2_tilde[nn_index(-t2[t2 < 0], t1)]
    B2_plus <- B2_tilde[seq_len(length(t2) - length(B2_minus))]
    B2 <- c(B2_minus, B2_plus)
  }
  ## NB: the authors obtain these coordinates with
  ##   apply(expand.grid(t2_tilde, t1_tilde), 1, function(x) crossprod(x, v))
  ## i.e. through BLAS, whose rounding is 1 ulp away from the elementwise
  ## expression `a*v1 + b*v2`.  On symmetric angles (alpha = pi/4) a large share
  ## of the queries sit exactly on a midpoint of the fBm grid, so that ulp
  ## decides the nearest neighbour.  A single BLAS matrix-vector product
  ## reproduces the authors' values bit for bit (checked in verify_equiv.R).
  M <- unname(as.matrix(expand.grid(t2_tilde = t2_tilde, t1_tilde = t1_tilde)))
  t1_u1 <- as.vector(M %*% c(cos(alpha), sin(alpha)))
  t2_u2 <- as.vector(M %*% c(-sin(alpha), cos(alpha)))
  i1 <- nn_index(t1_u1, t1)
  i2 <- nn_index(t2_u2, t2)
  X <- if (type == "sum") B1[i1] + B2[i2] else B1[i1] * B2[i2]
  if (missing(sigma) || is.null(sigma)) {
    matrix(X, nrow = length(t1_tilde), ncol = length(t2_tilde))
  } else {
    noise <- rnorm(n = length(X), mean = 0, sd = sigma)
    matrix(X + noise, nrow = length(t1_tilde), ncol = length(t2_tilde))
  }
}
