# Frozen original Section 7 generator functions; source function bodies unchanged.
norm_geo1 <- function(v) {
  v <- as.numeric(v)
  v <- pmax(v, .Machine$double.xmin)
  v / exp(mean(log(v)))
}

sample_gnedin_labels <- function(n, gamma = 0.5, check = TRUE) {
  stopifnot(n >= 1, is.finite(gamma), gamma > 0, gamma < 1)
  
  x <- integer(n)
  H <- 0L
  sizes <- integer(0)
  
  for (i in seq_len(n)) {
    if (H == 0L) {
      H <- 1L; sizes <- 1L; x[i] <- 1L
      next
    }
    
    n_curr <- i - 1L
    if (check) {
      stopifnot(length(sizes) == H, sum(sizes) == n_curr, all(sizes >= 1L))
    }
    
    existing_weights <- (sizes + 1) * (n_curr - H + gamma)
    new_weight <- H * (H - gamma)
    w <- c(existing_weights, new_weight)
    
    if (check) {
      total <- n_curr * (n_curr + gamma)
      if (!all(is.finite(w)) || any(w < 0)) stop("Non-finite or negative weights.")
      if (abs(sum(w) - total) > 1e-8 * max(1, total)) stop("Weight sum check failed.")
    }
    
    a <- sample.int(H + 1L, size = 1L, prob = w)
    if (a <= H) {
      sizes[a] <- sizes[a] + 1L
      x[i] <- a
    } else {
      H <- H + 1L
      sizes <- c(sizes, 1L)
      x[i] <- H
    }
  }
  
  list(x = x, K = H, sizes = sizes)
}

draw_gnedin_sizes_given_K <- function(n, K_target, gamma = 0.6, max_tries = 1e6) {
  for (t in seq_len(max_tries)) {
    z <- sample_gnedin_labels(n, gamma, check = FALSE)$x
    if (length(unique(z)) == K_target) {
      return(tabulate(z, nbins = K_target))
    }
  }
  stop("Failed to draw GN sizes given K. Increase max_tries or adjust gamma.")
}

# Assign most active players to smallest clusters (label 1 = smallest)
make_z_activity_to_smallest <- function(sizes, activity, seed = NULL, shuffle_within_cluster = TRUE) {
  if (!is.null(seed)) set.seed(seed)
  stopifnot(sum(sizes) == length(activity))
  
  K <- length(sizes)
  ord_sizes <- order(sizes, decreasing = FALSE)  # smallest first
  sizes_sorted <- sizes[ord_sizes]
  ord_players <- order(activity, decreasing = TRUE)
  
  z <- integer(length(activity))
  idx <- 1L
  for (c in seq_len(K)) {
    m <- sizes_sorted[c]
    players_c <- ord_players[idx:(idx + m - 1L)]
    if (shuffle_within_cluster && m > 1L) players_c <- sample(players_c)
    z[players_c] <- c
    idx <- idx + m
  }
  z
}

check_top_player_in_smallest_cluster <- function(z, activity) {
  sizes <- tabulate(z, nbins = length(unique(z)))
  min_size <- min(sizes)
  top_player <- which.max(activity)
  if (sizes[z[top_player]] != min_size) stop("Constraint violated: top-activity player not in a smallest cluster.")
  TRUE
}

# -----------------------------
# 5) DGP (strengths + outcomes)
# -----------------------------
make_lambda_from_padj <- function(K, p_adj = 0.85, base = 1) {
  stopifnot(K >= 2, p_adj > 0.5, p_adj < 1)
  delta <- qlogis(p_adj)
  base * exp(delta * (0:(K - 1)))
}

simulate_W_from_blocks <- function(N, z_true, lambda_by_label, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  n <- nrow(N)
  W <- matrix(0L, n, n)
  
  for (i in 1:(n - 1)) {
    for (j in (i + 1):n) {
      nij <- N[i, j]
      if (nij > 0) {
        li <- lambda_by_label[z_true[i]]
        lj <- lambda_by_label[z_true[j]]
        p_ij <- li / (li + lj)
        w_ij <- rbinom(1, size = nij, prob = p_ij)
        W[i, j] <- w_ij
        W[j, i] <- nij - w_ij
      }
    }
  }
  diag(W) <- 0L
  W
}

simulate_dataset_design_based <- function(N, K_target, gamma_true = 0.71, p_adj = 0.85, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  stopifnot(is.matrix(N), nrow(N) == ncol(N))
  
  n <- nrow(N)
  activity <- rowSums(N)
  
  sizes <- draw_gnedin_sizes_given_K(n, K_target = K_target, gamma = gamma_true)
  z_true <- make_z_activity_to_smallest(sizes, activity, seed = seed, shuffle_within_cluster = TRUE)
  check_top_player_in_smallest_cluster(z_true, activity)
  
  # IMPORTANT: label 1 is smallest cluster -> give it the BIGGEST lambda
  lambda_by_label <- rev(make_lambda_from_padj(K_target, p_adj = p_adj))
  
  W <- simulate_W_from_blocks(N, z_true, lambda_by_label, seed = seed)
  
  lambda_true_players <- norm_geo1(lambda_by_label[z_true])
  
  list(
    N = N, W = W,
    z_true = z_true,
    lambda_by_label = lambda_by_label,
    lambda_true_players = lambda_true_players,
    sizes = tabulate(z_true, nbins = K_target),
    activity = activity,
    gamma_true = gamma_true,
    p_adj = p_adj
  )
}
