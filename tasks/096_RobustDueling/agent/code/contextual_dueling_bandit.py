import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class NeuralApproximator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NeuralApproximator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

class DuelingGLMLearner:
    def __init__(self, d, e, lambda_reg=1.0, alpha=0.1, C=1.0, Lambda=1.0, mu_tau=0.0, kappa=0.1):
        self.d = d
        self.e = e
        self.dim = d + e
        self.lambda_reg = lambda_reg
        self.ucb_alpha = alpha # For arm selection confidence interval (c_t in paper)
        
        # Robustness parameters
        self.C = C
        self.Lambda = Lambda
        self.mu_tau = mu_tau
        self.kappa = kappa # Conservative lower bound for derivative
        
        # Alpha from Theorem (Weighting parameter): alpha = sqrt(d) / (C + D)
        # with d = d_x + d_y (= self.dim) and robustness scale D = max(sqrt(Lambda), mu*tau),
        # matching the paper. (No extra 1/sqrt(kappa) factor.)
        D = max(np.sqrt(Lambda), mu_tau) if mu_tau > 0 else np.sqrt(Lambda)
        denom = C + D
        self.weight_alpha = np.sqrt(self.dim) / (denom + 1e-6)
        
        # V: Full history (for weighting)
        self.V = self.lambda_reg * np.eye(self.dim)
        self.V_inv = (1.0 / self.lambda_reg) * np.eye(self.dim)
        
        # W: Observed history (for estimation/confidence)
        self.W = self.lambda_reg * np.eye(self.dim)
        self.W_inv = (1.0 / self.lambda_reg) * np.eye(self.dim)
        
        self.theta_hat = np.zeros(self.dim)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.f_t = NeuralApproximator(d, e).to(self.device)
        self.optimizer_f = optim.Adam(self.f_t.parameters(), lr=1e-3)
        self.loss_fn_f = nn.MSELoss()
        
        self.data_buffer_x = []
        self.data_buffer_y = []

        # Arrived-outcome history for the weighted-MLE estimator (paper eq. (Theta_t)).
        # Each entry is the observed complete-feature (delta_z, outcome, omega)
        # of an arrived sample.
        self.hist_dz = []
        self.hist_o = []
        self.hist_w = []

        # RCDP-UCB uses predicted post-serving features only for arm selection.
        # The robust weight and full-history geometry V are updated after the
        # selected post-serving contexts are observed, matching Algorithm 1.
        self.use_observed_feature_weighting = True

    def _solve_mle(self, max_iter=50, tol=1e-7):
        """Theta_t = argmin_Theta (lambda/2)||Theta||^2
                       - sum_{s in H_t} omega_s log g((-1)^{1-o_s} <Theta, dz_s>),
        i.e. the weighted regularized MLE / estimating equation of the paper.
        Solved by warm-started Newton (IRLS) over the full arrived history."""
        if not self.hist_dz:
            return
        Z = np.asarray(self.hist_dz)          # (n, dim)
        o = np.asarray(self.hist_o)           # (n,)
        w = np.asarray(self.hist_w)           # (n,)
        lam = self.lambda_reg
        theta = self.theta_hat.copy()         # warm start
        eye = np.eye(self.dim)
        for _ in range(max_iter):
            p = 1.0 / (1.0 + np.exp(-(Z @ theta)))      # g(<theta, dz>)
            grad = lam * theta + Z.T @ (w * (p - o))    # estimating equation
            if np.linalg.norm(grad) < tol:
                break
            s = w * p * (1.0 - p)
            H = lam * eye + (Z * s[:, None]).T @ Z       # weighted Hessian
            try:
                step = np.linalg.solve(H, grad)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(H, grad, rcond=None)[0]
            theta = theta - step
        self.theta_hat = theta

    def train_approximator(self, epochs=5):
        if self.e == 0:
            return # No post-serving context to learn
            
        if not self.data_buffer_x:
            return
        
        X_tensor = torch.FloatTensor(np.array(self.data_buffer_x)).to(self.device)
        Y_tensor = torch.FloatTensor(np.array(self.data_buffer_y)).to(self.device)
        
        self.f_t.train()
        for _ in range(epochs):
            self.optimizer_f.zero_grad()
            outputs = self.f_t(X_tensor)
            loss = self.loss_fn_f(outputs, Y_tensor)
            loss.backward()
            self.optimizer_f.step()
            
    def predict_y(self, X):
        if self.e == 0:
            return np.zeros((len(X), 0))
            
        self.f_t.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            Y_pred = self.f_t(X_tensor).cpu().numpy()
        return Y_pred

    def get_features(self, X):
        Y_pred = self.predict_y(X)
        return np.concatenate([X, Y_pred], axis=1)

    def get_observed_delta(self, X, a_idx, b_idx, y_a_obs, y_b_obs):
        if self.e == 0:
            return X[a_idx] - X[b_idx]
        z_a = np.concatenate([X[a_idx], y_a_obs])
        z_b = np.concatenate([X[b_idx], y_b_obs])
        return z_a - z_b

    def record_observed_features(self, X, a_idx, b_idx, y_a_obs, y_b_obs):
        """
        Calculate omega_t and update V_t after observing the selected
        post-serving contexts. This is the paper's Step 3:
        omega_t = min(1, alpha / ||Delta z_t||_{V_{t-1}^{-1}}),
        V_t = V_{t-1} + kappa * omega_t * Delta z_t Delta z_t^T.
        """
        delta_z = self.get_observed_delta(X, a_idx, b_idx, y_a_obs, y_b_obs)
        norm_V = np.sqrt(delta_z @ self.V_inv @ delta_z)

        if norm_V < 1e-9:
            omega_t = 1.0
        else:
            omega_t = min(1.0, self.weight_alpha / norm_V)

        outer = np.outer(delta_z, delta_z)
        self.V += self.kappa * omega_t * outer

        v_vec = np.sqrt(self.kappa * omega_t) * delta_z
        Av = self.V_inv @ v_vec
        denom = 1 + np.dot(v_vec, Av)
        self.V_inv -= np.outer(Av, Av) / denom

        self.last_delta_z = delta_z
        self.last_omega_t = omega_t
        return delta_z, omega_t
    
    def select_arms(self, X, env=None):
        """
        PonLinRUCB Strategy with Adaptive Weighting.
        Returns: a_t, b_t, omega_t
        """
        Z = self.get_features(X) 
        
        # 1. Champion a_t (Greedy)
        utilities = Z @ self.theta_hat
        a_t = np.argmax(utilities)
        z_a = Z[a_t]
        
        # 2. Challenger b_t using Full History V
        Delta_Z_a = Z - z_a 
        mean_diff = Delta_Z_a @ self.theta_hat
        width_V = np.sqrt(np.sum((Delta_Z_a @ self.V_inv) * Delta_Z_a, axis=1))
        
        scores_b = mean_diff + self.ucb_alpha * width_V
        b_t = np.argmax(scores_b)
        
        # omega_t and V_t are updated only after post-serving contexts are
        # observed in run_simulation().
        return a_t, b_t, None

    def observe_context(self, X, a_idx, b_idx, y_a_obs, y_b_obs):
        """
        Post-serving context is revealed IMMEDIATELY after the pair is served
        (only the preference outcome is delayed/corrupted). So the mapping
        approximator f_t is trained on (x, y) right away, independent of the
        feedback delay queue.
        """
        if self.e == 0:
            return
        self.data_buffer_x.append(X[a_idx])
        self.data_buffer_y.append(y_a_obs)
        self.data_buffer_x.append(X[b_idx])
        self.data_buffer_y.append(y_b_obs)
        self.train_approximator(epochs=2)

    def update(self, X, a_idx, b_idx, outcome, y_a_obs, y_b_obs, omega_t):
        # Only the preference-outcome estimation (W, theta) is gated by feedback
        # delay; the approximator was already trained in observe_context.
        # Use the queued delta_z when available so the estimate stays consistent
        # with the V update performed after post-serving context observation.
        delta_z = getattr(self, 'current_delta_z', None)
        if delta_z is None:
            z = self.get_features(X)
            delta_z = z[a_idx] - z[b_idx]

        # Weighted update for Observed History W (paper W̃_t, eq. (widetildeW));
        # kept for subclasses whose confidence width is built from W_inv.
        v_vec = np.sqrt(self.kappa * omega_t) * delta_z
        self.W += self.kappa * omega_t * np.outer(delta_z, delta_z)

        Aw = self.W_inv @ v_vec
        denom_w = 1 + np.dot(v_vec, Aw)
        self.W_inv -= np.outer(Aw, Aw) / denom_w

        # Theta_t: re-solve the weighted regularized MLE over ALL arrived
        # outcomes (paper eq. (Theta_t)/estimating equation), not a single
        # online step. delta_z/outcome/omega are the selection-time values.
        self.hist_dz.append(delta_z)
        self.hist_o.append(outcome)
        self.hist_w.append(omega_t)
        self._solve_mle()

class BaselineDuelingGLMLearner:
    """
    Baseline 1: X-only (Standard Dueling Contextual Bandit)
    Uses only pre-serving context X.
    """
    def __init__(self, d, lambda_reg=1.0, alpha=0.1):
        self.d = d
        self.dim = d 
        self.lambda_reg = lambda_reg
        self.alpha = alpha 
        
        self.theta_hat = np.zeros(self.dim)
        # Using W (Observed History) for Baselines as requested
        self.W = self.lambda_reg * np.eye(self.dim)
        self.W_inv = (1.0 / self.lambda_reg) * np.eye(self.dim)
        
    def select_arms(self, X, env=None):
        Z = X 
        utilities = Z @ self.theta_hat
        a_t = np.argmax(utilities)
        z_a = Z[a_t]
        
        Delta_Z = Z - z_a 
        mean_diff = Delta_Z @ self.theta_hat
        weighted_norm = np.sqrt(np.sum((Delta_Z @ self.W_inv) * Delta_Z, axis=1))
        
        ucb_scores = mean_diff + self.alpha * weighted_norm
        b_t = np.argmax(ucb_scores)

        self.last_delta_z = Z[a_t] - Z[b_t]
        # Return 1.0 for omega_t (not used)
        return a_t, b_t, 1.0

    def observe_context(self, X, a_idx, b_idx, y_a_obs, y_b_obs):
        # X-only baseline: no post-serving context / approximator to train.
        return

    def update(self, X, a_idx, b_idx, outcome, y_a_obs, y_b_obs, omega_t=1.0):
        # Ignore y_a_obs, y_b_obs, omega_t
        Z = X
        delta_z = Z[a_idx] - Z[b_idx]
        
        mu_val = 1 / (1 + np.exp(- np.dot(self.theta_hat, delta_z)))
        
        outer = np.outer(delta_z, delta_z)
        self.W += outer
        
        Aw = self.W_inv @ delta_z
        denom = 1 + np.dot(delta_z, Aw)
        self.W_inv -= np.outer(Aw, Aw) / denom
        
        step = self.W_inv @ (delta_z * (outcome - mu_val))
        self.theta_hat += step

# --- Advanced Robust Baselines ---

def joint_pair_ucb_select(Z, theta_hat, M_inv, beta):
    """Joint pair-UCB selection.

    Used by RCDB (Di et al. 2025, Algorithm 1, line 6) and by the single-layer
    MaxPairUCB reduction of VACDB (Di et al. 2024, Algorithm 1, line 8):
        (a_t, b_t) = argmax_{i,j} <Z_i + Z_j, theta> + beta * ||Z_i - Z_j||_{M^{-1}}.
    The pair is chosen jointly (not greedy-leader-then-challenger). M^{-1} is the
    design matrix over played pair-differences. O(K^2), vectorized via the Gram
    matrix identity ||Z_i - Z_j||^2 = G_ii + G_jj - 2 G_ij with G = Z M^{-1} Z^T.
    """
    util = Z @ theta_hat
    G = Z @ (M_inv @ Z.T)
    diagG = np.diag(G)
    d2 = diagG[:, None] + diagG[None, :] - 2.0 * G
    width = np.sqrt(np.maximum(d2, 0.0))
    score = util[:, None] + util[None, :] + beta * width
    i_t, j_t = np.unravel_index(int(np.argmax(score)), score.shape)
    return int(i_t), int(j_t)


def maxinp_select(Z, theta_hat, V_inv, alpha):
    """Maximum-Informative-Pair selection (Saha 2021), regret-minimizing variant.

    Leader  i_t = argmax_k <Z_k, theta>.
    Challenger j_t = the most informative arm relative to i_t,
        argmax_{k in C} ||Z_k - Z_{i_t}||_{V^{-1}},  among the plausibly-optimal
        set C = { k != i_t : <Z_k, theta> + alpha*||Z_k||_{V^{-1}} >= <Z_{i_t}, theta> }.

    Restricting the most-informative search to plausible winners keeps the regret
    sublinear; the unrestricted "most informative pair over all arms" rule
    over-explores irrelevant arms and is linear in this contextual setting.
    """
    util = Z @ theta_hat
    i_t = int(np.argmax(util))
    self_w = np.sqrt(np.maximum(np.sum((Z @ V_inv) * Z, axis=1), 0.0))
    ucb = util + alpha * self_w
    cand = np.where(ucb >= util[i_t])[0]
    cand = cand[cand != i_t]
    if len(cand) == 0:                       # very confident: verify the next-best UCB arm
        ucb2 = ucb.copy(); ucb2[i_t] = -np.inf
        return i_t, int(np.argmax(ucb2))
    diff = Z[cand] - Z[i_t]
    unc = np.sqrt(np.maximum(np.sum((diff @ V_inv) * diff, axis=1), 0.0))
    return i_t, int(cand[int(np.argmax(unc))])


class RCDBLearner(BaselineDuelingGLMLearner):
    """
    RCDB (Robust Contextual Dueling Bandits)
    From "Nearly Optimal Algorithms for Contextual Dueling Bandits from Adversarial Feedback".
    
    Features:
    1. Weighted ONS/MLE update based on uncertainty: w_t = min(1, alpha / ||z_{i,j}||_{V^{-1}})
    2. Exploration Bonus: beta * ||z_{i,j}||_{V^{-1}}
    
    We adapt a simplified version compatible with the current GLM structure:
    - Update uses weights.
    - Selection uses UCB with beta scaling.
    """
    def __init__(self, d, lambda_reg=1.0, alpha=0.1, rcdb_alpha=1.0, rcdb_beta=1.0, kappa=0.1):
        super().__init__(d, lambda_reg, alpha)
        self.rcdb_alpha = rcdb_alpha
        self.rcdb_beta = rcdb_beta
        self.kappa = kappa
        
    def select_arms(self, X, env=None):
        # Faithful RCDB joint pair-UCB (Di et al. 2025, Alg. 1 line 6).
        i_t, j_t = joint_pair_ucb_select(X, self.theta_hat, self.W_inv, self.rcdb_beta)
        self.last_delta_z = X[i_t] - X[j_t]
        return i_t, j_t, 1.0

    def update(self, X, a_idx, b_idx, outcome, y_a_obs, y_b_obs, omega_t=1.0):
        # Weight Calculation
        Z = X
        delta_z = Z[a_idx] - Z[b_idx]
        norm_val = np.sqrt(delta_z @ self.W_inv @ delta_z)

        # w_t = min(1, alpha / norm)   (RCDB Eq. 4.3)
        if norm_val < 1e-9:
            w_t = 1.0
        else:
            w_t = min(1.0, self.rcdb_alpha / norm_val)

        # Weighted online MLE step.
        mu_val = 1 / (1 + np.exp(- np.dot(self.theta_hat, delta_z)))

        # Sigma_t = lambda I + sum_i w_i (phi_a - phi_b)(phi_a - phi_b)^T
        # (RCDB's design matrix has NO kappa scaling; kappa enters only via alpha/beta).
        w_delta_z = np.sqrt(w_t) * delta_z

        outer = np.outer(w_delta_z, w_delta_z)
        self.W += outer

        Av = self.W_inv @ w_delta_z
        denom = 1 + np.dot(w_delta_z, Av)
        self.W_inv -= np.outer(Av, Av) / denom

        # Gradient Step weighted
        step = self.W_inv @ (delta_z * w_t * (outcome - mu_val))
        self.theta_hat += step


class MaxInPLearner(BaselineDuelingGLMLearner):
    """
    MaxInP (Maximum-Informative-Pair, Saha 2021, Algorithm 1).
    Faithful selection: pick the most uncertain pair ||Z_i - Z_j||_{V^{-1}} among
    the "promising" pairs whose optimistic relative score is positive
    (<Z_i - Z_j, theta> + eta*||Z_i - Z_j||_{V^{-1}} > 0). The exploration
    constant eta maps to self.alpha. (Saha's plain MLE + initial exploration
    phase is replaced by the regularized estimator inherited from the base class.)
    """
    def select_arms(self, X, env=None):
        i_t, j_t = maxinp_select(X, self.theta_hat, self.W_inv, self.alpha)
        self.last_delta_z = X[i_t] - X[j_t]
        return i_t, j_t, 1.0


class MaxPairUCBLearner(BaselineDuelingGLMLearner):
    """
    MaxPairUCB (single-layer pair-UCB reduction of VACDB, Di et al. 2024).
    Faithful selection: jointly maximize the pair-UCB
        <Z_i + Z_j, theta> + alpha * ||Z_i - Z_j||_{W^{-1}}.
    Update is the standard (unweighted) online MLE step (no variance weighting /
    multi-layer elimination of the full VACDB algorithm).
    """
    def select_arms(self, X, env=None):
        i_t, j_t = joint_pair_ucb_select(X, self.theta_hat, self.W_inv, self.alpha)
        self.last_delta_z = X[i_t] - X[j_t]
        return i_t, j_t, 1.0


def colstim_select_pair(Z, theta_hat, W_inv, c1, t,
                        perturb='gumbel', c_thresh=5.0, tau=0, coupling_const=None):
    """Faithful COLSTIM arm-pair selection (Bengs et al. 2022, Algorithm 1).

    M^{-1} = W_inv is the design matrix over the played pair-differences
    z_{i_s,j_s} (M_{t+1} = M_t + z z^T, line 17).

    First arm  (line 14, perturbed leader / follow-the-perturbed-leader):
        i_t = argmax_i  <x_i, theta> + eps_i * ||x_i||_{M^{-1}},
    where the eps_i are perturbations from the comparison-imitating
    distribution G, thresholded to [-c_thresh, c_thresh] (line 13). We default
    to Gumbel noise, whose argmax reproduces the Bradley-Terry-Luce (logistic)
    choice used as the link g here; pass perturb='gaussian' to imitate the
    Thurstone-Mosteller model instead.

    Second arm (line 15, optimistic toughest competitor of i_t):
        j_t = argmax_i  <x_i - x_{i_t}, theta> + c1 * ||x_i - x_{i_t}||_{M^{-1}}.

    Coupling (lines 7-12): with probability p_t the per-arm perturbations are
    independent (exploration), otherwise a single shared perturbation is used.
    Following Thm. 3.2, p_t decays as ~1/sqrt(t - tau); coupling_const=None
    keeps p_t = 1 (always-independent perturbations).
    """
    n = Z.shape[0]
    util = Z @ theta_hat
    widths = np.sqrt(np.maximum(np.sum((Z @ W_inv) * Z, axis=1), 0.0))  # ||x_i||_{M^{-1}}

    # Coupling: B_t ~ Ber(p_t).
    if coupling_const is None:
        p_t = 1.0
    else:
        p_t = min(1.0, coupling_const / np.sqrt(max(1, t - tau)))
    independent = (np.random.rand() < p_t)

    # Perturbations from G (Gumbel imitates BTL; Gaussian imitates Thurstone-Mosteller).
    if perturb == 'gaussian':
        draw = lambda size: np.random.randn(size)
    else:
        draw = lambda size: np.random.gumbel(0.0, 1.0, size=size)
    eps = draw(n) if independent else np.full(n, draw(1)[0])
    eps = np.clip(eps, -c_thresh, c_thresh)

    # First arm: perturbed leader.
    i_t = int(np.argmax(util + eps * widths))

    # Second arm: optimistic challenger relative to i_t (exclude i_t itself).
    diff = Z - Z[i_t]                                   # x_i - x_{i_t}
    mean_diff = diff @ theta_hat
    w_diff = np.sqrt(np.maximum(np.sum((diff @ W_inv) * diff, axis=1), 0.0))
    chall = mean_diff + c1 * w_diff
    chall[i_t] = -np.inf
    j_t = int(np.argmax(chall))
    return i_t, j_t


class ColSTIMLearner(BaselineDuelingGLMLearner):
    """
    COLSTIM (Contextual Linear Stochastic Transitivity Imitator, Bengs et al. 2022).
    Faithful Algorithm-1 selection: perturbed-leader first arm (Gumbel noise
    imitating the BTL link) + optimistic-challenger second arm. The parameter
    estimate uses the online weighted-MLE step (explicitly sanctioned in the
    paper, Sec. 3.2.3), inherited from BaselineDuelingGLMLearner.update.
    """
    def __init__(self, d, lambda_reg=1.0, alpha=0.1, perturb='gumbel',
                 c_thresh=5.0, coupling_const=None, tau=0):
        super().__init__(d, lambda_reg, alpha)
        self.perturb = perturb
        self.c_thresh = c_thresh
        self.coupling_const = coupling_const
        self.tau = tau
        self._t = 0

    def select_arms(self, X, env=None):
        self._t += 1
        i_t, j_t = colstim_select_pair(
            X, self.theta_hat, self.W_inv, c1=self.alpha, t=self._t,
            perturb=self.perturb, c_thresh=self.c_thresh,
            tau=self.tau, coupling_const=self.coupling_const)
        self.last_delta_z = X[i_t] - X[j_t]
        return i_t, j_t, 1.0

# --- Post-serving baselines (X + learned Y predictor) -----------------------
# Each baseline keeps its own faithful selection rule but runs it on the
# complete feature Z = [X, f_t(X)], where f_t is the shared neural predictor
# inherited from DuelingGLMLearner. Parameter estimation is the online step.

class RCDBPostServingLearner(DuelingGLMLearner):
    """RCDB (Di et al. 2025) with the learned post-serving predictor."""
    def __init__(self, d, e, lambda_reg=1.0, alpha=0.1, rcdb_alpha=1.0, rcdb_beta=1.0, kappa=0.1):
        super().__init__(d, e, lambda_reg, alpha, C=0.0, Lambda=0.0, mu_tau=0.0, kappa=kappa)
        self.use_observed_feature_weighting = False
        self.rcdb_alpha = rcdb_alpha
        self.rcdb_beta = rcdb_beta

    def select_arms(self, X, env=None):
        Z = self.get_features(X)
        i_t, j_t = joint_pair_ucb_select(Z, self.theta_hat, self.W_inv, self.rcdb_beta)
        self.last_delta_z = Z[i_t] - Z[j_t]
        return i_t, j_t, 1.0

    def update(self, X, a_idx, b_idx, outcome, y_a_obs, y_b_obs, omega_t=1.0):
        delta_z = getattr(self, 'current_delta_z', None)
        if delta_z is None:
            Z = self.get_features(X)
            delta_z = Z[a_idx] - Z[b_idx]
        norm_val = np.sqrt(delta_z @ self.W_inv @ delta_z)
        w_t = 1.0 if norm_val < 1e-9 else min(1.0, self.rcdb_alpha / norm_val)
        self.W += np.outer(delta_z, delta_z) * w_t
        Aw = self.W_inv @ delta_z
        denom = 1 + np.dot(delta_z, Aw) * w_t
        self.W_inv -= (np.outer(Aw, Aw) * w_t) / denom
        mu_val = 1 / (1 + np.exp(- np.dot(self.theta_hat, delta_z)))
        self.theta_hat += self.W_inv @ (delta_z * (outcome - mu_val)) * w_t


class _UnweightedPSUpdate(DuelingGLMLearner):
    """Shared unweighted online-MLE update on Z = [X, f(X)] for the
    information-/UCB-based post-serving baselines (ColSTIM, MaxInP, MaxPair)."""
    def update(self, X, a_idx, b_idx, outcome, y_a_obs, y_b_obs, omega_t=1.0):
        delta_z = getattr(self, 'current_delta_z', None)
        if delta_z is None:
            Z = self.get_features(X)
            delta_z = Z[a_idx] - Z[b_idx]
        self.W += np.outer(delta_z, delta_z) * omega_t
        Aw = self.W_inv @ delta_z
        denom = 1 + np.dot(delta_z, Aw) * omega_t
        self.W_inv -= (np.outer(Aw, Aw) * omega_t) / denom
        mu_val = 1 / (1 + np.exp(- np.dot(self.theta_hat, delta_z)))
        self.theta_hat += self.W_inv @ (delta_z * (outcome - mu_val)) * omega_t


class ColSTIMPostServingLearner(_UnweightedPSUpdate):
    """COLSTIM (Bengs et al. 2022) with the learned post-serving predictor."""
    def __init__(self, d, e, lambda_reg=1.0, alpha=0.1, C=0.0, Lambda=0.0, mu_tau=0.0, kappa=0.1,
                 perturb='gumbel', c_thresh=5.0, coupling_const=None, tau=0):
        super().__init__(d, e, lambda_reg, alpha, C=C, Lambda=Lambda, mu_tau=mu_tau, kappa=kappa)
        self.use_observed_feature_weighting = False
        self.perturb = perturb
        self.c_thresh = c_thresh
        self.coupling_const = coupling_const
        self.tau = tau
        self._t = 0

    def select_arms(self, X, env=None):
        self._t += 1
        Z = self.get_features(X)
        i_t, j_t = colstim_select_pair(Z, self.theta_hat, self.W_inv, c1=self.ucb_alpha, t=self._t,
                                       perturb=self.perturb, c_thresh=self.c_thresh,
                                       tau=self.tau, coupling_const=self.coupling_const)
        self.last_delta_z = Z[i_t] - Z[j_t]
        return i_t, j_t, 1.0


class MaxPairUCBPostServingLearner(_UnweightedPSUpdate):
    """MaxPairUCB (Di et al. 2024) with the learned post-serving predictor."""
    def __init__(self, d, e, lambda_reg=1.0, alpha=0.1, rcdb_alpha=1.0, rcdb_beta=1.0, kappa=0.1):
        super().__init__(d, e, lambda_reg, alpha, C=0.0, Lambda=0.0, mu_tau=0.0, kappa=kappa)
        self.use_observed_feature_weighting = False
        self.rcdb_beta = rcdb_beta

    def select_arms(self, X, env=None):
        Z = self.get_features(X)
        i_t, j_t = joint_pair_ucb_select(Z, self.theta_hat, self.W_inv, self.rcdb_beta)
        self.last_delta_z = Z[i_t] - Z[j_t]
        return i_t, j_t, 1.0


class MaxInPPostServingLearner(_UnweightedPSUpdate):
    """MaxInP (Saha 2021) with the learned post-serving predictor."""
    def __init__(self, d, e, lambda_reg=1.0, alpha=0.1, rcdb_alpha=1.0, rcdb_beta=1.0, kappa=0.1):
        super().__init__(d, e, lambda_reg, alpha, C=0.0, Lambda=0.0, mu_tau=0.0, kappa=kappa)
        self.use_observed_feature_weighting = False
        self.rcdb_beta = rcdb_beta

    def select_arms(self, X, env=None):
        Z = self.get_features(X)
        i_t, j_t = maxinp_select(Z, self.theta_hat, self.W_inv, self.rcdb_beta)
        self.last_delta_z = Z[i_t] - Z[j_t]
        return i_t, j_t, 1.0
