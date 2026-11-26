import numpy as np
import pandas as pd

from scipy.stats import binom


class Weights:
    """Base class for computing weighted objectives and constraints with probability bounds."""

    def __init__(self, Z_0, Z_1, alpha):
        self.Z_0 = Z_0
        self.Z_1 = Z_1
        self.alpha = alpha
        self.delta = 0.1

    def get_obj_and_constr(self, a_policy):
        Z = np.where(a_policy == 0, self.Z_0, self.Z_1)
        obj = np.mean(Z)

        prop = np.mean(a_policy)
        if prop == 0:
            return obj, 1, len(Z)

        Z = np.where(a_policy == 0, 0, Z) / prop
        constr = np.mean(Z)

        if np.mean(Z) == 0:
            constr = 1

        return obj, constr, len(Z)

    def get_hoeffding(self, a_policy, prop = None, max_Z = None):
        if prop is None:
            prop = np.mean(a_policy)
        if prop == 0:
            return 1
        Z = np.where(a_policy == 0, 0, self.Z_1) / prop

        if max_Z is None:
            max_Z = np.max(Z)
        if max_Z == 0:
            return 1
        Z_standardized = Z / max_Z
        R_hat = np.mean(Z_standardized)
        hoeffding = self.upper_confidence_bound(R_hat, len(Z))

        return hoeffding * max_Z

    def h1(self, t, R):
        """Compute h_1(t; R) as defined in the inequality."""
        if R == 0 or R == 1 or t == 0 or t == 1:
            return np.inf  # Avoid log(0)
        return t * np.log(t / R) + (1 - t) * np.log((1 - t) / (1 - R))

    def hoeffding_bound(self, n, R_lambda, t):
        """Compute the Bentkus inequality bound correctly."""
        return np.exp(-n * self.h1(t, R_lambda))

    def bentkus_bound(self, n, R_lambda, t):
        binom_prob = binom.cdf(int(n * t), n, R_lambda)
        return np.exp(1) * binom_prob

    def lower_tail_prob(self, n, R_lambda, t):
        """Compute the lower tail probability bound g^HB."""
        return min(self.hoeffding_bound(n, R_lambda, t), self.bentkus_bound(n, R_lambda, t))

    def upper_confidence_bound(self, R_hat_lambda, n):
        """Compute the (1 - delta) upper confidence bound \hat{R}^+_{HB}(\lambda)."""
        R = 0.0
        in_interval = False
        bound = self.lower_tail_prob(n, R, R_hat_lambda)
        while (bound >= self.delta or not in_interval) and R < 1:
            R += 0.001  # Increment step size
            if bound >= self.delta:
                in_interval = True
            bound = self.lower_tail_prob(n, R, R_hat_lambda)
        return R - 0.001

    def get_conformal_weight(self, Z, Z_max):
        n = len(Z)
        Z = np.sort(Z)
        ind = int(np.ceil((1 - self.alpha)*(n+1)))-1
        if ind >= n:
            return Z_max
        return Z[ind]


class WeightsDecision(Weights):
    """Weights for observational data using sensitivity model with parameter gamma."""

    def __init__(self, df, x_names, gamma, get_p_a_x, alpha):

        self.df = df.copy().reset_index(drop=True)
        self.x_names = x_names
        assert "y" in df.columns, "No 'y' column in df"
        assert "a" in df.columns, "No 'a' column in df"
        for x in x_names:
            assert x in df.columns, "No {} column in df".format(x)

        self.gamma = gamma

        df = self.add_weights(get_p_a_x, gamma)

        super().__init__(df['Z_0'], df['Z_1'], alpha)

    def add_weights(self, get_p_a_x, gamma):
        def get_weight(p_a_x, a, gamma_i):
            weight = 1 + gamma_i * (1 / p_a_x - 1)
            return np.where(self.df['a'] == a, weight, 0)

        def get_weights(a):
            p_a_x = get_p_a_x(self.df[self.x_names], a)

            weight_low = get_weight(p_a_x, a, 1 / gamma)
            weight_high = get_weight(p_a_x, a, gamma)

            weight = np.where(self.df['y'] == 1, weight_high, weight_low)
            return weight
        df = pd.DataFrame()
        for a in [0, 1]:
            df[f'weight_{a}'] = get_weights(a)
            df[f'Z_{a}'] = df[f'weight_{a}'] * self.df['y']

        return df


class WeightsSampling(Weights):
    """Weights for data with selection bias, combining treatment and sampling sensitivity."""

    def __init__(self, df, x_names, gamma, get_p_s1, get_p_a_x, get_p_s_x, alpha):

        self.df = df.copy().reset_index(drop=True)
        self.x_names = x_names
        assert "y" in df.columns, "No 'y' column in df"
        assert "a" in df.columns, "No 'a' column in df"
        for x in x_names:
            assert x in df.columns, "No {} column in df".format(x)

        self.gamma = gamma

        self.p_s1 = get_p_s1()

        df = self.add_weights(get_p_a_x, get_p_s_x, gamma)

        super().__init__(df['Z_0'], df['Z_1'], alpha)

    def add_weights(self, get_p_a_x, get_p_s_x, gamma):
        def get_weight(p_a_x, p_s1_x, a, gamma_s_i):
            weight = gamma_s_i * (1 - p_s1_x) / p_s1_x * self.p_s1 / (1 - self.p_s1) / p_a_x
            return np.where(self.df['a'] == a, weight, 0)

        def get_weights(a):
            p_a_x = get_p_a_x(self.df[self.x_names], a)
            p_s1_x = get_p_s_x(self.df[self.x_names], 1)

            weight_low = get_weight(p_a_x, p_s1_x, a, 1 / gamma)
            weight_high = get_weight(p_a_x, p_s1_x, a, gamma)

            weight = np.where(self.df['y'] == 1, weight_high, weight_low)
            return weight

        df = pd.DataFrame()
        for a in [0, 1]:
            df[f'weight_{a}'] = get_weights(a)
            df[f'Z_{a}'] =df[f'weight_{a}'] * self.df['y']

        return df
