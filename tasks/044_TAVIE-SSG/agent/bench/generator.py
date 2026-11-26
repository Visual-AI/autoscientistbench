"""Student-t observations; identical RNG stream and draw order to notebook cell 9. No fixed truth table."""
import numpy as np

def generate_student_regression_data(n, p, beta_true, tau2_true, df, seed=None):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    error = rng.standard_t(df=df, size=n) / np.sqrt(tau2_true)
    y = beta_true[0] + X @ beta_true[1:] + error
    return X, y

