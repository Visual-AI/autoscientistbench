"""Candidate method entry point. run_trial is also used by the replay probe.
Return the actual particle cloud and theta after `steps` iterations. No history is kept:
it would cost 629 MB per run and nothing reads it.
"""
import optax as ox
from coinem.zoo import m_svgd_em

def run_trial(model, data, latent_init, theta_init, config, key, steps=500):
    latent_optimiser = ox.adagrad(config['learning_rate'])
    theta_optimiser = ox.adagrad(config['learning_rate'])
    # metrics={} leaves the update loop as it is and makes the scan store nothing per iteration.
    latent, theta, _ = m_svgd_em(model, data, latent_init, theta_init, latent_optimiser, theta_optimiser, steps,
                                 config['alpha'], config['alpha'], key=key, metrics={})
    return latent, theta

def oracle_counts(config, steps=500):
    # Declaration in particle-image evaluations, per iteration: one latent-score call and one
    # theta-gradient call, each on every particle and every training image. Checked by the
    # replay probe; kernel derivatives are excluded.
    import numpy as np
    return np.full((steps, 2), config['particles'] * config['train_images'], dtype=np.int32)
