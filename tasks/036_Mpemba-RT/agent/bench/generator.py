"""Frozen Fig. 2 instance family; no solutions or evaluation seeds live here."""
from itertools import product
import numpy as np
import scipy.linalg as la

REPLICAS = 10
WALL_BUDGET_S = 3600.0

def sample_parameters(seed):
    rng = np.random.default_rng(seed)
    return dict(Ns=int(rng.choice([6, 7, 8])), J=float(rng.uniform(-.5, -.3)),
                h=float(rng.uniform(.1, .3)), beta_initial=float(rng.uniform(.4, .6)))

def ising_energies(
    n_sites: int, coupling: float, field: float
) -> tuple[np.ndarray, np.ndarray]:
    configurations = np.asarray(
        list(product([1, -1], repeat=n_sites)), dtype=float
    )
    interactions = np.sum(
        configurations[:, :-1] * configurations[:, 1:], axis=1
    )
    interactions += configurations[:, 0] + configurations[:, -1]
    energies = -coupling * interactions - field * configurations.sum(axis=1)
    return energies, configurations


def classical_ising_generator(
    n_sites: int, coupling: float, field: float, beta: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Eq. (11) for the fixed-boundary Ising chain used in Fig. 2."""
    energies, configurations = ising_energies(n_sites, coupling, field)
    differences = configurations[:, None, :] != configurations[None, :, :]
    single_flip = differences.sum(axis=2) == 1
    generator = np.where(
        single_flip,
        np.exp(-0.5 * beta * (energies[:, None] - energies[None, :])),
        0.0,
    )
    np.fill_diagonal(generator, -generator.sum(axis=0))
    weights = np.exp(-beta * (energies - energies.min()))
    equilibrium = weights / weights.sum()
    return generator, energies, configurations, equilibrium


def weighted_eigendecomposition(
    generator: np.ndarray, equilibrium: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    values, vectors = la.eig(generator)
    order = np.argsort(values.real)[::-1]
    values, vectors = values[order].real, vectors[:, order].real
    for column in range(vectors.shape[1]):
        norm = np.sqrt(np.sum(vectors[:, column] ** 2 / equilibrium))
        vectors[:, column] /= norm
    return values, vectors



def make_instance(Ns, J, h, beta_initial):
    L, energies, configurations, pi = classical_ising_generator(Ns, J, h, 1.0)
    eigenvalues, modes = weighted_eigendecomposition(L, pi)
    p = np.exp(-beta_initial * (energies - energies.min()))
    p /= p.sum()
    return dict(p=p, pi=pi, modes=modes[:, 1:3], L=L,
                energies=energies, eigenvalues=eigenvalues, right_modes=modes,
                configurations=configurations)
