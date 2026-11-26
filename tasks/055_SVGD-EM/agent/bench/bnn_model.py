"""Bayesian neural network, verbatim from cell 10 of the authors' experiment notebook.
Frozen definition: the prior scale is exp(2*theta), not a variance, and log_prob is divided by n.
"""
import jax
import jax.numpy as jnp
from jax import vmap
from jaxtyping import Array, Float
from dataclasses import dataclass
import distrax as dx

from coinem.model import AbstractModel
from coinem.dataset import Dataset


@dataclass
class BayesNN(AbstractModel):
    """Base class for p(θ, x)."""
    
    def log_prob(self, latent: Float[Array, "D"], theta: Float[Array, "Q"], data: Dataset) -> Float[Array, ""]:
        """Compute gradient of the objective function at x.

        Args:
            latent (Float[Array, "D"]): Input weights of shape (D,).
            theta (Float[Array, "Q"]): Parameters of shape (2,).

        Returns:
            Float[Array, ""]: log-probability of the data.
        """
        X = data.X # [Batchsize, 28, 28]
        y = data.y # [Batchsize, 1]

        # Compute prior:
        log_prior_w = dx.Normal(0.0, jnp.exp(2.0 * theta["alpha"].squeeze())).log_prob(latent["w"].ravel()).sum() 
        log_prior_v = dx.Normal(0.0, jnp.exp(2.0 * theta["beta"].squeeze())).log_prob(latent["v"].ravel()).sum()


        def _log_nn(image):
            # Log of the network's output when evaluated at image with weights w, v.
            return jax.nn.log_softmax(jnp.dot(latent["v"], jnp.tanh(jnp.dot(latent["w"], image.flatten())))) 


        def _log_nn_vec(images):
            # _log_nn vectorized over particles.
            return jax.vmap(_log_nn)(images)


        def _log_likelihood(images, labels):
            # Log-likelihood for set of images and labels, vectorized over particles.
            return vmap(lambda v, l : v[l])(_log_nn_vec(images), labels).sum()


        # Compute log-probability.
        return  (_log_likelihood(X, y) + log_prior_w + log_prior_v) / data.n # log p(y|x) + log p(x|theta) 
    

    def optimal_theta(self, latent_particles: Float[Array, "N D *"]) -> Float[Array, "Q *"]:
        """Optimal parameter for weight cloud w."""
        mom2 = (jax.vmap(_normsq)(latent_particles["w"])).mean()  # Second moment
        mom4 = (jax.vmap(_normsq)(latent_particles["v"])).mean()  # Fourth moment


        return {"alpha": jnp.log(mom2/(latent_particles["w"][0].size))/2, 
                "beta": jnp.log(mom4/(latent_particles["v"][0].size))/2}
        
def _normsq(x):
    # Squared Frobenius norm of x.
    v = x.reshape((x.size))
    return jnp.dot(v, v)
