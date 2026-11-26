"""Frozen sbibm forward simulators, source methods preserved verbatim.
Only dataset metadata/loaders are removed; seed comes from the caller.
This module contains no reference observations or posterior samples.
"""
import math
from typing import Callable, Optional
import numpy as np
import torch
import pyro
from pyro import distributions as pdist

class Simulator:
    def __init__(self, task, simulator, max_calls=None):
        self.task, self.simulator = task, simulator
    def __call__(self, parameters):
        return self.simulator(parameters).reshape(parameters.shape[0], -1)

class GaussianLinear:
    def __init__(
        self, dim: int = 10, prior_scale: float = 0.1, simulator_scale: float = 0.1
    ):
        self.dim_parameters = dim
        """Gaussian Linear

        Inference of mean under Gaussian prior in 2D

        Args:
            dim: Dimensionality of parameters and data
            prior_scale: Standard deviation of prior
            simulator_scale: Standard deviation of noise in simulator
        """

        self.prior_params = {
            "loc": torch.zeros((self.dim_parameters,)),
            "precision_matrix": torch.inverse(
                prior_scale * torch.eye(self.dim_parameters)
            ),
        }

        self.prior_dist = pdist.MultivariateNormal(**self.prior_params)
        self.prior_dist.set_default_validate_args(False)

        self.simulator_params = {
            "precision_matrix": torch.inverse(
                simulator_scale * torch.eye(self.dim_parameters),
            )
        }
    def get_prior(self) -> Callable:
        def prior(num_samples=1):
            return pyro.sample("parameters", self.prior_dist.expand_by([num_samples]))

        return prior
    def get_simulator(self, max_calls: Optional[int] = None) -> Simulator:
        """Get function returning samples from simulator given parameters

        Args:
            max_calls: Maximum number of function calls. Additional calls will
                result in SimulationBudgetExceeded exceptions. Defaults to None
                for infinite budget

        Return:
            Simulator callable
        """

        def simulator(parameters):
            return pyro.sample(
                "data",
                pdist.MultivariateNormal(
                    loc=parameters,
                    precision_matrix=self.simulator_params["precision_matrix"],
                ),
            )

        return Simulator(task=self, simulator=simulator, max_calls=max_calls)

class GaussianLinearUniform:
    def __init__(
        self, dim: int = 10, prior_bound: float = 1.0, simulator_scale: float = 0.1
    ):
        self.dim_parameters = dim
        """Gaussian Linear Uniform

        Inference of mean under uniform prior.

        Args:
            dim: Dimensionality of parameters and data.
            prior_bound: Prior is uniform in [-prior_bound, +prior_bound].
            simulator_scale: Standard deviation of noise in simulator.
        """

        self.prior_params = {
            "low": -prior_bound * torch.ones((self.dim_parameters,)),
            "high": +prior_bound * torch.ones((self.dim_parameters,)),
        }

        self.prior_dist = pdist.Uniform(**self.prior_params).to_event(1)
        self.prior_dist.set_default_validate_args(False)

        self.simulator_params = {
            "precision_matrix": torch.inverse(
                simulator_scale * torch.eye(self.dim_parameters),
            )
        }
    def get_prior(self) -> Callable:
        def prior(num_samples=1):
            return pyro.sample("parameters", self.prior_dist.expand_by([num_samples]))

        return prior
    def get_simulator(self, max_calls: Optional[int] = None) -> Simulator:
        """Get function returning samples from simulator given parameters

        Args:
            max_calls: Maximum number of function calls. Additional calls will
                result in SimulationBudgetExceeded exceptions. Defaults to None
                for infinite budget

        Return:
            Simulator callable
        """

        def simulator(parameters):
            return pyro.sample(
                "data",
                pdist.MultivariateNormal(
                    loc=parameters,
                    precision_matrix=self.simulator_params["precision_matrix"],
                ),
            )

        return Simulator(task=self, simulator=simulator, max_calls=max_calls)

class SLCP:
    def __init__(self, distractors: bool = False):
        self.dim_parameters = 5
        """SLCP"""
        self.num_data = 4
        self.distractors = distractors

        if not self.distractors:
            dim_data = 2 * self.num_data
            name = "slcp"
            name_display = "SLCP"
        else:
            dim_data = 100
            name = "slcp_distractors"
            name_display = "SLCP Distractors"



        self.prior_params = {
            "low": torch.tensor([-3.0 for _ in range(self.dim_parameters)]),
            "high": torch.tensor([+3.0 for _ in range(self.dim_parameters)]),
        }
        self.prior_dist = pdist.Uniform(**self.prior_params).to_event(1)
        self.prior_dist.set_default_validate_args(False)
    def get_prior(self) -> Callable:
        def prior(num_samples=1):
            return pyro.sample("parameters", self.prior_dist.expand_by([num_samples]))

        return prior
    def get_simulator(self, max_calls: Optional[int] = None) -> Simulator:
        """Get function returning samples from simulator given parameters

        Args:
            max_calls: Maximum number of function calls. Additional calls will
                result in SimulationBudgetExceeded exceptions. Defaults to None
                for infinite budget

        Return:
            Simulator callable
        """

        def simulator(parameters):
            num_samples = parameters.shape[0]

            m = torch.stack(
                (parameters[:, [0]].squeeze(), parameters[:, [1]].squeeze())
            ).T
            if m.dim() == 1:
                m.unsqueeze_(0)

            s1 = parameters[:, [2]].squeeze() ** 2
            s2 = parameters[:, [3]].squeeze() ** 2
            rho = torch.nn.Tanh()(parameters[:, [4]]).squeeze()

            S = torch.empty((num_samples, 2, 2))
            S[:, 0, 0] = s1 ** 2
            S[:, 0, 1] = rho * s1 * s2
            S[:, 1, 0] = rho * s1 * s2
            S[:, 1, 1] = s2 ** 2

            # Add eps to diagonal to ensure PSD
            eps = 0.000001
            S[:, 0, 0] += eps
            S[:, 1, 1] += eps

            data_dist = pdist.MultivariateNormal(
                m.unsqueeze(1).float(), S.unsqueeze(1).float()
            ).expand(
                (
                    num_samples,
                    self.num_data,
                )
            )

            if not self.distractors:
                return pyro.sample("data", data_dist)
            else:
                raise ValueError("Distractor observations are outside this task")

        return Simulator(task=self, simulator=simulator, max_calls=max_calls)

class GaussianMixture:
    def __init__(
        self,
        dim: int = 2,
        prior_bound: float = 10.0,
    ):
        self.dim_parameters = 2
        """Gaussian Mixture.

        Inference of mean under uniform prior.

        Args:
            dim: Dimensionality of parameters and data.
            prior_bound: Prior is uniform in [-prior_bound, +prior_bound].
        """

        self.prior_params = {
            "low": -prior_bound * torch.ones((self.dim_parameters,)),
            "high": +prior_bound * torch.ones((self.dim_parameters,)),
        }

        self.prior_dist = pdist.Uniform(**self.prior_params).to_event(1)
        self.prior_dist.set_default_validate_args(False)

        self.simulator_params = {
            "mixture_locs_factor": torch.tensor([1.0, 1.0]),
            "mixture_scales": torch.tensor([1.0, 0.1]),
            "mixture_weights": torch.tensor([0.5, 0.5]),
        }
    def get_prior(self) -> Callable:
        def prior(num_samples=1):
            return pyro.sample("parameters", self.prior_dist.expand_by([num_samples]))

        return prior
    def get_simulator(self, max_calls: Optional[int] = None) -> Simulator:
        """Get function returning samples from simulator given parameters

        Args:
            max_calls: Maximum number of function calls. Additional calls will
                result in SimulationBudgetExceeded exceptions. Defaults to None
                for infinite budget

        Return:
            Simulator callable
        """

        def simulator(parameters):
            # Sample mixture index for each parameter in batch
            idx = pyro.sample(
                "mixture_idx",
                pdist.Categorical(
                    probs=self.simulator_params["mixture_weights"]
                ).expand_by([parameters.shape[0]]),
            ).unsqueeze(1)

            # Select loc and scales according to mixture index
            loc = self.simulator_params["mixture_locs_factor"][idx] * parameters
            scale = self.simulator_params["mixture_scales"][idx]

            return pyro.sample(
                "data", pdist.Normal(loc=loc, scale=scale).to_event(1)
            )

        return Simulator(task=self, simulator=simulator, max_calls=max_calls)

class TwoMoons:
    def __init__(self):
        self.dim_parameters = 2
        """Two Moons"""



        prior_bound = 1.0
        self.prior_params = {
            "low": -prior_bound * torch.ones((self.dim_parameters,)),
            "high": +prior_bound * torch.ones((self.dim_parameters,)),
        }
        self.prior_dist = pdist.Uniform(**self.prior_params).to_event(1)
        self.prior_dist.set_default_validate_args(False)

        self.simulator_params = {
            "a_low": -math.pi / 2.0,
            "a_high": +math.pi / 2.0,
            "base_offset": 0.25,
            "r_loc": 0.1,
            "r_scale": 0.01,
        }
    def get_prior(self) -> Callable:
        def prior(num_samples=1):
            return pyro.sample("parameters", self.prior_dist.expand_by([num_samples]))

        return prior
    def get_simulator(self, max_calls: Optional[int] = None) -> Simulator:
        """Get function returning samples from simulator given parameters

        Args:
            max_calls: Maximum number of function calls. Additional calls will
                result in SimulationBudgetExceeded exceptions. Defaults to None
                for infinite budget

        Return:
            Simulator callable
        """

        def simulator(parameters):
            num_samples = parameters.shape[0]

            a_dist = (
                pdist.Uniform(
                    low=self.simulator_params["a_low"],
                    high=self.simulator_params["a_high"],
                )
                .expand_by((num_samples, 1))
                .to_event(1)
            )
            a = a_dist.sample()

            r_dist = (
                pdist.Normal(
                    self.simulator_params["r_loc"], self.simulator_params["r_scale"]
                )
                .expand_by((num_samples, 1))
                .to_event(1)
            )
            r = r_dist.sample()

            p = torch.cat(
                (
                    torch.cos(a) * r + self.simulator_params["base_offset"],
                    torch.sin(a) * r,
                ),
                dim=1,
            )

            return self._map_fun(parameters, p)

        return Simulator(task=self, simulator=simulator, max_calls=max_calls)
    @staticmethod
    def _map_fun(parameters: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        ang = torch.tensor([-math.pi / 4.0])
        c = torch.cos(ang)
        s = torch.sin(ang)
        z0 = (c * parameters[:, 0] - s * parameters[:, 1]).reshape(-1, 1)
        z1 = (s * parameters[:, 0] + c * parameters[:, 1]).reshape(-1, 1)
        return p + torch.cat((-torch.abs(z0), z1), dim=1)

MODELS = {"task1": GaussianLinear, "task2": GaussianLinearUniform, "task3": SLCP, "task4": GaussianMixture, "task5": TwoMoons}

def generate(task, seed):
    pyro.set_rng_seed(int(seed))
    model = MODELS[task]()
    theta = model.get_prior()(1)
    observation = model.get_simulator()(theta)
    return theta[0], observation[0]
