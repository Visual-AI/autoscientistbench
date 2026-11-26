import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tqdm

class ConformalResidualSampler(nn.Module):
    def __init__(
        self,
        cal_residuals,
        reservoir,
        alpha,
        T,
        past_residuals_window,
        eta,
        n_quantiles,
        similarity="cosine",
        decay="linear",
        decay_rate=0.99,
    ):
        super(ConformalResidualSampler, self).__init__()
        self.cal_states = None
        self.cal_residuals = cal_residuals
        self.reservoir = reservoir
        self.alpha = alpha
        self.T = T
        self.past_residuals_window = past_residuals_window
        self.eta = eta
        self.n_quantiles = int(n_quantiles)
        self.similarity = similarity
        self.decay = decay
        self.decay_rate = decay_rate
        self.sample = lambda x, size, p: x[
            torch.multinomial(p, num_samples=size, replacement=True)
        ]
        self.in_sweep = False

    def get_states(self, X, previous_state=None):
        """Computes the reservoir states for the input time series.
        Parameters
        ----------
        X : np.ndarray
             Input time series. Shape (n_samples, n_features)

        Returns
        -------
        states : np.ndarray
            Reservoir states of the input time series. Shape (n_samples, n_internal_units)
        """

        assert isinstance(X, torch.Tensor), f"Expected {torch.Tensor}, got {type(X)}"

        states = self.reservoir._reservoir.get_states(
            X[None, :, :], bidir=False, initial_state=previous_state
        )
        return states[0]

    def compute_calibration_states(self, X_cal):
        """Computes the reservoir states of the calibration set, which will be used to compare the new states.
        Parameters
        ----------
        X_cal : np.ndarray or torch.Tensor
            Calibration set. Shape (n_cal, n_features)

        Returns
        -------
        cal_states : np.ndarray or torch.Tensor
            Reservoir states of the calibration set. Shape (n_cal, n_internal_units)
        """
        self.cal_states = self.reservoir._reservoir.get_states(
            X_cal[None, :, :], bidir=False
        )[0]
        return self.cal_states

    def compute_similarity(self, cal_states, test_states, T=1):
        """Computes the similarity between the test set states and the calibration set states.
        Parameters
        ----------
        test_states : torch.Tensor
            Reservoir states of the test set. Shape (n_test, n_internal_units)
        T : float, optional
            Temperature of the softmax. By default ``1``.

        Returns
        -------
        similarity : torch.Tensor
            Similarity between the test set states and the calibration set states. Shape (n_test, n_cal)
        """

        if self.similarity == "cosine":
            unnorm_similarity = (cal_states @ test_states.T).flatten()
        elif self.similarity == "euclidean":
            # Euclidean distance similarity
            unnorm_similarity = -torch.norm(
                cal_states[:, None, :] - test_states[None, :, :], dim=2
            ).flatten()
        elif self.similarity == "angular":
            unnorm_similarity = (
                1
                - torch.arccos(torch.clamp(cal_states @ test_states.T, -1, 1)).flatten()
            )

        if self.decay == "exponential":
            # exponential decay
            rho = self.decay_rate
            self.weights = rho ** (
                torch.arange(cal_states.shape[0], 0, -1, device=cal_states.device)
            )
            self.weights = self.weights / self.weights.sum()

            similarity = F.softmax(unnorm_similarity / T, dim=0)

            similarity = similarity * self.weights
            similarity = similarity / similarity.sum()
        elif self.decay == "linear":
            # linear decay
            self.weights = torch.arange(cal_states.shape[0], device=cal_states.device)
            self.weights = self.weights / self.weights.sum()

            similarity = F.softmax(unnorm_similarity / T, dim=0)

            similarity = similarity * self.weights
            similarity = similarity / similarity.sum()
        else:
            # no decay
            similarity = F.softmax(unnorm_similarity / T, dim=0)

        assert torch.isclose(similarity.sum(), torch.tensor(1.0)), (
            "The similarity does not sum to 1. " f"Got {similarity.sum()} instead."
        )

        return similarity

    def sample_residuals(
        self,
        cal_residuals: torch.Tensor,
        similarity: torch.Tensor,
        sample_size: int,
    ) -> torch.Tensor:
        """Samples the residuals from the calibration set based on the similarity.
        Parameters
        ----------
        similarity : torch.Tensor
            Similarity between the test set states and the calibration set states. Shape (n_test, n_cal)
        sample_size : int
            Number of samples to draw from the calibration set.

        Returns
        -------
        sampled_residuals : torch.Tensor
            Sampled residuals from the calibration set. Shape (sample_size,)
        """

        sampled_residuals = self.sample(cal_residuals, size=sample_size, p=similarity)
        return sampled_residuals

    def forward(self, x, y_hat, y_true):  # x are the reservoir states
        target_coverage = 1 - self.alpha
        running_alpha = self.alpha

        coverages = torch.zeros(x.shape[0], device=x.device)
        rolling_coverage = torch.zeros(x.shape[0], device=x.device)
        miscoverage_history = torch.zeros(x.shape[0], device=x.device)
        alpha_history = torch.zeros(x.shape[0], device=x.device)
        lower_quantiles = torch.zeros((x.shape[0], self.n_quantiles), device=x.device)
        upper_quantiles = torch.zeros((x.shape[0], self.n_quantiles), device=x.device)

        # similarities = torch.zeros(
        #     (x.shape[0], self.past_residuals_window), device=x.device
        # )
        # all_sampled_residuals = torch.zeros(
        #     (x.shape[0], self.past_residuals_window), device=x.device
        # )
        # past_residuals = torch.zeros(
        #     (x.shape[0], self.past_residuals_window), device=x.device
        # )
        if not self.in_sweep:
            similarities = [None] * x.shape[0]
            all_sampled_residuals = [None] * x.shape[0]
            past_residuals = [None] * x.shape[0]

        if self.past_residuals_window < self.cal_residuals.shape[0]:
            print(
                "Calibration set is larger than the past residuals window. Sliding residuals and states."
            )
            # if the calibration set is larger than the past residuals window,
            # keep the last residuals and states, and then slide them as new data comes in
            self.cal_residuals = self.cal_residuals[-self.past_residuals_window :]
            self.cal_states = self.cal_states[-self.past_residuals_window :]
        else:
            # if the calibration set is smaller than the past residuals window,
            # add new residuals and states to fill the window
            print(
                "Calibration set is smaller than the past residuals window. Adding new residuals and states."
            )

        sample_size = self.past_residuals_window
        running_sample_size = sample_size

        for i in tqdm.tqdm(range(len(x))):
            test_state = x[i].reshape(1, -1)
            assert torch.isclose(
                torch.linalg.norm(test_state, dim=1), torch.tensor(1.0)
            ).all(), f"Test state {i} is not normalized. Got norm {torch.norm(test_state)} instead."

            similarity = self.compute_similarity(
                self.cal_states,
                test_state,
                T=self.T,
            )

            sampled_residuals = self.sample_residuals(
                self.cal_residuals.squeeze(),
                similarity,
                running_sample_size,
            )

            if not self.in_sweep:
                similarities[i] = similarity
                past_residuals[i] = self.cal_residuals.squeeze()
                all_sampled_residuals[i] = sampled_residuals

            if self.n_quantiles > 1:
                lower_betas = torch.linspace(
                    1.0e-3,
                    running_alpha - 1.0e-3,
                    self.n_quantiles,
                    device=x.device,
                )
                upper_betas = 1 - running_alpha + lower_betas
            else:
                lower_betas = torch.tensor([running_alpha / 2], device=x.device)
                upper_betas = torch.tensor([1 - running_alpha / 2], device=x.device)

            lower_quantiles[i] = torch.quantile(sampled_residuals, lower_betas)
            upper_quantiles[i] = torch.quantile(sampled_residuals, upper_betas)

            if self.n_quantiles > 1:
                idx_min = torch.argmin(upper_quantiles[i] - lower_quantiles[i])
                coverages[i] = (
                    (y_hat[i] + lower_quantiles[i][idx_min]) <= y_true[i]
                ) & (y_true[i] <= (y_hat[i] + upper_quantiles[i][idx_min]))
            else:
                coverages[i] = (y_hat[i] + lower_quantiles[i] <= y_true[i]) & (
                    y_true[i] <= (y_hat[i] + upper_quantiles[i])
                )

            rolling_coverage[i] = coverages[: i + 1].mean()

            miscoverage = rolling_coverage[i] - target_coverage
            miscoverage_history[i] = miscoverage
            # if miscoverage is negative, undercoverage -> decrease alpha
            # if miscoverage is positive, overcoverage -> increase alpha
            running_alpha = torch.clip(
                running_alpha + self.eta * miscoverage,
                1.0e-3,
                0.3,
            )
            alpha_history[i] = running_alpha

            if y_true is not None:
                if self.past_residuals_window < self.cal_residuals.shape[0]:
                    old_len = self.cal_states.shape[0]
                    # shift the calibration states by one position to the left
                    # (i.e. remove the first state)
                    self.cal_states = torch.roll(self.cal_states, -1, dims=0)
                    # add the new state to the end of the calibration states
                    self.cal_states[-1] = test_state.unsqueeze(0)
                    assert old_len == self.cal_states.shape[0]
                    # do the same for the residuals
                    old_len = self.cal_residuals.shape[0]
                    self.cal_residuals = torch.roll(self.cal_residuals, -1, dims=0)
                    self.cal_residuals[-1] = (y_true[i] - y_hat[i]).unsqueeze(0)
                    assert old_len == self.cal_residuals.shape[0]
                else:
                    old_len = self.cal_states.shape[0]
                    # if the calibration set is smaller than the past residuals window,
                    # add the new state and residual to the calibration set
                    self.cal_states = torch.cat(
                        [self.cal_states, test_state.unsqueeze(0)], dim=0
                    )
                    assert self.cal_states.shape[0] == old_len + 1
                    # do the same for the residuals
                    old_len = self.cal_residuals.shape[0]
                    self.cal_residuals = torch.cat(
                        [self.cal_residuals, (y_true[i] - y_hat[i]).unsqueeze(0)],
                        dim=0,
                    )
                    assert self.cal_residuals.shape[0] == old_len + 1
                    running_sample_size += 1

        self.alpha_history = alpha_history
        self.rolling_coverage = rolling_coverage
        self.miscoverage_history = miscoverage_history
        if not self.in_sweep:
            self.similarities = similarities
            self.all_sampled_residuals = all_sampled_residuals
            self.past_residuals = past_residuals

        return torch.hstack([lower_quantiles, upper_quantiles])

class ConformalResidualSamplerParallel(nn.Module):
    def __init__(
        self,
        cal_residuals,
        reservoir,
        alpha,
        T,
        past_residuals_window,
        eta,
        n_quantiles,
        similarity="cosine",
        decay="linear",
        decay_rate=0.99,
    ):
        super(ConformalResidualSamplerParallel, self).__init__()
        self.cal_states = None
        self.cal_residuals = cal_residuals  # n_cal, n_nodes, 1
        self.reservoir = reservoir
        self.alpha = alpha
        self.T = T
        self.past_residuals_window = past_residuals_window
        self.eta = eta
        self.n_quantiles = int(n_quantiles)
        self.similarity = similarity
        self.decay = decay
        self.decay_rate = decay_rate
        self.sample = lambda x, size, p: x[
            torch.multinomial(p, num_samples=size, replacement=True)
        ]
        self.in_sweep = False

    def compute_similarity(self, cal_states, test_states, T=1):
        """Computes the similarity between the test set states and the calibration set states.

        Parameters
        ----------
        cal_states : torch.Tensor
            Reservoir states of the calibration set. Shape (n_cal, n_nodes, n_internal_units)

        test_states : torch.Tensor
            Reservoir states of the test set. Shape (1, n_nodes, n_internal_units)

        T : float, optional
            Temperature of the softmax. By default ``1``.

        Returns
        -------
        similarity : torch.Tensor
            Similarity between the test set states and the calibration set states. Shape (n_cal, n_nodes)
        """

        if self.similarity == "cosine":
            unnorm_similarity = torch.einsum(
                "cnh,tnh->cn", cal_states, test_states
            )  # n_cal, n_nodes
        else:
            raise NotImplementedError(
                f"Similarity {self.similarity} not implemented for parallel reservoirs."
            )

        if self.decay == "linear":
            # linear decay
            self.weights = torch.arange(
                cal_states.shape[0], device=cal_states.device
            ).unsqueeze(
                1
            )  # n_cal, 1
            self.weights = self.weights / self.weights.sum()

            similarity = F.softmax(unnorm_similarity / T, dim=0)  # n_cal, n_nodes

            similarity = similarity * self.weights  # n_cal, n_nodes
            similarity = similarity / similarity.sum(dim=0, keepdim=True)
        else:
            # no decay
            similarity = F.softmax(unnorm_similarity / T, dim=0)  # n_cal, n_nodes

        assert torch.isclose(similarity.sum(dim=0), torch.tensor(1.0)).all(), (
            "The similarity does not sum to 1. " f"Got {similarity.sum()} instead."
        )

        return similarity  # n_cal, n_nodes

    def sample_residuals(
        self,
        cal_residuals: torch.Tensor,
        similarity: torch.Tensor,
        sample_size: int,
    ) -> torch.Tensor:
        """Samples the residuals from the calibration set based on the similarity.

        Parameters
        ----------
        cal_residuals : torch.Tensor
            Calibration residuals. Shape (n_cal, n_nodes)
        similarity : torch.Tensor
            Similarity between the test set states and the calibration set states. Shape (n_cal, n_nodes)
        sample_size : int
            Number of samples to draw from the calibration set.

        Returns
        -------
        sampled_residuals : torch.Tensor
            Sampled residuals from the calibration set. Shape (sample_size, n_nodes)
        """
        idxs = torch.multinomial(
            similarity.t(), num_samples=sample_size, replacement=True
        )
        col_idx = torch.arange(cal_residuals.shape[1], dtype=torch.long).unsqueeze(1)
        sampled_residuals = cal_residuals[idxs, col_idx].squeeze().t()

        # sample_size, n_nodes
        return sampled_residuals

    def forward(
        self, x, y_hat, y_true
    ):  # x are the reservoir states (n_samples, n_nodes, n_internal_units)
        """Computes the prediction intervals for the given reservoir states.
        It performs a quantile regression on the residuals of the calibration set to compute the prediction intervals.

        Parameters
        ----------
        x : torch.Tensor
            Reservoir states of the test set. Shape (n_samples, n_nodes, n_internal_units)
        y_hat : torch.Tensor
            Point predictions for the given time series. Shape (n_samples, n_nodes)
        y_true : torch.Tensor
            Ground truth values for the given time series. Shape (n_samples, n_nodes)

        Returns
        -------
        quantiles : torch.Tensor
            Prediction intervals for the given time series. Shape (n_samples, n_nodes, 2*n_quantiles)
        """
        target_coverage = 1 - self.alpha
        running_alpha = self.alpha

        coverages = torch.zeros(x.shape[0], x.shape[1], device=x.device)
        rolling_coverage = torch.zeros(x.shape[0], device=x.device)
        miscoverage_history = torch.zeros(x.shape[0], device=x.device)
        alpha_history = torch.zeros(x.shape[0], device=x.device)
        lower_quantiles = torch.zeros(
            (x.shape[0], x.shape[1], self.n_quantiles), device=x.device
        )
        upper_quantiles = torch.zeros(
            (x.shape[0], x.shape[1], self.n_quantiles), device=x.device
        )

        # if not self.in_sweep:
        #     similarities = np.zeros(x.shape[0], x.shape[1])
        #     all_sampled_residuals = np.zeros(x.shape[0], x.shape[1])
        #     past_residuals = np.zeros(x.shape[0], x.shape[1])

        if self.past_residuals_window < self.cal_residuals.shape[0]:
            print(
                "Calibration set is larger than the past residuals window. Sliding residuals and states."
            )
            # if the calibration set is larger than the past residuals window,
            # keep the last residuals and states, and then slide them as new data comes in
            self.cal_residuals = self.cal_residuals[-self.past_residuals_window :]
            self.cal_states = self.cal_states[-self.past_residuals_window :]
        else:
            # if the calibration set is smaller than the past residuals window,
            # add new residuals and states to fill the window
            print(
                "Calibration set is smaller than the past residuals window. Adding new residuals and states."
            )

        sample_size = self.past_residuals_window
        running_sample_size = sample_size

        for i in tqdm.tqdm(range(len(x))):
            test_state = x[i].reshape(
                1, x.shape[1], x.shape[2]
            )  # 1, n_nodes, n_internal_units
            assert torch.isclose(
                torch.linalg.norm(test_state, dim=2), torch.tensor(1.0)
            ).all(), f"Test state {i} is not normalized. Got norm {torch.norm(test_state)} instead."

            similarity = self.compute_similarity(
                self.cal_states,
                test_state,
                T=self.T,
            )  # n_cal, n_nodes

            sampled_residuals = self.sample_residuals(
                self.cal_residuals.squeeze(),  # n_cal, n_nodes
                similarity,  # n_cal, n_nodes
                running_sample_size,
            )  # sample_size, n_nodes

            # if not self.in_sweep:
            #     similarities[i] = similarity
            #     past_residuals[i] = self.cal_residuals.squeeze()
            #     all_sampled_residuals[i] = sampled_residuals

            if self.n_quantiles > 1:
                lower_betas = torch.linspace(
                    1.0e-3,
                    running_alpha - 1.0e-3,
                    self.n_quantiles,
                    device=x.device,
                )
                upper_betas = 1 - running_alpha + lower_betas
            else:
                lower_betas = torch.tensor([running_alpha / 2], device=x.device)
                upper_betas = torch.tensor([1 - running_alpha / 2], device=x.device)

            lower_quantiles[i] = torch.quantile(
                sampled_residuals, lower_betas, dim=0
            ).t()  # n_quantiles, n_nodes
            upper_quantiles[i] = torch.quantile(
                sampled_residuals, upper_betas, dim=0
            ).t()  # n_quantiles, n_nodes -> n_nodes, n_quantiles

            if self.n_quantiles > 1:
                _, idx_min = torch.min(
                    upper_quantiles[i] - lower_quantiles[i], dim=1
                )  # n_nodes
                coverages[i] = (
                    (
                        y_hat[i].squeeze()
                        + lower_quantiles[i][
                            torch.arange(lower_quantiles[i].shape[0]), idx_min
                        ]
                    )
                    <= y_true[i].squeeze()
                ) & (
                    y_true[i].squeeze()
                    <= (
                        y_hat[i].squeeze()
                        + upper_quantiles[i][
                            torch.arange(upper_quantiles[i].shape[0]), idx_min
                        ]
                    )
                )
            else:
                coverages[i] = (y_hat[i] + lower_quantiles[i] <= y_true[i]) & (
                    y_true[i] <= (y_hat[i] + upper_quantiles[i])
                )

            rolling_coverage[i] = coverages[: i + 1].mean()

            miscoverage = rolling_coverage[i] - target_coverage
            miscoverage_history[i] = miscoverage
            # if miscoverage is negative, undercoverage -> decrease alpha
            # if miscoverage is positive, overcoverage -> increase alpha
            running_alpha = torch.clip(
                running_alpha + self.eta * miscoverage,
                1.0e-3,
                0.3,
            )
            alpha_history[i] = running_alpha

            if y_true is not None:
                if self.past_residuals_window < self.cal_residuals.shape[0]:
                    old_len = self.cal_states.shape[0]
                    # shift the calibration states by one position to the left
                    # (i.e. remove the first state)
                    self.cal_states = torch.roll(self.cal_states, -1, dims=0)
                    # add the new state to the end of the calibration states
                    self.cal_states[-1] = test_state
                    assert old_len == self.cal_states.shape[0]
                    # do the same for the residuals
                    old_len = self.cal_residuals.shape[0]
                    self.cal_residuals = torch.roll(self.cal_residuals, -1, dims=0)
                    self.cal_residuals[-1] = (y_true[i] - y_hat[i]).unsqueeze(0)
                    assert old_len == self.cal_residuals.shape[0]
                else:
                    old_len = self.cal_states.shape[0]
                    # if the calibration set is smaller than the past residuals window,
                    # add the new state and residual to the calibration set
                    self.cal_states = torch.cat([self.cal_states, test_state], dim=0)
                    assert self.cal_states.shape[0] == old_len + 1
                    # do the same for the residuals
                    old_len = self.cal_residuals.shape[0]
                    self.cal_residuals = torch.cat(
                        [self.cal_residuals, (y_true[i] - y_hat[i]).unsqueeze(0)],
                        dim=0,
                    )
                    assert self.cal_residuals.shape[0] == old_len + 1
                    running_sample_size += 1

        self.alpha_history = alpha_history
        self.rolling_coverage = rolling_coverage
        self.miscoverage_history = miscoverage_history
        # if not self.in_sweep:
        #     self.similarities = similarities
        #     self.all_sampled_residuals = all_sampled_residuals
        #     self.past_residuals = past_residuals

        # (n_samples, n_nodes, 2*n_quantiles)
        return torch.cat([lower_quantiles, upper_quantiles], dim=2)
