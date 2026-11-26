"""Cache conditional distributions within each inverse-transform bisection.

Copied inverse routine retains original bounds, tolerances, iterations and dtype.
Original transformation2uniform.py remains available.
"""
import torch, math
from torch.distributions.normal import Normal
from transformation2uniform import inverse_transform_uniform_from_unitcube

def inversetransform(x,cfg,bounds,means=None,covariances=None,weights=None):
    if cfg.exp.q_dist=='uniform' or cfg.exp.target=='llm_prior':
        return inverse_transform_uniform_from_unitcube(x,bounds)
    return inverse_transform2unitcube(x,means,covariances,weights,bounds[0][0],bounds[0][1])

def inverse_rosenblatt_transform_gmm(u_batch, means, covariances, weights, lowerbound, upperbound, tol=1e-5, max_iter=100):
    """
    Applies the inverse Rosenblatt transform to a batch of points u_batch.
    This version is vectorized and uses a parallelized bisection search.

    Args:
        u_batch (Tensor): Input samples from [-0.5, 0.5]^D, shape (N, D).
        means, covariances, weights: GMM parameters.
        lowerbound, upperbound: Search bounds for the bisection algorithm.
        tol, max_iter: Parameters for the bisection algorithm.

    Returns:
        Tensor: Transformed samples from the GMM distribution, shape (N, D).
    """
    u_batch = u_batch + 0.5  # Map from [-0.5, 0.5] to the [0, 1] hypercube
    N, D = u_batch.shape
    K = means.shape[0]
    device = u_batch.device
    
    # Pre-compute Cholesky factors and constants
    cholesky_factors = torch.linalg.cholesky(covariances)
    log_weights = torch.log(weights)
    log_2pi = math.log(2 * math.pi)

    x_batch = torch.empty_like(u_batch)

    cached = {}
    # Helper function to compute the conditional CDF for a batch
    def vectorized_cdf_mix(xd_batch, x_prev_batch, d):
        if d in cached:
            dist, responsibilities = cached[d]
            component_cdfs = dist.cdf(xd_batch.unsqueeze(1))
            if d > 0:
                component_cdfs = torch.clamp(component_cdfs, 1e-7, 1 - 1e-7)
            return torch.sum(responsibilities * component_cdfs, dim=1)
        if d == 0: # Marginal case
            dist = Normal(means[:, 0], torch.sqrt(covariances[:, 0, 0]))
            component_cdfs = dist.cdf(xd_batch.unsqueeze(1))
            responsibilities = weights.unsqueeze(0)
        else: # Conditional case (logic copied from forward transform)
            mu_1, mu_2 = means[:, :d], means[:, d]
            Sigma_12, Sigma_22 = covariances[:, :d, d], covariances[:, d, d]
            L_11 = cholesky_factors[:, :d, :d]
            
            A_T = torch.cholesky_solve(Sigma_12.unsqueeze(2), L_11)
            A = A_T.squeeze(2)
            
            diff = x_prev_batch.unsqueeze(1) - mu_1.unsqueeze(0)
            cond_mean = mu_2.unsqueeze(0) + torch.einsum('nkd,kd->nk', diff, A)
            cond_var = torch.clamp(Sigma_22 - torch.einsum('kd,kd->k', A, Sigma_12), min=1e-9)
            cond_std = torch.sqrt(cond_var)
            
            dist = Normal(cond_mean, cond_std.unsqueeze(0))
            component_cdfs = dist.cdf(xd_batch.unsqueeze(1))
            component_cdfs = torch.clamp(component_cdfs, 1e-7, 1 - 1e-7) #Fix to improve numerical stability
            
            log_det_Sigma = 2 * torch.sum(torch.log(torch.diagonal(L_11, dim1=-2, dim2=-1)), dim=1)
            v = torch.linalg.solve_triangular(L_11, diff.permute(1, 2, 0), upper=False)
            quad_term = torch.sum(v**2, dim=1).T
            
            log_prob_x_prev = -0.5 * (d * log_2pi + log_det_Sigma.unsqueeze(0) + quad_term)
            log_responsibilities = log_weights.unsqueeze(0) + log_prob_x_prev
            log_responsibilities_sum = torch.logsumexp(log_responsibilities, dim=1, keepdim=True)
            responsibilities = torch.exp(log_responsibilities - log_responsibilities_sum)
            #Fix to improve numerical stability
            responsibilities = torch.clamp(responsibilities, 1e-12, 1.0)
            responsibilities = responsibilities / responsibilities.sum(dim=1, keepdim=True)
            
        cached[d] = (dist, responsibilities)
        return torch.sum(responsibilities * component_cdfs, dim=1)

    for d in range(D):
        u_d_batch = u_batch[:, d]
        x_prev_batch = x_batch[:, :d]
        
        # Perform bisection search in parallel for all N samples
        lb_batch = torch.full((N,), lowerbound, device=device, dtype=torch.float32)
        ub_batch = torch.full((N,), upperbound, device=device, dtype=torch.float32)

        for _ in range(max_iter):
            mid_batch = (lb_batch + ub_batch) / 2
            val_batch = vectorized_cdf_mix(mid_batch, x_prev_batch, d)
            
            if torch.all(torch.abs(val_batch - u_d_batch) < tol):
                break
            
            mask = val_batch < u_d_batch
            lb_batch[mask] = mid_batch[mask]
            ub_batch[~mask] = mid_batch[~mask]
        
        x_batch[:, d] = (lb_batch + ub_batch) / 2

        
    return x_batch


def inverse_transform2unitcube(u_samples, means, covariances, weights, lowerbound, upperbound):
    return inverse_rosenblatt_transform_gmm(u_samples if torch.is_tensor(u_samples) else torch.from_numpy(u_samples), means, covariances, weights,lowerbound=lowerbound, upperbound=upperbound)