"""Opt-in CPU sampler; original forward implementations remain untouched.

Pool slices preserve the release's window+1 behavior and multinomial draw order.
Only states strictly preceding each query enter its probability distribution.
"""
import os
import torch
import torch.nn.functional as F


def sorted_quantiles(samples, levels):
    ordered = samples.sort(dim=0).values
    ranks = levels * (len(samples) - 1)
    low = ranks.floor().long()
    high = ranks.ceil().long()
    weight = ranks - low
    if samples.ndim > 1:
        weight = weight[:, None]
    return torch.lerp(ordered[low], ordered[high], weight)


def forward(self, x, y_hat, y_true):
    scalar = self.__class__.__name__ == 'ConformalResidualSampler'
    no_window = getattr(self, '_cpu_no_window', False)
    chunk_size = int(os.environ.get('REPRO_SIM_CHUNK', '32'))
    use_gemm = os.environ.get('REPRO_SIM_GEMM', '1') == '1'
    steps, nodes = len(x), (1 if scalar else x.shape[1])
    initial = self.cal_residuals.shape[0]
    window = self.past_residuals_window
    if not no_window and window < initial:
        self.cal_states = self.cal_states[-window:]
        self.cal_residuals = self.cal_residuals[-window:]
    initial = len(self.cal_residuals)
    # Single allocation, subsequently only logical slices; never roll or append.
    states = torch.cat([self.cal_states, x], dim=0)
    residuals = torch.cat([self.cal_residuals, y_true-y_hat], dim=0)
    starts = [0 if no_window else max(0, initial+i-window-1) for i in range(steps+1)]
    lower = torch.zeros((steps, nodes, self.n_quantiles), device=x.device)
    upper = torch.zeros_like(lower)
    self.alpha_history = torch.zeros(steps, device=x.device)
    self.rolling_coverage = torch.zeros_like(self.alpha_history)
    self.miscoverage_history = torch.zeros_like(self.alpha_history)
    running_alpha = self.alpha
    coverage_sum = torch.tensor(0., device=x.device)
    draws = initial if no_window else window
    initial_draws = draws
    probabilities = None
    weights = {}
    for i in range(steps):
        lo, hi = starts[i], initial+i
        pool = residuals[lo:hi].squeeze()
        if use_gemm:
            if i % chunk_size == 0:
                end = min(i+chunk_size, steps)
                block_lo, block_hi = starts[i], initial+end-1
                if scalar:
                    dots = states[block_lo:block_hi].reshape(block_hi-block_lo,-1) @ x[i:end].reshape(end-i,-1).T
                else:
                    dots = torch.bmm(states[block_lo:block_hi].permute(1,0,2).contiguous(), x[i:end].permute(1,2,0).contiguous())
            raw = dots[lo-block_lo:hi-block_lo, i % chunk_size] if scalar else dots[:,lo-block_lo:hi-block_lo,i % chunk_size].T.contiguous()
            probabilities = F.softmax(raw / self.T, dim=0)
            if self.decay in ('linear', 'exponential'):
                if hi-lo not in weights:
                    w = torch.arange(hi-lo, device=x.device) if self.decay == 'linear' else self.decay_rate ** torch.arange(hi-lo,0,-1,device=x.device)
                    w = w / w.sum()
                    weights[hi-lo] = w if scalar else w[:,None]
                probabilities = probabilities * weights[hi-lo]
                probabilities = probabilities / probabilities.sum(dim=0, keepdim=not scalar)
        else:
            test = x[i].reshape(1,-1) if scalar else x[i:i+1]
            probabilities = self.compute_similarity(states[lo:hi], test, T=self.T)
        samples = self.sample_residuals(pool, probabilities, draws)
        if self.n_quantiles > 1:
            lb = torch.linspace(.001, running_alpha-.001, self.n_quantiles, device=x.device)
            ub = 1-running_alpha+lb
        else:
            lb = torch.tensor([running_alpha/2],device=x.device)
            ub = torch.tensor([1-running_alpha/2],device=x.device)
        q = sorted_quantiles(samples, torch.cat([lb,ub]))
        if scalar:
            q = q[:,None]
        lower[i], upper[i] = q[:self.n_quantiles].T, q[self.n_quantiles:].T
        idx = (upper[i]-lower[i]).argmin(dim=1)
        idx_nodes = torch.arange(nodes)
        covered = ((y_hat[i].reshape(nodes)+lower[i,idx_nodes,idx] <= y_true[i].reshape(nodes)) &
                   (y_true[i].reshape(nodes) <= y_hat[i].reshape(nodes)+upper[i,idx_nodes,idx]))
        coverage_sum += covered.sum()
        self.rolling_coverage[i] = coverage_sum / ((i+1)*nodes)
        miscoverage = self.rolling_coverage[i] - (1-self.alpha)
        self.miscoverage_history[i] = miscoverage
        running_alpha = torch.clip(running_alpha+self.eta*miscoverage, .001,.3)
        self.alpha_history[i] = running_alpha
        if no_window or hi-lo <= window:
            draws += 1
    self.cal_states = states[starts[steps]:initial+steps]
    self.cal_residuals = residuals[starts[steps]:initial+steps]
    self.cpu_audit = dict(initial_pool=initial, final_pool=len(self.cal_residuals),
                          steps=steps, initial_draws=initial_draws, last_draws=draws-(1 if no_window else 0),
                          no_window=no_window, gemm=use_gemm, chunk_size=chunk_size)
    if no_window:
        assert draws == initial + steps
        assert len(self.cal_residuals) == initial + steps
    result = torch.cat([lower,upper],dim=-1)
    return result[:,0] if scalar else result


def install_sampling(module, no_window=False):
    for cls in [module.ConformalResidualSampler,module.ConformalResidualSamplerParallel]:
        cls.forward_reference = cls.forward
        cls.forward = forward
        cls._cpu_no_window = no_window

