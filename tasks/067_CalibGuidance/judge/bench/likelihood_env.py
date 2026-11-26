"""Frozen likelihood interface: only the supplied observation is used.
Call finish() once to obtain the number of evaluated parameter vectors.
"""
import math
import torch

def gaussian_linear_loglik(theta: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """y ~ N(theta, 0.1 I_d)  (Tasks 1 and 2)."""
    th = theta.double()
    y2 = y.to(th.device).double()[None, None, :]
    diff = y2 - th
    r2 = (diff * diff).sum(-1)
    s2 = 0.1
    d = th.shape[-1]
    const = d * math.log(2.0 * math.pi)
    logdet = d * math.log(s2)
    ll = -0.5 * (const + logdet + r2 / s2)
    return ll.to(theta.dtype)

def slcp_loglik(theta: torch.Tensor, y_flat: torch.Tensor) -> torch.Tensor:
    """SLCP: y is 4 iid 2D points given theta=(m1,m2,s1,s2,rho)  (Task 3)."""
    th = theta.double()
    y = y_flat.view(4, 2).to(th.device).double()
    m1 = th[..., 0]
    m2 = th[..., 1]
    s1p = (th[..., 2] ** 2).clamp_min(1e-3)
    s2p = (th[..., 3] ** 2).clamp_min(1e-3)
    rho = torch.tanh(th[..., 4]).clamp(-0.999, 0.999)
    v1 = s1p ** 2
    v2 = s2p ** 2
    c12 = rho * s1p * s2p
    det = (v1 * v2 - c12 * c12).clamp_min(1e-12)
    logdet = torch.log(det)
    z1 = y[None, None, :, 0] - m1[..., None]
    z2 = y[None, None, :, 1] - m2[..., None]
    qf = ((v2[..., None] * z1 * z1)
          - (2.0 * c12[..., None] * z1 * z2)
          + (v1[..., None] * z2 * z2)) / det[..., None]
    const2 = 2.0 * math.log(2.0 * math.pi)
    loglik_per = -0.5 * (const2 + logdet[..., None] + qf)
    return loglik_per.sum(dim=-1).to(theta.dtype)

def gaussian_mixture_loglik(theta: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """0.5 N(y; theta, I) + 0.5 N(y; theta, 0.01 I), 2D  (Task 4)."""
    th = theta.double()
    y2 = y.to(th.device).double()[None, None, :]
    diff = y2 - th
    r2 = (diff * diff).sum(-1)
    const2 = 2.0 * math.log(2.0 * math.pi)
    ll1 = -0.5 * (const2 + 0.0 + r2)
    s2 = 0.01
    ll2 = -0.5 * (const2 + 2.0 * math.log(s2) + r2 / s2)
    a = ll1 + math.log(0.5)
    b = ll2 + math.log(0.5)
    m = torch.maximum(a, b)
    loglik = m + torch.log(torch.exp(a - m) + torch.exp(b - m))
    return loglik.to(theta.dtype)

def two_moons_loglik(theta: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Two-moons likelihood with offset +0.25 and shift s(theta), 2D  (Task 5)."""
    th = theta.double()
    yv = y.to(th.device).double()[None, None, :]
    s1 = -(th[..., 0] + th[..., 1]).abs() / math.sqrt(2.0)
    s2 = (-th[..., 0] + th[..., 1]) / math.sqrt(2.0)
    s = torch.stack([s1, s2], dim=-1)
    z = yv - s
    zx = z[..., 0] - 0.25
    zy = z[..., 1]
    rho = torch.sqrt(zx * zx + zy * zy).clamp_min(1e-12)
    phi = torch.atan2(zy, zx)
    mask = (phi > -math.pi / 2) & (phi < math.pi / 2)
    mu_r, sig_r = 0.1, 0.01
    logN = -0.5 * ((rho - mu_r) ** 2 / (sig_r ** 2)) - 0.5 * math.log(2.0 * math.pi * (sig_r ** 2))
    logpdf = logN - (math.log(math.pi) + torch.log(rho))
    logpdf = torch.where(mask, logpdf, torch.full_like(logpdf, -1e300))
    return logpdf.to(theta.dtype)

FUNCTIONS = {
    'gaussian_linear': gaussian_linear_loglik,
    'gaussian_linear_uniform': gaussian_linear_loglik,
    'slcp': slcp_loglik,
    'gaussian_mixture': gaussian_mixture_loglik,
    'two_moons': two_moons_loglik,
}

class Likelihood:
    def __init__(self, ident, model, observation):
        self.ident, self.model, self.observation = ident, model, observation
        self.calls = 0
        self.closed = False
    def __call__(self, theta):
        if self.closed:
            raise RuntimeError('likelihood already closed')
        self.calls += theta.numel() // theta.shape[-1]
        ll = FUNCTIONS[self.model](theta, self.observation).double()
        return torch.nan_to_num(ll, nan=-1e300, neginf=-1e300, posinf=1e300)
    def finish(self):
        self.closed = True
        return self.calls
