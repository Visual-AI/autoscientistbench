from typing import Tuple
import torch
from functools import lru_cache


@lru_cache(maxsize=16)
def get_freqs(dim, theta, device):
    freqs = 1.0 / (
        theta ** (torch.arange(0, dim, 2, device=device)[: (dim // 2)].float() / dim)
    )
    return freqs


class RoPEnD(torch.nn.Module):
    def __init__(self, head_dim, theta=10000.0, n=3):
        # we need 4D RoPE if we want to differentiate different trees
        super().__init__()
        self.n = n  # number of axes
        self.head_dim = head_dim
        self.theta = theta
        self.num_angles_per_axis = (self.head_dim // 2) // n
        self.remaining_angles = (self.head_dim // 2) % n
        self.cur_freq_cis = None
        self.repr_str = f"RoPE3D(head_dim={head_dim}, theta={theta})"
        self.encoder_cur_freq_cis = None
    
    def calc_and_cache(self, coords):  # coords: [B, L, 3]
        assert coords.shape[-1] == 3, "Input must be 3D coordinates"
        x, y, z = coords[..., 0], coords[..., 1], coords[..., 2]
        
        base_freqs = get_freqs(2 * self.num_angles_per_axis, self.theta, coords.device)
        
        # Compute angles per axis
        theta_x = torch.einsum('bl,d->bld', x, base_freqs)
        theta_y = torch.einsum('bl,d->bld', y, base_freqs)
        theta_z = torch.einsum('bl,d->bld', z, base_freqs)
        
        angles = torch.cat([theta_x, theta_y, theta_z], dim=-1)  # [B, L, 3 * D_axis]
        
        if self.remaining_angles > 0:
            angles = torch.cat([
                angles,
                torch.zeros((*angles.shape[:-1], self.remaining_angles), device=angles.device)
            ], dim=-1)
        
        freqs_cis = torch.polar(torch.ones_like(angles), angles)
        self.cur_freq_cis = freqs_cis.unsqueeze(-2)  # [B, L, 1, head_dim//2]
    # def calc_and_cache(self, positions):
    #     freqs = get_freqs(self.num_angles_per_axis * 2, self.theta, positions.device)
    #     freqs = torch.outer(positions.flatten(), freqs).view(*positions.shape[:-1], -1)
    #     freqs = torch.cat(
    #         [
    #             freqs,
    #             torch.zeros(
    #                 (*freqs.shape[:-1], self.remaining_angles), device=freqs.device
    #             ),
    #         ],
    #         dim=-1,
    #     )
    #     freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    #     self.cur_freq_cis = freqs_cis.view(*positions.shape[:-1], 1, -1)

    def apply_qk(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        def apply(x, cur_freq_cis):
            tx = torch.view_as_complex(x.view(*x.shape[:-1], -1, 2))
            return torch.view_as_real(tx * cur_freq_cis).view(*x.shape)

        return apply(xq.float(), self.cur_freq_cis).type_as(xq), apply(
            xk.float(), self.cur_freq_cis
        ).type_as(xk)

    def __repr__(self):
        return self.repr_str
