import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from layers import Linear, RMSNorm

class Attention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.1,
        bias=False,
        rope=None,
        use_qk_layernorm=False,
        apply_selective_rope="selective",
    ):
        super().__init__()
        self.embed_dim = embed_dim

        self.num_heads = num_heads
        self.dropout = dropout

        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        
        self.apply_selective_rope = apply_selective_rope
        if self.apply_selective_rope == "selective" or self.apply_selective_rope == "separate":
            self.rope_dim = embed_dim // 2
        elif self.apply_selective_rope == "full":
            self.rope_dim = embed_dim
            
        self.q_proj = Linear(self.rope_dim, self.rope_dim, bias=bias, init="bert")
        self.k_proj = Linear(self.rope_dim, self.rope_dim, bias=bias, init="bert")
        self.v_proj = Linear(self.rope_dim, self.rope_dim, bias=bias, init="bert")
        
        if self.apply_selective_rope == "separate":
            nystrom_dim = self.embed_dim - self.rope_dim
            self.q_proj_separate = Linear(nystrom_dim, nystrom_dim, bias=bias, init="bert")
            self.k_proj_separate = Linear(nystrom_dim, nystrom_dim, bias=bias, init="bert")
            self.v_proj_separate = Linear(nystrom_dim, nystrom_dim, bias=bias, init="bert")
        
        self.out_proj = Linear(embed_dim, embed_dim, bias=bias, init="final")
        self.rope = rope
        self.scale = self.head_dim**-0.5
        self.use_qk_layernorm = use_qk_layernorm

        
        if self.use_qk_layernorm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

    def forward(
        self,
        query: Tensor,
        key_padding_mask: Tensor,
        dist_bias: Tensor = None,
        kv_cache: dict = None,
    ) -> Tensor:

        # === 1. A linear map gives Q / K / V ===
        if self.apply_selective_rope == "selective" or self.apply_selective_rope == "separate":
            # Slice-first, Only the first half of the RoPE is applied
            query_rope = query[..., : self.embed_dim // 2]
            query_nystrom = query[..., self.embed_dim // 2 :]

            q_rope_proj = self.q_proj(query_rope)
            k_rope_proj = self.k_proj(query_rope)
            v_rope_proj = self.v_proj(query_rope)

            query_size = query.size()
            rope_head_dim = self.embed_dim  // self.num_heads

            q_rope = q_rope_proj.view(*query_size[:-1], self.num_heads // 2, rope_head_dim)
            k_rope = k_rope_proj.view(*query_size[:-1], self.num_heads // 2, rope_head_dim)
            v_rope = v_rope_proj.view(*query_size[:-1], self.num_heads // 2, rope_head_dim)

            if self.rope is not None:
                q_rope, k_rope = self.rope.apply_qk(q_rope, k_rope)

            nystrom_head_dim = self.embed_dim // self.num_heads
            if self.apply_selective_rope == "selective":
                query_ny = query_nystrom.view(*query_size[:-1], self.num_heads // 2, nystrom_head_dim)

                q = torch.cat([q_rope, query_ny], dim=-2)
                k = torch.cat([k_rope, query_ny], dim=-2)
                v = torch.cat([v_rope, query_ny], dim=-2)
            elif self.apply_selective_rope == "separate":
                q_nystrom_proj = self.q_proj_separate(query_nystrom)
                k_nystrom_proj = self.k_proj_separate(query_nystrom)
                v_nystrom_proj = self.v_proj_separate(query_nystrom)
                
                q_nystrom = q_nystrom_proj.view(*query_size[:-1], self.num_heads // 2, nystrom_head_dim)
                k_nystrom = k_nystrom_proj.view(*query_size[:-1], self.num_heads // 2, nystrom_head_dim)
                v_nystrom = v_nystrom_proj.view(*query_size[:-1], self.num_heads // 2, nystrom_head_dim)
                
                q = torch.cat([q_rope, q_nystrom], dim=-2)
                k = torch.cat([k_rope, k_nystrom], dim=-2)
                v = torch.cat([v_rope, v_nystrom], dim=-2)
                
        elif self.apply_selective_rope == "full":
            # Standard RoPE Branch (applied to all features) - "full" mode
            query_size = query.size()
            assert query_size[-1] == self.embed_dim
            q, k, v = (
                self.q_proj(query).view(*query_size[:-1], self.num_heads, -1),
                self.k_proj(query).view(*query_size[:-1], self.num_heads, -1),
                self.v_proj(query).view(*query_size[:-1], self.num_heads, -1),
            )

            if self.rope is not None:
                q, k = self.rope.apply_qk(q, k)

        # LayerNorm on Q/K
        if self.use_qk_layernorm:
            q = self.q_norm(q)
            k = self.k_norm(k)


        if kv_cache is not None:
            out = self._forward_incremental(q, k, v, kv_cache)
        else:
            # B×T×C
            assert query.dim() == 3, "query : (B, T, C)"
            out = self._forward_batch(q, k, v, key_padding_mask=key_padding_mask, dist_bias=dist_bias)

        out = out.view(*query.size()[:-1], -1)  
        return self.out_proj(out)

    def _forward_batch(self, q: Tensor, k: Tensor, v: Tensor, key_padding_mask: Tensor, dist_bias: Tensor = None) -> Tensor:
        """shape(B, T, H, D) Q/K/V"""
        B, T, H, D = q.shape
        # (B, H, T, D)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        att_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, T, T)

        if dist_bias is not None:
            att_scores = att_scores + dist_bias

        if key_padding_mask is not None:
            att_scores = att_scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )

        attn = torch.softmax(att_scores, dim=-1)
        if self.training and self.dropout > 0.0:
            attn = F.dropout(attn, p=self.dropout, training=True)

        y = torch.matmul(attn, v)  # (B, H, T, D)
        y = y.permute(0, 2, 1, 3).contiguous()  # (B, T, H, D)
        return y

    def _forward_incremental(self, q: Tensor, k: Tensor, v: Tensor, kv_cache: dict) -> Tensor:
        """ Attention implementation of incremental decoding (1 token input at a time) .q shape [B, 1, H, D]"""
        # (B, H, 1, D)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if "k" in kv_cache:
            k_cached = kv_cache["k"]
            v_cached = kv_cache["v"]
            k = torch.cat([k_cached, k], dim=2)
            v = torch.cat([v_cached, v], dim=2)

        kv_cache["k"] = k
        kv_cache["v"] = v
        kv_cache["step"] = kv_cache.get("step", 0) + 1

        # seq_q=1，seq_k=L
        q_scaled = q * self.scale  # (B, H, 1, D)
        att_scores = torch.matmul(q_scaled, k.transpose(-1, -2))  # (B, H, 1, L)

        attn = torch.softmax(att_scores, dim=-1)
        y = torch.matmul(attn, v)  # (B, H, 1, D)
        y = y.permute(0, 2, 1, 3)  # (B, 1, H, D)
        return y