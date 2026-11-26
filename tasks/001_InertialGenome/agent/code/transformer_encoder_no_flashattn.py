import torch
from torch import nn
from layers import RMSNorm, DropPath, FeedForward
from attention_no_flashattn import Attention


class TransformerLayer(nn.Module):

    def __init__(
        self,
        dim,
        heads,
        mlp_dim,
        residual_dropout=0.1,
        attn_dropout=0.1,
        rope=None,
        use_qk_layernorm=False,
        apply_selective_rope="selective",
    ):
        super().__init__()

        self.attn_norm = RMSNorm(dim)
        self.attn = Attention(
            dim,
            num_heads=heads,
            dropout=attn_dropout,
            bias=False,
            rope=rope,
            use_qk_layernorm=use_qk_layernorm,
            apply_selective_rope=apply_selective_rope,
        )
        self.ffn_norm = RMSNorm(dim)
        self.ffn = FeedForward(
            dim,
            mlp_dim,
            256,
        )

        self.dropout = DropPath(residual_dropout)

    def forward(
        self,
        x,
        key_padding_mask,
        dist_bias=None,
        kv_cache=None,
    ):
        x = x + self.dropout(
            self.attn(self.attn_norm(x), key_padding_mask=key_padding_mask, dist_bias=dist_bias, kv_cache=kv_cache)
        )
        x = x + self.dropout(self.ffn(self.ffn_norm(x)))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        mlp_dim,
        residual_dropout=0.1,
        attn_dropout=0.1,
        rope=None,
        use_qk_layernorm=False,
        apply_selective_rope="selective",
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        droppath_probs = [x.item() for x in torch.linspace(0, residual_dropout, depth)]
        for i in range(depth):
            self.layers.append(
                TransformerLayer(
                    dim,
                    heads,
                    mlp_dim,
                    droppath_probs[i],
                    attn_dropout,
                    rope=rope,
                    use_qk_layernorm=use_qk_layernorm,
                    apply_selective_rope=apply_selective_rope,
                )
            )

    def forward(
        self,
        x,
        key_padding_mask,
        dist_bias=None,
        kv_cache=None,
    ):
        for i, layer in enumerate(self.layers):
            cur_kv_cache = None if kv_cache is None else kv_cache[i]
            x = layer(
                x,
                key_padding_mask,
                dist_bias=dist_bias,
                kv_cache=cur_kv_cache,
            )
            if kv_cache is not None:
                kv_cache[i] = cur_kv_cache
        return x
