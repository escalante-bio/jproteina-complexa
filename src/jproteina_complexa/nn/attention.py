"""Pair-biased attention modules."""

import jax
import jax.numpy as jnp
import equinox as eqx
from einops import rearrange

from jproteina_complexa.backend import from_torch
from jproteina_complexa.nn.primitives import Linear, LayerNorm, Identity
from jproteina_complexa.nn.adaptive import AdaptiveLayerNorm, AdaptiveOutputScale


class PairBiasAttention(eqx.Module):
    node_norm: LayerNorm
    to_qkv: Linear
    to_g: Linear
    to_out_node: Linear
    q_layer_norm: LayerNorm | Identity
    k_layer_norm: LayerNorm | Identity
    to_bias: Linear | None = None
    pair_norm: LayerNorm | None = None
    heads: int = 12
    scale: float = 1.0

    @classmethod
    def from_torch(cls, model):
        return cls(
            **{n: from_torch(c) for n, c in model.named_children()},
            heads=model.heads,
            scale=model.scale,
        )

    def __call__(self, node_feats, pair_feats, mask):
        h = self.heads
        node_feats = self.node_norm(node_feats)
        pair_feats = self.pair_norm(pair_feats) if self.pair_norm is not None else None

        q, k, v = jnp.split(self.to_qkv(node_feats), 3, axis=-1)
        q = self.q_layer_norm(q)
        k = self.k_layer_norm(k)
        g = self.to_g(node_feats)

        # Pair bias: [n, n, pair_dim] -> [n, n, heads] -> [heads, n, n]
        if pair_feats is not None:
            bias = rearrange(self.to_bias(pair_feats), "i j h -> h i j")
        else:
            bias = None

        # Mask -> additive bias
        if mask is not None:
            mask_bias = jnp.where(rearrange(mask, "i j -> () i j"), 0.0, -1e4)
            bias = mask_bias if bias is None else bias + mask_bias

        # Reshape to [n, h, d] for jax.nn.dot_product_attention
        q, k, v, g = (rearrange(t, "n (h d) -> n h d", h=h) for t in (q, k, v, g))

        # Use JAX's dot_product_attention (flash attention on GPU)
        out = jax.nn.dot_product_attention(q, k, v, bias=bias, scale=self.scale)

        # Gate and project: [n, h, d] -> [n, h*d]
        g = rearrange(g, "n h d -> n (h d)")
        out = rearrange(out, "n h d -> n (h d)")
        return self.to_out_node(jax.nn.sigmoid(g) * out)


class MultiHeadBiasedAttentionADALN_MM(eqx.Module):
    adaln: AdaptiveLayerNorm
    mha: PairBiasAttention
    scale_output: AdaptiveOutputScale

    @classmethod
    def from_torch(cls, model):
        return cls(**{n: from_torch(c) for n, c in model.named_children()})

    def __call__(self, x, pair_rep, cond, mask):
        pair_mask = mask[:, None] * mask[None, :]
        x = self.adaln(x, cond, mask)
        x = self.mha(node_feats=x, pair_feats=pair_rep, mask=pair_mask)
        x = self.scale_output(x, cond, mask)
        return x * mask[..., None]
