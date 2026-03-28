"""Reusable sub-transformer building blocks: primitives, adaptive norm, transition, attention."""

import jax
import jax.numpy as jnp
import equinox as eqx
import einops
from einops import rearrange
from jaxtyping import Array, Float

from jproteina_complexa.backend import AbstractFromTorch, from_torch


# ---- Primitives ----

class Linear(AbstractFromTorch):
    weight: Float[Array, "Out In"]
    bias: Float[Array, "Out"] | None = None

    def __call__(self, x):
        o = einops.einsum(x, self.weight, "... In, Out In -> ... Out")
        if self.bias is not None:
            o = o + jnp.broadcast_to(self.bias, o.shape)
        return o


class LayerNorm(eqx.Module):
    weight: Float[Array, "D"] | None
    bias: Float[Array, "D"] | None
    eps: float = 1e-5

    @classmethod
    def from_torch(cls, model):
        w = from_torch(model.weight) if model.elementwise_affine else None
        b = from_torch(model.bias) if model.elementwise_affine else None
        return cls(weight=w, bias=b, eps=model.eps)

    def __call__(self, x):
        mean = x.mean(axis=-1, keepdims=True)
        var = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
        x = (x - mean) * jax.lax.rsqrt(var + self.eps)
        if self.weight is not None:
            x = x * self.weight
        if self.bias is not None:
            x = x + self.bias
        return x


class Identity(eqx.Module):
    def __call__(self, x):
        return x


class Sigmoid(eqx.Module):
    def __call__(self, x):
        return jax.nn.sigmoid(x)


class SwiGLU(eqx.Module):
    def __call__(self, x):
        x, gates = jnp.split(x, 2, axis=-1)
        return jax.nn.silu(gates) * x


class Sequential(eqx.Module):
    _modules: dict[str, eqx.Module]

    @classmethod
    def from_torch(cls, model):
        return cls(_modules={
            name: from_torch(child) for name, child in model.named_children()
        })

    def __call__(self, x):
        for idx in range(len(self._modules)):
            x = self._modules[str(idx)](x)
        return x


def _register_primitives():
    try:
        import torch
    except ImportError:
        return
    from_torch.register(torch.nn.Linear, Linear.from_torch)
    from_torch.register(torch.nn.LayerNorm, LayerNorm.from_torch)
    from_torch.register(torch.nn.Identity, lambda _: Identity())
    from_torch.register(torch.nn.Sigmoid, lambda _: Sigmoid())
    from_torch.register(torch.nn.Sequential, Sequential.from_torch)
    from_torch.register(torch.nn.ModuleList, lambda m: [from_torch(c) for c in m])

_register_primitives()


# ---- Adaptive normalization ----

class AdaptiveLayerNorm(AbstractFromTorch):
    norm: LayerNorm
    norm_cond: LayerNorm
    to_gamma: Sequential
    to_beta: Linear

    def __call__(self, x, cond, mask):
        normed = self.norm(x)
        normed_cond = self.norm_cond(cond)
        gamma = self.to_gamma(normed_cond)
        beta = self.to_beta(normed_cond)
        out = normed * gamma + beta
        return out * mask[..., None]


class AdaptiveOutputScale(AbstractFromTorch):
    to_adaln_zero_gamma: Sequential

    def __call__(self, x, cond, mask):
        gamma = self.to_adaln_zero_gamma(cond)
        return x * gamma * mask[..., None]


# ---- Transition ----

class Transition(AbstractFromTorch):
    swish_linear: Sequential
    linear_out: Linear

    def __call__(self, x, mask):
        x = self.linear_out(self.swish_linear(x))
        return x * mask[..., None]


class TransitionADALN(AbstractFromTorch):
    adaln: AdaptiveLayerNorm
    transition: Transition
    scale_output: AdaptiveOutputScale

    def __call__(self, x, cond, mask):
        x = self.adaln(x, cond, mask)
        x = self.transition(x, mask)
        x = self.scale_output(x, cond, mask)
        return x * mask[..., None]


# ---- Attention ----

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
