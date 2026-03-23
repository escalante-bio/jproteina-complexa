"""AdaptiveLayerNorm and AdaptiveOutputScale."""

import jax
import jax.numpy as jnp
import equinox as eqx

from jproteina_complexa.backend import AbstractFromTorch, from_torch
from jproteina_complexa.nn.primitives import LayerNorm, Linear, Sequential, Sigmoid


class AdaptiveLayerNorm(AbstractFromTorch):
    norm: LayerNorm          # LN without affine (elementwise_affine=False)
    norm_cond: LayerNorm     # LN on conditioning
    to_gamma: Sequential     # Linear → Sigmoid
    to_beta: Linear          # Linear (no bias)

    def __call__(self, x, cond, mask):
        normed = self.norm(x)
        normed_cond = self.norm_cond(cond)
        gamma = self.to_gamma(normed_cond)
        beta = self.to_beta(normed_cond)
        out = normed * gamma + beta
        return out * mask[..., None]


class AdaptiveOutputScale(AbstractFromTorch):
    to_adaln_zero_gamma: Sequential  # Linear → Sigmoid

    def __call__(self, x, cond, mask):
        gamma = self.to_adaln_zero_gamma(cond)
        return x * gamma * mask[..., None]
