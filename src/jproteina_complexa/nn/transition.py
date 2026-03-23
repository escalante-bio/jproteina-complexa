"""Transition and TransitionADALN modules."""

import jax.numpy as jnp
import equinox as eqx

from jproteina_complexa.backend import AbstractFromTorch, from_torch
from jproteina_complexa.nn.primitives import Linear, Sequential
from jproteina_complexa.nn.adaptive import AdaptiveLayerNorm, AdaptiveOutputScale


class Transition(AbstractFromTorch):
    swish_linear: Sequential   # Linear(dim, dim_inner*2) → SwiGLU
    linear_out: Linear         # Linear(dim_inner, dim)

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
