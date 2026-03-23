"""Leaf modules: Linear, LayerNorm, SwiGLU, Identity, Sigmoid, Sequential."""

import jax
import jax.numpy as jnp
import equinox as eqx
import einops

from jaxtyping import Array, Float

from jproteina_complexa.backend import AbstractFromTorch, from_torch


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


# ---- Register converters (only when torch is available) ----

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
