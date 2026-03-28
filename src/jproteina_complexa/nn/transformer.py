"""Transformer blocks and scanned transformer stack."""

import jax
import jax.numpy as jnp
import equinox as eqx

from jproteina_complexa.backend import from_torch
from jproteina_complexa.nn.layers import MultiHeadBiasedAttentionADALN_MM, TransitionADALN


class MultiheadAttnAndTransition(eqx.Module):
    mhba: MultiHeadBiasedAttentionADALN_MM
    transition: TransitionADALN
    parallel: bool = False
    residual_mha: bool = True
    residual_transition: bool = True

    @classmethod
    def from_torch(cls, model):
        return cls(
            **{n: from_torch(c) for n, c in model.named_children()},
            parallel=model.parallel,
            residual_mha=model.residual_mha,
            residual_transition=model.residual_transition,
        )

    def __call__(self, x, pair_rep, cond, mask):
        x = x * mask[..., None]
        if self.parallel:
            x = self._apply_mha(x, pair_rep, cond, mask) + self._apply_transition(x, cond, mask)
        else:
            x = self._apply_mha(x, pair_rep, cond, mask)
            x = self._apply_transition(x, cond, mask)
        return x * mask[..., None]

    def _apply_mha(self, x, pair_rep, cond, mask):
        x_attn = self.mhba(x, pair_rep, cond, mask)
        if self.residual_mha:
            x_attn = x_attn + x
        return x_attn * mask[..., None]

    def _apply_transition(self, x, cond, mask):
        x_tr = self.transition(x, cond, mask)
        if self.residual_transition:
            x_tr = x_tr + x
        return x_tr * mask[..., None]


class TransformerStack(eqx.Module):
    """N identical transformer layers executed via jax.lax.scan."""
    stacked_params: MultiheadAttnAndTransition  # arrays stacked along dim 0
    static: MultiheadAttnAndTransition           # non-array structure (shared)
    nlayers: int
    cond_dim: int

    @classmethod
    def from_layers(cls, layers: list[MultiheadAttnAndTransition]):
        n = len(layers)
        cond_dim = layers[0].transition.adaln.norm_cond.weight.shape[0]
        _, static = eqx.partition(layers[0], eqx.is_inexact_array)
        stacked = jax.tree.map(
            lambda *vs: jnp.stack(vs, axis=0),
            *[eqx.filter(layer, eqx.is_inexact_array) for layer in layers],
        )
        return cls(stacked_params=stacked, static=static, nlayers=n, cond_dim=cond_dim)

    def __call__(self, seqs, pair_rep, cond, mask):
        @jax.checkpoint
        def body(carry, params):
            block = eqx.combine(params, self.static)
            seqs = block(carry, pair_rep, cond, mask)
            return seqs, None

        seqs, _ = jax.lax.scan(body, seqs, self.stacked_params)
        return seqs
