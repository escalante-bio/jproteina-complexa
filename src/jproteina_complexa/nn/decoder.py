"""DecoderTransformer — JAX/Equinox translation."""

import jax.numpy as jnp
import equinox as eqx
from einops import rearrange

from jproteina_complexa.backend import from_torch
from jproteina_complexa.constants import RESTYPE_ATOM37_MASK as _RESTYPE_ATOM37_MASK_NP
from jproteina_complexa.types import DecoderBatch, DecoderOutput
from jproteina_complexa.nn.primitives import Sequential
from jproteina_complexa.nn.transformer import TransformerStack
from jproteina_complexa.nn.features import DecoderSeqFeatures, DecoderPairFeatures

RESTYPE_ATOM37_MASK = jnp.array(_RESTYPE_ATOM37_MASK_NP, dtype=jnp.bool_)


class DecoderTransformer(eqx.Module):
    seq_features: DecoderSeqFeatures
    pair_features: DecoderPairFeatures
    trunk: TransformerStack
    logit_linear: Sequential
    struct_linear: Sequential
    abs_coors: bool = False

    @classmethod
    def from_torch(cls, pt):
        layers = [from_torch(l) for l in pt.transformer_layers]
        return cls(
            seq_features=DecoderSeqFeatures(linear=from_torch(pt.init_repr_factory.linear_out)),
            pair_features=DecoderPairFeatures(linear=from_torch(pt.pair_rep_factory.linear_out)),
            trunk=TransformerStack.from_layers(layers),
            logit_linear=from_torch(pt.logit_linear),
            struct_linear=from_torch(pt.struct_linear),
            abs_coors=pt.abs_coors,
        )

    def __call__(self, batch: DecoderBatch) -> DecoderOutput:
        mask = batch.mask.astype(jnp.float32)
        b, n = batch.mask.shape
        c = jnp.zeros((b, n, self.trunk.cond_dim))

        seqs = self.trunk(self.seq_features(batch), self.pair_features(batch), c, mask)

        logits = self.logit_linear(seqs) * mask[..., None]

        coors = rearrange(self.struct_linear(seqs) * mask[..., None], "b n (a t) -> b n a t", a=37, t=3)
        if self.abs_coors:
            coors = coors.at[..., 1, :].set(batch.ca_coors_nm)
        else:
            coors = coors.at[..., 1, :].set(jnp.zeros_like(batch.ca_coors_nm))
            coors = coors + batch.ca_coors_nm[:, :, None, :]

        aatype = jnp.argmax(logits, axis=-1) * batch.mask.astype(jnp.int32)

        return DecoderOutput(
            coors_nm=coors,
            seq_logits=logits,
            aatype=aatype,
            atom_mask=RESTYPE_ATOM37_MASK[aatype] * batch.mask[..., None],
            mask=batch.mask,
        )
