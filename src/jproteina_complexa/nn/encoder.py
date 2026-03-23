"""EncoderTransformer — JAX/Equinox translation."""

import jax
import jax.numpy as jnp
import equinox as eqx

from jproteina_complexa.backend import from_torch
from jproteina_complexa.types import EncoderBatch, EncoderOutput
from jproteina_complexa.nn.primitives import Sequential, Identity
from jproteina_complexa.nn.transformer import TransformerStack
from jproteina_complexa.nn.features import EncoderSeqFeatures, EncoderPairFeatures


class EncoderTransformer(eqx.Module):
    seq_features: EncoderSeqFeatures
    pair_features: EncoderPairFeatures
    trunk: TransformerStack
    latent_projection: Sequential
    ln_z: Identity

    @classmethod
    def from_torch(cls, pt):
        layers = [from_torch(l) for l in pt.transformer_layers]
        return cls(
            seq_features=EncoderSeqFeatures(linear=from_torch(pt.init_repr_factory.linear_out)),
            pair_features=EncoderPairFeatures(
                linear=from_torch(pt.pair_rep_factory.linear_out),
                ln=from_torch(pt.pair_rep_factory.ln_out),
            ),
            trunk=TransformerStack.from_layers(layers),
            latent_projection=from_torch(pt.latent_decoder_mean_n_log_scale),
            ln_z=from_torch(pt.ln_z),
        )

    def _trunk(self, batch: EncoderBatch):
        mask = batch.mask.astype(jnp.float32)
        b, n = batch.mask.shape
        c = jnp.zeros((b, n, self.trunk.cond_dim))
        seqs = self.trunk(self.seq_features(batch), self.pair_features(batch), c, mask)
        flat = self.latent_projection(seqs) * mask[..., None]
        return jnp.split(flat * mask[..., None], 2, axis=-1)

    def __call__(self, batch: EncoderBatch, *, key) -> EncoderOutput:
        mean, log_scale = self._trunk(batch)
        mask = batch.mask.astype(jnp.float32)
        z = mean + jax.random.normal(key, log_scale.shape) * jnp.exp(log_scale)
        return EncoderOutput(mean=mean, log_scale=log_scale, z_latent=self.ln_z(z) * mask[..., None])

    def encode_deterministic(self, batch: EncoderBatch) -> EncoderOutput:
        mean, log_scale = self._trunk(batch)
        mask = batch.mask.astype(jnp.float32)
        return EncoderOutput(mean=mean, log_scale=log_scale, z_latent=self.ln_z(mean) * mask[..., None])
