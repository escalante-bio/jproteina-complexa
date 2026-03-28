"""Top-level model classes: encoder, decoder, and denoiser."""

import jax
import jax.numpy as jnp
import equinox as eqx
from einops import rearrange

from jproteina_complexa.backend import from_torch
from jproteina_complexa.constants import RESTYPE_ATOM37_MASK as _RESTYPE_ATOM37_MASK_NP
from jproteina_complexa.types import (
    EncoderBatch, EncoderOutput,
    DecoderBatch, DecoderOutput,
    DenoiserBatch, DenoiserOutput,
)
from jproteina_complexa.nn.layers import Sequential, Identity, Transition
from jproteina_complexa.nn.transformer import TransformerStack
from jproteina_complexa.nn.features import (
    EncoderSeqFeatures, EncoderPairFeatures,
    DecoderSeqFeatures, DecoderPairFeatures,
    DenoiserSeqFeatures, DenoiserCondFeatures,
    DenoiserPairFeatures, DenoiserPairCondFeatures,
    PairReprBuilder, TargetConcatFeatures,
    bin_and_one_hot, relative_seq_sep,
)

RESTYPE_ATOM37_MASK = jnp.array(_RESTYPE_ATOM37_MASK_NP, dtype=jnp.bool_)


# ---- Encoder ----

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
        (n,) = batch.mask.shape
        c = jnp.zeros((n, self.trunk.cond_dim))
        seqs = self.trunk(self.seq_features(batch), self.pair_features(batch), c, mask)
        flat = self.latent_projection(seqs) * mask[..., None]
        return jnp.split(flat, 2, axis=-1)

    def __call__(self, batch: EncoderBatch, *, key) -> EncoderOutput:
        mean, log_scale = self._trunk(batch)
        mask = batch.mask.astype(jnp.float32)
        z = mean + jax.random.normal(key, log_scale.shape) * jnp.exp(log_scale)
        return EncoderOutput(mean=mean, log_scale=log_scale, z_latent=self.ln_z(z) * mask[..., None])

    def encode_deterministic(self, batch: EncoderBatch) -> EncoderOutput:
        mean, log_scale = self._trunk(batch)
        mask = batch.mask.astype(jnp.float32)
        return EncoderOutput(mean=mean, log_scale=log_scale, z_latent=self.ln_z(mean) * mask[..., None])


# ---- Decoder ----

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
        (n,) = batch.mask.shape
        c = jnp.zeros((n, self.trunk.cond_dim))

        seqs = self.trunk(self.seq_features(batch), self.pair_features(batch), c, mask)

        logits = self.logit_linear(seqs) * mask[..., None]

        ca_nm = batch.ca_coors * 0.1
        coors = rearrange(self.struct_linear(seqs) * mask[..., None], "n (a t) -> n a t", a=37, t=3)
        if self.abs_coors:
            coors = coors.at[..., 1, :].set(ca_nm)
        else:
            coors = coors.at[..., 1, :].set(jnp.zeros_like(ca_nm))
            coors = coors + ca_nm[:, None, :]

        aatype = jnp.argmax(logits, axis=-1) * batch.mask.astype(jnp.int32)

        return DecoderOutput(
            coors=coors * 10.0,
            seq_logits=logits,
            aatype=aatype,
            atom_mask=RESTYPE_ATOM37_MASK[aatype] * batch.mask[..., None],
            mask=batch.mask,
        )


# ---- Denoiser ----

class LocalLatentsTransformer(eqx.Module):
    seq_features: DenoiserSeqFeatures
    cond_features: DenoiserCondFeatures
    pair_repr_builder: PairReprBuilder
    transition_c_1: Transition
    transition_c_2: Transition
    trunk: TransformerStack
    local_latents_linear: Sequential
    ca_linear: Sequential

    concat_features: TargetConcatFeatures | None = None
    concat_pair_linear: eqx.Module | None = None
    concat_pair_ln: eqx.Module | None = None

    use_concat: bool = False
    use_advanced_pair: bool = False

    @classmethod
    def from_torch(cls, pt):
        prb_pt = pt.pair_repr_builder
        pair_cond = None
        adaln = None
        if prb_pt.cond_factory is not None:
            pair_cond = DenoiserPairCondFeatures(
                linear=from_torch(prb_pt.cond_factory.linear_out),
                ln=from_torch(prb_pt.cond_factory.ln_out),
            )
            adaln = from_torch(prb_pt.adaln)

        use_concat = pt.use_concat
        use_advanced_pair = getattr(pt, "use_advanced_pair", False)
        concat = None
        cpair_linear = None
        cpair_ln = None
        if use_concat and pt.concat_factory is not None:
            concat = TargetConcatFeatures(
                linear=from_torch(pt.concat_factory.linear_out),
                ln=from_torch(pt.concat_factory.ln_out),
            )
        if use_advanced_pair and hasattr(pt, "concat_pair_factory"):
            cpair_linear = from_torch(pt.concat_pair_factory.linear_out)
            cpair_ln = from_torch(pt.concat_pair_factory.ln_out)

        return cls(
            seq_features=DenoiserSeqFeatures(linear=from_torch(pt.init_repr_factory.linear_out)),
            cond_features=DenoiserCondFeatures(linear=from_torch(pt.cond_factory.linear_out)),
            pair_repr_builder=PairReprBuilder(
                pair_features=DenoiserPairFeatures(
                    linear=from_torch(prb_pt.init_repr_factory.linear_out),
                    ln=from_torch(prb_pt.init_repr_factory.ln_out),
                ),
                pair_cond=pair_cond,
                adaln=adaln,
            ),
            transition_c_1=from_torch(pt.transition_c_1),
            transition_c_2=from_torch(pt.transition_c_2),
            trunk=TransformerStack.from_layers([from_torch(l) for l in pt.transformer_layers]),
            local_latents_linear=from_torch(pt.local_latents_linear),
            ca_linear=from_torch(pt.ca_linear),
            concat_features=concat,
            concat_pair_linear=cpair_linear,
            concat_pair_ln=cpair_ln,
            use_concat=use_concat,
            use_advanced_pair=use_advanced_pair,
        )

    def __call__(self, batch: DenoiserBatch) -> DenoiserOutput:
        mask_f = batch.mask.astype(jnp.float32)
        orig_mask = mask_f

        c = self.cond_features(batch)
        c = self.transition_c_2(self.transition_c_1(c, mask_f), mask_f)

        seqs = self.seq_features(batch)
        n_orig = seqs.shape[0]

        n_concat = 0
        if self.use_concat and self.concat_features is not None:
            concat_feats, concat_mask = self.concat_features(batch)
            n_concat = concat_feats.shape[0]
            if n_concat > 0:
                seqs = jnp.concatenate([seqs, concat_feats], axis=0)
                mask_f = jnp.concatenate([mask_f, concat_mask.astype(jnp.float32)], axis=0)
                c = jnp.concatenate([c, jnp.zeros((n_concat, c.shape[-1]))], axis=0)

        pair_rep = self.pair_repr_builder(batch)
        if n_concat > 0 and self.use_advanced_pair and self.concat_pair_linear is not None:
            pair_rep = self._extend_pair(batch, pair_rep, n_orig, n_concat)
        elif n_concat > 0:
            n_ext, d = seqs.shape[0], pair_rep.shape[-1]
            pair_rep = jnp.concatenate([pair_rep, jnp.zeros((n_concat, n_orig, d))], axis=0)
            pair_rep = jnp.concatenate([pair_rep, jnp.zeros((n_ext, n_concat, d))], axis=1)

        seqs = self.trunk(seqs, pair_rep, c, mask_f)

        latents = self.local_latents_linear(seqs) * mask_f[..., None]
        ca = self.ca_linear(seqs) * mask_f[..., None]

        if n_concat > 0:
            latents = latents[:n_orig] * orig_mask[..., None]
            ca = ca[:n_orig] * orig_mask[..., None]

        return DenoiserOutput(bb_ca=ca, local_latents=latents)

    def _pairwise_bb_dists(self, src, tgt_coords, has_cb, d_bins):
        """Pairwise binned distances from src to target N/CA/C/CB atoms.

        Args:
            src: [n_src, 1, 3] — source coords (unsqueezed for broadcasting)
            tgt_coords: [n_tgt, 37, 3] — target atom37 coords
            has_cb: [n_tgt] — whether each target residue has CB
            d_bins: 1-D bin edges
        Returns:
            [n_src, n_tgt, 4*(n_bins+1)] — binned distances to N, CA, C, CB
        """
        dists = jnp.sqrt(jnp.sum(jnp.square(src[..., None, :] - tgt_coords[None, :, :4, :]), axis=-1) + 1e-10)
        dists = dists.at[..., 3].multiply(has_cb[None, :])
        binned = bin_and_one_hot(dists, d_bins)
        return binned.reshape(*binned.shape[:2], -1)

    def _extend_pair(self, batch, pair_rep, n_orig, n_concat):
        pair_dim = self.concat_pair_linear.weight.shape[0]
        n_ext = n_orig + n_concat
        tgt = batch.target
        tgt_nm = tgt.coords * 0.1

        d_bins = jnp.linspace(0.1, 2.0, 20)
        has_cb_target = tgt.atom_mask[:, 3]

        h_tgt = tgt.hotspot_mask if tgt.hotspot_mask is not None else jnp.zeros((n_concat,), dtype=jnp.bool_)

        # Upper-right block: binder->target distances (bb_ca is already nm from flow matching)
        binder_ca = batch.x_t.bb_ca[:, None, :]
        ur_dists = self._pairwise_bb_dists(binder_ca, tgt_nm, has_cb_target, d_bins)
        ur_sep = jnp.zeros((n_orig, n_concat, 127))
        ur_hotspot = jnp.broadcast_to(h_tgt[None, :], (n_orig, n_concat)).astype(jnp.float32)[..., None]
        ur_raw = jnp.concatenate([ur_sep, ur_dists, jnp.ones((n_orig, n_concat, 1)), ur_hotspot], axis=-1)
        ur_proj = self.concat_pair_ln(self.concat_pair_linear(ur_raw))

        # Lower-right block: target->target distances
        t_ca = tgt_nm[:, 1, :][:, None, :]
        lr_dists = self._pairwise_bb_dists(t_ca, tgt_nm, has_cb_target, d_bins)
        lr_sep = relative_seq_sep(n_concat, 127)
        lr_hotspot = (h_tgt[:, None] | h_tgt[None, :]).astype(jnp.float32)[..., None]
        lr_raw = jnp.concatenate([lr_sep, lr_dists, jnp.zeros((n_concat, n_concat, 1)), lr_hotspot], axis=-1)
        lr_proj = self.concat_pair_ln(self.concat_pair_linear(lr_raw))

        ext = jnp.zeros((n_ext, n_ext, pair_dim))
        ext = ext.at[:n_orig, :n_orig].set(pair_rep)
        ext = ext.at[:n_orig, n_orig:].set(ur_proj)
        ext = ext.at[n_orig:, :n_orig].set(jnp.swapaxes(ur_proj, 0, 1))
        ext = ext.at[n_orig:, n_orig:].set(lr_proj)
        return ext
