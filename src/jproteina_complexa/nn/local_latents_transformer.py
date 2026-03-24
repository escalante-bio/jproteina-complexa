"""LocalLatentsTransformer — the denoising model for flow matching."""

import jax.numpy as jnp
import equinox as eqx

from jproteina_complexa.backend import from_torch
from jproteina_complexa.types import DenoiserBatch, DenoiserOutput
from jproteina_complexa.nn.primitives import Sequential
from jproteina_complexa.nn.transition import Transition
from jproteina_complexa.nn.transformer import MultiheadAttnAndTransition, TransformerStack
from jproteina_complexa.nn.features import (
    DenoiserSeqFeatures, DenoiserCondFeatures,
    DenoiserPairFeatures, DenoiserPairCondFeatures,
    PairReprBuilder, TargetConcatFeatures,
    bin_and_one_hot, relative_seq_sep,
)


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
        # src: [n_src, 1, 3] -> [n_src, 1, 1, 3] to broadcast with [1, n_tgt, 4, 3]
        # result: [n_src, n_tgt, 4]
        dists = jnp.sqrt(jnp.sum(jnp.square(src[..., None, :] - tgt_coords[None, :, :4, :]), axis=-1) + 1e-10)
        # Zero CB distances for residues without CB
        dists = dists.at[..., 3].multiply(has_cb[None, :])
        # Bin each atom channel and concatenate: [n_src, n_tgt, 4*(n_bins+1)]
        binned = bin_and_one_hot(dists, d_bins)  # [n_src, n_tgt, 4, n_bins+1]
        return binned.reshape(*binned.shape[:2], -1)

    def _extend_pair(self, batch, pair_rep, n_orig, n_concat):
        pair_dim = self.concat_pair_linear.weight.shape[0]
        n_ext = n_orig + n_concat
        tgt = batch.target

        d_bins = jnp.linspace(0.1, 2.0, 20)
        has_cb_target = tgt.atom_mask[:, 3]  # [n_concat]

        # Upper-right block: binder->target distances
        binder_ca = batch.x_t.bb_ca[:, None, :]  # [n_orig, 1, 3]
        ur_dists = self._pairwise_bb_dists(binder_ca, tgt.coords, has_cb_target, d_bins)
        ur_sep = jnp.zeros((n_orig, n_concat, 127))
        h_binder = jnp.zeros((n_orig,), dtype=jnp.bool_)
        h_target_bool = tgt.hotspot_mask if tgt.hotspot_mask is not None else jnp.zeros((n_concat,), dtype=jnp.bool_)
        ur_hotspot = (h_binder[:, None] | h_target_bool[None, :]).astype(jnp.float32)[..., None]
        ur_raw = jnp.concatenate([ur_sep, ur_dists, jnp.ones((n_orig, n_concat, 1)), ur_hotspot], axis=-1)
        ur_proj = self.concat_pair_ln(self.concat_pair_linear(ur_raw))

        # Lower-right block: target->target distances
        t_ca = tgt.coords[:, 1, :][:, None, :]  # [n_concat, 1, 3]
        lr_dists = self._pairwise_bb_dists(t_ca, tgt.coords, has_cb_target, d_bins)
        lr_sep = relative_seq_sep(n_concat, 127)
        h_bool = tgt.hotspot_mask if tgt.hotspot_mask is not None else jnp.zeros((n_concat,), dtype=jnp.bool_)
        lr_hotspot = (h_bool[:, None] | h_bool[None, :]).astype(jnp.float32)[..., None]
        lr_raw = jnp.concatenate([lr_sep, lr_dists, jnp.zeros((n_concat, n_concat, 1)), lr_hotspot], axis=-1)
        lr_proj = self.concat_pair_ln(self.concat_pair_linear(lr_raw))

        ext = jnp.zeros((n_ext, n_ext, pair_dim))
        ext = ext.at[:n_orig, :n_orig].set(pair_rep)
        ext = ext.at[:n_orig, n_orig:].set(ur_proj)
        ext = ext.at[n_orig:, :n_orig].set(jnp.swapaxes(ur_proj, 0, 1))
        ext = ext.at[n_orig:, n_orig:].set(lr_proj)
        return ext
