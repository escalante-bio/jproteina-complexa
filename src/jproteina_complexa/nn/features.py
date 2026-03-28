"""Feature computation — direct implementations for decoder, encoder, and denoiser."""

import math
import jax
import jax.numpy as jnp
import equinox as eqx
from einops import rearrange
from jaxtyping import Array, Float

from jproteina_complexa.types import DecoderBatch, EncoderBatch, DenoiserBatch
from jproteina_complexa.nn.layers import Linear, LayerNorm, Identity, AdaptiveLayerNorm


# ---- Shared primitives ----

def bin_and_one_hot(x, bin_limits):
    return jax.nn.one_hot(jnp.searchsorted(bin_limits, x), len(bin_limits) + 1)


def bin_pairwise_distances(coords, min_dist, max_dist, n_bins):
    d = jnp.sqrt(jnp.sum(jnp.square(coords[:, None] - coords[None, :]), axis=-1) + 1e-10)
    return bin_and_one_hot(d, jnp.linspace(min_dist, max_dist, n_bins - 1))


def relative_seq_sep(n, dim=127):
    idx = jnp.arange(1, n + 1, dtype=jnp.float32)
    return bin_and_one_hot(idx[:, None] - idx[None, :], jnp.linspace(-(dim / 2.0 - 1), dim / 2.0 - 1, dim - 1))


def time_embedding(t, dim, max_pos=2000):
    t = t * max_pos
    half = dim // 2
    freqs = jnp.exp(jnp.arange(half) * -(math.log(max_pos) / (half - 1)))
    args = t * freqs
    emb = jnp.concatenate([jnp.sin(args), jnp.cos(args)], axis=-1)
    return jnp.pad(emb, (0, dim % 2)) if dim % 2 == 1 else emb


def index_embedding(indices, dim, max_len=2056):
    K = jnp.arange(dim // 2)
    s = jnp.sin(indices[..., None] * math.pi / (max_len ** (2 * K / dim)))
    c = jnp.cos(indices[..., None] * math.pi / (max_len ** (2 * K / dim)))
    return jnp.concatenate([s, c], axis=-1)


def _normalize(v):
    return v / jnp.clip(jnp.sqrt(jnp.sum(v * v, axis=-1, keepdims=True) + 1e-16), 1e-8)


def signed_dihedral(a, b, c, d):
    n1 = _normalize(jnp.cross(b - a, c - b))
    n2 = _normalize(jnp.cross(c - b, d - c))
    n1xn2 = jnp.cross(n1, n2)
    return jnp.arctan2(
        jnp.sign(jnp.sum(n1xn2 * (c - b), axis=-1)) * jnp.sqrt(jnp.sum(n1xn2 ** 2, axis=-1) + 1e-16),
        jnp.sum(n1 * n2, axis=-1),
    )


def bond_angle(a, b, c):
    u, v = _normalize(b - a), _normalize(c - a)
    return jnp.arctan2(jnp.sqrt(jnp.sum(jnp.cross(u, v) ** 2, axis=-1) + 1e-16), jnp.sum(u * v, axis=-1))


# ---- Decoder features ----

class DecoderSeqFeatures(eqx.Module):
    linear: Linear

    def __call__(self, batch: DecoderBatch):
        mask = batch.mask.astype(jnp.float32)
        feats = jnp.concatenate([batch.ca_coors_nm, batch.z_latent], axis=-1)
        return self.linear(feats * mask[..., None]) * mask[..., None]


class DecoderPairFeatures(eqx.Module):
    linear: Linear
    seq_sep_dim: int = 127

    def __call__(self, batch: DecoderBatch):
        n = batch.ca_coors_nm.shape[0]
        mask = batch.mask.astype(jnp.float32)
        pair_mask = mask[None, :] * mask[:, None]
        sep = relative_seq_sep(n, self.seq_sep_dim)
        dists = bin_pairwise_distances(batch.ca_coors_nm, 0.1, 3.0, 30)
        feats = jnp.concatenate([sep, dists], axis=-1)
        return self.linear(feats * pair_mask[..., None]) * pair_mask[..., None]


# ---- Encoder features ----

class EncoderSeqFeatures(eqx.Module):
    linear: Linear

    def __call__(self, batch: EncoderBatch):
        mask = batch.mask.astype(jnp.float32)
        (n,) = mask.shape

        chain_break = jnp.zeros((n, 1))
        aatype = jax.nn.one_hot(batch.residue_type * mask.astype(jnp.int32), 20) * mask[..., None]

        c_abs = batch.coords_nm * batch.coord_mask[..., None]
        abs_feat = jnp.concatenate([rearrange(c_abs, "n a t -> n (a t)"), batch.coord_mask], axis=-1)

        ca = batch.coords_nm[:, 1, :]
        c_rel = (batch.coords_nm - ca[:, None, :]) * batch.coord_mask[..., None]
        rel_feat = jnp.concatenate([rearrange(c_rel, "n a t -> n (a t)"), batch.coord_mask], axis=-1)

        N, CA, C = batch.coords[:, 0], batch.coords[:, 1], batch.coords[:, 2]
        psi = signed_dihedral(N[:-1], CA[:-1], C[:-1], N[1:])
        omega = signed_dihedral(CA[:-1], C[:-1], N[1:], CA[1:])
        phi = signed_dihedral(C[:-1], N[1:], CA[1:], C[1:])
        bb = jnp.concatenate([jnp.stack([psi, omega, phi], axis=-1), jnp.zeros((1, 3))], axis=0)
        bb_feat = rearrange(bin_and_one_hot(bb, jnp.linspace(-jnp.pi, jnp.pi, 20)), "n t d -> n (t d)")

        feats = jnp.concatenate([chain_break, aatype, abs_feat, rel_feat, bb_feat, batch.sidechain_angles_feat], axis=-1)
        return self.linear(feats * mask[..., None]) * mask[..., None]


class EncoderPairFeatures(eqx.Module):
    linear: Linear
    ln: LayerNorm | Identity

    def __call__(self, batch: EncoderBatch):
        (n,) = batch.mask.shape
        mask = batch.mask.astype(jnp.float32)
        pair_mask = mask[None, :] * mask[:, None]
        has_cb = batch.coord_mask[:, 3]

        sep = relative_seq_sep(n, 127)

        CA_i = batch.coords_nm[:, 1, :][:, None, :]
        dist = lambda v: jnp.sqrt(jnp.sum(jnp.square(CA_i - v[None, :, :]), axis=-1) + 1e-10)
        d_bins = jnp.linspace(0.1, 2.0, 20)
        bb_dists = jnp.concatenate([
            bin_and_one_hot(dist(batch.coords_nm[:, i, :]), d_bins) for i in range(4)
        ], axis=-1)
        bb_dists = bb_dists.at[..., 63:84].multiply(has_cb[None, :, None])  # zero CB dists for glycine
        bb_dists = bb_dists * pair_mask[..., None]

        Nc, CAc, CBc = batch.coords[:, 0], batch.coords[:, 1], batch.coords[:, 3]
        N1, CA1, CB1 = (v[:, None, :] for v in (Nc, CAc, CBc))
        N2, CA2, CB2 = (v[None, :, :] for v in (Nc, CAc, CBc))
        angles = jnp.stack([
            signed_dihedral(N1, CA1, CB1, CB2), signed_dihedral(N2, CA2, CB2, CB1),
            bond_angle(CA1, CB1, CB2), bond_angle(CA2, CB2, CB1),
            signed_dihedral(CA1, CB1, CB2, CA2),
        ], axis=-1)
        orient_mask = pair_mask * has_cb[:, None] * has_cb[:, None]  # PyTorch quirk
        orient = rearrange(bin_and_one_hot(angles, jnp.linspace(-jnp.pi, jnp.pi, 20)), "n m f d -> n m (f d)")
        orient = orient * orient_mask[..., None]

        feats = jnp.concatenate([sep, bb_dists, orient], axis=-1)
        return self.ln(self.linear(feats * pair_mask[..., None])) * pair_mask[..., None]


# ---- Denoiser features ----

class DenoiserSeqFeatures(eqx.Module):
    linear: Linear
    latent_dim: int = 8
    idx_emb_dim: int = 256

    def __call__(self, batch: DenoiserBatch):
        mask = batch.mask.astype(jnp.float32)
        (n,) = mask.shape
        xsc_bb = batch.x_sc.bb_ca if batch.x_sc is not None else jnp.zeros((n, 3))
        xsc_lat = batch.x_sc.local_latents if batch.x_sc is not None else jnp.zeros((n, self.latent_dim))
        idx = jnp.arange(1, n + 1, dtype=jnp.float32)
        feats = jnp.concatenate([
            batch.x_t.bb_ca, batch.x_t.local_latents, xsc_bb, xsc_lat,
            jnp.zeros((n, 3)), jnp.zeros((n, 20)),  # optional features (disabled)
            jnp.zeros((n, 1)),  # hotspot
            index_embedding(idx, self.idx_emb_dim),
        ], axis=-1)
        return self.linear(feats * mask[..., None]) * mask[..., None]


class DenoiserCondFeatures(eqx.Module):
    linear: Linear
    t_emb_dim: int = 256

    def __call__(self, batch: DenoiserBatch):
        mask = batch.mask.astype(jnp.float32)
        (n,) = mask.shape
        emb_bb = jnp.broadcast_to(time_embedding(batch.t.bb_ca, self.t_emb_dim), (n, self.t_emb_dim))
        emb_lat = jnp.broadcast_to(time_embedding(batch.t.local_latents, self.t_emb_dim), (n, self.t_emb_dim))
        feats = jnp.concatenate([emb_bb, emb_lat], axis=-1)
        return self.linear(feats * mask[..., None]) * mask[..., None]


class DenoiserPairFeatures(eqx.Module):
    linear: Linear
    ln: LayerNorm | Identity
    seq_sep_dim: int = 127

    def __call__(self, batch: DenoiserBatch):
        (n,) = batch.mask.shape
        mask = batch.mask.astype(jnp.float32)
        pair_mask = mask[None, :] * mask[:, None]

        sep = relative_seq_sep(n, self.seq_sep_dim)
        xt_dists = bin_pairwise_distances(batch.x_t.bb_ca, 0.1, 3.0, 30)
        xsc_dists = bin_pairwise_distances(batch.x_sc.bb_ca, 0.1, 3.0, 30) if batch.x_sc is not None else jnp.zeros((n, n, 30))

        feats = jnp.concatenate([
            sep, xt_dists, xsc_dists,
            jnp.zeros((n, n, 30)),  # optional CA dists (disabled)
            jnp.zeros((n, n, 1)),   # chain idx
            jnp.zeros((n, n, 1)),   # hotspot
        ], axis=-1)
        return self.ln(self.linear(feats * pair_mask[..., None])) * pair_mask[..., None]


class DenoiserPairCondFeatures(eqx.Module):
    linear: Linear
    ln: LayerNorm | Identity
    t_emb_dim: int = 256

    def __call__(self, batch: DenoiserBatch):
        (n,) = batch.mask.shape
        mask = batch.mask.astype(jnp.float32)
        pair_mask = mask[None, :] * mask[:, None]
        # Time embeddings are spatially uniform — project once, then broadcast.
        emb = jnp.concatenate([
            time_embedding(batch.t.bb_ca, self.t_emb_dim),
            time_embedding(batch.t.local_latents, self.t_emb_dim),
        ])
        proj = self.ln(self.linear(emb))
        return jnp.broadcast_to(proj, (n, n, proj.shape[0])) * pair_mask[..., None]


# ---- PairReprBuilder ----

class PairReprBuilder(eqx.Module):
    pair_features: DenoiserPairFeatures | DecoderPairFeatures | EncoderPairFeatures
    pair_cond: DenoiserPairCondFeatures | None = None
    adaln: AdaptiveLayerNorm | None = None

    def __call__(self, batch):
        mask = batch.mask.astype(jnp.float32)
        pair_mask = mask[None, :] * mask[:, None]
        rep = self.pair_features(batch)
        if self.pair_cond is not None:
            rep = self.adaln(rep, self.pair_cond(batch), pair_mask)
        return rep


# ---- Target concat features ----

class TargetConcatFeatures(eqx.Module):
    linear: Linear
    ln: LayerNorm | Identity

    def __call__(self, batch: DenoiserBatch):
        """Returns (projected [n_target, dim], mask [n_target])."""
        if batch.target is None:
            return jnp.zeros((0, self.linear.weight.shape[0])), jnp.zeros((0,), dtype=jnp.bool_)

        tgt = batch.target
        c_abs = tgt.coords * tgt.atom_mask[..., None]
        abs_feat = jnp.concatenate([rearrange(c_abs, "n a t -> n (a t)"), tgt.atom_mask], axis=-1)

        ca = tgt.coords[:, 1, :]
        c_rel = (tgt.coords - ca[:, None, :]) * tgt.atom_mask[..., None]
        rel_feat = jnp.concatenate([rearrange(c_rel, "n a t -> n (a t)"), tgt.atom_mask], axis=-1)

        seq_oh = jax.nn.one_hot(tgt.seq * tgt.seq_mask.astype(jnp.int32), 20) * tgt.seq_mask[..., None]
        mask_feat = tgt.atom_mask.astype(jnp.float32)
        hotspot = (tgt.hotspot_mask.astype(jnp.float32) if tgt.hotspot_mask is not None else jnp.zeros(tgt.seq.shape))[..., None]
        sc = tgt.sidechain_feat if tgt.sidechain_feat is not None else jnp.zeros((*tgt.seq.shape, 88))
        tor = tgt.torsion_feat if tgt.torsion_feat is not None else jnp.zeros((*tgt.seq.shape, 63))

        raw = jnp.concatenate([abs_feat, seq_oh, mask_feat, hotspot, rel_feat, sc, tor], axis=-1)
        proj = self.ln(self.linear(raw * tgt.seq_mask[..., None]))
        return proj * tgt.seq_mask[..., None], tgt.seq_mask
