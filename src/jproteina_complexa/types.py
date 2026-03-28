"""Typed containers and model outputs (unbatched — use jax.vmap for batching).

All coordinates are in Angstroms at the public API boundary.
Internal nm conversion is handled by the models and feature computation.
"""

import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float, Bool, Int


# ---- Noisy state (used by denoiser, internal to flow matching — nm) ----

class NoisyState(eqx.Module):
    bb_ca: Float[Array, "N 3"]
    local_latents: Float[Array, "N D"]


class Timesteps(eqx.Module):
    bb_ca: Float[Array, ""]
    local_latents: Float[Array, ""]


# ---- Target conditioning ----

class TargetCond(eqx.Module):
    coords: Float[Array, "Nt 37 3"]       # target atom coords (Angstroms)
    atom_mask: Float[Array, "Nt 37"]       # target atom mask
    seq: Int[Array, "Nt"]                  # target residue types
    seq_mask: Bool[Array, "Nt"]            # target residue mask
    hotspot_mask: Bool[Array, "Nt"] | None = None
    sidechain_feat: Float[Array, "Nt 88"] | None = None
    torsion_feat: Float[Array, "Nt 63"] | None = None


# ---- Input types ----

class DecoderBatch(eqx.Module):
    z_latent: Float[Array, "N D"]
    ca_coors: Float[Array, "N 3"]          # Angstroms
    mask: Bool[Array, "N"]


class EncoderBatch(eqx.Module):
    coords: Float[Array, "N 37 3"]         # Angstroms
    coord_mask: Float[Array, "N 37"]
    residue_type: Int[Array, "N"]
    mask: Bool[Array, "N"]
    sidechain_angles_feat: Float[Array, "N 88"]


class DenoiserBatch(eqx.Module):
    x_t: NoisyState
    t: Timesteps
    mask: Bool[Array, "N"]
    x_sc: NoisyState | None = None
    target: TargetCond | None = None


# ---- Output types ----

class DecoderOutput(eqx.Module):
    coors: Float[Array, "N 37 3"]          # Angstroms
    seq_logits: Float[Array, "N 20"]
    aatype: Int[Array, "N"]
    atom_mask: Bool[Array, "N 37"]
    mask: Bool[Array, "N"]


class EncoderOutput(eqx.Module):
    mean: Float[Array, "N D"]
    log_scale: Float[Array, "N D"]
    z_latent: Float[Array, "N D"]


class DenoiserOutput(eqx.Module):
    bb_ca: Float[Array, "N 3"]
    local_latents: Float[Array, "N D"]
