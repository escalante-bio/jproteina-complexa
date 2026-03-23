"""Typed batch containers and model outputs."""

import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float, Bool, Int


# ---- Noisy state (used by denoiser) ----

class NoisyState(eqx.Module):
    bb_ca: Float[Array, "B N 3"]
    local_latents: Float[Array, "B N D"]


class Timesteps(eqx.Module):
    bb_ca: Float[Array, "B"]
    local_latents: Float[Array, "B"]


# ---- Target conditioning ----

class TargetCond(eqx.Module):
    coords: Float[Array, "B Nt 37 3"]       # target atom coords (nm)
    atom_mask: Float[Array, "B Nt 37"]       # target atom mask
    seq: Int[Array, "B Nt"]                  # target residue types
    seq_mask: Bool[Array, "B Nt"]            # target residue mask
    hotspot_mask: Bool[Array, "B Nt"] | None = None
    sidechain_feat: Float[Array, "B Nt 88"] | None = None
    torsion_feat: Float[Array, "B Nt 63"] | None = None


# ---- Batch types ----

class DecoderBatch(eqx.Module):
    z_latent: Float[Array, "B N D"]
    ca_coors_nm: Float[Array, "B N 3"]
    mask: Bool[Array, "B N"]


class EncoderBatch(eqx.Module):
    coords: Float[Array, "B N 37 3"]         # Angstroms
    coords_nm: Float[Array, "B N 37 3"]      # nanometers
    coord_mask: Float[Array, "B N 37"]
    residue_type: Int[Array, "B N"]
    mask: Bool[Array, "B N"]
    sidechain_angles_feat: Float[Array, "B N 88"]


class DenoiserBatch(eqx.Module):
    x_t: NoisyState
    t: Timesteps
    mask: Bool[Array, "B N"]
    x_sc: NoisyState | None = None
    target: TargetCond | None = None


# ---- Output types ----

class DecoderOutput(eqx.Module):
    coors_nm: Float[Array, "B N 37 3"]
    seq_logits: Float[Array, "B N 20"]
    aatype: Int[Array, "B N"]
    atom_mask: Bool[Array, "B N 37"]
    mask: Bool[Array, "B N"]


class EncoderOutput(eqx.Module):
    mean: Float[Array, "B N D"]
    log_scale: Float[Array, "B N D"]
    z_latent: Float[Array, "B N D"]


class DenoiserOutput(eqx.Module):
    bb_ca: Float[Array, "B N 3"]
    local_latents: Float[Array, "B N D"]
