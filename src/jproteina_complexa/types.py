"""Typed containers and model outputs (unbatched — use jax.vmap for batching).

All coordinates are in Angstroms at the public API boundary.
Internal nm conversion is handled by the models and feature computation.
"""

import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float, Bool, Int


# ---- Noisy state (used by denoiser, internal to flow matching — nm) ----

class NoisyState(eqx.Module):
    bb_ca: Float[Array, "N 3"]            # nm
    local_latents: Float[Array, "N D"]


class Timesteps(eqx.Module):
    bb_ca: Float[Array, ""]
    local_latents: Float[Array, ""]


# ---- Target conditioning ----

class TargetCond(eqx.Module):
    coords: Float[Array, "Nt 37 3"]       # target atom coords (Angstroms)
    atom_mask: Float[Array, "Nt 37"]       # target atom mask
    seq: Int[Array, "Nt"]                  # target residue types
    hotspot_mask: Bool[Array, "Nt"] | None = None
    sidechain_feat: Float[Array, "Nt 88"] | None = None
    torsion_feat: Float[Array, "Nt 63"] | None = None


# ---- Motif conditioning (protein motif scaffolding, no ligand) ----

class MotifCond(eqx.Module):
    """Fixed protein-motif residues, injected as extra sequence tokens.

    Compact representation: only the Nm motif residues are provided (already
    extracted from the source structure). The motif conditions generation purely
    through the appended sequence tokens; it adds nothing to the pair rep (the
    upstream motif pair features are multiplied by zero — see models.py).
    """
    x_motif: Float[Array, "Nm 37 3"]        # motif atom coords (Angstroms)
    motif_mask: Float[Array, "Nm 37"]        # per-atom mask (which atom37 slots are present)
    seq_motif: Int[Array, "Nm"]              # motif residue types (0..19)
    seq_motif_mask: Float[Array, "Nm"] | None = None  # per-residue validity (defaults to ones)


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
    motif: MotifCond | None = None


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
    bb_ca: Float[Array, "N 3"]             # velocity field (nm)
    local_latents: Float[Array, "N D"]     # velocity field
