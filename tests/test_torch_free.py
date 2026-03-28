"""Verify the full pipeline works without torch installed.

We block torch imports to prove no runtime dependency exists.
"""

import sys
import builtins

# Block torch at the __import__ level BEFORE anything else
_real_import = builtins.__import__
BLOCKED = {"torch", "proteinfoundation", "openfold"}

def _guarded_import(name, *args, **kwargs):
    if any(name == b or name.startswith(b + ".") for b in BLOCKED):
        raise ImportError(f"BLOCKED: {name}")
    return _real_import(name, *args, **kwargs)

builtins.__import__ = _guarded_import

import time  # noqa: E402
import numpy as np  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp
import equinox as eqx

from jproteina_complexa.serialization import load_model
from jproteina_complexa.types import DenoiserBatch, DecoderBatch, NoisyState, Timesteps, TargetCond, DecoderOutput
from jproteina_complexa.flow_matching import get_schedule, sample_noise, predict_x1_from_v, force_zero_com

print("=" * 60)
print("Torch-free inference test")
print("=" * 60)

# Verify torch is truly blocked
try:
    import torch
    print("FAIL: torch imported successfully (should be blocked)")
    sys.exit(1)
except ImportError:
    print("  torch is blocked — good")

# Load models
print("\nLoading models from .eqx files...")
t0 = time.perf_counter()
denoiser = load_model("weights/denoiser")
decoder = load_model("weights/decoder")
print(f"  Loaded in {time.perf_counter() - t0:.1f}s")

# Load target protein (crambin CA coords from a numpy file)
# In a real scenario you'd parse a PDB, but we'll create coords directly
import gemmi
from jproteina_complexa.pdb import load_target
from jproteina_complexa.constants import RESTYPE_ATOM37_MASK

_chain = gemmi.read_structure("/tmp/pdb/pdb1crn.ent")[0][0]
target_a37, target_a37_mask, target_seq_idx, N_target = load_target(_chain, center=False)

print(f"  Target: {N_target} residues")

# Generate
N_binder = 30
N_STEPS = 100
B = 1
key = jax.random.PRNGKey(99)

target = TargetCond(
    coords=jnp.array(target_a37 / 10.0)[None],
    atom_mask=jnp.array(target_a37_mask)[None],
    seq=jnp.array(target_seq_idx)[None],
    seq_mask=jnp.ones((B, N_target), dtype=jnp.bool_),
    hotspot_mask=jnp.ones((B, N_target), dtype=jnp.bool_),
)

mask = jnp.ones((B, N_binder), dtype=jnp.bool_)
ts = get_schedule("uniform", N_STEPS)

key, k1, k2 = jax.random.split(key, 3)
x_bb = sample_noise(k1, (B, N_binder, 3), mask.astype(jnp.float32), zero_com=True)
x_lat = sample_noise(k2, (B, N_binder, 8), mask.astype(jnp.float32), zero_com=False)

@eqx.filter_jit
def step_denoiser(model, batch):
    return model(batch)

@eqx.filter_jit
def step_decoder(model, batch):
    return model(batch)

print(f"\nGenerating {N_binder}-residue binder ({N_STEPS} steps, torch-free)...")
x_sc_bb = jnp.zeros_like(x_bb)
x_sc_lat = jnp.zeros_like(x_lat)

t0 = time.perf_counter()
for step in range(N_STEPS):
    t_now = float(ts[step])
    dt = float(ts[step + 1] - ts[step])

    batch = DenoiserBatch(
        x_t=NoisyState(bb_ca=x_bb, local_latents=x_lat),
        t=Timesteps(bb_ca=jnp.array([t_now]), local_latents=jnp.array([t_now])),
        mask=mask,
        x_sc=NoisyState(bb_ca=x_sc_bb, local_latents=x_sc_lat),
        target=target,
    )
    out = step_denoiser(denoiser, batch)
    x_sc_bb = predict_x1_from_v(x_bb, out.bb_ca, jnp.array([t_now]))
    x_sc_lat = predict_x1_from_v(x_lat, out.local_latents, jnp.array([t_now]))
    x_bb = x_bb + out.bb_ca * dt
    x_lat = x_lat + out.local_latents * dt
    x_bb = force_zero_com(x_bb, mask.astype(jnp.float32))

jax.block_until_ready(x_bb)
gen_time = time.perf_counter() - t0
print(f"  Generation: {gen_time:.1f}s")

# Decode
dec_out = step_decoder(decoder, DecoderBatch(z_latent=x_lat, ca_coors=x_bb, mask=mask))
jax.block_until_ready(dec_out.coors)

# Verify
AA_CODES = "ACDEFGHIKLMNPQRSTVWY"
pred_seq = "".join(AA_CODES[i] for i in np.array(dec_out.aatype[0]))
ca_dists = np.linalg.norm(np.diff(np.array(x_bb[0]) * 10, axis=0), axis=1)

print(f"\n  Sequence: {pred_seq}")
print(f"  CA-CA distances: mean={ca_dists.mean():.2f}Å (expected ~3.8Å)")
print(f"  Sequence length: {len(pred_seq)}")

assert 3.5 < ca_dists.mean() < 4.1, f"Bad backbone geometry: {ca_dists.mean():.2f}Å"

print(f"\n{'=' * 60}")
print("PASSED: Full generation pipeline runs without torch")
print(f"{'=' * 60}")
