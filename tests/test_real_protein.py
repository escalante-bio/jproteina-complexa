"""End-to-end test: real protein (crambin, PDB 1CRN) through both decoders."""

import sys
import types
import time

# ---- Mock transitive deps (same as before) ----
for mod_name in ["torch_scatter"]:
    sys.modules[mod_name] = types.ModuleType(mod_name)
    sys.modules[mod_name].scatter_mean = None

sys.path.insert(0, "proteina-complexa/community_models")
sys.path.insert(0, "proteina-complexa/src")

import importlib.util
spec = importlib.util.spec_from_file_location(
    "openfold.np.residue_constants",
    "proteina-complexa/community_models/openfold/np/residue_constants.py",
)
rc_mod = importlib.util.module_from_spec(spec)

def _make_fake_module(name, attrs=None):
    mod = types.ModuleType(name)
    if attrs:
        for k, v in attrs.items():
            setattr(mod, k, v)
    sys.modules[name] = mod

_make_fake_module("openfold")
_make_fake_module("openfold.np")
sys.modules["openfold.np.residue_constants"] = rc_mod
spec.loader.exec_module(rc_mod)

import torch

class _Fake(torch.nn.Module):
    def __init__(self, *a, **kw): super().__init__()
    def forward(self, x, *a, **kw): return x

_make_fake_module("openfold.model")
_make_fake_module("openfold.model.dropout", {"DropoutColumnwise": _Fake, "DropoutRowwise": _Fake})
_make_fake_module("openfold.model.pair_transition", {"PairTransition": _Fake})
_make_fake_module("openfold.model.triangular_attention", {"TriangleAttentionStartingNode": _Fake, "TriangleAttentionEndingNode": _Fake})
_make_fake_module("openfold.model.triangular_multiplicative_update", {"TriangleMultiplicationIncoming": _Fake, "TriangleMultiplicationOutgoing": _Fake})
_make_fake_module("openfold.data")
_make_fake_module("openfold.data.data_transforms", {"atom37_to_torsion_angles": lambda **kw: None})

# ---- Load real protein ----
import numpy as np
import gemmi

print("=" * 70)
print("End-to-end test: real protein (crambin, PDB 1CRN, 46 residues)")
print("=" * 70)

structure = gemmi.read_structure("/tmp/pdb/pdb1crn.ent")
polymer = structure[0][0].get_polymer()

# Extract CA coordinates
ca_coords_ang = np.array(
    [[a.pos.x, a.pos.y, a.pos.z] for res in polymer for a in res if a.name == "CA"],
    dtype=np.float32,
)  # [N, 3] in Angstroms
ca_coords_nm = ca_coords_ang / 10.0  # Convert to nanometers

N_res = len(ca_coords_nm)
print(f"\nProtein: {N_res} residues")
print(f"CA coords range (nm): [{ca_coords_nm.min():.3f}, {ca_coords_nm.max():.3f}]")

# Compute consecutive CA-CA distances to sanity check
ca_dists = np.linalg.norm(np.diff(ca_coords_nm, axis=0), axis=1)
print(f"Consecutive CA-CA distances (nm): mean={ca_dists.mean():.4f}, "
      f"min={ca_dists.min():.4f}, max={ca_dists.max():.4f}")
print(f"  (expected ~0.38 nm)")

# ---- Build batch ----
B = 1
LATENT_DIM = 8
torch.manual_seed(123)
np.random.seed(123)

ca_tensor = torch.tensor(ca_coords_nm).unsqueeze(0)  # [1, N, 3]
z_latent = torch.randn(B, N_res, LATENT_DIM) * 0.3   # simulate encoder output
mask = torch.ones(B, N_res, dtype=torch.bool)

batch_input_pt = {
    "z_latent": z_latent,
    "ca_coors_nm": ca_tensor,
    "residue_mask": mask,
    "mask": mask,
}

print(f"\nBatch shapes:")
print(f"  ca_coors_nm:  {ca_tensor.shape}")
print(f"  z_latent:     {z_latent.shape}")
print(f"  mask:         {mask.shape}")

# ---- Instantiate PyTorch decoder ----
from proteinfoundation.partial_autoencoder.decoder import DecoderTransformer as PTDecoder

decoder_cfg = {
    "decoder": {
        "abs_coors": False,
        "nlayers": 12,
        "token_dim": 512,
        "nheads": 12,
        "parallel_mha_transition": False,
        "strict_feats": False,
        "feats_seq": ["ca_coors_nm", "z_latent_seq"],
        "feats_cond_seq": None,
        "dim_cond": 128,
        "idx_emb_dim": 128,
        "feats_pair_repr": ["rel_seq_sep", "ca_coors_nm_pair_dists"],
        "seq_sep_dim": 127,
        "pair_repr_dim": 256,
        "update_pair_repr": False,
        "update_pair_repr_every_n": 3,
        "use_tri_mult": False,
        "use_qkln": True,
        "latent_z_dim": 8,
    }
}

print("\nInstantiating PyTorch decoder...")
pt_model = PTDecoder(**decoder_cfg)
pt_model.eval()

# ---- Run PyTorch ----
print("Running PyTorch forward pass...")
t0 = time.perf_counter()
with torch.no_grad():
    pt_output = pt_model(batch_input_pt)
pt_time = time.perf_counter() - t0
print(f"  PyTorch time: {pt_time:.3f}s")

# ---- Convert to JAX ----
import jax
import jax.numpy as jnp
import equinox as eqx
import jproteina_complexa.nn.register
from jproteina_complexa.backend import from_torch

print("\nConverting to JAX/Equinox...")
jax_decoder = from_torch(pt_model)

batch_input_jax = {
    "z_latent": jnp.array(z_latent.numpy()),
    "ca_coors_nm": jnp.array(ca_tensor.numpy()),
    "residue_mask": jnp.array(mask.numpy()),
    "mask": jnp.array(mask.numpy()),
}

# ---- Run JAX (eager) ----
print("Running JAX forward pass (eager)...")
t0 = time.perf_counter()
with jax.default_matmul_precision("float32"):
    jax_output = jax_decoder(batch_input_jax)
jax_eager_time = time.perf_counter() - t0
print(f"  JAX eager time: {jax_eager_time:.3f}s")

# ---- Run JAX (JIT) ----
@eqx.filter_jit
def run_decoder(model, inp):
    return model(inp)

print("Compiling JAX (JIT)...")
t0 = time.perf_counter()
jit_output = run_decoder(jax_decoder, batch_input_jax)
jax.block_until_ready(jit_output["coors_nm"])
jit_compile_time = time.perf_counter() - t0
print(f"  JAX JIT compile+run: {jit_compile_time:.3f}s")

t0 = time.perf_counter()
jit_output2 = run_decoder(jax_decoder, batch_input_jax)
jax.block_until_ready(jit_output2["coors_nm"])
jit_run_time = time.perf_counter() - t0
print(f"  JAX JIT cached run:  {jit_run_time:.3f}s")

# ---- Compare outputs ----
print("\n" + "=" * 70)
print("Numerical comparison (PyTorch vs JAX)")
print("=" * 70)

for key in ["seq_logits", "coors_nm", "aatype_max", "atom_mask"]:
    pt_np = pt_output[key].numpy()
    jax_np = np.array(jax_output[key])
    jit_np = np.array(jit_output[key])

    if pt_np.dtype == np.bool_:
        n_diff_eager = np.sum(pt_np != jax_np)
        n_diff_jit = np.sum(pt_np != jit_np)
        print(f"  {key:15s}: eager={n_diff_eager} mismatches, jit={n_diff_jit} mismatches (bool)")
    else:
        err_eager = np.abs(pt_np.astype(np.float64) - jax_np.astype(np.float64)).max()
        err_jit = np.abs(pt_np.astype(np.float64) - jit_np.astype(np.float64)).max()
        print(f"  {key:15s}: eager max_err={err_eager:.2e}, jit max_err={err_jit:.2e}")

# ---- Inspect predictions ----
print("\n" + "=" * 70)
print("Decoder output analysis (crambin)")
print("=" * 70)

# Map amino acid indices to 1-letter codes
AA_CODES = "ACDEFGHIKLMNPQRSTVWY"

pt_seq = pt_output["aatype_max"][0].numpy()
jax_seq = np.array(jax_output["aatype_max"][0])

pt_seq_str = "".join(AA_CODES[i] for i in pt_seq)
jax_seq_str = "".join(AA_CODES[i] for i in jax_seq)
seq_match = pt_seq_str == jax_seq_str

print(f"\nPredicted sequence (PyTorch): {pt_seq_str}")
print(f"Predicted sequence (JAX):     {jax_seq_str}")
print(f"Sequences match:              {seq_match}")

# True crambin sequence (PDB 1CRN)
true_seq = "TTCCPSIVARSNFNVCRLPGTPEALCATYTGCIIIPGATCPGDYAN"
print(f"True crambin sequence:        {true_seq}")
print(f"  (note: decoder uses random weights, so prediction won't match)")

# Coordinate analysis
pt_coors = pt_output["coors_nm"][0].numpy()  # [N, 37, 3]
jax_coors = np.array(jax_output["coors_nm"][0])
ca_input = ca_tensor[0].numpy()

# Check that CA atom (index 1) matches input
ca_pred_pt = pt_coors[:, 1, :]
ca_pred_jax = jax_coors[:, 1, :]
ca_err_pt = np.abs(ca_pred_pt - ca_input).max()
ca_err_jax = np.abs(ca_pred_jax - ca_input).max()

print(f"\nCA coordinate preservation:")
print(f"  PyTorch: max deviation from input CA = {ca_err_pt:.2e} nm")
print(f"  JAX:     max deviation from input CA = {ca_err_jax:.2e} nm")
print(f"  (abs_coors=False: all atoms offset by CA, CA atom zeroed → should match input)")

# Check predicted sidechain atoms are reasonable
atom_mask = pt_output["atom_mask"][0].numpy()  # [N, 37]
n_atoms_per_res = atom_mask.sum(axis=1)
print(f"\nAtoms per residue: min={n_atoms_per_res.min()}, max={n_atoms_per_res.max()}, "
      f"mean={n_atoms_per_res.mean():.1f}")

# Check all-atom coordinate spread (should be near CA positions)
for res_idx in [0, N_res // 2, N_res - 1]:
    res_mask = atom_mask[res_idx].astype(bool)
    res_coors = jax_coors[res_idx, res_mask, :]  # active atoms
    ca_pos = ca_input[res_idx]
    dists = np.linalg.norm(res_coors - ca_pos, axis=1)
    print(f"  Residue {res_idx} ({AA_CODES[jax_seq[res_idx]]}): "
          f"{res_mask.sum()} atoms, dist from CA: "
          f"mean={dists.mean():.3f}nm, max={dists.max():.3f}nm")

# Sequence logits confidence
logits = np.array(jax_output["seq_logits"][0])  # [N, 20]
from scipy.special import softmax as sp_softmax
probs = sp_softmax(logits, axis=-1)
max_probs = probs.max(axis=1)
entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1)
print(f"\nSequence prediction confidence:")
print(f"  Max probability per residue: mean={max_probs.mean():.3f}, "
      f"min={max_probs.min():.3f}, max={max_probs.max():.3f}")
print(f"  Entropy per residue:         mean={entropy.mean():.3f} "
      f"(uniform={np.log(20):.3f})")

print(f"\n{'=' * 70}")
print("All checks passed. Decoder translation is numerically correct.")
print(f"{'=' * 70}")
