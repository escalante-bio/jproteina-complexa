"""Full encoder→decoder round-trip on crambin with real pretrained weights."""

import sys
import types
import time
import importlib

# ---- Setup sys.path FIRST ----
sys.path.insert(0, "proteina-complexa/community_models")
sys.path.insert(0, "proteina-complexa/src")

# ---- Mock only what we must ----
# torch_scatter (used in seq_cond_feats, not by our features)
sys.modules["torch_scatter"] = types.ModuleType("torch_scatter")
sys.modules["torch_scatter"].scatter_mean = None

# We need real openfold.np, openfold.data, openfold.utils, openfold.config
# But openfold.model eagerly imports everything including heads.py → loss.py → ml_collections
# So we fake just openfold.model and its children

import torch

class _F(torch.nn.Module):
    def __init__(self, *a, **kw): super().__init__()
    def forward(self, x, *a, **kw): return x

# Pre-register openfold package so submodule imports work
import openfold  # this loads __init__.py from community_models/openfold/

# Patch openfold.np.__init__ to not eagerly import protein.py (needs Bio.PDB)
# We only need residue_constants
import openfold.np.residue_constants  # this works

# Patch openfold.model to not eagerly import everything
fake_model = types.ModuleType("openfold.model")
sys.modules["openfold.model"] = fake_model

# Create fake submodules for pair_update.py imports
for name, attrs in [
    ("openfold.model.dropout", {"DropoutColumnwise": _F, "DropoutRowwise": _F}),
    ("openfold.model.pair_transition", {"PairTransition": _F}),
    ("openfold.model.triangular_attention", {"TriangleAttentionStartingNode": _F, "TriangleAttentionEndingNode": _F}),
    ("openfold.model.triangular_multiplicative_update", {"TriangleMultiplicationIncoming": _F, "TriangleMultiplicationOutgoing": _F}),
]:
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod

# Now import the real data_transforms (needs openfold.config, openfold.utils.rigid_utils)
from openfold.data import data_transforms

# ---------------------------------------------------------------------------
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import gemmi
from omegaconf import OmegaConf
from openfold.np import residue_constants as rc

print("=" * 70)
print("Full autoencoder round-trip: encode → decode on crambin (PDB 1CRN)")
print("=" * 70)

# ---- Load real protein ----
print("\nLoading crambin...")
structure = gemmi.read_structure("/tmp/pdb/pdb1crn.ent")
polymer = structure[0][0].get_polymer()

AA_3TO1 = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
}
AA_1TO_IDX = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}

true_seq = "".join(AA_3TO1.get(res.name, "X") for res in polymer)
N_res = len(polymer)
print(f"  {N_res} residues: {true_seq}")

# Build atom37 representation
atom37_coords = np.zeros((N_res, 37, 3), dtype=np.float32)
atom37_mask = np.zeros((N_res, 37), dtype=np.float32)
residue_types = np.zeros(N_res, dtype=np.int64)

atom37_names = rc.atom_types

for i, res in enumerate(polymer):
    residue_types[i] = AA_1TO_IDX.get(AA_3TO1.get(res.name, "A"), 0)
    for atom in res:
        if atom.name in atom37_names:
            idx = atom37_names.index(atom.name)
            atom37_coords[i, idx] = [atom.pos.x, atom.pos.y, atom.pos.z]
            atom37_mask[i, idx] = 1.0

ca_coords_nm = atom37_coords[:, 1, :] / 10.0
coords_nm = atom37_coords / 10.0

print(f"  Atoms populated: {atom37_mask.sum():.0f} / {N_res * 37}")

# ---- Load checkpoint ----
print("\nLoading checkpoint...")
t0 = time.perf_counter()
ckpt = torch.load("proteina-complexa/ckpts/complexa_ae.ckpt", map_location="cpu", weights_only=False)
print(f"  Loaded in {time.perf_counter() - t0:.1f}s")
cfg_ae = ckpt["hyper_parameters"]["cfg_ae"]
cfg_dict = OmegaConf.to_container(cfg_ae.nn_ae, resolve=True)

# ---- Build PyTorch models ----
from proteinfoundation.partial_autoencoder.encoder import EncoderTransformer as PTEncoder
from proteinfoundation.partial_autoencoder.decoder import DecoderTransformer as PTDecoder

print("\nInstantiating PyTorch encoder + decoder...")
pt_encoder = PTEncoder(**cfg_dict)
enc_sd = {k.removeprefix("encoder."): v for k, v in ckpt["state_dict"].items() if k.startswith("encoder.")}
pt_encoder.load_state_dict(enc_sd)
pt_encoder.eval()
print(f"  Encoder: {sum(p.numel() for p in pt_encoder.parameters()):,} params")

pt_decoder = PTDecoder(**cfg_dict)
dec_sd = {k.removeprefix("decoder."): v for k, v in ckpt["state_dict"].items() if k.startswith("decoder.")}
pt_decoder.load_state_dict(dec_sd)
pt_decoder.eval()
print(f"  Decoder: {sum(p.numel() for p in pt_decoder.parameters()):,} params")

del ckpt

# ---- Precompute sidechain angles (using PyTorch/OpenFold) ----
print("\nPrecomputing sidechain angles via OpenFold...")
sc_input = {
    "aatype": torch.tensor(residue_types).unsqueeze(0),
    "all_atom_positions": torch.tensor(atom37_coords, dtype=torch.float64).unsqueeze(0),
    "all_atom_mask": torch.tensor(atom37_mask, dtype=torch.float64).unsqueeze(0),
}
# curry1 decorator: atom37_to_torsion_angles(prefix="")(input_dict)
sc_result = data_transforms.atom37_to_torsion_angles(prefix="")(sc_input)
torsion_sin_cos = sc_result["torsion_angles_sin_cos"].float()
torsion_mask = sc_result["torsion_angles_mask"].float()

import einops
torsion_sin_cos_normed = torsion_sin_cos / (torch.linalg.norm(torsion_sin_cos, dim=-1, keepdim=True) + 1e-10)
angles = torch.atan2(torsion_sin_cos_normed[..., 0], torsion_sin_cos_normed[..., 1])
angles = angles * torsion_mask
sc_angles = angles[..., -4:]
sc_mask = torsion_mask[..., -4:].bool()
from proteinfoundation.nn.feature_factory.feature_utils import bin_and_one_hot as pt_bin_oh
angles_feat = pt_bin_oh(sc_angles, torch.linspace(-torch.pi, torch.pi, 20))
angles_feat = angles_feat * sc_mask[..., None]
angles_feat = einops.rearrange(angles_feat, "b n s d -> b n (s d)")
sidechain_feat = torch.cat([angles_feat, sc_mask.float()], dim=-1)
print(f"  Sidechain feature shape: {sidechain_feat.shape}")

# ---- Build encoder batch ----
B = 1
mask = torch.ones(B, N_res, dtype=torch.bool)

encoder_batch_pt = {
    "coords": torch.tensor(atom37_coords).unsqueeze(0),
    "coords_nm": torch.tensor(coords_nm).unsqueeze(0),
    "coord_mask": torch.tensor(atom37_mask).unsqueeze(0),
    "residue_type": torch.tensor(residue_types).unsqueeze(0),
    "mask": mask,
    "strict_feats": False,
}

# ---- Run PyTorch encoder ----
print("\nRunning PyTorch encoder...")
t0 = time.perf_counter()
with torch.no_grad():
    pt_enc_out = pt_encoder(encoder_batch_pt)
pt_enc_time = time.perf_counter() - t0
print(f"  Time: {pt_enc_time:.3f}s")
print(f"  z_latent shape: {pt_enc_out['z_latent'].shape}")
print(f"  mean range: [{pt_enc_out['mean'].min():.3f}, {pt_enc_out['mean'].max():.3f}]")

# ---- Run PyTorch decoder ----
print("\nRunning PyTorch decoder (using encoder mean as z)...")
decoder_batch_pt = {
    "z_latent": pt_enc_out["mean"].detach(),
    "ca_coors_nm": torch.tensor(ca_coords_nm).unsqueeze(0),
    "residue_mask": mask,
    "mask": mask,
}
with torch.no_grad():
    pt_dec_out = pt_decoder(decoder_batch_pt)

# ---- Convert to JAX ----
print("\nConverting to JAX/Equinox...")
import jproteina_complexa.nn.register
from jproteina_complexa.backend import from_torch

jax_encoder = from_torch(pt_encoder)
jax_decoder = from_torch(pt_decoder)

encoder_batch_jax = {
    "coords": jnp.array(atom37_coords)[None],
    "coords_nm": jnp.array(coords_nm)[None],
    "coord_mask": jnp.array(atom37_mask)[None],
    "residue_type": jnp.array(residue_types)[None],
    "mask": jnp.ones((B, N_res), dtype=jnp.bool_),
    "sidechain_angles_feat": jnp.array(sidechain_feat.numpy()),
}

# ---- Run JAX encoder ----
print("\nRunning JAX encoder (deterministic)...")
t0 = time.perf_counter()
with jax.default_matmul_precision("float32"):
    jax_enc_out = jax_encoder.encode_deterministic(encoder_batch_jax)
jax_enc_time = time.perf_counter() - t0
print(f"  Time: {jax_enc_time:.3f}s")

pt_mean = pt_enc_out["mean"].numpy()
jax_mean = np.array(jax_enc_out["mean"])
enc_err = np.abs(pt_mean - jax_mean).max()
print(f"  Encoder mean max_err: {enc_err:.2e}")

# ---- Run JAX decoder ----
print("\nRunning JAX decoder...")
decoder_batch_jax = {
    "z_latent": jax_enc_out["mean"],
    "ca_coors_nm": jnp.array(ca_coords_nm)[None],
    "residue_mask": jnp.ones((B, N_res), dtype=jnp.bool_),
    "mask": jnp.ones((B, N_res), dtype=jnp.bool_),
}
with jax.default_matmul_precision("float32"):
    jax_dec_out = jax_decoder(decoder_batch_jax)

# ---- Pipeline comparison ----
print("\n" + "=" * 70)
print("Pipeline comparison (PyTorch vs JAX)")
print("=" * 70)

for key in ["seq_logits", "coors_nm", "aatype_max"]:
    pt_v = pt_dec_out[key].numpy()
    jax_v = np.array(jax_dec_out[key])
    if pt_v.dtype == np.bool_:
        print(f"  {key}: {np.sum(pt_v != jax_v)} mismatches")
    else:
        err = np.abs(pt_v.astype(np.float64) - jax_v.astype(np.float64)).max()
        print(f"  {key}: max_err={err:.2e}")

# ---- Reconstruction quality ----
print("\n" + "=" * 70)
print("Reconstruction quality (autoencoder round-trip on crambin)")
print("=" * 70)

AA_CODES = "ACDEFGHIKLMNPQRSTVWY"
pred_seq = "".join(AA_CODES[i] for i in np.array(jax_dec_out["aatype_max"][0]))
n_match = sum(a == b for a, b in zip(true_seq, pred_seq))

print(f"\n  True sequence:      {true_seq}")
print(f"  Reconstructed:      {pred_seq}")
print(f"  Sequence identity:  {n_match}/{N_res} ({100*n_match/N_res:.1f}%)")

# CA RMSD (should be ~0 since CA is provided as input)
pred_coors = np.array(jax_dec_out["coors_nm"][0])
pred_ca_ang = pred_coors[:, 1, :] * 10.0
true_ca_ang = atom37_coords[:, 1, :]
ca_rmsd = np.sqrt(np.mean(np.sum((pred_ca_ang - true_ca_ang) ** 2, axis=-1)))
print(f"  CA RMSD:            {ca_rmsd:.4f} Å")

# All-atom RMSD
pred_all = pred_coors * 10.0
true_all = atom37_coords
mask_bool = atom37_mask.astype(bool)

per_res_rmsd = []
for i in range(N_res):
    m = mask_bool[i]
    if m.sum() < 2:
        continue
    diff = pred_all[i, m] - true_all[i, m]
    per_res_rmsd.append(np.sqrt(np.mean(np.sum(diff ** 2, axis=-1))))

per_res_rmsd = np.array(per_res_rmsd)
print(f"\n  Per-residue all-atom RMSD:")
print(f"    mean   = {per_res_rmsd.mean():.3f} Å")
print(f"    median = {np.median(per_res_rmsd):.3f} Å")
print(f"    max    = {per_res_rmsd.max():.3f} Å (residue {np.argmax(per_res_rmsd)})")

all_pred = pred_all[mask_bool]
all_true = true_all[mask_bool]
global_rmsd = np.sqrt(np.mean(np.sum((all_pred - all_true) ** 2, axis=-1)))
print(f"  Global all-atom RMSD: {global_rmsd:.3f} Å")

# Backbone RMSD (N, CA, C)
bb_mask_res = mask_bool[:, :3].all(axis=1)
pred_bb = pred_all[bb_mask_res][:, :3, :].reshape(-1, 3)
true_bb = true_all[bb_mask_res][:, :3, :].reshape(-1, 3)
bb_rmsd = np.sqrt(np.mean(np.sum((pred_bb - true_bb) ** 2, axis=-1)))
print(f"  Backbone RMSD (N,CA,C): {bb_rmsd:.3f} Å")

# Logits confidence
from scipy.special import softmax as sp_softmax
logits = np.array(jax_dec_out["seq_logits"][0])
probs = sp_softmax(logits, axis=-1)
max_probs = probs.max(axis=1)
print(f"\n  Sequence confidence: mean={max_probs.mean():.3f}, "
      f"min={max_probs.min():.3f}, max={max_probs.max():.3f}")

# ---- JIT benchmark ----
print("\n" + "=" * 70)
print("JIT performance")
print("=" * 70)

@eqx.filter_jit
def encode_jit(model, batch):
    return model.encode_deterministic(batch)

@eqx.filter_jit
def decode_jit(model, batch):
    return model(batch)

print("Compiling encoder+decoder...")
t0 = time.perf_counter()
jit_enc = encode_jit(jax_encoder, encoder_batch_jax)
jax.block_until_ready(jit_enc["mean"])
print(f"  Encoder JIT compile: {time.perf_counter() - t0:.1f}s")

t0 = time.perf_counter()
jit_enc = encode_jit(jax_encoder, encoder_batch_jax)
jax.block_until_ready(jit_enc["mean"])
print(f"  Encoder JIT cached:  {time.perf_counter() - t0:.3f}s")

t0 = time.perf_counter()
jit_dec = decode_jit(jax_decoder, decoder_batch_jax)
jax.block_until_ready(jit_dec["coors_nm"])
print(f"  Decoder JIT compile: {time.perf_counter() - t0:.1f}s")

t0 = time.perf_counter()
jit_dec = decode_jit(jax_decoder, decoder_batch_jax)
jax.block_until_ready(jit_dec["coors_nm"])
print(f"  Decoder JIT cached:  {time.perf_counter() - t0:.3f}s")

print(f"\n{'=' * 70}")
print("Round-trip complete!")
print(f"{'=' * 70}")
