"""Compare JAX vs PyTorch decoder on the same generated binder structure."""

import sys
import types
import time

sys.modules["torch_scatter"] = types.ModuleType("torch_scatter")
sys.modules["torch_scatter"].scatter_mean = None
sys.path.insert(0, "proteina-complexa/community_models")
sys.path.insert(0, "proteina-complexa/src")

import openfold, openfold.np.residue_constants
import torch

class _F(torch.nn.Module):
    def __init__(self, *a, **kw): super().__init__()
    def forward(self, x, *a, **kw): return x

for n, a in [("openfold.model",{}),("openfold.model.dropout",{"DropoutColumnwise":_F,"DropoutRowwise":_F}),
    ("openfold.model.pair_transition",{"PairTransition":_F}),
    ("openfold.model.triangular_attention",{"TriangleAttentionStartingNode":_F,"TriangleAttentionEndingNode":_F}),
    ("openfold.model.triangular_multiplicative_update",{"TriangleMultiplicationIncoming":_F,"TriangleMultiplicationOutgoing":_F}),
    ("openfold.model.structure_module",{"InvariantPointAttention":_F}),
    ("openfold.utils",{}),("openfold.utils.rigid_utils",{"Rigid":None}),
    ("openfold.data",{}),("openfold.data.data_transforms",{"atom37_to_torsion_angles":lambda **kw: None})]:
    m=types.ModuleType(n); [setattr(m,k,v) for k,v in a.items()]; sys.modules[n]=m

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import gemmi
from omegaconf import OmegaConf
from openfold.np import residue_constants as rc

print("=" * 70)
print("Decode comparison: same latent structure → PyTorch vs JAX decoder")
print("=" * 70)

# ---- Regenerate the binder (same seed as test_binder_generation.py) ----
print("\nRegenerating binder (same seed)...")

# Load target
AA_3TO1 = {'ALA':'A','CYS':'C','ASP':'D','GLU':'E','PHE':'F','GLY':'G','HIS':'H','ILE':'I','LYS':'K','LEU':'L','MET':'M','ASN':'N','PRO':'P','GLN':'Q','ARG':'R','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'}
AA_1TO_IDX = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}

structure = gemmi.read_structure("/tmp/pdb/pdb1crn.ent")
polymer = structure[0][0].get_polymer()
N_target = len(polymer)

target_a37 = np.zeros((N_target, 37, 3), dtype=np.float32)
target_a37_mask = np.zeros((N_target, 37), dtype=np.float32)
target_seq_idx = np.zeros(N_target, dtype=np.int64)
for i, res in enumerate(polymer):
    target_seq_idx[i] = AA_1TO_IDX.get(AA_3TO1.get(res.name, "A"), 0)
    for atom in res:
        if atom.name in rc.atom_types:
            j = rc.atom_types.index(atom.name)
            target_a37[i, j] = [atom.pos.x, atom.pos.y, atom.pos.z]
            target_a37_mask[i, j] = 1.0
target_nm = target_a37 / 10.0

# Load denoiser
ckpt = torch.load("proteina-complexa/ckpts/complexa.ckpt", map_location="cpu", weights_only=False)
nn_cfg = OmegaConf.to_container(ckpt["hyper_parameters"]["cfg_exp"].nn)
for k in ckpt["state_dict"]:
    if "local_latents_linear.1.weight" in k:
        nn_cfg["latent_dim"] = ckpt["state_dict"][k].shape[0]
        break
LATENT_DIM = nn_cfg["latent_dim"]

from proteinfoundation.nn.local_latents_transformer import LocalLatentsTransformer as PTModel
import jproteina_complexa.nn.register
from jproteina_complexa.backend import from_torch

pt_denoiser = PTModel(**nn_cfg)
nn_sd = {k.removeprefix("nn."): v for k, v in ckpt["state_dict"].items() if k.startswith("nn.")}
pt_denoiser.load_state_dict(nn_sd)
pt_denoiser.eval()
jax_denoiser = from_torch(pt_denoiser)
del ckpt, pt_denoiser

# Load decoder (both PyTorch and JAX)
ae_ckpt = torch.load("proteina-complexa/ckpts/complexa_ae.ckpt", map_location="cpu", weights_only=False)
ae_cfg = OmegaConf.to_container(ae_ckpt["hyper_parameters"]["cfg_ae"].nn_ae, resolve=True)

from proteinfoundation.partial_autoencoder.decoder import DecoderTransformer as PTDecoder
pt_decoder = PTDecoder(**ae_cfg)
dec_sd = {k.removeprefix("decoder."): v for k, v in ae_ckpt["state_dict"].items() if k.startswith("decoder.")}
pt_decoder.load_state_dict(dec_sd)
pt_decoder.eval()
jax_decoder = from_torch(pt_decoder)
del ae_ckpt

# ---- Generate binder ----
from jproteina_complexa.flow_matching import get_schedule, sample_noise, predict_x1_from_v, force_zero_com

N_binder = 50
N_STEPS = 200
B = 1
key = jax.random.PRNGKey(42)
ts = get_schedule("uniform", N_STEPS)

key, k1, k2 = jax.random.split(key, 3)
mask_jax = jnp.ones((B, N_binder), dtype=jnp.bool_)
x_bb = sample_noise(k1, (B, N_binder, 3), mask_jax.astype(jnp.float32), zero_com=True)
x_lat = sample_noise(k2, (B, N_binder, LATENT_DIM), mask_jax.astype(jnp.float32), zero_com=False)

base_batch = {
    "mask": mask_jax,
    "x_target": jnp.array(target_nm)[None],
    "target_mask": jnp.array(target_a37_mask)[None],
    "seq_target": jnp.array(target_seq_idx)[None],
    "seq_target_mask": jnp.ones((B, N_target), dtype=jnp.bool_),
    "target_hotspot_mask": jnp.ones((B, N_target), dtype=jnp.bool_),
    "strict_feats": False,
}

@eqx.filter_jit
def run_denoiser(model, batch):
    return model(batch)

x_sc_bb = jnp.zeros_like(x_bb)
x_sc_lat = jnp.zeros_like(x_lat)

print(f"Running {N_STEPS}-step ODE...")
for step in range(N_STEPS):
    t_now = float(ts[step])
    dt = float(ts[step + 1] - ts[step])
    step_batch = dict(base_batch)
    step_batch["x_t"] = {"bb_ca": x_bb, "local_latents": x_lat}
    step_batch["t"] = {"bb_ca": jnp.array([t_now]), "local_latents": jnp.array([t_now])}
    step_batch["x_sc"] = {"bb_ca": x_sc_bb, "local_latents": x_sc_lat}
    nn_out = run_denoiser(jax_denoiser, step_batch)
    v_bb = nn_out["bb_ca"]["v"]
    v_lat = nn_out["local_latents"]["v"]
    t_arr = jnp.array([t_now])
    x_sc_bb = predict_x1_from_v(x_bb, v_bb, t_arr)
    x_sc_lat = predict_x1_from_v(x_lat, v_lat, t_arr)
    x_bb = x_bb + v_bb * dt
    x_lat = x_lat + v_lat * dt
    x_bb = force_zero_com(x_bb, mask_jax.astype(jnp.float32))
    if step % 50 == 0:
        jax.block_until_ready(x_bb)
        print(f"  Step {step}/{N_STEPS}")

jax.block_until_ready(x_bb)
print("  Done.")

# ---- Decode with BOTH decoders ----
print("\nDecoding with JAX decoder...")
dec_batch_jax = {
    "z_latent": x_lat,
    "ca_coors_nm": x_bb,
    "residue_mask": mask_jax,
    "mask": mask_jax,
}
with jax.default_matmul_precision("float32"):
    jax_dec_out = jax_decoder(dec_batch_jax)

print("Decoding with PyTorch decoder...")
dec_batch_pt = {
    "z_latent": torch.tensor(np.array(x_lat)),
    "ca_coors_nm": torch.tensor(np.array(x_bb)),
    "residue_mask": torch.ones(B, N_binder, dtype=torch.bool),
    "mask": torch.ones(B, N_binder, dtype=torch.bool),
}
with torch.no_grad():
    pt_dec_out = pt_decoder(dec_batch_pt)

# ---- Compare ----
print("\n" + "=" * 70)
print("Decoder output comparison")
print("=" * 70)

for key in ["seq_logits", "coors_nm", "aatype_max", "atom_mask"]:
    pt_v = pt_dec_out[key].numpy()
    jax_v = np.array(jax_dec_out[key])
    if pt_v.dtype == np.bool_:
        n_diff = np.sum(pt_v != jax_v)
        print(f"  {key:15s}: {n_diff} mismatches")
    else:
        err = np.abs(pt_v.astype(np.float64) - jax_v.astype(np.float64)).max()
        print(f"  {key:15s}: max_err={err:.2e}")

AA_CODES = "ACDEFGHIKLMNPQRSTVWY"
pt_seq = "".join(AA_CODES[i] for i in pt_dec_out["aatype_max"][0].numpy())
jax_seq = "".join(AA_CODES[i] for i in np.array(jax_dec_out["aatype_max"][0]))
print(f"\n  PyTorch seq: {pt_seq}")
print(f"  JAX seq:     {jax_seq}")
print(f"  Match:       {pt_seq == jax_seq}")

# ---- Save both PDBs ----
print("\n" + "=" * 70)
print("Saving PDBs")
print("=" * 70)

atom37_names = rc.atom_types
AA_3LETTER = {
    'A':'ALA','C':'CYS','D':'ASP','E':'GLU','F':'PHE','G':'GLY','H':'HIS',
    'I':'ILE','K':'LYS','L':'LEU','M':'MET','N':'ASN','P':'PRO','Q':'GLN',
    'R':'ARG','S':'SER','T':'THR','V':'VAL','W':'TRP','Y':'TYR',
}

def write_pdb(path, binder_coords_ang, binder_amask, binder_seq, label):
    """Write binder (chain A) + target (chain B) PDB."""
    binder_resnames = [AA_3LETTER.get(aa, "UNK") for aa in binder_seq]
    target_resnames = [AA_3LETTER.get(AA_CODES[idx], "UNK") for idx in target_seq_idx]

    serial = 1
    with open(path, "w") as f:
        f.write(f"REMARK   {label}\n")
        f.write(f"REMARK   Binder: {N_binder} res (chain A), Target: {N_target} res (chain B)\n")
        f.write(f"REMARK   Sequence: {binder_seq}\n")

        # Chain A: binder
        for i in range(N_binder):
            for j in range(37):
                if binder_amask[i, j] < 0.5:
                    continue
                aname = atom37_names[j]
                x, y, z = binder_coords_ang[i, j]
                elem = aname[0] if aname[0] in "CNOS" else aname.strip()[:2]
                f.write(f"ATOM  {serial:5d} {aname:^4s} {binder_resnames[i]:>3s} A{i+1:4d}    "
                        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {elem:>2s}\n")
                serial += 1
        f.write(f"TER   {serial:5d}      {binder_resnames[-1]:>3s} A{N_binder:4d}\n")
        serial += 1

        # Chain B: target
        for i in range(N_target):
            for j in range(37):
                if target_a37_mask[i, j] < 0.5:
                    continue
                aname = atom37_names[j]
                x, y, z = target_a37[i, j]
                elem = aname[0] if aname[0] in "CNOS" else aname.strip()[:2]
                f.write(f"ATOM  {serial:5d} {aname:^4s} {target_resnames[i]:>3s} B{i+1:4d}    "
                        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {elem:>2s}\n")
                serial += 1
        f.write(f"TER   {serial:5d}      {target_resnames[-1]:>3s} B{N_target:4d}\n")
        f.write("END\n")

# JAX decoder PDB
jax_coors = np.array(jax_dec_out["coors_nm"][0]) * 10.0
jax_amask = np.array(jax_dec_out["atom_mask"][0]).astype(np.float32)
write_pdb("binder_complex_jax.pdb", jax_coors, jax_amask, jax_seq, "JAX decoder")
print(f"  JAX:     binder_complex_jax.pdb")

# PyTorch decoder PDB
pt_coors = pt_dec_out["coors_nm"][0].numpy() * 10.0
pt_amask = pt_dec_out["atom_mask"][0].numpy().astype(np.float32)
write_pdb("binder_complex_pytorch.pdb", pt_coors, pt_amask, pt_seq, "PyTorch decoder")
print(f"  PyTorch: binder_complex_pytorch.pdb")

# ---- Structural comparison ----
print("\n" + "=" * 70)
print("Structural comparison")
print("=" * 70)

# All-atom RMSD between the two decoder outputs
common_mask = (jax_amask > 0.5) & (pt_amask > 0.5)
jax_atoms = jax_coors[common_mask]
pt_atoms = pt_coors[common_mask]
rmsd = np.sqrt(np.mean(np.sum((jax_atoms - pt_atoms) ** 2, axis=-1)))
print(f"\n  All-atom RMSD (JAX vs PyTorch decoder): {rmsd:.4f} Å")
print(f"  Max atom deviation: {np.max(np.linalg.norm(jax_atoms - pt_atoms, axis=-1)):.4f} Å")

# Bond length analysis for both
for label, coors in [("JAX", jax_coors), ("PyTorch", pt_coors)]:
    n_ca = np.linalg.norm(coors[:, 1, :] - coors[:, 0, :], axis=1)
    ca_c = np.linalg.norm(coors[:, 2, :] - coors[:, 1, :], axis=1)
    ca_ca = np.linalg.norm(np.diff(coors[:, 1, :], axis=0), axis=1)
    print(f"\n  {label} decoder:")
    print(f"    N-CA bond:  mean={n_ca.mean():.3f}Å, std={n_ca.std():.3f}Å (expected ~1.46Å)")
    print(f"    CA-C bond:  mean={ca_c.mean():.3f}Å, std={ca_c.std():.3f}Å (expected ~1.52Å)")
    print(f"    CA-CA dist: mean={ca_ca.mean():.3f}Å, std={ca_ca.std():.3f}Å (expected ~3.8Å)")

    # Check for clashes (any two atoms < 1.5Å apart)
    amask = jax_amask if label == "JAX" else pt_amask
    all_atoms = []
    for i in range(N_binder):
        for j in range(37):
            if amask[i, j] > 0.5:
                all_atoms.append(coors[i, j])
    all_atoms = np.array(all_atoms)
    dists = np.linalg.norm(all_atoms[:, None, :] - all_atoms[None, :, :], axis=-1)
    np.fill_diagonal(dists, 999)
    # Exclude bonded atoms (rough: same residue)
    n_clash = np.sum(dists < 1.5) // 2  # each pair counted twice
    min_dist = dists.min()
    print(f"    Inter-atom clashes (<1.5Å): {n_clash}")
    print(f"    Min inter-atom distance: {min_dist:.3f}Å")

print(f"\n{'=' * 70}")
print("Done!")
print(f"{'=' * 70}")
