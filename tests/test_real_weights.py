"""End-to-end test with real pretrained weights on crambin (PDB 1CRN)."""

import sys
import types
import time

# ---- Mock transitive deps ----
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

# ---------------------------------------------------------------------------
import pickle
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import gemmi
from omegaconf import OmegaConf
from proteinfoundation.partial_autoencoder.decoder import DecoderTransformer as PTDecoder

print("=" * 70)
print("Real weights test: pretrained decoder on crambin (PDB 1CRN)")
print("=" * 70)

# ---- Load checkpoint ----
print("\nLoading checkpoint...")
t0 = time.perf_counter()
ckpt = torch.load(
    "proteina-complexa/ckpts/complexa_ae.ckpt",
    map_location="cpu",
    weights_only=False,
)
print(f"  Loaded in {time.perf_counter() - t0:.1f}s")

# Extract decoder config from checkpoint
cfg_ae = ckpt["hyper_parameters"]["cfg_ae"]
decoder_cfg_omegaconf = cfg_ae.nn_ae.decoder
decoder_cfg = OmegaConf.to_container(cfg_ae.nn_ae, resolve=True)
print(f"  token_dim={decoder_cfg['decoder']['token_dim']}, "
      f"nlayers={decoder_cfg['decoder']['nlayers']}, "
      f"latent_z_dim={decoder_cfg['decoder']['latent_z_dim']}")

# Build PyTorch decoder with real config and load weights
print("\nInstantiating PyTorch decoder with real config...")
pt_model = PTDecoder(**decoder_cfg)

# Load only decoder weights
decoder_sd = {
    k.removeprefix("decoder."): v
    for k, v in ckpt["state_dict"].items()
    if k.startswith("decoder.")
}
pt_model.load_state_dict(decoder_sd)
pt_model.eval()

n_params = sum(p.numel() for p in pt_model.parameters())
print(f"  Loaded {len(decoder_sd)} weight tensors ({n_params:,} parameters)")

# Free checkpoint memory
del ckpt

# ---- Load real protein ----
print("\nLoading crambin (PDB 1CRN)...")
structure = gemmi.read_structure("/tmp/pdb/pdb1crn.ent")
polymer = structure[0][0].get_polymer()

AA_3TO1 = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
}

ca_coords_ang = np.array(
    [[a.pos.x, a.pos.y, a.pos.z] for res in polymer for a in res if a.name == "CA"],
    dtype=np.float32,
)
ca_coords_nm = ca_coords_ang / 10.0
N_res = len(ca_coords_nm)
print(f"  {N_res} residues")

true_seq = "".join(AA_3TO1.get(res.name, "X") for res in polymer)
print(f"  True sequence: {true_seq}")

# ---- Build batch ----
B = 1
LATENT_DIM = decoder_cfg["decoder"]["latent_z_dim"]
torch.manual_seed(42)

ca_tensor = torch.tensor(ca_coords_nm).unsqueeze(0)
z_latent = torch.randn(B, N_res, LATENT_DIM) * 0.3
mask = torch.ones(B, N_res, dtype=torch.bool)

batch_pt = {
    "z_latent": z_latent,
    "ca_coors_nm": ca_tensor,
    "residue_mask": mask,
    "mask": mask,
}

# ---- PyTorch forward pass ----
print("\nRunning PyTorch forward pass...")
# Capture per-module data for bottom-up testing
captures = {}
def make_hook(name):
    def hook(module, input, output):
        def detach(x):
            if isinstance(x, torch.Tensor):
                return x.detach().cpu()
            elif isinstance(x, dict):
                return {k: detach(v) for k, v in x.items()}
            elif isinstance(x, (tuple, list)):
                return type(x)(detach(v) for v in x)
            return x
        captures[name] = {
            "input": detach(input),
            "output": detach(output),
        }
    return hook

for name, module in pt_model.named_modules():
    if name:
        module.register_forward_hook(make_hook(name))

t0 = time.perf_counter()
with torch.no_grad():
    pt_output = pt_model(batch_pt)
pt_time = time.perf_counter() - t0
print(f"  PyTorch time: {pt_time:.3f}s")

# ---- Convert to JAX ----
print("\nConverting to JAX/Equinox...")
import jproteina_complexa.nn.register
from jproteina_complexa.backend import from_torch

t0 = time.perf_counter()
jax_decoder = from_torch(pt_model)
convert_time = time.perf_counter() - t0
print(f"  Conversion time: {convert_time:.1f}s")

batch_jax = {
    "z_latent": jnp.array(z_latent.numpy()),
    "ca_coors_nm": jnp.array(ca_tensor.numpy()),
    "residue_mask": jnp.array(mask.numpy()),
    "mask": jnp.array(mask.numpy()),
}

# ---- JAX forward pass ----
print("\nRunning JAX forward pass (eager)...")
t0 = time.perf_counter()
with jax.default_matmul_precision("float32"):
    jax_output = jax_decoder(batch_jax)
jax_eager_time = time.perf_counter() - t0
print(f"  JAX eager time: {jax_eager_time:.3f}s")

@eqx.filter_jit
def run_decoder(model, inp):
    return model(inp)

print("Compiling JAX (JIT)...")
t0 = time.perf_counter()
jit_output = run_decoder(jax_decoder, batch_jax)
jax.block_until_ready(jit_output["coors_nm"])
jit_compile_time = time.perf_counter() - t0
print(f"  JAX JIT compile+run: {jit_compile_time:.1f}s")

t0 = time.perf_counter()
jit_output2 = run_decoder(jax_decoder, batch_jax)
jax.block_until_ready(jit_output2["coors_nm"])
jit_run_time = time.perf_counter() - t0
print(f"  JAX JIT cached run:  {jit_run_time:.3f}s")

# ---- Bottom-up numerical comparison ----
print("\n" + "=" * 70)
print("Bottom-up numerical comparison (real weights)")
print("=" * 70)

test_paths = [
    # Level 1
    "init_repr_factory.linear_out",
    "logit_linear.0",
    "logit_linear.1",
    "transition_c_1.swish_linear.1",
    # Level 2
    "logit_linear",
    "transformer_layers.0.mhba.adaln",
    "transformer_layers.0.mhba.scale_output",
    "transition_c_1",
    # Level 3
    "transformer_layers.0.transition",
    "transformer_layers.0.mhba",
    # Level 4
    "transformer_layers.0",
    "transformer_layers.5",
    "transformer_layers.11",
]

all_ok = True
for path in test_paths:
    if path not in captures:
        print(f"  SKIP {path}: not captured")
        continue
    cap = captures[path]
    pt_out = cap["output"]
    if not isinstance(pt_out, torch.Tensor):
        continue

    # Get the JAX module
    parts = path.split(".")
    pt_mod = pt_model
    for p in parts:
        pt_mod = getattr(pt_mod, p) if not p.isdigit() else list(pt_mod.children())[int(p)]
    jax_mod = from_torch(pt_mod)

    # Run with captured inputs
    inp = cap["input"]
    jax_args = []
    for a in inp:
        if isinstance(a, torch.Tensor):
            jax_args.append(jnp.array(a.numpy()))
        else:
            jax_args.append(a)

    if len(jax_args) > 0:
        with jax.default_matmul_precision("float32"):
            jax_out = jax_mod(*jax_args)
        pt_np = pt_out.numpy()
        jax_np = np.array(jax_out)
        max_err = np.abs(pt_np.astype(np.float64) - jax_np.astype(np.float64)).max()
        ok = max_err < 1e-4
        status = "OK" if ok else "FAIL"
        if not ok:
            all_ok = False
        print(f"  {status} {path}: max_err={max_err:.2e}")

# ---- Full model comparison ----
print("\n" + "=" * 70)
print("Full model output comparison")
print("=" * 70)

AA_CODES = "ACDEFGHIKLMNPQRSTVWY"

for key in ["seq_logits", "coors_nm", "aatype_max", "atom_mask"]:
    pt_np = pt_output[key].numpy()
    jax_np = np.array(jax_output[key])
    jit_np = np.array(jit_output[key])

    if pt_np.dtype == np.bool_:
        n_diff = np.sum(pt_np != jax_np)
        print(f"  {key:15s}: {n_diff} mismatches (bool)")
        if n_diff > 0:
            all_ok = False
    else:
        err = np.abs(pt_np.astype(np.float64) - jax_np.astype(np.float64)).max()
        err_jit = np.abs(pt_np.astype(np.float64) - jit_np.astype(np.float64)).max()
        print(f"  {key:15s}: eager max_err={err:.2e}, jit max_err={err_jit:.2e}")
        if err > 1e-3:
            all_ok = False

# ---- Decoder predictions with real weights ----
print("\n" + "=" * 70)
print("Pretrained decoder predictions on crambin")
print("=" * 70)

pt_seq_idx = pt_output["aatype_max"][0].numpy()
jax_seq_idx = np.array(jax_output["aatype_max"][0])
pt_seq_str = "".join(AA_CODES[i] for i in pt_seq_idx)
jax_seq_str = "".join(AA_CODES[i] for i in jax_seq_idx)

print(f"\n  True sequence:      {true_seq}")
print(f"  Predicted (PyTorch): {pt_seq_str}")
print(f"  Predicted (JAX):     {jax_seq_str}")
print(f"  Sequences match:     {pt_seq_str == jax_seq_str}")

# Sequence recovery
n_match = sum(a == b for a, b in zip(true_seq, pt_seq_str))
print(f"\n  Sequence recovery vs true: {n_match}/{N_res} ({100*n_match/N_res:.1f}%)")
print(f"  (note: decoder sees random latent z, not encoder output — low recovery expected)")

# Logits confidence
from scipy.special import softmax as sp_softmax
logits = np.array(jax_output["seq_logits"][0])
probs = sp_softmax(logits, axis=-1)
max_probs = probs.max(axis=1)
print(f"\n  Sequence confidence: mean={max_probs.mean():.3f}, "
      f"min={max_probs.min():.3f}, max={max_probs.max():.3f}")

# CA preservation check
ca_pred = np.array(jax_output["coors_nm"][0, :, 1, :])  # atom index 1 = CA
ca_input = ca_coords_nm
ca_err = np.abs(ca_pred - ca_input).max()
print(f"  CA coord preservation: max_err={ca_err:.2e} nm")

# Sidechain extent
atom_mask_np = np.array(jax_output["atom_mask"][0])
coors_np = np.array(jax_output["coors_nm"][0])
n_atoms = atom_mask_np.sum(axis=1)
print(f"  Atoms per residue: mean={n_atoms.mean():.1f}, min={n_atoms.min()}, max={n_atoms.max()}")

# Check some specific residue predictions
for i in [0, 10, 20, 30, N_res-1]:
    res_mask = atom_mask_np[i].astype(bool)
    res_coors = coors_np[i, res_mask, :]
    ca_pos = ca_input[i]
    dists = np.linalg.norm(res_coors - ca_pos, axis=1)
    print(f"  Residue {i:2d} true={true_seq[i]} pred={jax_seq_str[i]}: "
          f"{res_mask.sum():2d} atoms, max dist from CA={dists.max():.3f}nm")

# ---- Summary ----
print(f"\n{'=' * 70}")
if all_ok:
    print("ALL CHECKS PASSED — real weights, real protein, numerically equivalent")
else:
    print("SOME CHECKS FAILED — investigate above")
print(f"{'=' * 70}")
