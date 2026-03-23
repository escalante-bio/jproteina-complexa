"""Step 0 + 0.5: Instantiate the PyTorch DecoderTransformer and capture test data."""

import sys
import types

# ---------------------------------------------------------------------------
# Mock out heavy transitive dependencies the decoder doesn't actually use.
# The decoder only needs: Linear, LayerNorm, SwiGLU, PairBiasAttention,
# AdaptiveLayerNorm, FeatureFactory (with 4 simple features), and
# RESTYPE_ATOM37_MASK from openfold.
# ---------------------------------------------------------------------------

def _make_fake_module(name, attrs=None):
    """Create a fake module with optional attributes."""
    mod = types.ModuleType(name)
    if attrs:
        for k, v in attrs.items():
            setattr(mod, k, v)
    sys.modules[name] = mod
    return mod

# torch_scatter — used in seq_cond_feats.py (not by decoder features)
_make_fake_module("torch_scatter", {"scatter_mean": None})

# Prevent openfold's eager __init__.py from importing everything.
# We need:
#   openfold.np.residue_constants.RESTYPE_ATOM37_MASK
#   openfold.model.dropout.DropoutColumnwise/DropoutRowwise
#   openfold.model.pair_transition.PairTransition
#   openfold.model.triangular_attention.*
#   openfold.model.triangular_multiplicative_update.*
# But pair_update is not used (update_pair_repr=False), so we just need
# the imports to not crash.

# Patch the openfold __init__.py files to not eagerly import everything
sys.path.insert(0, "proteina-complexa/community_models")
sys.path.insert(0, "proteina-complexa/src")

# Load residue_constants directly (it only needs dm-tree and numpy)
import importlib.util
spec = importlib.util.spec_from_file_location(
    "openfold.np.residue_constants",
    "proteina-complexa/community_models/openfold/np/residue_constants.py",
)
rc_mod = importlib.util.module_from_spec(spec)

# Pre-register openfold package hierarchy so the import chain works
_make_fake_module("openfold")
_make_fake_module("openfold.np")
sys.modules["openfold.np.residue_constants"] = rc_mod
spec.loader.exec_module(rc_mod)

# Now fake the openfold.model imports (pair_update.py needs these at import time)
import torch

class _FakeDropout(torch.nn.Module):
    def __init__(self, *a, **kw): super().__init__()
    def forward(self, x, *a, **kw): return x

class _FakePairTransition(torch.nn.Module):
    def __init__(self, *a, **kw): super().__init__()
    def forward(self, x, *a, **kw): return x

class _FakeTriUpdate(torch.nn.Module):
    def __init__(self, *a, **kw): super().__init__()
    def forward(self, x, *a, **kw): return x

_make_fake_module("openfold.model")
_make_fake_module("openfold.model.dropout", {
    "DropoutColumnwise": _FakeDropout,
    "DropoutRowwise": _FakeDropout,
})
_make_fake_module("openfold.model.pair_transition", {
    "PairTransition": _FakePairTransition,
})
_make_fake_module("openfold.model.triangular_attention", {
    "TriangleAttentionStartingNode": _FakeTriUpdate,
    "TriangleAttentionEndingNode": _FakeTriUpdate,
})
_make_fake_module("openfold.model.triangular_multiplicative_update", {
    "TriangleMultiplicationIncoming": _FakeTriUpdate,
    "TriangleMultiplicationOutgoing": _FakeTriUpdate,
})

# Also fake openfold.data (used by seq_feats.py for sidechain angles — not our features)
_make_fake_module("openfold.data")
_make_fake_module("openfold.data.data_transforms", {
    "atom37_to_torsion_angles": lambda **kw: None,
})

# Now import the decoder
import pickle
from proteinfoundation.partial_autoencoder.decoder import DecoderTransformer

# ---- Build config matching configs/nn_ae/nn_60m.yaml decoder section ----
decoder_cfg = {
    "decoder": {
        "abs_coors": False,
        "nlayers": 12,
        "token_dim": 512,
        "nheads": 12,
        "parallel_mha_transition": False,
        "strict_feats": False,
        "feats_seq": ["ca_coors_nm", "z_latent_seq"],
        "feats_cond_seq": None,  # empty → ZeroFeat
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

print("Instantiating DecoderTransformer...")
model = DecoderTransformer(**decoder_cfg)
model.eval()

n_params = sum(p.numel() for p in model.parameters())
print(f"Decoder parameters: {n_params:,}")

# Print module tree
print("\nModule tree:")
for name, mod in model.named_modules():
    if name:
        print(f"  {name}: {mod.__class__.__name__}")

# ---- Create realistic inputs ----
B, N = 2, 64
LATENT_DIM = 8
torch.manual_seed(42)

# CA coordinates in nanometers — helix-like arrangement
ca_coors_nm = torch.zeros(B, N, 3)
for i in range(N):
    ca_coors_nm[:, i, 0] = i * 0.38 * 0.3
    ca_coors_nm[:, i, 1] = torch.sin(torch.tensor(i * 0.5)) * 0.5
    ca_coors_nm[:, i, 2] = torch.cos(torch.tensor(i * 0.5)) * 0.5
ca_coors_nm[1] = ca_coors_nm[0] + torch.randn(N, 3) * 0.05

z_latent = torch.randn(B, N, LATENT_DIM) * 0.5

mask = torch.ones(B, N, dtype=torch.bool)
mask[1, 50:] = False

batch_input = {
    "z_latent": z_latent,
    "ca_coors_nm": ca_coors_nm,
    "residue_mask": mask,
    "mask": mask,
}

print(f"\nInput shapes:")
print(f"  z_latent:      {z_latent.shape}")
print(f"  ca_coors_nm:   {ca_coors_nm.shape}")
print(f"  residue_mask:  {mask.shape}")

# ---- Step 0.5: Capture per-module ground truth ----
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

for name, module in model.named_modules():
    if name:
        module.register_forward_hook(make_hook(name))

print("\nRunning forward pass...")
with torch.no_grad():
    output = model(batch_input)

print(f"\nOutput shapes:")
for k, v in output.items():
    if isinstance(v, torch.Tensor):
        print(f"  {k}: {v.shape} ({v.dtype})")

print(f"\nCaptured {len(captures)} module activations:")
for name in sorted(captures.keys()):
    out = captures[name]["output"]
    if isinstance(out, torch.Tensor):
        print(f"  {name}: output {out.shape}, range [{out.min():.4f}, {out.max():.4f}]")

# ---- Save test data ----
save_data = {
    "input": {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in batch_input.items()},
    "output": {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in output.items()},
    "captures": captures,
    "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
    "config": decoder_cfg,
}

with open("decoder_test_data.pkl", "wb") as f:
    pickle.dump(save_data, f)

print(f"\nSaved decoder_test_data.pkl ({len(save_data['state_dict'])} weight tensors)")
print("Done!")
