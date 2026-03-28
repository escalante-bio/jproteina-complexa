"""Test JAX/Equinox decoder against PyTorch ground truth, bottom-up."""

import sys
import types

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

for n, a in [("openfold.model",{}),("openfold.model.dropout",{"DropoutColumnwise":_Fake,"DropoutRowwise":_Fake}),
    ("openfold.model.pair_transition",{"PairTransition":_Fake}),
    ("openfold.model.triangular_attention",{"TriangleAttentionStartingNode":_Fake,"TriangleAttentionEndingNode":_Fake}),
    ("openfold.model.triangular_multiplicative_update",{"TriangleMultiplicationIncoming":_Fake,"TriangleMultiplicationOutgoing":_Fake}),
    ("openfold.model.structure_module",{"InvariantPointAttention":_Fake}),
    ("openfold.utils",{}),("openfold.utils.rigid_utils",{"Rigid":None}),
    ("openfold.data",{}),("openfold.data.data_transforms",{"atom37_to_torsion_angles":lambda **kw:None})]:
    _make_fake_module(n, a)

import pickle
import numpy as np
import jax
import jax.numpy as jnp

# Load test data
print("Loading test data...")
with open("decoder_test_data.pkl", "rb") as f:
    test_data = pickle.load(f)

captures = test_data["captures"]

# Reconstruct PyTorch model
from proteinfoundation.partial_autoencoder.decoder import DecoderTransformer as PTDecoder
pt_model = PTDecoder(**test_data["config"])
pt_model.load_state_dict(test_data["state_dict"])
pt_model.eval()

# Import JAX modules and register converters
import jproteina_complexa.nn.register
from jproteina_complexa.backend import from_torch
from jproteina_complexa.nn.models import DecoderTransformer

from jproteina_complexa.types import DecoderBatch

jax_input = DecoderBatch(
    z_latent=jnp.array(test_data["input"]["z_latent"].numpy()),
    ca_coors_nm=jnp.array(test_data["input"]["ca_coors_nm"].numpy()),
    mask=jnp.array(test_data["input"]["mask"].numpy()),
)

def check(name, jax_out, pt_out_tensor, atol=1e-5):
    pt_np = pt_out_tensor.numpy() if isinstance(pt_out_tensor, torch.Tensor) else np.array(pt_out_tensor)
    jax_np = np.array(jax_out)
    if pt_np.shape != jax_np.shape:
        print(f"  FAIL {name}: shape mismatch pt={pt_np.shape} jax={jax_np.shape}")
        return False
    if pt_np.dtype == np.bool_:
        n_diff = np.sum(pt_np != jax_np)
        print(f"  {'OK' if n_diff == 0 else 'FAIL'} {name}: {n_diff} mismatches (bool)")
        return n_diff == 0
    max_err = np.abs(pt_np.astype(np.float64) - jax_np.astype(np.float64)).max()
    ok = max_err <= atol
    print(f"  {'OK' if ok else 'FAIL'} {name}: max_err={max_err:.2e} (atol={atol:.0e})")
    return ok

# ---- Test leaf modules individually ----
print("\n=== Leaf modules ===")

for path in ["logit_linear.0", "logit_linear.1",
             "transformer_layers.0.mhba.adaln.norm",
             "transition_c_1.swish_linear.1"]:
    cap = captures[path]
    pt_in, pt_out = cap["input"][0], cap["output"]
    parts = path.split(".")
    pt_mod = pt_model
    for p in parts:
        pt_mod = getattr(pt_mod, p) if not p.isdigit() else list(pt_mod.children())[int(p)]
    jax_mod = from_torch(pt_mod)
    jax_out = jax_mod(jnp.array(pt_in.numpy()))
    check(path, jax_out, pt_out)

# ---- Test composites ----
print("\n=== Composites ===")

for path in ["logit_linear", "transformer_layers.0.mhba.adaln",
             "transformer_layers.0.mhba.scale_output", "transition_c_1",
             "transformer_layers.0.transition", "transformer_layers.0.mhba"]:
    cap = captures[path]
    pt_out = cap["output"]
    pt_in = cap["input"]
    parts = path.split(".")
    pt_mod = pt_model
    for p in parts:
        pt_mod = getattr(pt_mod, p) if not p.isdigit() else list(pt_mod.children())[int(p)]
    jax_mod = from_torch(pt_mod)
    jax_args = [jnp.array(a.numpy()) for a in pt_in if isinstance(a, torch.Tensor)]
    if jax_args:
        jax_out = jax_mod(*jax_args)
        check(path, jax_out, pt_out)

# ---- Test transformer block ----
print("\n=== Transformer block ===")
cap = captures["transformer_layers.0"]
pt_in = cap["input"]
pt_out = cap["output"]
jax_mod = from_torch(pt_model.transformer_layers[0])
jax_out = jax_mod(*[jnp.array(a.numpy()) for a in pt_in])
check("transformer_layers.0", jax_out, pt_out)

# ---- Full decoder ----
print("\n=== Full DecoderTransformer ===")
jax_decoder = DecoderTransformer.from_torch(pt_model)

with jax.default_matmul_precision("float32"):
    jax_output = jax_decoder(jax_input)

pt_output = test_data["output"]
check("seq_logits", jax_output.seq_logits, pt_output["seq_logits"], atol=1e-4)
check("coors_nm", jax_output.coors_nm, pt_output["coors_nm"], atol=1e-4)
check("aatype", jax_output.aatype, pt_output["aatype_max"], atol=0)
check("atom_mask", jax_output.atom_mask, pt_output["atom_mask"])

# ---- JIT test ----
print("\n=== JIT test ===")
import equinox as eqx

@eqx.filter_jit
def run_decoder(model, batch):
    return model(batch)

jit_output = run_decoder(jax_decoder, jax_input)
check("jit seq_logits", jit_output.seq_logits, pt_output["seq_logits"], atol=1e-4)
check("jit coors_nm", jit_output.coors_nm, pt_output["coors_nm"], atol=1e-4)

print("\nDone!")
