"""Head-to-head comparison: PyTorch vs JAX denoiser on identical inputs."""

import sys, types
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
    ("openfold.model.pair_transition",{"PairTransition":_F}),("openfold.model.triangular_attention",{"TriangleAttentionStartingNode":_F,"TriangleAttentionEndingNode":_F}),
    ("openfold.model.triangular_multiplicative_update",{"TriangleMultiplicationIncoming":_F,"TriangleMultiplicationOutgoing":_F}),
    ("openfold.model.structure_module",{"InvariantPointAttention":_F}),("openfold.utils",{}),("openfold.utils.rigid_utils",{"Rigid":None}),
    ("openfold.data",{}),("openfold.data.data_transforms",{"atom37_to_torsion_angles":lambda **kw:None})]:
    m=types.ModuleType(n); [setattr(m,k,v) for k,v in a.items()]; sys.modules[n]=m

import numpy as np
import jax, jax.numpy as jnp
from omegaconf import OmegaConf

# ---- Load PyTorch model ----
ckpt = torch.load("proteina-complexa/ckpts/complexa.ckpt", map_location="cpu", weights_only=False)
nn_cfg = OmegaConf.to_container(ckpt["hyper_parameters"]["cfg_exp"].nn)
for k in ckpt["state_dict"]:
    if "local_latents_linear.1.weight" in k:
        nn_cfg["latent_dim"] = ckpt["state_dict"][k].shape[0]
        break

from proteinfoundation.nn.local_latents_transformer import LocalLatentsTransformer as PTModel
pt_model = PTModel(**nn_cfg)
pt_model.load_state_dict({k.removeprefix("nn."): v for k, v in ckpt["state_dict"].items() if k.startswith("nn.")})
pt_model.eval()

# ---- Load JAX model ----
from jproteina_complexa.serialization import load_model
jax_model = load_model("weights/denoiser")

# ---- Create identical inputs (no target, simple case first) ----
B, N, D = 1, 20, 8
torch.manual_seed(42)
x_t_bb = torch.randn(B, N, 3) * 0.1
x_t_lat = torch.randn(B, N, D) * 0.1
t_val = torch.tensor([0.5])
mask = torch.ones(B, N, dtype=torch.bool)

batch_pt = {
    "x_t": {"bb_ca": x_t_bb, "local_latents": x_t_lat},
    "t": {"bb_ca": t_val, "local_latents": t_val},
    "mask": mask,
    "x_sc": {"bb_ca": torch.zeros(B, N, 3), "local_latents": torch.zeros(B, N, D)},
    "strict_feats": False,
}

# ---- Compare features at each stage ----
print("=" * 60)
print("Denoiser feature comparison: PyTorch vs JAX")
print("=" * 60)

# 1. Sequence features
with torch.no_grad():
    pt_seq_feats = pt_model.init_repr_factory(batch_pt).numpy()

from jproteina_complexa.types import DenoiserBatch, NoisyState, Timesteps
jax_batch = DenoiserBatch(
    x_t=NoisyState(bb_ca=jnp.array(x_t_bb.numpy()), local_latents=jnp.array(x_t_lat.numpy())),
    t=Timesteps(bb_ca=jnp.array([0.5]), local_latents=jnp.array([0.5])),
    mask=jnp.ones((B, N), dtype=jnp.bool_),
    x_sc=NoisyState(bb_ca=jnp.zeros((B, N, 3)), local_latents=jnp.zeros((B, N, D))),
    target=None,
)
jax_seq_feats = np.array(jax_model.seq_features(jax_batch))
err = np.abs(pt_seq_feats - jax_seq_feats).max()
print(f"\nSeq features: max_err={err:.2e}")

# Check raw feature inputs
print("\n  Raw feature breakdown:")
# PyTorch computes: cat(xt_bb, xt_lat, xsc_bb, xsc_lat, opt_ca, opt_res, hotspot, idx_emb)
# Let's check each
pt_feats = []
for fc in pt_model.init_repr_factory.feat_creators:
    with torch.no_grad():
        f = fc(batch_pt)
    pt_feats.append(f.numpy())
    print(f"    PT {fc.__class__.__name__}: shape={f.shape}, range=[{f.min():.4f}, {f.max():.4f}]")

# JAX features are computed inline, so let's compute them manually
from jproteina_complexa.nn.features import index_embedding
mask_f = jnp.ones((B, N))
idx = jnp.broadcast_to(jnp.arange(1, N+1, dtype=jnp.float32)[None], (B, N))
idx_emb = index_embedding(idx, 256)

jax_raw = [
    jnp.array(x_t_bb.numpy()),      # xt_bb [1,20,3]
    jnp.array(x_t_lat.numpy()),     # xt_lat [1,20,8]
    jnp.zeros((B, N, 3)),           # xsc_bb
    jnp.zeros((B, N, D)),           # xsc_lat
    jnp.zeros((B, N, 3)),           # opt_ca
    jnp.zeros((B, N, 20)),          # opt_res
    jnp.zeros((B, N, 1)),           # hotspot
    idx_emb,                         # idx_emb [1,20,256]
]
for i, (jf, pf) in enumerate(zip(jax_raw, pt_feats)):
    jf_np = np.array(jf)
    if jf_np.shape != pf.shape:
        print(f"    SHAPE MISMATCH feat {i}: JAX={jf_np.shape} PT={pf.shape}")
    else:
        err = np.abs(jf_np - pf).max()
        print(f"    JAX feat {i}: err={err:.2e}")

# 2. Conditioning features
with torch.no_grad():
    pt_cond = pt_model.cond_factory(batch_pt).numpy()
jax_cond = np.array(jax_model.cond_features(jax_batch))
err_cond = np.abs(pt_cond - jax_cond).max()
print(f"\nCond features: max_err={err_cond:.2e}")

# Check raw cond features
print("  PT cond features:")
for fc in pt_model.cond_factory.feat_creators:
    with torch.no_grad():
        f = fc(batch_pt)
    print(f"    {fc.__class__.__name__}: shape={f.shape}, range=[{f.min():.4f}, {f.max():.4f}]")

# 3. Pair features
with torch.no_grad():
    pt_pair = pt_model.pair_repr_builder(batch_pt).numpy()
jax_pair = np.array(jax_model.pair_repr_builder(jax_batch))
err_pair = np.abs(pt_pair - jax_pair).max()
print(f"\nPair features: max_err={err_pair:.2e}")
if err_pair > 1e-4:
    print(f"  PT pair shape={pt_pair.shape}, JAX pair shape={jax_pair.shape}")
    print(f"  PT range=[{pt_pair.min():.4f}, {pt_pair.max():.4f}]")
    print(f"  JAX range=[{jax_pair.min():.4f}, {jax_pair.max():.4f}]")

# 4. Full forward pass
with torch.no_grad():
    pt_out = pt_model(batch_pt)
jax_out = jax_model(jax_batch)

for dm in ["bb_ca", "local_latents"]:
    key = list(pt_out[dm].keys())[0]
    pt_v = pt_out[dm][key].numpy()
    jax_v = np.array(getattr(jax_out, dm.replace("local_latents", "local_latents")))
    err = np.abs(pt_v - jax_v).max()
    print(f"\nOutput {dm}[{key}]: max_err={err:.2e}")
    if err > 1e-3:
        print(f"  PT range=[{pt_v.min():.4f}, {pt_v.max():.4f}]")
        print(f"  JAX range=[{jax_v.min():.4f}, {jax_v.max():.4f}]")

print(f"\n{'=' * 60}")
