"""Compare one ODE step: PyTorch vs JAX, to find where sampling diverges."""

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
import equinox as eqx
from omegaconf import OmegaConf

# Load PyTorch model
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
del ckpt

# Load JAX model
from jproteina_complexa.serialization import load_model
jax_model = load_model("weights/denoiser")

# Run N steps of plain Euler ODE with BOTH models on same noise
B, N, D = 1, 30, 8
np.random.seed(42)
x_bb_np = np.random.randn(B, N, 3).astype(np.float32) * 0.5
x_lat_np = np.random.randn(B, N, D).astype(np.float32) * 0.5

# Center backbone
x_bb_np -= x_bb_np.mean(axis=1, keepdims=True)

NSTEPS = 10
ts = np.linspace(0, 1, NSTEPS + 1).astype(np.float32)

# PyTorch loop
x_bb_pt = torch.tensor(x_bb_np.copy())
x_lat_pt = torch.tensor(x_lat_np.copy())

# JAX loop
x_bb_jax = jnp.array(x_bb_np.copy())
x_lat_jax = jnp.array(x_lat_np.copy())

from jproteina_complexa.types import DenoiserBatch, NoisyState, Timesteps

print("=" * 60)
print(f"Step-by-step comparison ({NSTEPS} plain Euler steps)")
print("=" * 60)

for step in range(NSTEPS):
    t = float(ts[step])
    dt = float(ts[step + 1] - ts[step])

    # PyTorch
    batch_pt = {
        "x_t": {"bb_ca": x_bb_pt, "local_latents": x_lat_pt},
        "t": {"bb_ca": torch.tensor([t]), "local_latents": torch.tensor([t])},
        "mask": torch.ones(B, N, dtype=torch.bool),
        "x_sc": {"bb_ca": torch.zeros(B, N, 3), "local_latents": torch.zeros(B, N, D)},
        "strict_feats": False,
    }
    with torch.no_grad():
        pt_out = pt_model(batch_pt)
    v_bb_pt = pt_out["bb_ca"]["v"]
    v_lat_pt = pt_out["local_latents"]["v"]

    # JAX
    jax_batch = DenoiserBatch(
        x_t=NoisyState(bb_ca=x_bb_jax, local_latents=x_lat_jax),
        t=Timesteps(bb_ca=jnp.array([t]), local_latents=jnp.array([t])),
        mask=jnp.ones((B, N), dtype=jnp.bool_),
        x_sc=NoisyState(bb_ca=jnp.zeros((B, N, 3)), local_latents=jnp.zeros((B, N, D))),
        target=None,
    )
    jax_out = jax_model(jax_batch)

    # Compare velocities
    err_bb = np.abs(v_bb_pt.numpy() - np.array(jax_out.bb_ca)).max()
    err_lat = np.abs(v_lat_pt.numpy() - np.array(jax_out.local_latents)).max()

    # Euler step
    x_bb_pt = x_bb_pt + v_bb_pt * dt
    x_lat_pt = x_lat_pt + v_lat_pt * dt
    # Center backbone
    x_bb_pt = x_bb_pt - x_bb_pt.mean(dim=1, keepdim=True)

    x_bb_jax = x_bb_jax + jax_out.bb_ca * dt
    x_lat_jax = x_lat_jax + jax_out.local_latents * dt
    from jproteina_complexa.flow_matching import force_zero_com
    x_bb_jax = force_zero_com(x_bb_jax)

    # Compare states after step
    state_err_bb = np.abs(x_bb_pt.numpy() - np.array(x_bb_jax)).max()
    state_err_lat = np.abs(x_lat_pt.numpy() - np.array(x_lat_jax)).max()

    print(f"Step {step}: t={t:.2f} | v_err bb={err_bb:.2e} lat={err_lat:.2e} | state_err bb={state_err_bb:.2e} lat={state_err_lat:.2e}")

# Final check
ca_pt = x_bb_pt[0].numpy() * 10
ca_jax = np.array(x_bb_jax[0]) * 10
ca_dists_pt = np.linalg.norm(np.diff(ca_pt, axis=0), axis=1)
ca_dists_jax = np.linalg.norm(np.diff(ca_jax, axis=0), axis=1)

print(f"\nFinal CA-CA: PT={ca_dists_pt.mean():.3f}A  JAX={ca_dists_jax.mean():.3f}A")
print(f"Max state divergence: bb={np.abs(ca_pt - ca_jax).max():.4f}A")
