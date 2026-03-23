"""Run the actual PyTorch full_simulation and compare with JAX generate, step by step."""

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

print("=" * 70)
print("Full pipeline comparison: PyTorch full_simulation vs JAX generate")
print("=" * 70)

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
del ckpt

# ---- Load PyTorch flow matcher ----
from proteinfoundation.flow_matching.rdn_flow_matcher import RDNFlowMatcher
from proteinfoundation.flow_matching.product_space_flow_matcher import get_schedule as pt_get_schedule, get_gt as pt_get_gt

pt_fm_bb = RDNFlowMatcher(zero_com_noise=True, guidance_enabled=False, dim=3)
pt_fm_lat = RDNFlowMatcher(zero_com_noise=False, guidance_enabled=False, dim=8)

# ---- Load JAX model ----
from jproteina_complexa.serialization import load_model
from jproteina_complexa.flow_matching import get_schedule, get_gt, simulation_step, predict_x1_from_v, force_zero_com, sample_noise
from jproteina_complexa.types import DenoiserBatch, NoisyState, Timesteps

jax_model = load_model("weights/denoiser")

# ---- Config matching production YAML ----
B, N, D = 1, 30, 8
NSTEPS = 20  # Few steps for debugging

# Schedules
ts_bb_pt = pt_get_schedule("log", NSTEPS, p1=2.0)
ts_lat_pt = pt_get_schedule("power", NSTEPS, p1=2.0)
ts_bb_jax = get_schedule("log", NSTEPS, p1=2.0)
ts_lat_jax = get_schedule("power", NSTEPS, p1=2.0)

print("\nSchedule comparison:")
for s in [0, NSTEPS//2, NSTEPS]:
    print(f"  step {s}: bb PT={float(ts_bb_pt[s]):.6f} JAX={float(ts_bb_jax[s]):.6f} | "
          f"lat PT={float(ts_lat_pt[s]):.6f} JAX={float(ts_lat_jax[s]):.6f}")

# ---- Same initial noise ----
torch.manual_seed(99)
mask_pt = torch.ones(B, N, dtype=torch.bool)

# PyTorch noise
x_bb_pt = pt_fm_bb.sample_noise(N, torch.device("cpu"), shape=(B,), mask=mask_pt)
x_lat_pt = pt_fm_lat.sample_noise(N, torch.device("cpu"), shape=(B,), mask=mask_pt)

# Use same noise for JAX
x_bb_jax = jnp.array(x_bb_pt.numpy())
x_lat_jax = jnp.array(x_lat_pt.numpy())

print(f"\nInitial noise: bb range=[{x_bb_pt.min():.3f}, {x_bb_pt.max():.3f}], "
      f"lat range=[{x_lat_pt.min():.3f}, {x_lat_pt.max():.3f}]")

# Self-conditioning init
x_sc_bb_pt = torch.zeros_like(x_bb_pt)
x_sc_lat_pt = torch.zeros_like(x_lat_pt)
x_sc_bb_jax = jnp.zeros_like(x_bb_jax)
x_sc_lat_jax = jnp.zeros_like(x_lat_jax)

# ---- Step-by-step comparison ----
print(f"\nRunning {NSTEPS} steps with production config (log/power, sc mode)...")

step_params = {
    "sampling_mode": "sc",
    "sc_scale_noise": 0.1,
    "sc_scale_score": 1.0,
    "t_lim_ode": 0.98,
    "t_lim_ode_below": 0.02,
    "center_every_step": False,
}

for step in range(NSTEPS):
    t_bb = float(ts_bb_pt[step])
    t_lat = float(ts_lat_pt[step])
    dt_bb = float(ts_bb_pt[step + 1] - ts_bb_pt[step])
    dt_lat = float(ts_lat_pt[step + 1] - ts_lat_pt[step])

    # ---- PyTorch forward ----
    batch_pt = {
        "x_t": {"bb_ca": x_bb_pt, "local_latents": x_lat_pt},
        "t": {"bb_ca": torch.tensor([t_bb]), "local_latents": torch.tensor([t_lat])},
        "mask": mask_pt,
        "x_sc": {"bb_ca": x_sc_bb_pt, "local_latents": x_sc_lat_pt},
        "strict_feats": False,
    }
    with torch.no_grad():
        pt_out = pt_model(batch_pt)
    v_bb_pt = pt_out["bb_ca"]["v"]
    v_lat_pt = pt_out["local_latents"]["v"]

    # ---- JAX forward ----
    jax_batch = DenoiserBatch(
        x_t=NoisyState(bb_ca=x_bb_jax, local_latents=x_lat_jax),
        t=Timesteps(bb_ca=jnp.array([t_bb]), local_latents=jnp.array([t_lat])),
        mask=jnp.ones((B, N), dtype=jnp.bool_),
        x_sc=NoisyState(bb_ca=x_sc_bb_jax, local_latents=x_sc_lat_jax),
        target=None,
    )
    jax_out = jax_model(jax_batch)

    # Compare velocities
    v_err_bb = np.abs(v_bb_pt.numpy() - np.array(jax_out.bb_ca)).max()
    v_err_lat = np.abs(v_lat_pt.numpy() - np.array(jax_out.local_latents)).max()

    # ---- PyTorch self-conditioning ----
    t_bb_t = torch.tensor([t_bb])[:, None, None]
    t_lat_t = torch.tensor([t_lat])[:, None, None]
    x_sc_bb_pt = x_bb_pt + (1 - t_bb_t) * v_bb_pt
    x_sc_lat_pt = x_lat_pt + (1 - t_lat_t) * v_lat_pt

    # ---- JAX self-conditioning ----
    x_sc_bb_jax = predict_x1_from_v(x_bb_jax, jax_out.bb_ca, jnp.array([t_bb]))
    x_sc_lat_jax = predict_x1_from_v(x_lat_jax, jax_out.local_latents, jnp.array([t_lat]))

    sc_err_bb = np.abs(x_sc_bb_pt.numpy() - np.array(x_sc_bb_jax)).max()
    sc_err_lat = np.abs(x_sc_lat_pt.numpy() - np.array(x_sc_lat_jax)).max()

    # ---- PyTorch ODE step (plain Euler for now, matching "vf" mode) ----
    # Actually use the PT flow matcher's simulation_step for proper comparison
    gt_bb_pt = pt_get_gt(torch.tensor([t_bb]), "1/t", 1.0)
    gt_lat_pt = pt_get_gt(torch.tensor([t_lat]), "tan", 1.0)

    # For sc mode: need to check what PT does
    # PT simulation_step for "sc" mode:
    #   score = vf_to_score(x_t, v, t)
    #   if t < t_lim_ode:
    #     dx = (v + gt*score)*dt + sqrt(2*gt*noise_scale)*noise*dt
    #   else:
    #     dx = v*dt

    # For now just compare plain Euler to isolate the NN
    x_bb_pt = x_bb_pt + v_bb_pt * dt_bb
    x_lat_pt = x_lat_pt + v_lat_pt * dt_lat
    # Center backbone
    x_bb_pt = x_bb_pt - x_bb_pt.mean(dim=1, keepdim=True)

    x_bb_jax = x_bb_jax + jax_out.bb_ca * dt_bb
    x_lat_jax = x_lat_jax + jax_out.local_latents * dt_lat
    x_bb_jax = force_zero_com(x_bb_jax)

    state_err_bb = np.abs(x_bb_pt.numpy() - np.array(x_bb_jax)).max()
    state_err_lat = np.abs(x_lat_pt.numpy() - np.array(x_lat_jax)).max()

    if step % 5 == 0 or step == NSTEPS - 1:
        print(f"  step {step:2d}: t_bb={t_bb:.4f} t_lat={t_lat:.4f} | "
              f"v_err bb={v_err_bb:.1e} lat={v_err_lat:.1e} | "
              f"sc_err bb={sc_err_bb:.1e} lat={sc_err_lat:.1e} | "
              f"x_err bb={state_err_bb:.1e} lat={state_err_lat:.1e}")

# Final comparison
ca_pt = x_bb_pt[0].numpy() * 10
ca_jax = np.array(x_bb_jax[0]) * 10
d_pt = np.linalg.norm(np.diff(ca_pt, axis=0), axis=1)
d_jax = np.linalg.norm(np.diff(ca_jax, axis=0), axis=1)

print(f"\nFinal CA-CA: PT={d_pt.mean():.3f}±{d_pt.std():.3f}  JAX={d_jax.mean():.3f}±{d_jax.std():.3f}")
print(f"Max position divergence: {np.abs(ca_pt - ca_jax).max():.4f} A")

# Now check gt comparison
print("\n\ngt comparison:")
for t_val in [0.01, 0.1, 0.5, 0.9, 0.99]:
    gt_pt_bb = float(pt_get_gt(torch.tensor([t_val]), "1/t", 1.0))
    gt_pt_lat = float(pt_get_gt(torch.tensor([t_val]), "tan", 1.0))
    gt_jax_bb = float(get_gt(jnp.array([t_val]), "1/t", 1.0))
    gt_jax_lat = float(get_gt(jnp.array([t_val]), "tan", 1.0))
    print(f"  t={t_val}: bb PT={gt_pt_bb:.4f} JAX={gt_jax_bb:.4f} | lat PT={gt_pt_lat:.4f} JAX={gt_jax_lat:.4f}")

print(f"\n{'=' * 70}")
