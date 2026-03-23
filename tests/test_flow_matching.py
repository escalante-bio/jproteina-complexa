"""Test flow matching operations: schedules, interpolation, simulation steps."""

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
    ("openfold.model.triangular_multiplicative_update",{"TriangleMultiplicationIncoming":_F,"TriangleMultiplicationOutgoing":_F})]:
    m=types.ModuleType(n); [setattr(m,k,v) for k,v in a.items()]; sys.modules[n]=m

import numpy as np
import jax
import jax.numpy as jnp

from proteinfoundation.flow_matching.product_space_flow_matcher import get_schedule as pt_get_schedule, get_gt as pt_get_gt
from proteinfoundation.flow_matching.rdn_flow_matcher import RDNFlowMatcher

from jproteina_complexa.flow_matching import (
    get_schedule, get_gt, force_zero_com, interpolate,
    predict_x1_from_v, predict_v_from_x1, vf_to_score, score_to_vf,
    simulation_step,
)

print("=" * 60)
print("Flow matching tests")
print("=" * 60)

all_ok = True

def check(name, jax_v, pt_v, atol=1e-5):
    global all_ok
    err = np.abs(np.array(jax_v) - np.array(pt_v)).max()
    ok = err <= atol
    if not ok: all_ok = False
    print(f"  {'OK' if ok else 'FAIL'} {name}: max_err={err:.2e}")
    return ok

# ---- Schedules ----
print("\nSchedules:")
for mode in ["uniform", "power", "loglinear"]:
    p1 = 2.0 if mode in ("power",) else None
    pt_ts = pt_get_schedule(mode, 100, p1=p1).numpy()
    jax_ts = np.array(get_schedule(mode, 100, p1=p1))
    check(f"schedule({mode})", jax_ts, pt_ts, atol=1e-4)

# ---- gt ----
print("\nNoise injection gt:")
t = jnp.linspace(0.01, 0.99, 50)
t_pt = torch.tensor(np.array(t))
for mode in ["1-t/t", "tan", "1/t", "1/t2"]:
    pt_g = pt_get_gt(t_pt, mode, 1.0).numpy()
    jax_g = np.array(get_gt(t, mode, 1.0))
    check(f"gt({mode})", jax_g, pt_g, atol=1e-5)

# ---- Interpolation ----
print("\nInterpolation:")
np.random.seed(42)
x0 = np.random.randn(2, 20, 3).astype(np.float32)
x1 = np.random.randn(2, 20, 3).astype(np.float32)
t_val = np.array([0.3, 0.7], dtype=np.float32)

pt_xt = ((1 - torch.tensor(t_val)[:, None, None]) * torch.tensor(x0) +
         torch.tensor(t_val)[:, None, None] * torch.tensor(x1)).numpy()
jax_xt = np.array(interpolate(jnp.array(x0), jnp.array(x1), jnp.array(t_val)))
check("interpolate", jax_xt, pt_xt)

# ---- Velocity / x1 conversions ----
print("\nVelocity/x1 conversions:")
v = np.random.randn(2, 20, 3).astype(np.float32)
x1_from_v = np.array(predict_x1_from_v(jnp.array(pt_xt), jnp.array(v), jnp.array(t_val)))
v_back = np.array(predict_v_from_x1(jnp.array(pt_xt), jnp.array(x1_from_v), jnp.array(t_val)))
check("v→x1→v roundtrip", v_back, v, atol=1e-4)

# ---- Score conversions ----
print("\nScore conversions:")
score = np.array(vf_to_score(jnp.array(pt_xt), jnp.array(v), jnp.array(t_val)))
v_from_score = np.array(score_to_vf(jnp.array(pt_xt), jnp.array(score), jnp.array(t_val)))
check("v→score→v roundtrip", v_from_score, v, atol=2e-4)  # division by t and (1-t) amplifies fp error

# ---- Center of mass ----
print("\nCenter of mass:")
mask = np.ones((2, 20), dtype=np.float32)
mask[1, 15:] = 0
x_centered = np.array(force_zero_com(jnp.array(x0), jnp.array(mask)))
# Check COM is zero for valid residues
for b in range(2):
    n_valid = int(mask[b].sum())
    com = x_centered[b, :n_valid].mean(axis=0)
    com_err = np.abs(com).max()
    ok = com_err < 1e-6
    if not ok: all_ok = False
    print(f"  {'OK' if ok else 'FAIL'} COM batch {b}: {com_err:.2e}")

# ---- Simulation step (vf mode) ----
print("\nSimulation step (vf):")
key = jax.random.PRNGKey(0)
x_next = np.array(simulation_step(
    jnp.array(pt_xt), jnp.array(v), jnp.array(t_val),
    0.01, jnp.zeros(2), jnp.array(mask), key,
    {"sampling_mode": "vf"},
))
expected = pt_xt + v * 0.01
expected = expected * mask[..., None]
check("step(vf)", x_next, expected)

# ---- Simulation step (vf_ss mode) ----
print("\nSimulation step (vf_ss):")
x_next_ss = np.array(simulation_step(
    jnp.array(pt_xt), jnp.array(v), jnp.array(t_val),
    0.01, jnp.zeros(2), jnp.array(mask), key,
    {"sampling_mode": "vf_ss", "sc_scale_score": 0.5},
))
# Verify it's different from plain vf (score scaling changes the step)
diff = np.abs(x_next_ss - x_next).max()
print(f"  vf_ss differs from vf: max_diff={diff:.4f} (should be >0)")
if diff < 1e-10: all_ok = False

# ---- RDNFlowMatcher comparison ----
print("\nRDNFlowMatcher comparison:")
pt_fm = RDNFlowMatcher(zero_com_noise=True, guidance_enabled=False, dim=3)

# Compare interpolation
pt_x0 = torch.tensor(x0)
pt_x1 = torch.tensor(x1)
pt_t = torch.tensor(t_val)
pt_mask = torch.tensor(mask).bool()

with torch.no_grad():
    pt_interp = pt_fm.interpolate(pt_x0, pt_x1, pt_t, pt_mask).numpy()
jax_interp = np.array(interpolate(jnp.array(x0), jnp.array(x1), jnp.array(t_val), jnp.array(mask)))
check("RDN interpolate", jax_interp, pt_interp, atol=1e-5)

# Compare x1 prediction from v
pt_xt_t = torch.tensor(pt_xt)
nn_out = {"v": torch.tensor(v)}
with torch.no_grad():
    pt_fm.nn_out_add_clean_sample_prediction(pt_xt_t, pt_t, pt_mask, nn_out)
    pt_x1_pred = nn_out["x_1"].numpy()
jax_x1_pred = np.array(predict_x1_from_v(jnp.array(pt_xt), jnp.array(v), jnp.array(t_val)))
jax_x1_pred = jax_x1_pred * np.array(mask)[..., None]  # apply mask like PyTorch does
check("RDN predict_x1", jax_x1_pred, pt_x1_pred, atol=1e-5)

# ---- Summary ----
print(f"\n{'=' * 60}")
if all_ok:
    print("ALL FLOW MATCHING TESTS PASSED")
else:
    print("SOME TESTS FAILED")
print(f"{'=' * 60}")
