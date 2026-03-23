"""Run PyTorch binder generation against 1UBQ with Ile44 patch hotspots.
Same conditions as JAX: 80 residues, 400 steps, log/power schedules, sc mode, seed 42.
"""
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
    ("openfold.model.pair_transition",{"PairTransition":_F}),
    ("openfold.model.triangular_attention",{"TriangleAttentionStartingNode":_F,"TriangleAttentionEndingNode":_F}),
    ("openfold.model.triangular_multiplicative_update",{"TriangleMultiplicationIncoming":_F,"TriangleMultiplicationOutgoing":_F}),
    ("openfold.model.structure_module",{"InvariantPointAttention":_F})]:
    m = types.ModuleType(n); [setattr(m, k, v) for k, v in a.items()]; sys.modules[n] = m

import numpy as np, time
import gemmi
from openfold.np import residue_constants as rc
from omegaconf import OmegaConf
from proteinfoundation.flow_matching.product_space_flow_matcher import get_schedule
from proteinfoundation.flow_matching.rdn_flow_matcher import RDNFlowMatcher

# ---- Load target ----
structure = gemmi.read_structure("1UBQ.pdb")
polymer = structure[0][0].get_polymer()
N_t = len(polymer)
AA = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
AA3 = {"ALA":"A","CYS":"C","ASP":"D","GLU":"E","PHE":"F","GLY":"G","HIS":"H","ILE":"I","LYS":"K","LEU":"L","MET":"M","ASN":"N","PRO":"P","GLN":"Q","ARG":"R","SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V"}

t_a37 = np.zeros((N_t, 37, 3), dtype=np.float32)
t_mask = np.zeros((N_t, 37), dtype=np.float32)
t_seq = np.zeros(N_t, dtype=np.int64)
for i, res in enumerate(polymer):
    t_seq[i] = AA.get(AA3.get(res.name, "A"), 0)
    for atom in res:
        if atom.name in rc.atom_types:
            j = rc.atom_types.index(atom.name)
            t_a37[i, j] = [atom.pos.x, atom.pos.y, atom.pos.z]
            t_mask[i, j] = 1.0
com = t_a37[:, 1, :].mean(0)
t_a37 -= com[None, None, :]
t_nm = t_a37 / 10.0

# Hotspot mask: Ile44 patch (residues 8, 44, 68, 70 — 0-indexed: 7, 43, 67, 69)
hotspot = torch.zeros(1, N_t, dtype=torch.bool)
for r in [7, 43, 67, 69]:  # 0-indexed
    hotspot[0, r] = True
print(f"Target: {N_t} residues, hotspots: {hotspot.sum().item()}")

# ---- Load model ----
ckpt = torch.load("proteina-complexa/ckpts/complexa.ckpt", map_location="cpu", weights_only=False)
nn_cfg = OmegaConf.to_container(ckpt["hyper_parameters"]["cfg_exp"].nn)
for k in ckpt["state_dict"]:
    if "local_latents_linear.1.weight" in k:
        nn_cfg["latent_dim"] = ckpt["state_dict"][k].shape[0]
        break
from proteinfoundation.nn.local_latents_transformer import LocalLatentsTransformer as PTM
pt = PTM(**nn_cfg)
pt.load_state_dict({k.removeprefix("nn."): v for k, v in ckpt["state_dict"].items() if k.startswith("nn.")})
pt.eval()
del ckpt

# ---- Generate ----
N_b = 80
NSTEPS = 400
mask = torch.ones(1, N_b, dtype=torch.bool)
ts_bb = get_schedule("log", NSTEPS, p1=2.0)
ts_lat = get_schedule("power", NSTEPS, p1=2.0)

torch.manual_seed(42)
x_bb = torch.randn(1, N_b, 3)
x_bb = x_bb - x_bb.mean(1, keepdim=True)
x_lat = torch.randn(1, N_b, 8)
x_sc_bb = torch.zeros_like(x_bb)
x_sc_lat = torch.zeros_like(x_lat)

target_batch = {
    "x_target": torch.tensor(t_nm)[None],
    "target_mask": torch.tensor(t_mask)[None],
    "seq_target": torch.tensor(t_seq)[None],
    "seq_target_mask": torch.ones(1, N_t, dtype=torch.bool),
    "target_hotspot_mask": hotspot,
}

print(f"Generating {N_b}-residue binder ({NSTEPS} steps, seed=42)...")
t0 = time.perf_counter()
with torch.no_grad():
    for step in range(NSTEPS):
        t_bb = float(ts_bb[step])
        t_lat = float(ts_lat[step])
        dt_bb = float(ts_bb[step + 1] - ts_bb[step])
        dt_lat = float(ts_lat[step + 1] - ts_lat[step])
        batch = {
            "x_t": {"bb_ca": x_bb, "local_latents": x_lat},
            "t": {"bb_ca": torch.tensor([t_bb]), "local_latents": torch.tensor([t_lat])},
            "mask": mask,
            "x_sc": {"bb_ca": x_sc_bb, "local_latents": x_sc_lat},
            "strict_feats": False,
        }
        batch.update(target_batch)
        out = pt(batch)
        v_bb = out["bb_ca"]["v"]
        v_lat = out["local_latents"]["v"]
        x_sc_bb = x_bb + (1 - torch.tensor([t_bb])[:, None, None]) * v_bb
        x_sc_lat = x_lat + (1 - torch.tensor([t_lat])[:, None, None]) * v_lat
        x_bb = x_bb + v_bb * dt_bb
        x_lat = x_lat + v_lat * dt_lat
        x_bb = x_bb - x_bb.mean(1, keepdim=True)

gen_time = time.perf_counter() - t0
print(f"  {gen_time:.1f}s")

ca = x_bb[0].numpy() * 10
ca_dists = np.linalg.norm(np.diff(ca, axis=0), axis=1)
rg = np.sqrt(np.mean(np.sum((ca - ca.mean(0)) ** 2, axis=1)))
tca = t_a37[:, 1, :]
min_dists = np.min(np.linalg.norm(ca[:, None] - tca[None], axis=-1), axis=1)

print(f"\nPyTorch results:")
print(f"  CA-CA: {ca_dists.mean():.2f} +/- {ca_dists.std():.2f} A")
print(f"  Rg: {rg:.1f} A")
print(f"  Contacts <8A: {(min_dists < 8).sum()}/{N_b}")
print(f"  Contacts <12A: {(min_dists < 12).sum()}/{N_b}")
print(f"  Min dist to target: {min_dists.min():.1f} A")
print(f"  Mean dist to target: {min_dists.mean():.1f} A")
