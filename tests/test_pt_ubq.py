"""Run PyTorch binder generation against 1UBQ and report metrics."""
import sys, types
sys.modules["torch_scatter"] = types.ModuleType("torch_scatter")
sys.modules["torch_scatter"].scatter_mean = None
sys.path.insert(0, "proteina-complexa/community_models")
sys.path.insert(0, "proteina-complexa/src")
import openfold, openfold.np.residue_constants
import torch, numpy as np
from openfold.np import residue_constants as rc
class _F(torch.nn.Module):
    def __init__(self, *a, **kw): super().__init__()
    def forward(self, x, *a, **kw): return x
for n, a in [("openfold.model",{}),("openfold.model.dropout",{"DropoutColumnwise":_F,"DropoutRowwise":_F}),("openfold.model.pair_transition",{"PairTransition":_F}),("openfold.model.triangular_attention",{"TriangleAttentionStartingNode":_F,"TriangleAttentionEndingNode":_F}),("openfold.model.triangular_multiplicative_update",{"TriangleMultiplicationIncoming":_F,"TriangleMultiplicationOutgoing":_F}),("openfold.model.structure_module",{"InvariantPointAttention":_F})]:
    m=types.ModuleType(n); [setattr(m,k,v) for k,v in a.items()]; sys.modules[n]=m
# Need real openfold.data + openfold.utils for target feature computation
from omegaconf import OmegaConf
import gemmi

# Load target
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

# Load model
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
from proteinfoundation.flow_matching.product_space_flow_matcher import get_schedule

N_b = 60
mask = torch.ones(1, N_b, dtype=torch.bool)
ts_bb = get_schedule("log", 200, p1=2.0)
ts_lat = get_schedule("power", 200, p1=2.0)

torch.manual_seed(42)
x_bb = torch.randn(1, N_b, 3)
x_bb = x_bb - x_bb.mean(1, keepdim=True)
x_lat = torch.randn(1, N_b, 8)
x_sc_bb = torch.zeros_like(x_bb)
x_sc_lat = torch.zeros_like(x_lat)

tb = {
    "x_target": torch.tensor(t_nm)[None],
    "target_mask": torch.tensor(t_mask)[None],
    "seq_target": torch.tensor(t_seq)[None],
    "seq_target_mask": torch.ones(1, N_t, dtype=torch.bool),
    "target_hotspot_mask": torch.ones(1, N_t, dtype=torch.bool),
}

print("Running PyTorch generation with 1UBQ target...")
with torch.no_grad():
    for step in range(200):
        t_bb = float(ts_bb[step])
        t_lat = float(ts_lat[step])
        dt_bb = float(ts_bb[step + 1] - ts_bb[step])
        dt_lat = float(ts_lat[step + 1] - ts_lat[step])
        b = {
            "x_t": {"bb_ca": x_bb, "local_latents": x_lat},
            "t": {"bb_ca": torch.tensor([t_bb]), "local_latents": torch.tensor([t_lat])},
            "mask": mask,
            "x_sc": {"bb_ca": x_sc_bb, "local_latents": x_sc_lat},
            "strict_feats": False,
        }
        b.update(tb)
        o = pt(b)
        v_bb = o["bb_ca"]["v"]
        v_lat = o["local_latents"]["v"]
        x_sc_bb = x_bb + (1 - torch.tensor([t_bb])[:, None, None]) * v_bb
        x_sc_lat = x_lat + (1 - torch.tensor([t_lat])[:, None, None]) * v_lat
        x_bb = x_bb + v_bb * dt_bb
        x_lat = x_lat + v_lat * dt_lat
        x_bb = x_bb - x_bb.mean(1, keepdim=True)

ca = x_bb[0].numpy() * 10
d = np.linalg.norm(np.diff(ca, axis=0), axis=1)
tca = t_a37[:, 1, :]
md = np.min(np.linalg.norm(ca[:, None] - tca[None], axis=-1), axis=1)

print(f"PyTorch results:")
print(f"  CA-CA: {d.mean():.3f} +/- {d.std():.3f}")
print(f"  Contacts <8A: {(md < 8).sum()}/{N_b}")
print(f"  Closest approach: {md.min():.1f}A")

# Now run JAX on same initial noise
import jax, jax.numpy as jnp, equinox as eqx
from jproteina_complexa.serialization import load_model
from jproteina_complexa.flow_matching import generate, get_schedule as jax_get_schedule
from jproteina_complexa.types import DenoiserBatch, DecoderBatch, NoisyState, Timesteps, TargetCond

jax_model = load_model("weights/denoiser")

# Use SAME initial noise
x_bb_jax = jnp.array(x_bb_init := (torch.manual_seed(42), torch.randn(1, N_b, 3))[1].numpy())
x_bb_jax = x_bb_jax - x_bb_jax.mean(1, keepdims=True)
x_lat_jax = jnp.array(torch.randn(1, N_b, 8).numpy())

mask_jax = jnp.ones((1, N_b), dtype=jnp.bool_)
# Use preprocessed target (computed with PyTorch/openfold for exact match)
tgt_data = np.load("targets/1UBQ.npz")

target_jax = TargetCond(
    coords=jnp.array(tgt_data["coords"] / 10.0)[None],
    atom_mask=jnp.array(tgt_data["coord_mask"])[None],
    seq=jnp.array(tgt_data["seq"])[None],
    seq_mask=jnp.ones((1, N_t), dtype=jnp.bool_),
    hotspot_mask=jnp.ones((1, N_t), dtype=jnp.bool_),
    sidechain_feat=jnp.array(tgt_data["sidechain_feat"])[None],
    torsion_feat=jnp.array(tgt_data["torsion_feat"])[None],
)

# Manual JAX loop with same schedule, same Euler, no SDE
ts_bb_j = jax_get_schedule("log", 200, p1=2.0)
ts_lat_j = jax_get_schedule("power", 200, p1=2.0)
from jproteina_complexa.flow_matching import predict_x1_from_v, force_zero_com

x_bb_j = x_bb_jax
x_lat_j = x_lat_jax
x_sc_bb_j = jnp.zeros_like(x_bb_j)
x_sc_lat_j = jnp.zeros_like(x_lat_j)

jit_model = eqx.filter_jit(jax_model)

print("\nRunning JAX generation with same noise and target...")
for step in range(200):
    t_bb = float(ts_bb_j[step])
    t_lat = float(ts_lat_j[step])
    dt_bb = float(ts_bb_j[step + 1] - ts_bb_j[step])
    dt_lat = float(ts_lat_j[step + 1] - ts_lat_j[step])

    batch_j = DenoiserBatch(
        x_t=NoisyState(bb_ca=x_bb_j, local_latents=x_lat_j),
        t=Timesteps(bb_ca=jnp.array([t_bb]), local_latents=jnp.array([t_lat])),
        mask=mask_jax,
        x_sc=NoisyState(bb_ca=x_sc_bb_j, local_latents=x_sc_lat_j),
        target=target_jax,
    )
    out_j = jit_model(batch_j)

    x_sc_bb_j = predict_x1_from_v(x_bb_j, out_j.bb_ca, jnp.array([t_bb]))
    x_sc_lat_j = predict_x1_from_v(x_lat_j, out_j.local_latents, jnp.array([t_lat]))

    x_bb_j = x_bb_j + out_j.bb_ca * dt_bb
    x_lat_j = x_lat_j + out_j.local_latents * dt_lat
    x_bb_j = force_zero_com(x_bb_j)

jax.block_until_ready(x_bb_j)

ca_j = np.array(x_bb_j[0]) * 10
d_j = np.linalg.norm(np.diff(ca_j, axis=0), axis=1)
md_j = np.min(np.linalg.norm(ca_j[:, None] - tca[None], axis=-1), axis=1)

print(f"JAX results:")
print(f"  CA-CA: {d_j.mean():.3f} +/- {d_j.std():.3f}")
print(f"  Contacts <8A: {(md_j < 8).sum()}/{N_b}")
print(f"  Closest approach: {md_j.min():.1f}A")

# Direct comparison
state_err = np.abs(ca - ca_j).max()
print(f"\nMax position divergence PT vs JAX: {state_err:.4f} A")
