"""Head-to-head: PyTorch (LoRA, v2) vs JAX motif denoiser on identical inputs.

Validates both the LoRA merge (JAX weights are state-dict-merged; PT here uses the
real loralib path) and the motif sequence-concat feature port. Requires the converted
`weights/denoiser_motif.*` and the AME checkpoint under Proteina-Complexa/ckpts.

Run: JAX_PLATFORMS=cpu UPSTREAM=/home/ubuntu/Proteina-Complexa \
     uv run --extra convert python tests/test_motif_denoiser_compare.py
"""
import os, sys, types
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ["USE_V2_COMPLEXA_ARCH"] = "True"

UPSTREAM = os.environ.get("UPSTREAM", "/home/ubuntu/Proteina-Complexa")
CKPT = os.path.join(UPSTREAM, "ckpts", "complexa_ame.ckpt")

sys.modules["torch_scatter"] = types.ModuleType("torch_scatter")
sys.modules["torch_scatter"].scatter_mean = None
sys.path.insert(0, os.path.join(UPSTREAM, "community_models"))
sys.path.insert(0, os.path.join(UPSTREAM, "src"))

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
import jax.numpy as jnp
from omegaconf import OmegaConf

# ---- PyTorch model: v2 + LoRA (real upstream path) ----
ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
nn_cfg = OmegaConf.to_container(ckpt["hyper_parameters"]["cfg_exp"].nn)
sd = ckpt["state_dict"]
for k in sd:
    if "local_latents_linear.1.weight" in k:
        nn_cfg["latent_dim"] = sd[k].shape[0]; break

from proteinfoundation.nn.local_latents_transformer_v2 import LocalLatentsTransformer as PTModel
from proteinfoundation.utils.lora_utils import replace_lora_layers
pt_model = PTModel(**nn_cfg)
replace_lora_layers(pt_model, r=32, lora_alpha=64.0, lora_dropout=0.0)
pt_model.load_state_dict({k.removeprefix("nn."): v for k, v in sd.items() if k.startswith("nn.")}, strict=False)
pt_model.eval()

# ---- JAX model ----
from jproteina_complexa.serialization import load_model
jax_model = load_model("weights/denoiser_motif")

# ---- Inputs: scaffold + synthetic motif ----
B, N, D, Nm = 1, 20, 8, 4
torch.manual_seed(0)
x_t_bb = torch.randn(B, N, 3) * 0.1
x_t_lat = torch.randn(B, N, D) * 0.1
t_val = torch.tensor([0.5])
mask = torch.ones(B, N, dtype=torch.bool)
x_motif_nm = torch.randn(B, Nm, 37, 3) * 0.3       # upstream passes x_motif as coords_nm
motif_mask = torch.ones(B, Nm, 37)
seq_motif = torch.randint(0, 20, (B, Nm))
seq_motif_mask = torch.ones(B, Nm, dtype=torch.bool)

batch_pt = {
    "x_t": {"bb_ca": x_t_bb, "local_latents": x_t_lat},
    "t": {"bb_ca": t_val, "local_latents": t_val},
    "mask": mask,
    "x_sc": {"bb_ca": torch.zeros(B, N, 3), "local_latents": torch.zeros(B, N, D)},
    "x_motif": x_motif_nm,
    "motif_mask": motif_mask,
    "seq_motif": seq_motif,
    "seq_motif_mask": seq_motif_mask,
    "strict_feats": False,
}

from jproteina_complexa.types import DenoiserBatch, NoisyState, Timesteps, MotifCond
jax_batch = DenoiserBatch(
    x_t=NoisyState(bb_ca=jnp.array(x_t_bb[0].numpy()), local_latents=jnp.array(x_t_lat[0].numpy())),
    t=Timesteps(bb_ca=jnp.array(0.5), local_latents=jnp.array(0.5)),
    mask=jnp.ones((N,), dtype=jnp.bool_),
    x_sc=NoisyState(bb_ca=jnp.zeros((N, 3)), local_latents=jnp.zeros((N, D))),
    motif=MotifCond(
        x_motif=jnp.array(x_motif_nm[0].numpy()) * 10.0,   # Å at the JAX boundary
        motif_mask=jnp.array(motif_mask[0].numpy()),
        seq_motif=jnp.array(seq_motif[0].numpy()),
        seq_motif_mask=jnp.array(seq_motif_mask[0].numpy(), dtype=jnp.float32),
    ),
)

print("=" * 60)
print("Motif denoiser comparison: PyTorch(LoRA,v2) vs JAX")
print("=" * 60)

# Seq concat features (motif tokens): compare PT concat_factory motif projection vs JAX module
with torch.no_grad():
    pt_seqrepr = torch.zeros(B, N, nn_cfg["token_dim"])
    pt_ext, pt_extmask = pt_model.concat_factory(dict(batch_pt), pt_seqrepr, mask)
    pt_motif_tokens = pt_ext[0, N:].numpy()           # [Nm, token_dim]
jax_motif_tokens, jax_motif_mask = jax_model.concat_features(jax_batch)
jax_motif_tokens = np.array(jax_motif_tokens)
print(f"\nMotif tokens: PT{pt_motif_tokens.shape} JAX{jax_motif_tokens.shape} "
      f"max_err={np.abs(pt_motif_tokens - jax_motif_tokens).max():.2e}")

# Full forward
with torch.no_grad():
    pt_out = pt_model(batch_pt)
jax_out = jax_model(jax_batch)
ok = True
for dm in ["bb_ca", "local_latents"]:
    key = list(pt_out[dm].keys())[0]
    pt_v = pt_out[dm][key][0].numpy()
    jax_v = np.array(getattr(jax_out, dm))
    err = np.abs(pt_v - jax_v).max()
    ok = ok and err < 1e-3
    print(f"Output {dm}[{key}]: shape PT{pt_v.shape} JAX{jax_v.shape} max_err={err:.2e}")

# ---- Decoder parity (motif AE: complexa_ame_ae.ckpt) ----
print("\n--- Decoder (complexa_ame_ae) ---")
ae_ckpt = torch.load(os.path.join(UPSTREAM, "ckpts", "complexa_ame_ae.ckpt"), map_location="cpu", weights_only=False)
ae_cfg = OmegaConf.to_container(ae_ckpt["hyper_parameters"]["cfg_ae"].nn_ae, resolve=True)
from proteinfoundation.partial_autoencoder.decoder import DecoderTransformer as PTDecoder
pt_decoder = PTDecoder(**ae_cfg)
pt_decoder.load_state_dict({k.removeprefix("decoder."): v for k, v in ae_ckpt["state_dict"].items() if k.startswith("decoder.")})
pt_decoder.eval()
jax_decoder = load_model("weights/decoder_motif")

z_lat = torch.randn(B, N, D) * 0.5
ca_nm = torch.randn(B, N, 3) * 0.3
dec_mask = torch.ones(B, N, dtype=torch.bool)
with torch.no_grad():
    pt_dec = pt_decoder({"z_latent": z_lat, "ca_coors_nm": ca_nm, "mask": dec_mask, "residue_mask": dec_mask})
from jproteina_complexa.types import DecoderBatch
jax_dec = jax_decoder(DecoderBatch(
    z_latent=jnp.array(z_lat[0].numpy()),
    ca_coors=jnp.array(ca_nm[0].numpy()) * 10.0,   # Å at the JAX boundary
    mask=jnp.ones((N,), dtype=jnp.bool_),
))
for pk, jv in [("seq_logits", jax_dec.seq_logits), ("coors_nm", jax_dec.coors * 0.1)]:
    err = np.abs(pt_dec[pk][0].numpy() - np.array(jv)).max()
    ok = ok and err < 1e-3
    print(f"Decoder {pk}: max_err={err:.2e}")

# ---- End-to-end generate sanity (shapes + finiteness + trimming) ----
print("\n--- generate() sanity (motif conditioning) ---")
import jax
from jproteina_complexa.flow_matching import generate
bb, lat = generate(jax_model, jnp.ones((N,), dtype=jnp.bool_), jax.random.PRNGKey(0),
                   nsteps=5, motif=jax_batch.motif)
gen_ok = bb.shape == (N, 3) and lat.shape == (N, D) and bool(jnp.all(jnp.isfinite(bb))) and bool(jnp.all(jnp.isfinite(lat)))
ok = ok and gen_ok
print(f"generate: bb{tuple(bb.shape)} lat{tuple(lat.shape)} finite={bool(jnp.all(jnp.isfinite(bb)) & jnp.all(jnp.isfinite(lat)))} (trim to N={N})")

print("\n" + ("PASS" if ok else "FAIL"))
sys.exit(0 if ok else 1)
