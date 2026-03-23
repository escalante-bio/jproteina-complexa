"""Convert PyTorch checkpoints to torch-free .eqx files."""

import sys
import types
import time
import os

# Mocks for transitive deps
sys.modules["torch_scatter"] = types.ModuleType("torch_scatter")
sys.modules["torch_scatter"].scatter_mean = None
sys.path.insert(0, "proteina-complexa/community_models")
sys.path.insert(0, "proteina-complexa/src")
sys.path.insert(0, "src")

import openfold, openfold.np.residue_constants
import torch

class _F(torch.nn.Module):
    def __init__(self, *a, **kw): super().__init__()
    def forward(self, x, *a, **kw): return x

for n, a in [("openfold.model",{}),("openfold.model.dropout",{"DropoutColumnwise":_F,"DropoutRowwise":_F}),
    ("openfold.model.pair_transition",{"PairTransition":_F}),
    ("openfold.model.triangular_attention",{"TriangleAttentionStartingNode":_F,"TriangleAttentionEndingNode":_F}),
    ("openfold.model.triangular_multiplicative_update",{"TriangleMultiplicationIncoming":_F,"TriangleMultiplicationOutgoing":_F}),
    ("openfold.model.structure_module",{"InvariantPointAttention":_F}),
    ("openfold.utils",{}),("openfold.utils.rigid_utils",{"Rigid":None}),
    ("openfold.data",{}),("openfold.data.data_transforms",{"atom37_to_torsion_angles":lambda **kw: None})]:
    m=types.ModuleType(n); [setattr(m,k,v) for k,v in a.items()]; sys.modules[n]=m

from omegaconf import OmegaConf
import jproteina_complexa.nn.register
from jproteina_complexa.backend import from_torch
from jproteina_complexa.serialization import save_model

os.makedirs("weights", exist_ok=True)

# ---- Denoiser ----
print("Converting denoiser...")
ckpt = torch.load("proteina-complexa/ckpts/complexa.ckpt", map_location="cpu", weights_only=False)
nn_cfg = OmegaConf.to_container(ckpt["hyper_parameters"]["cfg_exp"].nn)
for k in ckpt["state_dict"]:
    if "local_latents_linear.1.weight" in k:
        nn_cfg["latent_dim"] = ckpt["state_dict"][k].shape[0]
        break

from proteinfoundation.nn.local_latents_transformer import LocalLatentsTransformer as PTDenoiser
pt = PTDenoiser(**nn_cfg)
pt.load_state_dict({k.removeprefix("nn."): v for k, v in ckpt["state_dict"].items() if k.startswith("nn.")})
pt.eval()
jax_denoiser = from_torch(pt)
save_model(jax_denoiser, "weights/denoiser")
print(f"  Saved weights/denoiser.eqx + weights/denoiser.skeleton.pkl")
del ckpt, pt, jax_denoiser

# ---- Decoder ----
print("Converting decoder...")
ae_ckpt = torch.load("proteina-complexa/ckpts/complexa_ae.ckpt", map_location="cpu", weights_only=False)
ae_cfg = OmegaConf.to_container(ae_ckpt["hyper_parameters"]["cfg_ae"].nn_ae, resolve=True)

from proteinfoundation.partial_autoencoder.decoder import DecoderTransformer as PTDecoder
pt = PTDecoder(**ae_cfg)
pt.load_state_dict({k.removeprefix("decoder."): v for k, v in ae_ckpt["state_dict"].items() if k.startswith("decoder.")})
pt.eval()
jax_decoder = from_torch(pt)
save_model(jax_decoder, "weights/decoder")
print(f"  Saved weights/decoder.eqx + weights/decoder.skeleton.pkl")

# ---- Encoder ----
print("Converting encoder...")
from proteinfoundation.partial_autoencoder.encoder import EncoderTransformer as PTEncoder
pt = PTEncoder(**ae_cfg)
pt.load_state_dict({k.removeprefix("encoder."): v for k, v in ae_ckpt["state_dict"].items() if k.startswith("encoder.")})
pt.eval()
jax_encoder = from_torch(pt)
save_model(jax_encoder, "weights/encoder")
print(f"  Saved weights/encoder.eqx + weights/encoder.skeleton.pkl")
del ae_ckpt, pt, jax_encoder

print("\nDone! All models saved to weights/")
