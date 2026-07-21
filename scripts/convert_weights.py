"""Convert PyTorch checkpoints to torch-free .eqx files.

Two variants:
  * base  (default): protein-binder denoiser (complexa.ckpt, v1) + AE (complexa_ae.ckpt)
  * motif (VARIANT=motif): monomer protein-motif denoiser (complexa_ame.ckpt, v2 +
           LoRA-merged) + motif AE (complexa_ame_ae.ckpt)

LoRA adapters are folded into the base weights directly in the state dict
(W += (alpha/r) * B @ A), so the plain modules load cleanly and no loralib
dependency or Equinox LoRA module is needed. Only the motif projections are used;
all ligand submodules are ignored by the JAX motif path.
"""

import sys
import types
import os

# Mocks for transitive deps
UPSTREAM = os.environ.get("UPSTREAM", os.path.join(os.path.dirname(__file__), "..", "proteina-complexa"))
CKPT_DIR = os.environ.get("CKPT_DIR", os.path.join(UPSTREAM, "ckpts"))
VARIANT = os.environ.get("VARIANT", "base")  # "base" | "motif"

sys.modules["torch_scatter"] = types.ModuleType("torch_scatter")
sys.modules["torch_scatter"].scatter_mean = None
sys.path.insert(0, os.path.join(UPSTREAM, "community_models"))
sys.path.insert(0, os.path.join(UPSTREAM, "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

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

OUT_DIR = os.environ.get("OUT_DIR", "weights")
os.makedirs(OUT_DIR, exist_ok=True)


def merge_lora_(state_dict, lora_alpha, r):
    """Fold LoRA adapters into base weights in-place and strip lora_A/lora_B keys.

    For a base weight `W` with companion `lora_A`/`lora_B`, sets
    `W += (lora_alpha / r) * (lora_B @ lora_A)` (transposing when the product
    matches W only after transpose, i.e. LoRA'd Embeddings). Returns the count merged.
    """
    scaling = lora_alpha / r
    prefixes = sorted({k[: -len("lora_A")] for k in state_dict if k.endswith("lora_A")})
    n = 0
    for p in prefixes:
        A = state_dict.pop(p + "lora_A")            # [r, in]
        B = state_dict.pop(p + "lora_B")            # [out, r]
        w_key = p + "weight"
        W = state_dict[w_key]
        delta = (B @ A) * scaling
        if delta.shape == W.shape:
            state_dict[w_key] = W + delta
        elif delta.t().shape == W.shape:            # LoRA'd Embedding
            state_dict[w_key] = W + delta.t()
        else:
            raise ValueError(f"LoRA delta {tuple(delta.shape)} incompatible with {w_key} {tuple(W.shape)}")
        n += 1
    return n


def _load_nn_cfg(ckpt):
    nn_cfg = OmegaConf.to_container(ckpt["hyper_parameters"]["cfg_exp"].nn)
    for k in ckpt["state_dict"]:
        if "local_latents_linear.1.weight" in k:
            nn_cfg["latent_dim"] = ckpt["state_dict"][k].shape[0]
            break
    return nn_cfg


def convert_denoiser(ckpt_name, out_name, *, motif=False, lora_alpha=64.0, r=32):
    print(f"Converting denoiser ({'motif/v2' if motif else 'base/v1'})...")
    ckpt = torch.load(os.path.join(CKPT_DIR, ckpt_name), map_location="cpu", weights_only=False)
    nn_cfg = _load_nn_cfg(ckpt)
    nn_sd = {k.removeprefix("nn."): v for k, v in ckpt["state_dict"].items() if k.startswith("nn.")}

    if motif:
        merged = merge_lora_(nn_sd, lora_alpha, r)
        print(f"  merged {merged} LoRA adapters (scaling={lora_alpha / r})")
        from proteinfoundation.nn.local_latents_transformer_v2 import LocalLatentsTransformer as PT
    else:
        from proteinfoundation.nn.local_latents_transformer import LocalLatentsTransformer as PT

    pt = PT(**nn_cfg)
    missing, unexpected = pt.load_state_dict(nn_sd, strict=False)
    # Ligand submodules are unused by the motif path; allow their keys to remain unexpected-free.
    assert not missing, f"missing keys: {missing[:8]}"
    pt.eval()
    jax_model = from_torch(pt)
    save_model(jax_model, os.path.join(OUT_DIR, out_name))
    print(f"  Saved {OUT_DIR}/{out_name}.eqx + .skeleton.pkl")


def convert_ae(ckpt_name, decoder_out, encoder_out):
    print("Converting autoencoder (decoder + encoder)...")
    ae_ckpt = torch.load(os.path.join(CKPT_DIR, ckpt_name), map_location="cpu", weights_only=False)
    ae_cfg = OmegaConf.to_container(ae_ckpt["hyper_parameters"]["cfg_ae"].nn_ae, resolve=True)

    from proteinfoundation.partial_autoencoder.decoder import DecoderTransformer as PTDecoder
    pt = PTDecoder(**ae_cfg)
    pt.load_state_dict({k.removeprefix("decoder."): v for k, v in ae_ckpt["state_dict"].items() if k.startswith("decoder.")})
    pt.eval()
    save_model(from_torch(pt), os.path.join(OUT_DIR, decoder_out))
    print(f"  Saved {OUT_DIR}/{decoder_out}.eqx + .skeleton.pkl")

    from proteinfoundation.partial_autoencoder.encoder import EncoderTransformer as PTEncoder
    pt = PTEncoder(**ae_cfg)
    pt.load_state_dict({k.removeprefix("encoder."): v for k, v in ae_ckpt["state_dict"].items() if k.startswith("encoder.")})
    pt.eval()
    save_model(from_torch(pt), os.path.join(OUT_DIR, encoder_out))
    print(f"  Saved {OUT_DIR}/{encoder_out}.eqx + .skeleton.pkl")


if VARIANT == "motif":
    convert_denoiser("complexa_ame.ckpt", "denoiser_motif", motif=True)
    convert_ae("complexa_ame_ae.ckpt", "decoder_motif", "encoder_motif")
else:
    convert_denoiser("complexa.ckpt", "denoiser")
    convert_ae("complexa_ae.ckpt", "decoder", "encoder")

print(f"\nDone! Models saved to {OUT_DIR}/")
