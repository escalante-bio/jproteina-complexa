"""Download and load pretrained models from HuggingFace Hub."""

import os

from jproteina_complexa.serialization import load_model

HF_REPO = "escalante-bio/jproteina-complexa"
DEFAULT_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "jproteina_complexa", "weights_v2")
_MODEL_NAMES = ("denoiser", "decoder", "encoder")


def ensure_weights(cache_dir: str = DEFAULT_CACHE, models: tuple[str, ...] = _MODEL_NAMES):
    """Download model weights from HuggingFace if not already cached.

    Args:
        cache_dir: local directory to store weights
        models: which models to download (default: all three)
    """
    from huggingface_hub import hf_hub_download

    for name in models:
        for ext in (".eqx", ".skeleton.pkl"):
            path = os.path.join(cache_dir, f"{name}{ext}")
            if not os.path.exists(path):
                os.makedirs(cache_dir, exist_ok=True)
                hf_hub_download(HF_REPO, f"{name}{ext}", local_dir=cache_dir)


def load_denoiser(cache_dir: str = DEFAULT_CACHE):
    """Load the denoiser (LocalLatentsTransformer), downloading if needed."""
    ensure_weights(cache_dir, models=("denoiser",))
    return load_model(os.path.join(cache_dir, "denoiser"))


def load_decoder(cache_dir: str = DEFAULT_CACHE):
    """Load the decoder (DecoderTransformer), downloading if needed."""
    ensure_weights(cache_dir, models=("decoder",))
    return load_model(os.path.join(cache_dir, "decoder"))


def load_encoder(cache_dir: str = DEFAULT_CACHE):
    """Load the encoder (EncoderTransformer), downloading if needed."""
    ensure_weights(cache_dir, models=("encoder",))
    return load_model(os.path.join(cache_dir, "encoder"))
