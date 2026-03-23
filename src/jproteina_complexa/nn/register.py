"""Register from_torch converters for proteina-complexa PyTorch module types.

Only needed during weight conversion. Not imported at inference time.
"""

from jproteina_complexa.backend import from_torch
from jproteina_complexa.nn.primitives import SwiGLU
from jproteina_complexa.nn.adaptive import AdaptiveLayerNorm, AdaptiveOutputScale
from jproteina_complexa.nn.transition import Transition, TransitionADALN
from jproteina_complexa.nn.attention import PairBiasAttention, MultiHeadBiasedAttentionADALN_MM
from jproteina_complexa.nn.transformer import MultiheadAttnAndTransition
from jproteina_complexa.nn.decoder import DecoderTransformer
from jproteina_complexa.nn.encoder import EncoderTransformer
from jproteina_complexa.nn.local_latents_transformer import LocalLatentsTransformer

# PyTorch modules (for type dispatch)
import proteinfoundation.nn.modules.swiglu as _sw
import proteinfoundation.nn.modules.adaptive_ln_scale as _ad
import proteinfoundation.nn.modules.seq_transition_af3 as _tr
import proteinfoundation.nn.modules.pair_bias_attn as _at
import proteinfoundation.nn.modules.attn_n_transition as _tf
import proteinfoundation.partial_autoencoder.decoder as _dec
import proteinfoundation.partial_autoencoder.encoder as _enc
import proteinfoundation.nn.local_latents_transformer as _llt

from_torch.register(_sw.SwiGLU, lambda _: SwiGLU())
from_torch.register(_ad.AdaptiveLayerNorm, AdaptiveLayerNorm.from_torch)
from_torch.register(_ad.AdaptiveOutputScale, AdaptiveOutputScale.from_torch)
from_torch.register(_tr.Transition, Transition.from_torch)
from_torch.register(_tr.TransitionADALN, TransitionADALN.from_torch)
from_torch.register(_at.PairBiasAttention, PairBiasAttention.from_torch)
from_torch.register(_at.MultiHeadBiasedAttentionADALN_MM, MultiHeadBiasedAttentionADALN_MM.from_torch)
from_torch.register(_tf.MultiheadAttnAndTransition, MultiheadAttnAndTransition.from_torch)
from_torch.register(_dec.DecoderTransformer, DecoderTransformer.from_torch)
from_torch.register(_enc.EncoderTransformer, EncoderTransformer.from_torch)
from_torch.register(_llt.LocalLatentsTransformer, LocalLatentsTransformer.from_torch)
