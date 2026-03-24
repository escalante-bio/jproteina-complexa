"""Flow matching for (R^d)^n — sampling and SDE integration."""

import jax
import jax.numpy as jnp
import equinox as eqx
from functools import partial
from typing import Callable


# ---- Schedule functions: (nsteps: int) -> Array[nsteps+1] ----

def _clamp_schedule(ts, eps=1e-5):
    """Clamp schedule to [0, 1] with endpoints pinned."""
    ts = jnp.clip(ts, eps, 1 - eps)
    ts = ts.at[0].set(0.0)
    ts = ts.at[-1].set(1.0)
    return ts


def power_schedule(nsteps, *, p=2.0, eps=1e-5):
    return _clamp_schedule(jnp.linspace(0, 1, nsteps + 1) ** p, eps)


def log_schedule(nsteps, *, p=2.0, eps=1e-5):
    return _clamp_schedule(1 - jnp.flip(jnp.logspace(-p, 0, nsteps + 1)), eps)


# ---- Noise injection functions: (t: Array) -> Array ----

def noise_k_over_t_pow(t, *, k=1.0, p=1.0, eps=1e-2):
    """Noise scale k / (t^p + eps)."""
    return k / (jnp.clip(t, 0, 1 - 1e-5) ** p + eps)


def noise_tan(t, *, eps=1e-2):
    """Noise scale (pi/2) * tan((1-t)*pi/2)."""
    t = jnp.clip(t, 0, 1 - 1e-5)
    return (jnp.pi / 2) * jnp.sin((1 - t) * jnp.pi / 2) / (jnp.cos((1 - t) * jnp.pi / 2) + eps)


noise_1_over_t = partial(noise_k_over_t_pow, k=1.0, p=1.0)


# ---- Core flow matching operations ----

def force_zero_com(x, mask=None):
    """Center x to zero center-of-mass along residue dimension."""
    if mask is None:
        return x - x.mean(axis=-2, keepdims=True)
    nres = mask.sum(axis=-1, keepdims=True)[..., None]  # [*, 1, 1]
    com = (x * mask[..., None]).sum(axis=-2, keepdims=True) / jnp.clip(nres, 1)
    return (x - com) * mask[..., None]


def sample_noise(key, shape, mask=None, zero_com=True):
    """Sample Gaussian noise. shape = (batch, n_residues, dim)."""
    x = jax.random.normal(key, shape)
    if mask is not None:
        x = x * mask[..., None]
    if zero_com:
        x = force_zero_com(x, mask)
    return x


def predict_x1_from_v(x_t, v, t):
    """Recover clean sample from velocity: x_1 = x_t + (1-t)*v."""
    t_exp = t[..., None, None]
    return x_t + (1 - t_exp) * v


def vf_to_score(x_t, v, t):
    """Convert velocity field to score: score = (t*v - x_t) / (1-t)."""
    t_exp = t[..., None, None]
    return (t_exp * v - x_t) / (1 - t_exp + 1e-5)


def score_to_vf(x_t, score, t):
    """Convert score to velocity field: v = (x_t + (1-t)*score) / t."""
    t_exp = t[..., None, None]
    return (x_t + (1 - t_exp) * score) / (t_exp + 1e-5)


# ---- Stepper ----

def _mask_and_center(x, mask, center):
    if mask is not None:
        x = x * mask[..., None]
    if center:
        x = force_zero_com(x, mask)
    return x


class SDEStepper(eqx.Module):
    """SDE integration with score-scaled ODE fallback near t=1."""
    sc_scale_noise: float = 0.1
    t_lim_ode: float = 0.98
    sc_scale_score_fallback: float = 1.5
    center_every_step: bool = False

    def __call__(self, x_t, v, t, dt, gt, mask, key):
        score = vf_to_score(x_t, v, t)
        gt_exp = gt[..., None, None]
        noise = jax.random.normal(key, x_t.shape)
        dx_sde = (v + gt_exp * score) * dt + jnp.sqrt(2 * gt_exp * self.sc_scale_noise * dt) * noise
        v_scaled = score_to_vf(x_t, score * self.sc_scale_score_fallback, t)
        dx_ode = v_scaled * dt
        use_sde = (t < self.t_lim_ode)[..., None, None]
        x_next = x_t + jnp.where(use_sde, dx_sde, dx_ode)
        return _mask_and_center(x_next, mask, self.center_every_step)


# ---- Typed config ----

class ChannelConfig(eqx.Module):
    """Configuration for one flow matching channel (backbone or latents)."""
    schedule_fn: Callable  # (nsteps: int) -> Array[nsteps+1]
    noise_fn: Callable     # (t: Array) -> Array
    stepper: SDEStepper
    zero_com: bool = False

    def time_schedule(self, nsteps):
        return self.schedule_fn(nsteps)

    def step(self, x_t, v, t, dt, mask, key):
        gt = self.noise_fn(t)
        return self.stepper(x_t, v, t, dt, gt, mask, key)


class SamplingConfig(eqx.Module):
    """Full sampling configuration for two-channel flow matching."""
    bb_ca: ChannelConfig
    local_latents: ChannelConfig
    nsteps: int = 400
    self_cond: bool = True


PRODUCTION_SAMPLING = SamplingConfig(
    bb_ca=ChannelConfig(
        schedule_fn=partial(log_schedule, p=2.0),
        noise_fn=noise_1_over_t,
        stepper=SDEStepper(sc_scale_noise=0.1, t_lim_ode=0.98),
        zero_com=True,
    ),
    local_latents=ChannelConfig(
        schedule_fn=partial(power_schedule, p=2.0),
        noise_fn=noise_tan,
        stepper=SDEStepper(sc_scale_noise=0.1, t_lim_ode=0.98),
    ),
)


# ---- Generation ----

class _ScanState(eqx.Module):
    """Carry for the generation scan loop."""
    x_bb: jax.Array
    x_lat: jax.Array
    x_sc_bb: jax.Array
    x_sc_lat: jax.Array
    key: jax.Array


def generate(
    model,
    mask,
    n_residues: int,
    latent_dim: int,
    key,
    *,
    cfg: SamplingConfig | None = None,
    nsteps: int | None = None,
    self_cond: bool | None = None,
    target=None,
):
    """Generate binder structures via flow matching SDE integration.

    Args:
        model: the denoiser (LocalLatentsTransformer), called as model(DenoiserBatch)
        mask: [n] boolean residue mask
        n_residues: number of binder residues
        latent_dim: dimension of latent space
        key: PRNG key
        cfg: sampling config (defaults to PRODUCTION_SAMPLING)
        nsteps: override cfg.nsteps
        self_cond: override cfg.self_cond
        target: TargetCond or None

    Returns:
        (bb_ca [n, 3], local_latents [n, latent_dim])
    """
    from jproteina_complexa.types import DenoiserBatch, NoisyState, Timesteps

    cfg = cfg or PRODUCTION_SAMPLING
    nsteps = nsteps if nsteps is not None else cfg.nsteps
    self_cond = self_cond if self_cond is not None else cfg.self_cond

    mask_f = mask.astype(jnp.float32)

    ts_bb = cfg.bb_ca.time_schedule(nsteps)
    ts_lat = cfg.local_latents.time_schedule(nsteps)

    key, k1, k2 = jax.random.split(key, 3)
    x_bb = sample_noise(k1, (n_residues, 3), mask_f, zero_com=cfg.bb_ca.zero_com)
    x_lat = sample_noise(k2, (n_residues, latent_dim), mask_f, zero_com=cfg.local_latents.zero_com)

    def step_fn(state, ts):
        t_bb, t_bb_next, t_lat, t_lat_next = ts

        batch = DenoiserBatch(
            x_t=NoisyState(bb_ca=state.x_bb, local_latents=state.x_lat),
            t=Timesteps(bb_ca=t_bb, local_latents=t_lat),
            mask=mask,
            x_sc=NoisyState(bb_ca=state.x_sc_bb, local_latents=state.x_sc_lat) if self_cond else None,
            target=target,
        )

        out = model(batch)
        v_bb, v_lat = out.bb_ca, out.local_latents

        x_sc_bb = predict_x1_from_v(state.x_bb, v_bb, t_bb) if self_cond else state.x_sc_bb
        x_sc_lat = predict_x1_from_v(state.x_lat, v_lat, t_lat) if self_cond else state.x_sc_lat

        key, k_bb, k_lat = jax.random.split(state.key, 3)
        x_bb = cfg.bb_ca.step(state.x_bb, v_bb, t_bb, t_bb_next - t_bb, mask_f, k_bb)
        x_lat = cfg.local_latents.step(state.x_lat, v_lat, t_lat, t_lat_next - t_lat, mask_f, k_lat)

        return _ScanState(x_bb=x_bb, x_lat=x_lat, x_sc_bb=x_sc_bb, x_sc_lat=x_sc_lat, key=key), None

    init = _ScanState(
        x_bb=x_bb, x_lat=x_lat,
        x_sc_bb=jnp.zeros_like(x_bb), x_sc_lat=jnp.zeros_like(x_lat),
        key=key,
    )
    scan_inputs = (ts_bb[:-1], ts_bb[1:], ts_lat[:-1], ts_lat[1:])

    final, _ = jax.lax.scan(step_fn, init, scan_inputs)
    return final.x_bb, final.x_lat
