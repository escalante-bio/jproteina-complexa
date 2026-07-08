"""Tests for SDEdit / partial-diffusion seeding.

Covers flow_matching.seed_state and generate(seed=..., start=...). Pure flow-matching math
plus a zero-velocity denoiser round-trip — no model weights required.
"""

import jax
import jax.numpy as jnp
import equinox as eqx

from jproteina_complexa.flow_matching import PRODUCTION_SAMPLING, seed_state, generate
from jproteina_complexa.types import DenoiserOutput


def _sched(nsteps):
    cfg = PRODUCTION_SAMPLING
    return cfg, cfg.bb_ca.time_schedule(nsteps), cfg.local_latents.time_schedule(nsteps)


def test_seed_state_t1_returns_clean():
    """start == nsteps (t = 1): the seeded state is exactly the clean input (no noise)."""
    n, D, nsteps = 10, 8, 100
    cfg, ts_bb, ts_lat = _sched(nsteps)
    mask = jnp.ones(n, bool)
    bb = jax.random.normal(jax.random.PRNGKey(1), (n, 3))
    lat = jax.random.normal(jax.random.PRNGKey(2), (n, D))
    s = seed_state(jax.random.PRNGKey(0), bb, lat, mask, cfg, ts_bb, ts_lat, nsteps)
    assert jnp.allclose(s.bb, bb, atol=1e-5)
    assert jnp.allclose(s.lat, lat, atol=1e-5)


def test_seed_state_t0_is_pure_noise():
    """start == 0 (t = 0): the seeded state is pure noise, independent of the clean input."""
    n, D, nsteps = 10, 8, 100
    cfg, ts_bb, ts_lat = _sched(nsteps)
    mask = jnp.ones(n, bool)
    bb = 100.0 + jax.random.normal(jax.random.PRNGKey(1), (n, 3))
    lat = 100.0 + jax.random.normal(jax.random.PRNGKey(2), (n, D))
    s = seed_state(jax.random.PRNGKey(0), bb, lat, mask, cfg, ts_bb, ts_lat, 0)
    assert not jnp.allclose(s.bb, bb, atol=1.0)
    assert not jnp.allclose(s.lat, lat, atol=1.0)


def test_seed_state_larger_start_stays_closer():
    """Larger start (t -> 1) stays closer to the clean input than a smaller start."""
    n, D, nsteps = 10, 8, 200
    cfg, ts_bb, ts_lat = _sched(nsteps)
    mask = jnp.ones(n, bool)
    bb = jax.random.normal(jax.random.PRNGKey(1), (n, 3))
    lat = jax.random.normal(jax.random.PRNGKey(2), (n, D))
    near = jnp.linalg.norm(seed_state(jax.random.PRNGKey(0), bb, lat, mask, cfg, ts_bb, ts_lat, 180).lat - lat)
    far = jnp.linalg.norm(seed_state(jax.random.PRNGKey(0), bb, lat, mask, cfg, ts_bb, ts_lat, 40).lat - lat)
    assert near < far


class _ZeroDenoiser(eqx.Module):
    """Denoiser stub returning zero velocity (only for wiring tests, never trained)."""
    def __call__(self, batch):
        return DenoiserOutput(
            bb_ca=jnp.zeros_like(batch.x_t.bb_ca),
            local_latents=jnp.zeros_like(batch.x_t.local_latents),
        )


def test_generate_seed_roundtrip():
    """generate(seed=..., start=nsteps) runs zero denoise steps and returns the seed:
    bb back in Angstroms, latent unchanged."""
    n, D, nsteps = 10, 8, 50
    mask = jnp.ones(n, bool)
    bb_A = jax.random.normal(jax.random.PRNGKey(1), (n, 3)) * 10.0
    lat = jax.random.normal(jax.random.PRNGKey(2), (n, D))
    out_bb, out_lat = generate(
        _ZeroDenoiser(), mask, jax.random.PRNGKey(0), nsteps=nsteps, seed=(bb_A, lat), start=nsteps
    )
    assert jnp.allclose(out_bb, bb_A, atol=1e-4)
    assert jnp.allclose(out_lat, lat, atol=1e-4)


def test_generate_seed_requires_positive_start():
    n, D = 10, 8
    mask = jnp.ones(n, bool)
    seed = (jnp.zeros((n, 3)), jnp.zeros((n, D)))
    try:
        generate(_ZeroDenoiser(), mask, jax.random.PRNGKey(0), seed=seed, start=0)
        assert False, "expected ValueError for start=0 with a seed"
    except ValueError:
        pass
