"""CLI for binder generation.

Usage:
    uv run jpc-generate --target /path/to/target.pdb --outdir designs/ --num-samples 8
    uv run jpc-generate --target /path/to/target.pdb --outdir designs/ --length 80 --steps 400 --seed 123
"""

import argparse
import os
import time

import gemmi
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
from jproteina_complexa.constants import AA_CODES, AA_3LETTER
from jproteina_complexa.pdb import load_target, load_target_cond, make_structure
from jproteina_complexa.hub import load_denoiser, load_decoder
from jproteina_complexa.flow_matching import generate
from jproteina_complexa.types import DecoderBatch


def main():
    p = argparse.ArgumentParser(description="Generate protein binders with jproteina_complexa")
    p.add_argument("--target", required=True, help="Path to target PDB file")
    p.add_argument("--chain", default=None, help="Target chain ID (default: first chain)")
    p.add_argument("--length", type=int, default=80, help="Binder length in residues")
    p.add_argument("--steps", type=int, default=400, help="Number of ODE integration steps")
    p.add_argument("--seed", type=int, default=0, help="Random seed")
    p.add_argument("--outdir", required=True, help="Output directory for generated PDB files")
    p.add_argument("--weights", default=None, help="Directory with .eqx weight files (default: auto-download from HuggingFace)")
    p.add_argument("--hotspots", default=None, help="Comma-separated 1-indexed residue numbers for hotspots (e.g. 8,44,68,70)")
    p.add_argument("--num-samples", type=int, default=1, help="Total number of designs to generate")
    p.add_argument("--batch", type=int, default=1, help="Number of designs to generate in parallel per round")
    p.add_argument("--no-self-cond", action="store_true", help="Disable self-conditioning")
    args = p.parse_args()

    if args.num_samples % args.batch != 0:
        p.error(f"--num-samples ({args.num_samples}) must be divisible by --batch ({args.batch})")

    os.makedirs(args.outdir, exist_ok=True)

    # Load target
    print(f"Loading target: {args.target}")
    structure = gemmi.read_structure(args.target)
    chain = structure[0][args.chain] if args.chain else structure[0][0]

    hotspots = None
    if args.hotspots:
        hotspots = [int(x.strip()) - 1 for x in args.hotspots.split(",")]
        print(f"  Hotspots: {len(hotspots)} residues ({args.hotspots})")

    target = load_target_cond(chain, hotspots=hotspots)
    target_coords, target_amask, target_seq = np.array(target.coords), np.array(target.atom_mask), np.array(target.seq)
    print(f"  {len(target_seq)} residues")

    # Load models (download from HuggingFace if needed)
    print("Loading models...")
    t0 = time.perf_counter()
    kwargs = {"cache_dir": args.weights} if args.weights else {}
    denoiser = load_denoiser(**kwargs)
    decoder = load_decoder(**kwargs)
    print(f"  Loaded in {time.perf_counter() - t0:.1f}s")

    LATENT_DIM = 8
    mask = jnp.ones(args.length, dtype=jnp.bool_)

    def _run_single(denoiser, decoder, key):
        x_bb, x_lat = generate(
            model=denoiser, mask=mask, n_residues=args.length, latent_dim=LATENT_DIM,
            key=key, nsteps=args.steps, self_cond=not args.no_self_cond, target=target,
        )
        dec_out = decoder(DecoderBatch(z_latent=x_lat, ca_coors=x_bb, mask=mask))
        return x_bb, dec_out

    B = args.batch
    N = args.num_samples
    n_rounds = N // B

    if B > 1:
        @eqx.filter_jit
        def _run(denoiser, decoder, keys):
            return jax.vmap(lambda k: _run_single(denoiser, decoder, k))(keys)
    else:
        @eqx.filter_jit
        def _run(denoiser, decoder, keys):
            return _run_single(denoiser, decoder, keys[0])

    print(f"Generating {N} x {args.length}-residue binders ({n_rounds} rounds of {B}, {args.steps} steps, seed={args.seed})...")
    target_resnames = [AA_3LETTER[AA_CODES[i]] for i in target_seq]
    target_ca = target_coords[:, 1, :]
    all_keys = jax.random.split(jax.random.PRNGKey(args.seed), N)
    sample_idx = 0
    round_times = []

    for ri in range(n_rounds):
        round_keys = all_keys[ri * B : (ri + 1) * B]
        t0 = time.perf_counter()
        x_bb, dec_out = _run(denoiser, decoder, round_keys)
        jax.block_until_ready(dec_out.coors)
        gen_time = time.perf_counter() - t0
        round_times.append(gen_time)
        print(f"\n  Round {ri+1}/{n_rounds}: {gen_time:.1f}s ({gen_time / args.steps * 1000:.0f}ms/step)")

        for bi in range(B):
            bb_i = x_bb[bi] if B > 1 else x_bb
            do_i = jax.tree.map(lambda x: x[bi] if B > 1 else x, dec_out)

            pred_seq = "".join(AA_CODES[i] for i in np.array(do_i.aatype))
            pred_coors = np.array(do_i.coors)
            pred_amask = np.array(do_i.atom_mask).astype(np.float32)
            binder_resnames = [AA_3LETTER[aa] for aa in pred_seq]

            out_path = os.path.join(args.outdir, f"sample_{sample_idx}.pdb")
            structure = make_structure([
                ("A", binder_resnames, pred_coors, pred_amask),
                ("B", target_resnames, target_coords, target_amask),
            ])
            structure.write_pdb(out_path)

            binder_ca = np.array(bb_i)
            ca_dists = np.linalg.norm(np.diff(binder_ca, axis=0), axis=1)
            min_dists = np.min(np.linalg.norm(binder_ca[:, None] - target_ca[None], axis=-1), axis=1)

            print(f"  [{sample_idx}] {pred_seq}")
            print(f"      CA-CA: {ca_dists.mean():.2f}+/-{ca_dists.std():.2f}A  "
                  f"contact: {(min_dists < 8).sum()}/{args.length}  {out_path}")
            sample_idx += 1

    # Timing summary
    total_time = sum(round_times)
    compile_time = round_times[0] - round_times[1] if n_rounds > 1 else None
    gen_only = total_time - round_times[0] + (round_times[1] if n_rounds > 1 else 0)
    per_sample = gen_only / N if n_rounds > 1 else round_times[0] / N

    print(f"\nTiming summary:")
    print(f"  Total:       {total_time:.1f}s for {N} samples")
    if n_rounds > 1:
        print(f"  JIT compile: ~{compile_time:.1f}s (first round overhead)")
        print(f"  Generation:  {gen_only:.1f}s ({per_sample:.2f}s/sample, {per_sample / args.steps * 1000:.0f}ms/step)")
    else:
        print(f"  Per sample:  {per_sample:.2f}s (includes JIT compilation)")
    print(f"  Output:      {args.outdir}/")
