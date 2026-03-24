"""CLI for binder generation.

Usage:
    uv run jpc-generate --target /path/to/target.pdb --length 50 --out binder.pdb
    uv run jpc-generate --target /path/to/target.pdb --length 80 --steps 400 --seed 123
"""

import argparse
import time
import sys

import gemmi
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
from jproteina_complexa.constants import AA_CODES, AA_3LETTER, ATOM_NAMES
from jproteina_complexa.pdb import load_target, make_structure
from jproteina_complexa.hub import load_denoiser, load_decoder
from jproteina_complexa.flow_matching import generate, PRODUCTION_SAMPLING
from jproteina_complexa.types import DecoderBatch, TargetCond
from jproteina_complexa.target_features import compute_target_sidechain_feat, compute_target_torsion_feat


def main():
    p = argparse.ArgumentParser(description="Generate protein binders with jproteina_complexa")
    p.add_argument("--target", required=True, help="Path to target PDB file")
    p.add_argument("--chain", default=None, help="Target chain ID (default: first chain)")
    p.add_argument("--length", type=int, default=80, help="Binder length in residues")
    p.add_argument("--steps", type=int, default=400, help="Number of ODE integration steps")
    p.add_argument("--seed", type=int, default=0, help="Random seed")
    p.add_argument("--out", default="binder.pdb", help="Output PDB path")
    p.add_argument("--weights", default=None, help="Directory with .eqx weight files (default: auto-download from HuggingFace)")
    p.add_argument("--hotspots", default=None, help="Comma-separated 1-indexed residue numbers for hotspots (e.g. 8,44,68,70)")
    p.add_argument("--batch", type=int, default=1, help="Number of designs to generate in parallel")
    p.add_argument("--no-self-cond", action="store_true", help="Disable self-conditioning")
    args = p.parse_args()

    # Load target
    print(f"Loading target: {args.target}")
    structure = gemmi.read_structure(args.target)
    chain = structure[0][args.chain] if args.chain else structure[0][0]
    target_coords, target_amask, target_seq, n_target = load_target(chain)
    target_nm = target_coords / 10.0
    target_sc = compute_target_sidechain_feat(target_coords, target_amask, target_seq)
    target_tor = compute_target_torsion_feat(target_coords)
    print(f"  {n_target} residues")

    # Load models (download from HuggingFace if needed)
    print("Loading models...")
    t0 = time.perf_counter()
    kwargs = {"cache_dir": args.weights} if args.weights else {}
    denoiser = load_denoiser(**kwargs)
    decoder = load_decoder(**kwargs)
    print(f"  Loaded in {time.perf_counter() - t0:.1f}s")

    LATENT_DIM = 8
    mask = jnp.ones(args.length, dtype=jnp.bool_)

    # Hotspot mask: sparse (matching training distribution of 0-6 hotspots)
    hotspot_mask = np.zeros(n_target, dtype=bool)
    if args.hotspots:
        for idx in args.hotspots.split(","):
            resnum = int(idx.strip()) - 1  # 1-indexed to 0-indexed
            if 0 <= resnum < n_target:
                hotspot_mask[resnum] = True
        print(f"  Hotspots: {hotspot_mask.sum()} residues ({args.hotspots})")
    else:
        print(f"  Hotspots: none (use --hotspots to specify)")

    target = TargetCond(
        coords=jnp.array(target_nm),
        atom_mask=jnp.array(target_amask),
        seq=jnp.array(target_seq),
        seq_mask=jnp.ones(n_target, dtype=jnp.bool_),
        hotspot_mask=jnp.array(hotspot_mask),
        sidechain_feat=jnp.array(target_sc),
        torsion_feat=jnp.array(target_tor),
    )

    def _run_single(denoiser, decoder, key):
        x_bb, x_lat = generate(
            model=denoiser, mask=mask, n_residues=args.length, latent_dim=LATENT_DIM,
            key=key, nsteps=args.steps, self_cond=not args.no_self_cond, target=target,
        )
        dec_out = decoder(DecoderBatch(z_latent=x_lat, ca_coors_nm=x_bb, mask=mask))
        return x_bb, dec_out

    B = args.batch
    if B > 1:
        @eqx.filter_jit
        def _run(denoiser, decoder, keys):
            return jax.vmap(lambda k: _run_single(denoiser, decoder, k))(keys)
    else:
        @eqx.filter_jit
        def _run(denoiser, decoder, keys):
            return _run_single(denoiser, decoder, keys[0])

    print(f"Generating {B}x {args.length}-residue binder ({args.steps} steps, seed={args.seed})...")
    t0 = time.perf_counter()
    keys = jax.random.split(jax.random.PRNGKey(args.seed), B)
    x_bb, dec_out = _run(denoiser, decoder, keys)
    jax.block_until_ready(dec_out.coors_nm)
    gen_time = time.perf_counter() - t0
    print(f"  {gen_time:.1f}s ({gen_time / args.steps * 1000:.0f}ms/step)")

    # Write one PDB per design
    out_stem, out_ext = (args.out.rsplit(".", 1) + ["pdb"])[:2]
    for bi in range(B):
        bb_i = x_bb[bi] if B > 1 else x_bb
        do_i = jax.tree.map(lambda x: x[bi] if B > 1 else x, dec_out)

        pred_seq = "".join(AA_CODES[i] for i in np.array(do_i.aatype))
        pred_coors = np.array(do_i.coors_nm) * 10.0
        pred_amask = np.array(do_i.atom_mask).astype(np.float32)
        binder_resnames = [AA_3LETTER[aa] for aa in pred_seq]
        target_resnames = [AA_3LETTER[AA_CODES[i]] for i in target_seq]

        out_path = f"{out_stem}_{bi}.{out_ext}" if B > 1 else args.out
        structure = make_structure([
            ("A", binder_resnames, pred_coors, pred_amask),
            ("B", target_resnames, target_coords, target_amask),
        ])
        structure.write_pdb(out_path)

        binder_ca = np.array(bb_i) * 10.0
        ca_dists = np.linalg.norm(np.diff(binder_ca, axis=0), axis=1)
        target_ca = target_nm[:, 1, :] * 10.0
        min_dists = np.min(np.linalg.norm(binder_ca[:, None] - target_ca[None], axis=-1), axis=1)

        print(f"\n[{bi}] Sequence: {pred_seq}")
        print(f"    CA-CA distances: {ca_dists.mean():.2f} +/- {ca_dists.std():.2f} A")
        print(f"    Residues within 8A of target: {(min_dists < 8).sum()}/{args.length}")
        print(f"    Saved: {out_path}")
