"""CLI for binder generation and protein-motif scaffolding.

Binder design (condition on a target):
    uv run jpc-generate --target target.pdb --outdir designs/ --num-samples 8

Protein-motif scaffolding (condition on a fixed protein motif, no ligand):
    uv run jpc-generate --motif-pdb 5tpn.pdb --motif-contig A163-181 \
        --length 60 --weights weights/ --outdir designs/
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
from jproteina_complexa.pdb import load_target_cond, load_motif_cond, make_structure
from jproteina_complexa.hub import load_denoiser, load_decoder
from jproteina_complexa.flow_matching import generate
from jproteina_complexa.types import DecoderBatch


def main():
    p = argparse.ArgumentParser(description="Generate protein binders or scaffold protein motifs with jproteina_complexa")
    # Binder (target) conditioning
    p.add_argument("--target", default=None, help="Path to target PDB (binder design mode)")
    p.add_argument("--chain", default=None, help="Target chain ID (default: first chain)")
    p.add_argument("--hotspots", default=None, help="Comma-separated 1-indexed target hotspot residues (e.g. 8,44,68)")
    # Motif conditioning
    p.add_argument("--motif-pdb", default=None, help="Path to source PDB for motif scaffolding mode")
    p.add_argument("--motif-contig", default=None, help="Contig string, e.g. 'A163-181' or '10-40/A163-181/10-40' (only chain segments extracted). Omit to use the whole PDB as the motif.")
    p.add_argument("--motif-atoms", default="all_atom", choices=["ca", "bb3o", "all_atom", "tip_atoms"], help="Which motif atoms to condition on")
    p.add_argument("--motif-only", action="store_true", help="Use all (non-hetero) residues of each contig chain, ignoring residue ranges")
    # Shared
    p.add_argument("--length", type=int, default=80, help="Designed protein length in residues")
    p.add_argument("--steps", type=int, default=400, help="Number of ODE integration steps")
    p.add_argument("--seed", type=int, default=0, help="Random seed")
    p.add_argument("--outdir", required=True, help="Output directory for generated PDB files")
    p.add_argument("--weights", default=None, help="Directory with .eqx weight files (default: auto-download binder weights; required for motif mode)")
    p.add_argument("--num-samples", type=int, default=1, help="Total number of designs to generate")
    p.add_argument("--batch", type=int, default=1, help="Number of designs to generate in parallel per round")
    p.add_argument("--no-self-cond", action="store_true", help="Disable self-conditioning")
    args = p.parse_args()

    motif_mode = args.motif_pdb is not None
    if motif_mode and args.target is not None:
        p.error("--target and --motif-pdb are mutually exclusive")
    if not motif_mode and args.target is None:
        p.error("provide --target (binder design) or --motif-pdb (motif scaffolding)")
    if args.num_samples % args.batch != 0:
        p.error(f"--num-samples ({args.num_samples}) must be divisible by --batch ({args.batch})")

    os.makedirs(args.outdir, exist_ok=True)
    kwargs = {"cache_dir": args.weights} if args.weights else {}

    # ---- Build conditioning + pick model variant ----
    target = motif = None
    ref_resnames = ref_coords = ref_amask = None  # chain-B reference (target or motif)
    if motif_mode:
        print(f"Loading motif: {args.motif_pdb}  contig={args.motif_contig or 'whole structure'}  atoms={args.motif_atoms}")
        structure = gemmi.read_structure(args.motif_pdb)
        structure.setup_entities()
        motif, motif_resnames = load_motif_cond(
            structure, args.motif_contig, atom_selection_mode=args.motif_atoms, motif_only=args.motif_only,
        )
        nm = int(motif.seq_motif.shape[0])
        print(f"  {nm} motif residues; scaffolding into {args.length}-residue proteins")
        ref_resnames = motif_resnames
        ref_coords = np.array(motif.x_motif)
        ref_amask = np.array(motif.motif_mask)
        denoiser_name, decoder_name = "denoiser_motif", "decoder_motif"
    else:
        print(f"Loading target: {args.target}")
        structure = gemmi.read_structure(args.target)
        structure.setup_entities()
        chain = structure[0][args.chain] if args.chain else structure[0][0]
        hotspots = None
        if args.hotspots:
            hotspots = [int(x.strip()) - 1 for x in args.hotspots.split(",")]
            print(f"  Hotspots: {len(hotspots)} residues ({args.hotspots})")
        target = load_target_cond(chain, hotspots=hotspots)
        target_seq = np.array(target.seq)
        print(f"  {len(target_seq)} residues")
        ref_resnames = [AA_3LETTER[AA_CODES[i]] for i in target_seq]
        ref_coords = np.array(target.coords)
        ref_amask = np.array(target.atom_mask)
        denoiser_name, decoder_name = "denoiser", "decoder"

    # ---- Load models ----
    print("Loading models...")
    t0 = time.perf_counter()
    denoiser = load_denoiser(name=denoiser_name, **kwargs)
    decoder = load_decoder(name=decoder_name, **kwargs)
    print(f"  Loaded in {time.perf_counter() - t0:.1f}s")

    mask = jnp.ones(args.length, dtype=jnp.bool_)

    def _run_single(denoiser, decoder, key):
        x_bb, x_lat = generate(
            model=denoiser, mask=mask, key=key, nsteps=args.steps,
            self_cond=not args.no_self_cond, target=target, motif=motif,
        )
        dec_out = decoder(DecoderBatch(z_latent=x_lat, ca_coors=x_bb, mask=mask))
        return x_bb, dec_out

    B, N = args.batch, args.num_samples
    n_rounds = N // B

    if B > 1:
        @eqx.filter_jit
        def _run(denoiser, decoder, keys):
            return jax.vmap(lambda k: _run_single(denoiser, decoder, k))(keys)
    else:
        @eqx.filter_jit
        def _run(denoiser, decoder, keys):
            return _run_single(denoiser, decoder, keys[0])

    mode_str = "motif scaffolds" if motif_mode else "binders"
    print(f"Generating {N} x {args.length}-residue {mode_str} ({n_rounds} rounds of {B}, {args.steps} steps, seed={args.seed})...")
    ref_ca = ref_coords[:, 1, :]
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
            design_resnames = [AA_3LETTER[aa] for aa in pred_seq]

            # Chain A = designed protein; chain B = reference (target, or the input motif).
            out_path = os.path.join(args.outdir, f"sample_{sample_idx}.pdb")
            make_structure([
                ("A", design_resnames, pred_coors, pred_amask),
                ("B", ref_resnames, ref_coords, ref_amask),
            ]).write_pdb(out_path)

            design_ca = np.array(bb_i)
            ca_dists = np.linalg.norm(np.diff(design_ca, axis=0), axis=1)
            print(f"  [{sample_idx}] {pred_seq}")
            geom = f"CA-CA: {ca_dists.mean():.2f}+/-{ca_dists.std():.2f}A"
            if not motif_mode:
                min_dists = np.min(np.linalg.norm(design_ca[:, None] - ref_ca[None], axis=-1), axis=1)
                geom += f"  contact: {(min_dists < 8).sum()}/{args.length}"
            print(f"      {geom}  {out_path}")
            sample_idx += 1

    # Timing summary
    total_time = sum(round_times)
    if n_rounds > 1:
        compile_time = round_times[0] - round_times[1]
        gen_only = total_time - round_times[0] + round_times[1]
        per_sample = gen_only / N
    else:
        gen_only = round_times[0]
        per_sample = round_times[0] / N
    print(f"\nTiming summary:")
    print(f"  Total:       {total_time:.1f}s for {N} samples")
    if n_rounds > 1:
        print(f"  JIT compile: ~{compile_time:.1f}s (first round overhead)")
        print(f"  Generation:  {gen_only:.1f}s ({per_sample:.2f}s/sample, {per_sample / args.steps * 1000:.0f}ms/step)")
    else:
        print(f"  Per sample:  {per_sample:.2f}s (includes JIT compilation)")
    print(f"  Output:      {args.outdir}/")
