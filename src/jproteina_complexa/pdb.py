"""PDB reading and writing using gemmi."""

import gemmi
import numpy as np
import jax.numpy as jnp

from jproteina_complexa.constants import (
    AA_3TO1, AA_1TO_IDX, AA_3LETTER, AA_CODES, ATOM_NAMES, SIDECHAIN_TIP_ATOMS,
)
from jproteina_complexa.types import TargetCond, MotifCond
from jproteina_complexa.target_features import compute_target_sidechain_feat, compute_target_torsion_feat

_BB3O = [ATOM_NAMES.index(a) for a in ("N", "CA", "C", "O")]  # [0, 1, 2, 4]


def load_target(chain: gemmi.Chain, center: bool = True):
    """Extract target protein arrays from a gemmi Chain.

    Returns (coords, mask, seq) where:
        coords: [n, 37, 3] float32 atom coordinates in Angstroms (centered on CA COM if center=True)
        mask:   [n, 37]    float32 atom presence mask
        seq:    [n]        int64   residue type indices (0-19)
    """
    polymer = chain.get_polymer()
    n = len(polymer)

    coords = np.zeros((n, 37, 3), dtype=np.float32)
    mask = np.zeros((n, 37), dtype=np.float32)
    seq = np.zeros(n, dtype=np.int64)

    for i, res in enumerate(polymer):
        aa1 = AA_3TO1.get(res.name, "A")
        seq[i] = AA_1TO_IDX.get(aa1, 0)
        for atom in res:
            if atom.name in ATOM_NAMES:
                j = ATOM_NAMES.index(atom.name)
                coords[i, j] = [atom.pos.x, atom.pos.y, atom.pos.z]
                mask[i, j] = 1.0

    if center:
        # Center on CA center of mass
        ca_mask = mask[:, 1]
        ca_coords = coords[:, 1, :]
        com = (ca_coords * ca_mask[:, None]).sum(0) / max(ca_mask.sum(), 1)
        coords = coords - com[None, None, :]

    return coords, mask, seq


def load_target_cond(chain: gemmi.Chain, hotspots: list[int] | None = None) -> TargetCond:
    """Build a TargetCond from a gemmi Chain.

    Args:
        chain: gemmi Chain object (e.g., structure[0]["A"])
        hotspots: optional list of 0-indexed residue numbers to mark as hotspots

    Returns:
        TargetCond with all coordinates in Angstroms (centered on CA COM).
    """
    coords, amask, seq = load_target(chain)
    n = len(seq)
    sc = compute_target_sidechain_feat(coords, amask, seq)
    tor = compute_target_torsion_feat(coords)

    hotspot_mask = None
    if hotspots is not None:
        h = np.zeros(n, dtype=bool)
        for idx in hotspots:
            if 0 <= idx < n:
                h[idx] = True
        hotspot_mask = jnp.array(h)

    return TargetCond(
        coords=jnp.array(coords),
        atom_mask=jnp.array(amask),
        seq=jnp.array(seq),
        hotspot_mask=hotspot_mask,
        sidechain_feat=jnp.array(sc),
        torsion_feat=jnp.array(tor),
    )


def _select_motif_atom_indices(available: set[int], mode: str, res_name: str) -> set[int]:
    """Atom37 indices to keep for a motif residue under a given selection mode.

    Mirrors proteina-complexa motif_utils._select_motif_atoms.
      ca        -> CA only
      bb3o      -> N, CA, C, O
      all_atom  -> every atom present
      tip_atoms -> functional side-chain atoms (SIDECHAIN_TIP_ATOMS)
    All are intersected with the atoms actually present in the structure.
    """
    if mode == "ca":
        want = {1}
    elif mode == "bb3o":
        want = set(_BB3O)
    elif mode == "all_atom":
        return set(available)
    elif mode == "tip_atoms":
        want = {ATOM_NAMES.index(a) for a in SIDECHAIN_TIP_ATOMS.get(res_name, []) if a in ATOM_NAMES}
    else:
        raise ValueError(f"unknown atom_selection_mode: {mode!r}")
    return want & available


def _parse_contig_segments(contig: str):
    """Yield (chain_id, start, end) for each motif segment of a contig string.

    Scaffold-gap segments (leading digit / range like ``10-40``) are skipped —
    only motif segments (leading chain letter, e.g. ``A163-181`` or ``A1051``)
    contribute residues. ``start``/``end`` are None when motif_only picks a whole chain.
    """
    for part in contig.split("/"):
        part = part.strip()
        if not part or not part[0].isalpha():
            continue  # scaffold gap
        chain_id = part[0]
        spec = part[1:]
        if not spec:
            yield chain_id, None, None
        elif "-" in spec:
            lo, hi = spec.split("-")
            yield chain_id, int(lo), int(hi)
        else:
            yield chain_id, int(spec), int(spec)


def load_motif_cond(
    structure: gemmi.Structure,
    contig: str | None = None,
    atom_selection_mode: str = "all_atom",
    motif_only: bool = False,
    center: bool = True,
) -> tuple[MotifCond, list[str]]:
    """Build a compact MotifCond from a source structure.

    ``contig`` names the motif residues (e.g. ``"A163-181"`` or
    ``"10-40/A163-181/10-40"``, scaffold gaps ignored). Pass ``contig=None`` when
    the structure *is* the motif — every standard amino-acid residue across all
    chains is used, in structure order.

    Keeps the atom37 slots selected by ``atom_selection_mode`` and (by default)
    centers on the all-atom motif center of mass — matching the upstream unindexed
    monomer path. Coordinates are Angstroms (the JAX API convention).

    Returns (MotifCond, resnames) where resnames are 3-letter codes for writing a
    reference copy of the motif.
    """
    model = structure[0]

    # Ordered-unique motif residues, preserving structure/contig order.
    picked = []
    seen = set()
    if contig is None:
        segments = [(chain.name, None, None) for chain in model]  # whole structure
    else:
        segments = list(_parse_contig_segments(contig))
    for chain_id, start, end in segments:
        chain = model[chain_id]
        for res in chain:
            if res.name not in AA_3TO1:  # skip water / ligands / hetero
                continue
            num = res.seqid.num
            if start is not None and not motif_only and not (start <= num <= end):
                continue
            key = (chain_id, num, res.seqid.icode)
            if key in seen:
                continue
            seen.add(key)
            picked.append(res)

    if not picked:
        raise ValueError(f"contig {contig!r} selected no motif residues from the structure")

    nm = len(picked)
    coords = np.zeros((nm, 37, 3), dtype=np.float32)
    mask = np.zeros((nm, 37), dtype=np.float32)
    seq = np.zeros(nm, dtype=np.int64)
    resnames = []
    for i, res in enumerate(picked):
        resnames.append(res.name)
        seq[i] = AA_1TO_IDX.get(AA_3TO1.get(res.name, "A"), 0)
        available = {ATOM_NAMES.index(a.name): a for a in res if a.name in ATOM_NAMES}
        keep = _select_motif_atom_indices(set(available), atom_selection_mode, res.name)
        for j in keep:
            a = available[j]
            coords[i, j] = [a.pos.x, a.pos.y, a.pos.z]
            mask[i, j] = 1.0

    if center:
        # All-atom center of mass over kept motif atoms (matches upstream all-atom mode).
        com = (coords * mask[..., None]).reshape(-1, 3).sum(0) / max(mask.sum(), 1)
        coords = (coords - com[None, None, :]) * mask[..., None]

    motif = MotifCond(
        x_motif=jnp.array(coords),
        motif_mask=jnp.array(mask),
        seq_motif=jnp.array(seq),
        seq_motif_mask=jnp.array((mask.sum(-1) > 0).astype(np.float32)),
    )
    return motif, resnames


def make_structure(chains) -> gemmi.Structure:
    """Build a gemmi Structure from chain data.

    chains = [(chain_id, resnames, coords, atom_mask), ...]
    where coords is [n_res, 37, 3] and atom_mask is [n_res, 37].
    """
    structure = gemmi.Structure()
    structure.name = "jproteina_complexa"
    model = gemmi.Model("1")

    for chain_id, resnames, coords, amask in chains:
        chain = gemmi.Chain(chain_id)
        for i in range(len(resnames)):
            res = gemmi.Residue()
            res.name = resnames[i]
            res.seqid = gemmi.SeqId(str(i + 1))
            for j in range(37):
                if amask[i, j] < 0.5:
                    continue
                atom = gemmi.Atom()
                atom.name = ATOM_NAMES[j]
                x, y, z = float(coords[i, j, 0]), float(coords[i, j, 1]), float(coords[i, j, 2])
                atom.pos = gemmi.Position(x, y, z)
                elem_str = atom.name[0] if atom.name[0] in "CNOS" else atom.name.strip()[:2]
                atom.element = gemmi.Element(elem_str)
                atom.occ = 1.0
                atom.b_iso = 0.0
                res.add_atom(atom)
            chain.add_residue(res)
        model.add_chain(chain)

    structure.add_model(model)
    structure.setup_entities()
    structure.assign_serial_numbers()
    return structure
