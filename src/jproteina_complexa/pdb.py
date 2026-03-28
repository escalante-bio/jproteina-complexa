"""PDB reading and writing using gemmi."""

import gemmi
import numpy as np
import jax.numpy as jnp

from jproteina_complexa.constants import AA_3TO1, AA_1TO_IDX, AA_3LETTER, AA_CODES, ATOM_NAMES
from jproteina_complexa.types import TargetCond
from jproteina_complexa.target_features import compute_target_sidechain_feat, compute_target_torsion_feat


def load_target(chain: gemmi.Chain, center: bool = True):
    """Extract target protein arrays from a gemmi Chain.

    Returns (coords, mask, seq, n) where:
        coords: [n, 37, 3] float32 atom coordinates in Angstroms (centered on CA COM if center=True)
        mask:   [n, 37]    float32 atom presence mask
        seq:    [n]        int64   residue type indices (0-19)
        n:      int        number of residues
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

    return coords, mask, seq, n


def load_target_cond(chain: gemmi.Chain, hotspots: list[int] | None = None) -> TargetCond:
    """Build a TargetCond from a gemmi Chain.

    Args:
        chain: gemmi Chain object (e.g., structure[0]["A"])
        hotspots: optional list of 0-indexed residue numbers to mark as hotspots

    Returns:
        TargetCond with all coordinates in Angstroms (centered on CA COM).
    """
    coords, amask, seq, n = load_target(chain)
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
        seq_mask=jnp.ones(n, dtype=jnp.bool_),
        hotspot_mask=hotspot_mask,
        sidechain_feat=jnp.array(sc),
        torsion_feat=jnp.array(tor),
    )


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
