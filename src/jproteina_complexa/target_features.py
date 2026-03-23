"""Precompute target sidechain and torsion angle features (no torch dependency)."""

import numpy as np

from jproteina_complexa.constants import CHI_ATOM_INDICES, CHI_ANGLES_MASK


def _dihedral(p0, p1, p2, p3):
    """Signed dihedral angle between 4 points. All [n, 3]. Returns [n]."""
    b0, b1, b2 = p1 - p0, p2 - p1, p3 - p2
    n1 = np.cross(b0, b1)
    n2 = np.cross(b1, b2)
    n1 = n1 / (np.linalg.norm(n1, axis=-1, keepdims=True) + 1e-8)
    n2 = n2 / (np.linalg.norm(n2, axis=-1, keepdims=True) + 1e-8)
    cross = np.cross(n1, n2)
    return np.arctan2(
        np.sign(np.sum(cross * b1, axis=-1)) * np.sqrt(np.sum(cross ** 2, axis=-1) + 1e-16),
        np.sum(n1 * n2, axis=-1),
    )


def _bin_one_hot(values, n_bins=20):
    """Bin values in [-pi, pi] and one-hot encode. [...] -> [..., n_bins+1]."""
    bins = np.linspace(-np.pi, np.pi, n_bins)
    indices = np.searchsorted(bins, values)
    result = np.zeros((*values.shape, n_bins + 1), dtype=np.float32)
    np.put_along_axis(result, indices[..., None], 1.0, axis=-1)
    return result


def compute_target_torsion_feat(coords):
    """Backbone torsion features [n, 63] from atom37 coords."""
    n = len(coords)
    N, CA, C = coords[:, 0], coords[:, 1], coords[:, 2]

    psi = np.zeros(n)
    omega = np.zeros(n)
    phi = np.zeros(n)
    if n > 1:
        psi[:-1] = _dihedral(N[:-1], CA[:-1], C[:-1], N[1:])
        omega[:-1] = _dihedral(CA[:-1], C[:-1], N[1:], CA[1:])
        phi[:-1] = _dihedral(C[:-1], N[1:], CA[1:], C[1:])

    return _bin_one_hot(np.stack([psi, omega, phi], axis=-1)).reshape(n, -1).astype(np.float32)


def compute_target_sidechain_feat(coords, coord_mask, residue_types):
    """Sidechain angle features [n, 88] from atom37 coords."""
    n = len(coords)
    chi_angles = np.zeros((n, 4), dtype=np.float32)
    chi_mask = np.zeros((n, 4), dtype=np.float32)

    for i in range(n):
        rt = int(residue_types[i])
        if rt >= 20:
            continue
        for ci in range(4):
            if CHI_ANGLES_MASK[rt, ci] < 0.5:
                continue
            aids = CHI_ATOM_INDICES[rt, ci]
            if all(coord_mask[i, aids[j]] > 0.5 for j in range(4)):
                p = [coords[i, aids[j]] for j in range(4)]
                chi_angles[i, ci] = _dihedral(p[0][None], p[1][None], p[2][None], p[3][None])[0]
                chi_mask[i, ci] = 1.0

    binned = _bin_one_hot(chi_angles) * chi_mask[..., None]  # [n, 4, 21]
    return np.concatenate([binned.reshape(n, -1), chi_mask], axis=-1).astype(np.float32)
