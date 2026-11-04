"""
R-vector and G-vector nucleobase-pair featurization.

Definitions follow Bottaro, Di Palma & Bussi (NAR 2014) and Herron et al.
(RNAnneal, 2026), exactly as in the standalone rg_vectors.py:

  ring center  c_i = mean(N1, C2, N3, C4, C5, C6) of residue i
               (six-membered ring shared by A, G, C, U -> one name lookup
               works for purines, pyrimidines and AMBER terminal variants)

  R-vector     r_ij = c_j - c_i                                (3 comps)

  G-vector     gamma = pi / r_c
               G_ij  = [ sin(gamma|r_ij|) r_ij/|r_ij| , 1 + cos(gamma|r_ij|) ]
               for |r_ij| < r_c, else 0                        (4 comps)

Only the i<j upper triangle is stored, flattened pair-major / component-
minor, so column names are  G_<res_i>-<res_j>_<comp>.

r_c defaults to 15 A (see rg_vectors.py for the TAR calibration that set it).

What REVEAL adds over the standalone script
-------------------------------------------
* one global topology OR one topology per trajectory (or a manifest file);
  ring-atom indices are resolved per topology, and every topology must give
  the same residue sequence so columns mean the same thing in every traj
* --stride, applied while reading (mdtraj iterload), plus chunked reading so
  long trajectories never have to fit in memory as full-atom coordinates
* only the ring atoms are read from disk (atom_indices), ~20x less I/O
* optional residue selection (MDTraj selection syntax)
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import numpy as np

from .io import Dataset

RING_ATOMS = ["N1", "C2", "N3", "C4", "C5", "C6"]
R_COMPONENTS = ["x", "y", "z"]
G_COMPONENTS = ["sin_x", "sin_y", "sin_z", "1+cos"]


def ring_atom_indices(topology, selection: Optional[str] = None):
    """(N, 6) ring-atom indices and '<resName><resSeq>' labels for every residue
    that carries all six ring atoms (optionally restricted to a selection)."""
    allowed = None
    if selection:
        allowed = set(topology.select(selection).tolist())
        if not allowed:
            raise ValueError(f"selection '{selection}' matched no atoms")
    idx, labels = [], []
    for res in topology.residues:
        by_name = {a.name: a.index for a in res.atoms}
        if all(n in by_name for n in RING_ATOMS):
            ring = [by_name[n] for n in RING_ATOMS]
            if allowed is not None and not all(a in allowed for a in ring):
                continue
            idx.append(ring)
            labels.append(f"{res.name}{res.resSeq}")
    if not idx:
        raise ValueError("no residues with a full N1/C2/N3/C4/C5/C6 ring found"
                         + (f" in selection '{selection}'" if selection else ""))
    return np.array(idx), labels


def ring_centers(xyz_ang: np.ndarray, ring_idx: np.ndarray) -> np.ndarray:
    """(n_frames, n_atoms, 3) Angstrom -> (n_frames, N, 3) ring centers."""
    return xyz_ang[:, ring_idx, :].mean(axis=2)


def compute_rg_vectors(centers: np.ndarray, r_c: float = 15.0):
    """centers (n_frames, N, 3) in Angstrom -> r (n_frames, n_pairs, 3),
    g (n_frames, n_pairs, 4), pairs [(i, j), ...] with i < j."""
    n_frames, N, _ = centers.shape
    iu, ju = np.triu_indices(N, k=1)
    r = centers[:, ju, :] - centers[:, iu, :]
    dist = np.linalg.norm(r, axis=-1)
    gamma = np.pi / r_c
    within = dist < r_c
    safe = np.where(dist > 0, dist, 1.0)
    direction = r / safe[..., None]
    sin_term = np.sin(gamma * dist)[..., None] * direction
    cos_term = (1.0 + np.cos(gamma * dist))[..., None]
    g = np.concatenate([sin_term, cos_term], axis=-1)
    g = np.where(within[..., None], g, 0.0)
    return r, g, list(zip(iu.tolist(), ju.tolist()))


def feature_names(labels: Sequence[str], kind: str) -> List[str]:
    iu, ju = np.triu_indices(len(labels), k=1)
    comps = R_COMPONENTS if kind == "r" else G_COMPONENTS
    tag = kind.upper()
    return [f"{tag}_{labels[i]}-{labels[j]}_{c}" for i, j in zip(iu, ju) for c in comps]


def featurize(traj_files: Sequence[str], top_files: Sequence[str], kinds=("g", "r"),
              r_cutoff: float = 15.0, stride: int = 1, chunk: int = 1000,
              selection: Optional[str] = None, dtype=np.float32,
              verbose: bool = True) -> Dict[str, Dataset]:
    """Compute R/G vectors for every trajectory.

    traj_files and top_files must be the same length (use io.pair_topologies
    to expand a single global topology).  Returns {"g": Dataset, "r": Dataset}.
    """
    import mdtraj as md

    if len(traj_files) != len(top_files):
        raise ValueError("traj_files and top_files must have the same length")
    stride = max(1, int(stride))
    kinds = tuple(k.lower() for k in kinds)

    top_cache = {}
    ref_labels = None
    out = {k: [] for k in kinds}
    n_frames_report = []

    for ti, (tpath, ppath) in enumerate(zip(traj_files, top_files)):
        if ppath not in top_cache:
            topo = md.load_topology(ppath)
            ring_idx, labels = ring_atom_indices(topo, selection)
            top_cache[ppath] = (ring_idx, labels)
        ring_idx, labels = top_cache[ppath]
        if ref_labels is None:
            ref_labels = labels
            if verbose:
                print(f"tracking {len(labels)} nucleotides: {labels[0]} .. {labels[-1]} "
                      f"({len(labels) * (len(labels) - 1) // 2} pairs)")
        elif labels != ref_labels:
            raise ValueError(
                f"topology {ppath} gives residues {labels[:3]}..{labels[-1:]} ({len(labels)}), "
                f"which differs from the first topology ({ref_labels[:3]}..{ref_labels[-1:]}, "
                f"{len(ref_labels)}). All trajectories must share the same residue sequence.")

        # read only the ring atoms; remap ring indices into that subset
        atom_subset = np.unique(ring_idx.ravel())
        remap = {a: i for i, a in enumerate(atom_subset)}
        local_ring = np.vectorize(remap.get)(ring_idx)

        pieces = {k: [] for k in kinds}
        n_read = 0
        for chunk_traj in md.iterload(tpath, top=ppath, chunk=chunk, stride=stride,
                                      atom_indices=atom_subset):
            centers = ring_centers(chunk_traj.xyz * 10.0, local_ring)  # nm -> Angstrom
            r, g, _ = compute_rg_vectors(centers, r_c=r_cutoff)
            if "g" in kinds:
                pieces["g"].append(g.reshape(g.shape[0], -1).astype(dtype))
            if "r" in kinds:
                pieces["r"].append(r.reshape(r.shape[0], -1).astype(dtype))
            n_read += chunk_traj.n_frames
        if n_read == 0:
            raise ValueError(f"{tpath}: no frames read")
        for k in kinds:
            out[k].append(np.concatenate(pieces[k], axis=0))
        n_frames_report.append(n_read)
        if verbose:
            print(f"  [{ti:02d}] {os.path.basename(tpath)} (top: {os.path.basename(ppath)}) "
                  f"-> traj_{ti:02d}: {n_read} frames")

    iu, ju = np.triu_indices(len(ref_labels), k=1)
    pairs = np.stack([iu, ju], axis=1)
    result = {}
    for k in kinds:
        meta = {
            "kind": "features",
            "method": f"{k.upper()}-vector",
            "frame_stride": stride,
            "frame_start": 0,
            "r_cutoff_angstrom": r_cutoff,
            "selection": selection,
            "traj_files": [os.path.abspath(p) for p in traj_files],
            "top_files": [os.path.abspath(p) for p in top_files],
            "n_residues": len(ref_labels),
        }
        result[k] = Dataset(out[k], feature_names(ref_labels, k), meta,
                            {"residues": np.array(ref_labels), "pairs": pairs})
    return result
