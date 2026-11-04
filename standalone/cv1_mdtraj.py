"""
CV1 for HIV-1 TAR GS <-> ES2 conformational state, computed with MDTraj.

CV1 (pivot form) = d(C13 N3 -- G22 N1)  -  d(C13 N3 -- G20 N1)
                    [GS-register partner]   [ES2-register partner]

Residue numbers are CONSTRUCT numbering (1-31) for the 31-nt sequence
GGCAGAUCUGAGCCUUCGGGAGCUCUCUGCC (5'-G5 ... C3-3', UUCG-stabilised TAR).
C13 is the pivot base that pairs G22 in the ground state and G20 in the
excited state (ES2, PDB 8U3M); its partner switch is the single most
diagnostic distance for the register shift.

Sign convention:  CV1 > 0  ->  ES2-like register
                   CV1 < 0  ->  GS-like register
Empirical decision bands from 86 experimental NMR models (76 GS + 10 ES2,
PDB 1ANR/8THV/1UTS/2KDQ/6D2U/8U3M):
    GS   :  CV1 <= -0.28 Å   (GS models: mean -4.83 +/- 2.62 Å)
    ES2  :  CV1 >= +0.75 Å   (ES2 models: mean +3.70 +/- 0.33 Å)
    intermediate: -0.28 < CV1 < 0.75  (no experimental structure observed here)

A second, coarser form (mean of 4 GS-defining pairs minus mean of 6
ES2-defining pairs) is also provided as `cv1_deltaD` -- it is the exact
CV1 used in the accompanying HIV-1 TAR analysis and agrees with the pivot
form to r > 0.9 on the validation set.

DATA INPUT
----------
This script reads trajectories from a `Data/` folder (default: ./Data,
override with the first positional argument). It expects:
  - one or more `.xtc` trajectory files in that folder
  - a topology file in the same folder: `.pdb`, `.gro`, or `.prmtop`/`.parm7`
    (auto-detected; pass --top to override)

Usage:
    python cv1_mdtraj.py                  # reads ./Data
    python cv1_mdtraj.py /path/to/Data
    python cv1_mdtraj.py Data --top Data/topology.pdb --out results.csv
"""
import os
import numpy as np
import mdtraj as md

WC = {"G5": "N1", "G": "N1", "A": "N1", "C": "N3", "C3": "N3", "U": "N3"}

GS_PAIRS = [(10, 25), (11, 24), (12, 23), (13, 22)]   # GS-register base pairs
ES_PAIRS = [(9, 24), (10, 23), (11, 22), (12, 21), (13, 20), (14, 19)]  # ES2-register


def _atom_index(top, resSeq, atom_name):
    for res in top.residues:
        if res.resSeq == resSeq:
            for atom in res.atoms:
                if atom.name == atom_name:
                    return atom.index
    raise ValueError(f"Atom {atom_name} not found in residue {resSeq}")


def _resname(top, resSeq):
    for res in top.residues:
        if res.resSeq == resSeq:
            return res.name
    raise ValueError(f"Residue {resSeq} not found")


def _wc_distance(traj, i, j):
    """Watson-Crick heavy-atom (N1/N3) distance between residues i and j, in Angstrom."""
    ai = _atom_index(traj.topology, i, WC[_resname(traj.topology, i)])
    aj = _atom_index(traj.topology, j, WC[_resname(traj.topology, j)])
    return md.compute_distances(traj, [[ai, aj]])[:, 0] * 10.0  # nm -> Angstrom


def cv1_pivot(traj, gs_partner=22, es_partner=20, pivot=13):
    """CV1 from a single pivot base's two candidate partners. One value per frame."""
    return _wc_distance(traj, pivot, gs_partner) - _wc_distance(traj, pivot, es_partner)


def cv1_deltaD(traj, gs_pairs=GS_PAIRS, es_pairs=ES_PAIRS):
    """CV1 as <d_GS> - <d_ES2> averaged over the defining pair sets. One value per frame."""
    d_gs = np.mean([_wc_distance(traj, i, j) for i, j in gs_pairs], axis=0)
    d_es = np.mean([_wc_distance(traj, i, j) for i, j in es_pairs], axis=0)
    return d_gs - d_es


def classify(cv1_value, gs_hi=-0.282, es_lo=0.745):
    if cv1_value <= gs_hi:
        return "GS"
    if cv1_value >= es_lo:
        return "ES2"
    return "intermediate"


def find_topology(data_dir):
    """Pick a topology file from data_dir: prefer .pdb, then .gro, then .prmtop/.parm7."""
    import glob
    for ext in ("*.pdb", "*.gro", "*.prmtop", "*.parm7"):
        hits = sorted(glob.glob(os.path.join(data_dir, ext)))
        if hits:
            return hits[0]
    raise FileNotFoundError(
        f"No topology file (.pdb/.gro/.prmtop/.parm7) found in '{data_dir}'. "
        f"Place one alongside the .xtc files, or pass --top explicitly."
    )


def find_xtc_files(data_dir):
    import glob
    hits = sorted(glob.glob(os.path.join(data_dir, "*.xtc")))
    if not hits:
        raise FileNotFoundError(f"No .xtc files found in '{data_dir}'.")
    return hits


def compute_cv1_for_file(xtc_path, topology_path):
    """Load one .xtc against the shared topology and compute both CV1 forms for every frame."""
    traj = md.load(xtc_path, top=topology_path)
    pivot = cv1_pivot(traj)
    deltaD = cv1_deltaD(traj)
    rows = []
    for frame in range(traj.n_frames):
        rows.append({
            "file": os.path.basename(xtc_path),
            "frame": frame,
            "CV1_pivot": float(pivot[frame]),
            "state_pivot": classify(pivot[frame]),
            "CV1_deltaD": float(deltaD[frame]),
            "state_deltaD": classify(deltaD[frame]),
        })
    return rows, pivot, deltaD


def run(data_dir="Data", topology_path=None, out_csv="cv1_results.csv", out_pkl="cv1_results.pkl",
        out_npz="cv1_cv2.npz"):
    """
    Scan `data_dir` for .xtc trajectories, load each against a shared topology
    (auto-detected in `data_dir` unless `topology_path` is given explicitly),
    compute CV1 per frame, and write the combined results to CSV and/or pickle.

    Also writes `out_npz`: one (n_frames, 2) array per trajectory, columns
    [CV1_pivot, CV1_deltaD] (labelled CV1/CV2 for downstream 2D clustering),
    keyed 'traj_00', 'traj_01', ... in the same order as `xtc_files` below.
    This is npz rather than pickle so it loads cleanly across numpy versions
    (no numpy._core / numpy.core reference issues) -- use it directly as the
    `sfa` input list for pyemma.coordinates.cluster_kmeans, e.g.:
        data = np.load('cv1_cv2.npz')
        sfa = [data[k] for k in sorted(data.files)]
    """
    if topology_path is None:
        topology_path = find_topology(data_dir)
    xtc_files = find_xtc_files(data_dir)

    print(f"topology: {topology_path}")
    print(f"found {len(xtc_files)} .xtc file(s) in '{data_dir}':")
    for f in xtc_files:
        print(f"  {f}")

    all_rows = []
    npz_arrays = {}
    for idx, xtc_path in enumerate(xtc_files):
        rows, pivot, deltaD = compute_cv1_for_file(xtc_path, topology_path)
        all_rows.extend(rows)
        npz_arrays[f"traj_{idx:02d}"] = np.column_stack([pivot, deltaD])  # columns: CV1, CV2
        print(f"\n{os.path.basename(xtc_path)}: {len(rows)} frame(s)")
        for r in rows:
            print(f"  frame {r['frame']:5d}  CV1_pivot = {r['CV1_pivot']:+7.3f} A "
                  f"({r['state_pivot']:<12s})  CV1_deltaD = {r['CV1_deltaD']:+7.3f} A "
                  f"({r['state_deltaD']})")

    if out_csv:
        import csv
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nwrote {len(all_rows)} row(s) to {out_csv}")

    if out_pkl:
        import pickle
        try:
            import pandas as pd
            payload = pd.DataFrame(all_rows)
        except ImportError:
            payload = all_rows  # fall back to a plain list of dicts if pandas is unavailable
        with open(out_pkl, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"wrote {len(all_rows)} row(s) to {out_pkl} "
              f"({'DataFrame' if 'pd' in dir() and isinstance(payload, pd.DataFrame) else 'list of dict'})")

    if out_npz:
        np.savez(out_npz, **npz_arrays)
        print(f"wrote {len(npz_arrays)} trajector{'y' if len(npz_arrays) == 1 else 'ies'} "
              f"(CV1, CV2 columns) to {out_npz}")

    return all_rows


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Compute CV1 (GS/ES2 register coordinate) for every frame "
                    "of every .xtc trajectory found in a Data folder."
    )
    parser.add_argument("data_dir", nargs="?", default="Data",
                        help="Folder containing the .xtc file(s) and a topology "
                             "file (.pdb/.gro/.prmtop). Default: ./Data")
    parser.add_argument("--top", dest="topology_path", default=None,
                        help="Explicit topology file path (overrides auto-detection "
                             "inside data_dir).")
    parser.add_argument("--out", dest="out_csv", default="cv1_results.csv",
                        help="Output CSV path (default: cv1_results.csv). Pass "
                             "an empty string to skip writing a file.")
    parser.add_argument("--out-pkl", dest="out_pkl", default="cv1_results.pkl",
                        help="Output pickle path (default: cv1_results.pkl). Pass "
                             "an empty string to skip writing a file. Saves a "
                             "pandas DataFrame if pandas is available, else a "
                             "list of dicts.")
    parser.add_argument("--out-npz", dest="out_npz", default="cv1_cv2.npz",
                        help="Output npz path (default: cv1_cv2.npz). Pass an "
                             "empty string to skip writing a file. Stores one "
                             "(n_frames, 2) array per trajectory, columns "
                             "[CV1_pivot, CV1_deltaD], keyed traj_00, traj_01, ...")
    args = parser.parse_args()
    run(args.data_dir, args.topology_path, args.out_csv or None, args.out_pkl or None,
        args.out_npz or None)
