"""Build a tiny synthetic RNA-like system to exercise REVEAL end to end.

12 nucleotides on a helix; residues 5-7 (a "loop") switch between a stacked
and a flipped-out position via a slow two-state Markov chain, plus thermal
noise. Four trajectories x 3000 frames, written as .xtc, with
  * topology A (top_A.pdb): canonical atom order
  * topology B (traj_01.pdb, traj_03.pdb): same residues, atoms in a
    different order within every residue -- tests per-trajectory topologies.
"""
import os
import sys

import mdtraj as md
import numpy as np

RING = ["N1", "C2", "N3", "C4", "C5", "C6"]
OTHER = ["P", "C1'", "O4'"]
SEQ = ["G", "G", "C", "A", "G", "U", "A", "C", "U", "C", "C", "C"]
N_RES = len(SEQ)
LOOP = [5, 6, 7]


def build_top(order="A"):
    top = md.Topology()
    ch = top.add_chain()
    atom_map = []  # (res, atomname) in the order atoms are created
    for i, rn in enumerate(SEQ):
        r = top.add_residue(rn, ch, resSeq=17 + i)
        names = OTHER + RING if order == "A" else RING[::-1] + OTHER[::-1]
        for n in names:
            el = md.element.get_by_symbol(n[0])
            top.add_atom(n, el, r)
            atom_map.append((i, n))
    return top, atom_map


def ring_offsets():
    ang = np.linspace(0, 2 * np.pi, 7)[:-1]
    return {n: 0.14 * np.array([np.cos(a), np.sin(a), 0.0]) for n, a in zip(RING, ang)}


def coords(n_frames, rng):
    # helix ring centres (nm)
    t = np.arange(N_RES)
    base = np.stack([0.9 * np.cos(t * 0.6), 0.9 * np.sin(t * 0.6), 0.34 * t], axis=1)
    flipped = base.copy()
    flipped[LOOP] += np.array([1.2, -0.8, 0.0])
    state = np.zeros(n_frames, dtype=int)
    s = rng.integers(2)
    for f in range(n_frames):
        if rng.random() < 0.004:
            s = 1 - s
        state[f] = s
    centers = np.where(state[:, None, None] == 1, flipped[None], base[None]).astype(float)
    # smooth the switch a bit + noise
    centers += rng.normal(0, 0.05, centers.shape)
    return centers, state


def write(outdir, n_traj=4, n_frames=3000, seed=0):
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(seed)
    offs = ring_offsets()
    topA, mapA = build_top("A")
    topB, mapB = build_top("B")
    other_off = {"P": np.array([0.0, 0.0, 0.5]), "C1'": np.array([0.0, 0.3, 0.2]),
                 "O4'": np.array([0.2, 0.3, 0.2])}
    states = {}
    for k in range(n_traj):
        centers, state = coords(n_frames, rng)
        states[f"traj_{k:02d}"] = state
        use_B = k % 2 == 1
        top, amap = (topB, mapB) if use_B else (topA, mapA)
        xyz = np.zeros((n_frames, top.n_atoms, 3), dtype=np.float32)
        for a, (ri, n) in enumerate(amap):
            off = offs[n] if n in offs else other_off[n]
            xyz[:, a] = centers[:, ri] + off + rng.normal(0, 0.01, (n_frames, 3))
        tr = md.Trajectory(xyz, top)
        tr.save_xtc(os.path.join(outdir, f"traj_{k:02d}.xtc"))
        if use_B:
            tr[0].save_pdb(os.path.join(outdir, f"traj_{k:02d}.pdb"))
        if k == 0:
            tr[0].save_pdb(os.path.join(outdir, "top_A.pdb"))
    np.savez(os.path.join(outdir, "true_states.npz"), **states)
    print(f"wrote {n_traj} trajectories x {n_frames} frames to {outdir}")


if __name__ == "__main__":
    write(sys.argv[1] if len(sys.argv) > 1 else "synthetic")
