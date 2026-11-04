"""REVEAL test suite: `pytest -q tests/` (builds a synthetic RNA system in a temp dir)."""
import os
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)


def reveal(*args, cwd):
    r = subprocess.run([sys.executable, "-m", "reveal", *args], cwd=cwd, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


@pytest.fixture(scope="module")
def work(tmp_path_factory):
    d = tmp_path_factory.mktemp("reveal")
    from make_synthetic import write
    write(str(d / "syn"), n_traj=4, n_frames=1200, seed=0)
    with open(d / "manifest.txt", "w") as f:
        for k in range(4):
            top = "syn/top_A.pdb" if k % 2 == 0 else f"syn/traj_{k:02d}.pdb"
            f.write(f"syn/traj_{k:02d}.xtc {top}\n")
    reveal("featurize", "--manifest", "manifest.txt", "--stride", "2", "--chunk", "137",
           "--out-prefix", "feat/rg", cwd=d)
    return d


def test_featurize_matches_reference(work):
    import mdtraj as md
    from reveal.io import load_dataset
    from reveal.featurize import ring_atom_indices, compute_rg_vectors
    G = load_dataset(str(work / "feat/rg_g.npz"))
    assert G.frame_stride == 2 and G.lengths == [600] * 4
    for k in range(4):
        top = work / ("syn/top_A.pdb" if k % 2 == 0 else f"syn/traj_{k:02d}.pdb")
        t = md.load(str(work / f"syn/traj_{k:02d}.xtc"), top=str(top), stride=2)
        idx, _ = ring_atom_indices(t.topology)
        _, g, _ = compute_rg_vectors((t.xyz * 10)[:, idx].mean(2))
        assert np.abs(G.trajs[k] - g.reshape(len(g), -1)).max() < 1e-5


def test_pwcs_matches_legacy():
    from legacy_pwcs import pwcs_score as legacy
    from reveal.pwcs import pwcs_score
    rng = np.random.default_rng(0)
    X = np.concatenate([rng.normal(0, 1, (400, 2)), rng.normal(4, 0.5, (300, 2))])
    fs = {"a": X[:, [0]], "b": X[:, [1]], "ab": X}
    w = rng.random(len(X))
    assert np.allclose(legacy(fs), pwcs_score(fs), atol=1e-12)
    assert np.allclose(legacy(fs, weights=w), pwcs_score(fs, weights=w), atol=1e-12)


@pytest.mark.parametrize("kw", [dict(tau=1), dict(tau=3), dict(tau=1, cross_boundaries=True),
                                dict(tau=2, unequal=True)])
def test_sksfa_routes_match_native(kw):
    """Every sksfa route (batch / whitening+partial / cross-boundaries) equals native SFA."""
    from reveal.reduce.linear import sfa_native, sfa_sksfa, _sksfa_has_tau
    kw = dict(kw)
    unequal = kw.pop("unequal", False)
    rng = np.random.default_rng(0)
    M = rng.normal(size=(7, 7))
    trajs = []
    for n in ([700, 500, 600] if unequal else [600, 600, 600]):
        slow = np.cumsum(rng.normal(0, 0.05, (n, 2)), 0)
        trajs.append(np.hstack([slow, rng.normal(size=(n, 5))]) @ M)
    Ys, Ws, bs, es = sfa_sksfa(trajs, 2, **kw)
    Yn, *_ = sfa_native(trajs, 2, **kw)
    for i in range(2):
        assert abs(np.corrcoef(Ys[:, i], Yn[:, i])[0, 1]) > 0.99999
    assert np.allclose(np.concatenate(trajs) @ Ws.T + bs, Ys, atol=1e-6)


def test_fpca_equals_direct_skfda(work):
    """REVEAL's skfda backend reproduces a direct fpca_gvec.py-style skfda call."""
    skfda = pytest.importorskip("skfda")
    from skfda.preprocessing.dim_reduction import FPCA
    from reveal.io import load_dataset
    reveal("reduce", "fpca", "-i", "feat/rg_g.npz", "-n", "2", "-o", "red/fpca_direct", cwd=work)
    X = load_dataset(str(work / "feat/rg_g.npz")).concat(np.float64)
    fd = skfda.FDataGrid(data_matrix=X, grid_points=[np.arange(X.shape[1])])
    np.random.seed(0)                      # REVEAL seeds NumPy before skfda (randomized SVD)
    m = FPCA(n_components=2).fit(fd)
    Y = load_dataset(str(work / "red/fpca_direct/projection.npz")).concat()
    assert np.allclose(Y, m.transform(fd), atol=1e-4)
    assert np.allclose(np.load(work / "red/fpca_direct/fpca_weights.npy"),
                       m.components_.data_matrix[:, :, 0], atol=1e-10)


def test_sfa_cross_boundaries_equals_direct_sksfa(work):
    """--cross-boundaries reproduces sfa_gvec.py's call sksfa.SFA(n).fit(X, tau=tau) (mdml build,
    here the copy bundled in reveal._vendor)."""
    from reveal._vendor import sksfa
    from reveal.io import load_dataset
    reveal("reduce", "sfa", "-i", "feat/rg_g.npz", "-n", "2", "--tau", "3", "--cross-boundaries",
           "-o", "red/sfa_cb", cwd=work)
    X = load_dataset(str(work / "feat/rg_g.npz")).concat(np.float64)
    m = sksfa.SFA(n_components=2)
    m.fit(X, tau=3)
    Y = load_dataset(str(work / "red/sfa_cb/projection.npz")).concat()
    assert np.allclose(Y, m.transform(X), atol=1e-4)


@pytest.mark.parametrize("method,extra", [
    ("fpca", []), ("sfa", ["--tau", "2"]),
    ("tae", ["--lag", "2", "--max-steps", "100", "--max-rounds", "1"])])
def test_reduce_methods(work, method, extra):
    pytest.importorskip("torch") if method == "tae" else None
    reveal("reduce", method, "-i", "feat/rg_g.npz", "-n", "2", "-o", f"red/{method}", *extra, cwd=work)
    from reveal.io import load_dataset
    P = load_dataset(str(work / f"red/{method}/projection.npz"))
    assert P.n_dim == 2 and P.frame_stride == 2 and P.lengths == [600] * 4
    assert np.load(work / f"red/{method}/{method}_weights.npy").shape == (2, 264)


def test_tae_matches_tae_gvec(work):
    """REVEAL's TAE == tae_gvec.py (verbatim copy in tests/legacy_tae_gvec.py), same seed.

    tae_gvec.py does `from timelaggedcv import ...`. If mdml's timelaggedcv is installed, that one
    is used (bundled copy vs mdml's); otherwise the bundled copy is registered under that name.
    """
    pytest.importorskip("torch")
    import importlib
    try:
        importlib.import_module("timelaggedcv")
    except ImportError:
        import reveal._vendor.timelaggedcv as tl
        for sub in ("", ".dataset", ".dataset.dataset", ".dataset.utils", ".util", ".net",
                    ".estimator", ".model", ".vamp", ".vamp.vamp"):
            sys.modules["timelaggedcv" + sub] = importlib.import_module(tl.__name__ + sub)
    import argparse
    import random
    import torch
    import legacy_tae_gvec as L
    from reveal.io import load_dataset
    G = load_dataset(str(work / "feat/rg_g.npz"))
    leg = work / "legacy"
    leg.mkdir(exist_ok=True)
    np.savez(leg / "rg_vectors_g_part1.npz", traj_00=G.trajs[0], traj_01=G.trajs[1])
    np.savez(leg / "rg_vectors_g_part2.npz", traj_00=G.trajs[2], traj_01=G.trajs[3])
    np.save(leg / "rg_vectors_g_features.npy", np.array(G.feature_names))
    args = argparse.Namespace(stride=1, target_n_feat=30, max_steps=150, max_rounds=1, patience=2,
                              step_lam=0.2, batch_size=1024, valid_ratio=0.2, lr=1e-3,
                              weight_decay=0.1, top_k=20)
    cwd = os.getcwd()
    try:
        os.chdir(work)
        fp, names = L.load_and_cache_gvectors("legacy", "legacy_cache", 1)
        torch.manual_seed(0); np.random.seed(0); random.seed(0)
        L.run_one(3, 2, fp, names, names.shape[0], args, "legacy_out")
    finally:
        os.chdir(cwd)
    reveal("reduce", "tae", "-i", "legacy/rg_vectors_g_part*.npz", "-n", "2", "--lag", "3",
           "--seed", "0", "--max-steps", "150", "--max-rounds", "1", "-o", "reveal_tae", cwd=work)
    assert np.array_equal(np.load(work / "reveal_tae/tae_latent.npy"),
                          np.load(work / "legacy_out/tau3_dim2/tae_latent.npy"))
    assert np.array_equal(np.load(work / "reveal_tae/tae_weights.npy"),
                          np.load(work / "legacy_out/tau3_dim2/tae_weights.npy"))


def test_msm_and_pwcs_weighting(work):
    reveal("reduce", "fpca", "-i", "feat/rg_g.npz", "-n", "2", "-o", "red/fpca2", cwd=work)
    reveal("msm", "-i", "F=red/fpca2/projection.npz", "--k", "20", "--lag", "2",
           "--its-lags", "1", "2", "5", "-o", "msm", cwd=work)
    reveal("pwcs", "-i", "F=red/fpca2/projection.npz", "--d-max", "4", "-o", "pw_frame", cwd=work)
    reveal("pwcs", "-i", "F=red/fpca2/projection.npz", "--d-max", "4", "--weights",
           "msm/msm_weights.npz", "-o", "pw_w", cwd=work)
    reveal("pwcs", "-i", "F=red/fpca2/projection.npz", "--d-max", "4", "--msm", "--msm-k", "20",
           "--msm-lag", "2", "-o", "pw_msm", cwd=work)
    a = np.load(work / "pw_w/pwcs_score.npz")["pwcs_score"]
    b = np.load(work / "pw_msm/pwcs_score.npz")["pwcs_score"]
    c = np.load(work / "pw_frame/pwcs_score.npz")["pwcs_score"]
    assert np.allclose(a, b) and not np.allclose(a, c) and len(a) == 2400
    out = reveal("extract", "--pwcs", "pw_msm/pwcs.npz", "--top-n", "1", "--bottom-n", "1",
                 "-o", "frames", cwd=work)
    assert "wrote 2 PDB" in out


def test_stride_alignment(work):
    st = np.load(work / "syn/true_states.npz")
    np.savez(work / "cv_full.npz", **{k: np.stack([st[k], st[k] * 0.5], 1).astype("f4")
                                      for k in sorted(st.files)})
    reveal("reduce", "fpca", "-i", "feat/rg_g.npz", "-n", "1", "-o", "red/fpca1", cwd=work)
    out = reveal("pwcs", "-i", "CV=cv_full.npz:0", "-i", "F=red/fpca1/projection.npz",
                 "--d-max", "3", "-o", "pw_align", cwd=work)
    assert "frame_stride 1 -> 2" in out


def test_pwcs_user_variable_and_user_weights(tmp_path):
    """User-defined variable + user MSM weights in several formats, with --stride."""
    from reveal.pwcs import pwcs_score
    rng = np.random.default_rng(3)
    L = [400, 350, 450]
    trajs = [np.cumsum(rng.normal(0, 0.3, (n, 2)), 0) for n in L]
    X = np.concatenate(trajs)
    w = rng.random(X.shape[0])
    np.save(tmp_path / "rc.npy", X[:, 0])
    with open(tmp_path / "colvar.dat", "w") as f:
        f.write("#! FIELDS time a b\n")
        np.savetxt(f, np.column_stack([np.arange(len(X)), X]), fmt="%.10f")
    b = np.cumsum([0] + L)
    per = [w[b[i]:b[i + 1]] for i in range(3)]
    np.save(tmp_path / "w.npy", w)
    np.savez(tmp_path / "w_list.npz", *per)
    np.savetxt(tmp_path / "w.txt", w)
    ws = np.concatenate([p_[::5] for p_ in per])
    np.save(tmp_path / "w_strided.npy", ws)
    lens = [str(n) for n in L]
    X1 = np.concatenate([t[::5, [0]] for t in trajs])
    X2 = np.concatenate([t[::5] for t in trajs])
    ref1 = pwcs_score({"RC": X1}, d_range=range(2, 4), weights=ws)
    ref2 = pwcs_score({"RC": X2}, d_range=range(2, 4), weights=ws)
    for wf in ("w.npy", "w_list.npz", "w.txt", "w_strided.npy"):
        reveal("pwcs", "-i", "RC=rc.npy", "--lengths", *lens, "--stride", "5", "--d-max", "3",
               "--weights", wf, "-o", f"o_{wf}", cwd=tmp_path)
        S = np.load(tmp_path / f"o_{wf}/pwcs_score.npz")
        assert np.allclose(S["weights"], ws) and np.allclose(S["pwcs_score"], ref1, atol=1e-10), wf
    reveal("pwcs", "-i", "RC=colvar.dat:1,2", "--lengths", *lens, "--stride", "5", "--d-max", "3",
           "--weights", "w.npy", "-o", "o_colvar", cwd=tmp_path)
    assert np.allclose(np.load(tmp_path / "o_colvar/pwcs_score.npz")["pwcs_score"], ref2, atol=1e-6)
    # wrong count and negative weights are rejected
    np.save(tmp_path / "w_bad.npy", w[:-3])
    wn = w.copy(); wn[1] = -0.1; np.save(tmp_path / "w_neg.npy", wn)
    for bad in ("w_bad.npy", "w_neg.npy"):
        r = subprocess.run([sys.executable, "-m", "reveal", "pwcs", "-i", "RC=rc.npy", "--lengths", *lens,
                            "--stride", "5", "--weights", bad, "-o", "o_bad"],
                           cwd=tmp_path, capture_output=True, text=True)
        assert r.returncode != 0 and "error:" in r.stderr, bad



def test_pwcs_minimal_cv_and_weights(tmp_path):
    """Just a CV (+ optional weights): no --lengths, no --stride needed."""
    from reveal.pwcs import pwcs_score
    rng = np.random.default_rng(5)
    X = np.concatenate([rng.normal(0, 1, (500, 1)), rng.normal(4, 0.6, (400, 1))])
    w = rng.random(len(X))
    np.save(tmp_path / "cv.npy", X[:, 0])
    np.save(tmp_path / "w.npy", w)
    reveal("pwcs", "-i", "CV=cv.npy", "--weights", "w.npy", "--d-max", "4", "-o", "o_w", cwd=tmp_path)
    reveal("pwcs", "-i", "CV=cv.npy", "--d-max", "4", "-o", "o_frames", cwd=tmp_path)
    assert np.allclose(np.load(tmp_path / "o_w/pwcs_score.npz")["pwcs_score"],
                       pwcs_score({"CV": X}, d_range=range(2, 5), weights=w), atol=1e-12)
    S = np.load(tmp_path / "o_frames/pwcs_score.npz")
    assert "weights" not in S.files
    assert np.allclose(S["pwcs_score"], pwcs_score({"CV": X}, d_range=range(2, 5)), atol=1e-12)


def test_lengths_taken_from_per_trajectory_weights(tmp_path):
    """Flat CV, no --lengths, weights saved per trajectory -> boundaries come from the weights."""
    rng = np.random.default_rng(6)
    L = [300, 250, 320]
    X = rng.normal(size=(sum(L), 1))
    w = rng.random(sum(L))
    b = np.cumsum([0] + L)
    np.save(tmp_path / "cv.npy", X[:, 0])
    np.savez(tmp_path / "w_list.npz", *[w[b[i]:b[i + 1]] for i in range(3)])
    np.save(tmp_path / "w.npy", w)
    out = reveal("pwcs", "-i", "CV=cv.npy", "--weights", "w_list.npz", "--stride", "4", "--d-max", "3",
                 "-o", "o_auto", cwd=tmp_path)
    assert "trajectory lengths taken from the per-trajectory weights" in out
    reveal("pwcs", "-i", "CV=cv.npy", "--lengths", *map(str, L), "--weights", "w.npy", "--stride", "4",
           "--d-max", "3", "-o", "o_explicit", cwd=tmp_path)
    assert np.array_equal(np.load(tmp_path / "o_auto/pwcs_score.npz")["pwcs_score"],
                          np.load(tmp_path / "o_explicit/pwcs_score.npz")["pwcs_score"])


def test_reveal_lengths(work, tmp_path):
    out = reveal("lengths", "--traj", str(work / "syn/traj_*.xtc"), cwd=work)
    assert "--lengths 1200 1200 1200 1200" in out
    out = reveal("lengths", "--traj", str(work / "syn/traj_*.xtc"), "--stride", "7",
                 "-o", str(tmp_path / "l.txt"), cwd=work)
    assert "--lengths 172 172 172 172" in out                     # ceil(1200 / 7)
    import mdtraj as md
    assert md.load(str(work / "syn/traj_00.xtc"), top=str(work / "syn/top_A.pdb"), stride=7).n_frames == 172
    assert np.loadtxt(tmp_path / "l.txt").astype(int).tolist() == [172] * 4
    out = reveal("lengths", "--from", "feat/rg_g.npz", cwd=work)
    assert "--lengths 600 600 600 600" in out
    with open(tmp_path / "colvar", "w") as f:                     # time restarts per trajectory
        f.write("#! FIELDS time cv\n")
        for n in (5, 7, 3):
            np.savetxt(f, np.column_stack([np.arange(n) * 2.0, np.ones(n)]))
    out = reveal("lengths", "--colvar", str(tmp_path / "colvar"), cwd=work)
    assert "--lengths 5 7 3" in out


def test_bundled_packages_are_used():
    """The bundled sksfa has mdml's tau patch; the bundled timelaggedcv imports from inside REVEAL."""
    import inspect
    from reveal._vendor import sksfa
    from reveal.reduce.linear import _sksfa
    from reveal.reduce.tlcv import import_timelaggedcv
    assert "tau" in inspect.signature(sksfa.SFA.fit).parameters
    assert _sksfa() is sksfa
    pytest.importorskip("torch")
    assert import_timelaggedcv().__name__ == "reveal._vendor.timelaggedcv"

