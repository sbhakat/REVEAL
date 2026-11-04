"""
Linear reductions: functional PCA and slow feature analysis.

FPCA
----
As in fpca_gvec.py, each frame's feature vector is treated as a function
sampled on the feature-index grid t = 0..p-1 and decomposed with FPCA.
The native implementation reproduces skfda.FPCA on an FDataGrid without
regularization: Simpson-rule quadrature weights w on the grid (exactly
scipy.integrate.simpson(I, x=grid), as skfda computes them), PCA of
(X - mean) * sqrt(w), components phi = v / sqrt(w) (so that the weighted
inner product <phi_k, phi_k>_w = 1), scores = (X - mean) @ (phi * w).T.
Default backend: skfda itself; --backend native is the NumPy re-implementation
(identical results, no skfda dependency).

SFA
---
Slow feature analysis (Wiskott & Sejnowski 2002), as in sfa_gvec.py:
whiten X, then find the directions in whitened space whose lag-tau
differences have the smallest variance; outputs are unit-variance,
decorrelated and ordered slowest first, written in affine form
y = W x + b (same convention as sksfa's affine_parameters()).
Default backend: sksfa (see sfa_sksfa for how exact lag-tau, within-
trajectory SFA is obtained from stock sksfa). The native NumPy version
also differences only *within* trajectories. sfa_gvec.py differenced the
concatenated array, which adds spurious jumps at every trajectory
boundary; --cross-boundaries restores that legacy behaviour.
Rank-deficient inputs (G-vector columns that are always beyond the
cutoff, linear dependencies) are handled by dropping whitening
directions with eigenvalue < rcond * max.
"""
from __future__ import annotations

import numpy as np

from .common import column_stats, covariance


def quadrature_weights(p: int, rule: str = "simpson") -> np.ndarray:
    """Integration weights on the grid 0..p-1 (skfda's default is Simpson)."""
    if rule == "trapezoid":
        w = np.ones(p)
        if p > 1:
            w[0] = w[-1] = 0.5
        return w
    from scipy.integrate import simpson
    w = simpson(y=np.eye(p), x=np.arange(p, dtype=float))
    if np.any(w <= 0):
        raise ValueError("non-positive Simpson weights; use rule='trapezoid'")
    return w


def fpca_native(X: np.ndarray, n_components: int, rule: str = "simpson"):
    p = X.shape[1]
    w = quadrature_weights(p, rule)
    sw = np.sqrt(w)
    mean, _ = column_stats(X)
    C = covariance(X, mean, scale=sw)
    evals, evecs = np.linalg.eigh(C)
    order = np.argsort(evals)[::-1]
    evals, evecs = evals[order], evecs[:, order]
    k = n_components
    V = evecs[:, :k]
    # deterministic sign: largest-|loading| entry positive (sklearn's svd_flip convention)
    signs = np.sign(V[np.argmax(np.abs(V), axis=0), np.arange(k)])
    V = V * signs
    comps = (V / sw[:, None]).T                       # (k, p) functional components
    proj = (comps * w).T                              # (p, k) quadrature-weighted projector
    scores = np.empty((X.shape[0], k))
    for i in range(0, X.shape[0], 100_000):
        scores[i:i + 100_000] = (X[i:i + 100_000].astype(np.float64) - mean) @ proj
    evr = evals[:k] / evals[evals > 0].sum()
    return scores, comps, mean, {"explained_variance": evals[:k], "explained_variance_ratio": evr}


def fpca_skfda(X: np.ndarray, n_components: int, seed: int = 0):
    """skfda.FPCA on an FDataGrid, as in fpca_gvec.py.

    skfda's internal sklearn PCA uses svd_solver='auto', which picks a randomized SVD for
    some data shapes (e.g. 2400 x 264) and draws from NumPy's global RNG; NumPy is seeded
    first so results are reproducible run to run.
    """
    import skfda
    from skfda.preprocessing.dim_reduction import FPCA
    np.random.seed(seed)
    X = X.astype(np.float64)
    fd = skfda.FDataGrid(data_matrix=X, grid_points=[np.arange(X.shape[1])])
    model = FPCA(n_components=n_components)
    model.fit(fd)
    scores = model.transform(fd)
    comps = model.components_.data_matrix[:, :, 0]
    mean = model.mean_.data_matrix[0, :, 0]
    return scores, comps, mean, {"explained_variance": model.explained_variance_,
                                 "explained_variance_ratio": model.explained_variance_ratio_}


def sfa_native(trajs, n_components: int, tau: int = 1, cross_boundaries: bool = False,
               rcond: float = 1e-10):
    X = np.concatenate(trajs, axis=0)
    mean, _ = column_stats(X)
    C = covariance(X, mean, ddof=0)
    evals, evecs = np.linalg.eigh(C)
    keep = evals > rcond * evals.max()
    Wwhite = evecs[:, keep] / np.sqrt(evals[keep])   # (p, r)
    r = Wwhite.shape[1]
    if r < n_components:
        raise ValueError(f"data has rank {r} < n_components={n_components}")

    segments = [X] if cross_boundaries else trajs
    Cd = np.zeros((r, r))
    n_d = 0
    for seg in segments:
        if seg.shape[0] <= tau:
            continue
        for i in range(0, seg.shape[0] - tau, 50_000):
            j = min(i + 50_000, seg.shape[0] - tau)
            a = (seg[i:j].astype(np.float64) - mean) @ Wwhite
            b = (seg[i + tau:j + tau].astype(np.float64) - mean) @ Wwhite
            d = b - a
            Cd += d.T @ d
            n_d += d.shape[0]
    if n_d == 0:
        raise ValueError(f"tau={tau} is longer than every trajectory")
    Cd /= n_d
    dvals, dvecs = np.linalg.eigh(Cd)                 # ascending: slowest first
    R = dvecs[:, :n_components]
    W = (Wwhite @ R).T                                # (k, p)
    signs = np.sign(W[np.arange(n_components), np.argmax(np.abs(W), axis=1)])
    W = W * signs[:, None]
    b = -W @ mean
    Y = np.empty((X.shape[0], n_components))
    for i in range(0, X.shape[0], 100_000):
        Y[i:i + 100_000] = X[i:i + 100_000].astype(np.float64) @ W.T + b
    return Y, W, b, {"delta_values": dvals[:n_components], "whitening_rank": int(r)}


def _sksfa():
    """sklearn-sfa with mdml's tau patch, bundled in reveal/_vendor/sksfa."""
    from .._vendor import sksfa
    return sksfa


def _sksfa_has_tau() -> bool:
    """True for the mdml-patched sklearn-sfa (fit(X, tau, y=None)) -- always the case for the
    bundled copy. Checked from the signature: stock fit(X, y=None) would silently take a
    positional tau as y."""
    import inspect
    return "tau" in inspect.signature(_sksfa().SFA.fit).parameters


def _sksfa_fit(model, X, tau):
    if _sksfa_has_tau():
        return model.fit(X, tau)
    if tau != 1:
        raise ValueError("this sklearn-sfa has no 'tau' argument (lag 1 only)")
    return model.fit(X)


def sfa_sksfa(trajs, n_components: int, tau: int = 1, cross_boundaries: bool = False,
              robustness_cutoff: float = 1e-15):
    """SFA with sklearn-sfa (sksfa) as the solver.

    Works with the mdml-patched sksfa (fit(X, tau): differences X[tau:] - X[:-tau]) and
    with stock sksfa (lag 1 only). Both difference *whatever array they are given*, i.e.
    across trajectory boundaries when trajectories are concatenated. Routes:

    cross_boundaries=True     fit(X, tau) on the concatenated frames -> exactly sfa_gvec.py
                              with mdml's sksfa (stock sksfa: tau=1 only)
    tau == 1, equal lengths   fit(X) with batch_size = trajectory length: sksfa's standard
                              method, differences only inside each trajectory. (Not used for
                              tau > 1: in mdml's patch the batch_size path still uses lag 1.)
    otherwise                 whiten exactly like sksfa's standard method (PCA whiten=True,
                              drop directions with explained-variance ratio <= robustness_cutoff),
                              then sksfa.partial() once per lag-tau sub-series x[o::tau] of every
                              trajectory (sksfa never differences across partial() calls), then
                              sksfa's generalized eigenproblem. Exact lag-tau, every frame in the
                              covariance, every within-trajectory (t, t+tau) pair in the slowness.
    """
    import contextlib
    import io as _io
    sksfa = _sksfa()
    from sklearn.decomposition import PCA

    lengths = [t.shape[0] for t in trajs]
    X = np.concatenate(trajs, axis=0).astype(np.float64)
    build = "mdml-patched" if _sksfa_has_tau() else "stock"
    quiet = contextlib.redirect_stdout(_io.StringIO())   # affine_parameters() prints a debug value

    if cross_boundaries:
        model = sksfa.SFA(n_components=n_components, fill_mode="zero")
        _sksfa_fit(model, X, tau)
        Y = model.transform(X)
        with quiet:
            W, b = model.affine_parameters()
        return Y, W, b, {"sksfa_build": build,
                         "sksfa_route": f"fit(X, tau={tau}) on concatenated frames (crosses boundaries)",
                         "delta_values": np.asarray(model.delta_values_)[:n_components]}

    if tau == 1 and len(set(lengths)) == 1 and lengths[0] > 1:
        model = sksfa.SFA(n_components=n_components, batch_size=lengths[0], fill_mode="zero")
        _sksfa_fit(model, X, 1)
        Y = model.transform(X)
        with quiet:
            W, b = model.affine_parameters()
        return Y, W, b, {"sksfa_build": build,
                         "sksfa_route": f"fit(X, batch_size={lengths[0]}): one batch per trajectory",
                         "delta_values": np.asarray(model.delta_values_)[:n_components]}

    pca = PCA(whiten=True, svd_solver="full").fit(X)
    keep = np.flatnonzero(pca.explained_variance_ratio_ > robustness_cutoff)
    Wp = (pca.components_[keep].T / np.sqrt(pca.explained_variance_[keep])).T   # (r, p)
    model = sksfa.SFA(n_components=n_components)
    n_series = 0
    for t in trajs:
        z = (np.asarray(t, dtype=np.float64) - pca.mean_) @ Wp.T
        for o in range(tau):
            sub = z[o::tau]
            if sub.shape[0] >= 2:
                model.partial(sub)
                n_series += 1
    if n_series == 0:
        raise ValueError(f"tau={tau} is longer than every trajectory")
    Z = (X - pca.mean_) @ Wp.T
    Y = model.transform(Z)                               # sksfa: eigh(Cdot, C) on accumulated stats
    V = model._partial_eigenvectors[:, :n_components]    # (r, k)
    zmean = model.sample_sum / model._n_partial_samples
    W = (Wp.T @ V).T                                     # (k, p)
    b = -(pca.mean_ @ Wp.T) @ V - zmean @ V
    return Y, W, b, {"sksfa_build": build,
                     "sksfa_route": f"sksfa whitening + partial() over {n_series} lag-{tau} sub-series",
                     "delta_values": np.asarray(model._partial_eigenvalues)[:n_components],
                     "whitening_rank": int(len(keep))}
