"""
Markov state models on any low-dimensional variables (CVs, FPCA/SFA/TAE
projections, ...), producing per-frame equilibrium weights.

Method (same as hivtar.ipynb / build_tae_msm*.py / pwcs.py --use-msm-weights):
k-means microstates -> (Bayesian) reversible MSM at lag tau -> per-frame
trajectory weights  w_t = pi_{s(t)} / N_{s(t)}  (stationary probability of
the frame's microstate divided by that microstate's frame count), so a
weighted histogram over frames reproduces the MSM equilibrium distribution.
Frames in microstates outside the largest connected set get weight 0.

Backend: deeptime (default; maintained successor of pyemma). If deeptime is
missing, pyemma is used with the exact calls from the original scripts.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np


@dataclass
class MSMResult:
    dtrajs: List[np.ndarray]
    centers: np.ndarray
    weights: List[np.ndarray]
    lag: int
    k: int
    bayesian: bool
    transition_matrix: np.ndarray
    stationary_distribution: np.ndarray
    active_set: np.ndarray
    active_state_fraction: float
    active_count_fraction: float
    timescales: np.ndarray
    backend: str
    model: object = field(default=None, repr=False)

    def flat_weights(self) -> np.ndarray:
        return np.concatenate(self.weights)

    def save(self, path):
        d = {k: getattr(self, k) for k in ("dtrajs", "centers", "weights", "lag", "k", "bayesian",
                                            "transition_matrix", "stationary_distribution",
                                            "active_set", "active_state_fraction",
                                            "active_count_fraction", "timescales", "backend")}
        with open(path, "wb") as f:
            pickle.dump(d, f)


def _kmeans_deeptime(trajs, k, max_iter, seed, fit_stride):
    from deeptime.clustering import KMeans
    X = np.concatenate([t[::fit_stride] for t in trajs]).astype(np.float64)
    km = KMeans(n_clusters=k, max_iter=max_iter, fixed_seed=seed, init_strategy="kmeans++")
    model = km.fit(X).fetch_model()
    return [model.transform(t.astype(np.float64)) for t in trajs], model.cluster_centers


def _active_fractions(dtrajs, counts_full, counts_sub, k):
    n_states_total = len(np.unique(np.concatenate(dtrajs)))
    asf = counts_sub.n_states / max(n_states_total, 1)
    acf = counts_sub.count_matrix.sum() / max(counts_full.count_matrix.sum(), 1e-12)
    return float(asf), float(acf)


def build_msm(trajs: Sequence[np.ndarray], k: int = 200, lag: int = 10, bayesian: bool = True,
              n_samples: int = 100, max_iter: int = 75, seed: int = 1, fit_stride: int = 1,
              nits: int = 5, backend: str = "auto", verbose: bool = True) -> MSMResult:
    trajs = [np.asarray(t, dtype=np.float64).reshape(len(t), -1) for t in trajs]
    if backend in ("auto", "deeptime"):
        try:
            import deeptime  # noqa: F401
            backend = "deeptime"
        except ImportError:
            if backend == "deeptime":
                raise
            backend = "pyemma"

    if backend == "pyemma":
        import pyemma
        cl = pyemma.coordinates.cluster_kmeans(trajs, k=k, max_iter=max_iter, fixed_seed=seed,
                                               stride=fit_stride)
        dtrajs = [np.asarray(d) for d in cl.dtrajs]
        m = (pyemma.msm.bayesian_markov_model(dtrajs, lag=lag, nsamples=n_samples) if bayesian
             else pyemma.msm.estimate_markov_model(dtrajs, lag=lag))
        weights = [np.asarray(w) for w in m.trajectory_weights()]
        res = MSMResult(dtrajs, np.asarray(cl.clustercenters), weights, lag, k, bayesian,
                        np.asarray(m.transition_matrix), np.asarray(m.stationary_distribution),
                        np.asarray(m.active_set), float(m.active_state_fraction),
                        float(m.active_count_fraction), np.asarray(m.timescales(nits)), "pyemma", m)
    else:
        from deeptime.markov import TransitionCountEstimator
        from deeptime.markov.msm import BayesianMSM, MaximumLikelihoodMSM
        dtrajs, centers = _kmeans_deeptime(trajs, k, max_iter, seed, fit_stride)
        mode = "effective" if bayesian else "sliding"
        counts = TransitionCountEstimator(lag, mode).fit_fetch(dtrajs)
        sub = counts.submodel_largest()
        if bayesian:
            post = BayesianMSM(n_samples=n_samples).fit_fetch(sub)
            model = post.prior
        else:
            model = MaximumLikelihoodMSM(reversible=True).fit_fetch(sub)
            post = model
        weights = [np.asarray(w) for w in model.compute_trajectory_weights(dtrajs)]
        asf, acf = _active_fractions(dtrajs, counts, sub, k)
        nts = min(nits, model.n_states - 1)
        res = MSMResult(dtrajs, np.asarray(centers), weights, lag, k, bayesian,
                        np.asarray(model.transition_matrix), np.asarray(model.stationary_distribution),
                        np.asarray(sub.state_symbols), asf, acf,
                        np.asarray(model.timescales(nts)) if nts > 0 else np.array([]), "deeptime", post)
    if verbose:
        print(f"MSM ({res.backend}, {'Bayesian' if bayesian else 'ML'}): k={k}, lag={lag} | "
              f"active_state_fraction = {res.active_state_fraction:.3f}, "
              f"active_count_fraction = {res.active_count_fraction:.3f}")
        if len(res.timescales):
            print("  slowest implied timescales (stored frames): "
                  + ", ".join(f"{t:.1f}" for t in res.timescales[:5]))
        zero = int((res.flat_weights() == 0).sum())
        if zero:
            print(f"  note: {zero} frame(s) lie outside the largest connected set -> weight 0")
    return res


def implied_timescales(dtrajs, lags: Sequence[int], nits: int = 4, backend: str = "deeptime"):
    """ML reversible MSM timescales for each lag -> (n_lags, nits) array (NaN-padded)."""
    out = np.full((len(lags), nits), np.nan)
    if backend == "pyemma":
        import pyemma
        its = pyemma.msm.its(dtrajs, lags=list(lags), nits=nits)
        ts = np.asarray(its.timescales)
        out[:, :ts.shape[1]] = ts[:, :nits]
        return out
    from deeptime.markov import TransitionCountEstimator
    from deeptime.markov.msm import MaximumLikelihoodMSM
    for i, lag in enumerate(lags):
        try:
            sub = TransitionCountEstimator(int(lag), "sliding").fit_fetch(dtrajs).submodel_largest()
            m = MaximumLikelihoodMSM(reversible=True).fit_fetch(sub)
            n = min(nits, m.n_states - 1)
            if n > 0:
                out[i, :n] = m.timescales(n)
        except Exception as e:  # lag too long for some trajectories, etc.
            print(f"  ITS at lag {lag} failed: {e}")
    return out
