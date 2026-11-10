"""
Population-Weighted Consensus Score (PWCS).

Per-frame consensus metric that averages normalized log-likelihoods across
several feature-set / density-model combinations: a frame well supported
under many featurizations and model complexities scores high, a frame that
is an outlier under most of them scores low. (Also known as the RNAnneal
score; see tests/legacy_pwcs.py for the original reference implementation.)

For every feature set v (here: a 1D or 2D variable -- CV1, CV2, (CV1, CV2),
an FPCA/SFA/TAE projection, ...) and every GMM size d in d_range:
    1. fit GaussianMixture(n_components=d) on the feature set
    2. score every frame: L_i = log p(x_i)
    3. normalize: l_i = (L_i - min_i L_i) / std_i(L_i)
PWCS s_i = mean over all (v, d) of l_i.

MSM weighting (optional)
    With per-frame equilibrium weights w_i (e.g. from an MSM), every GMM is
    fit on a weighted resample of the frames (sklearn's GaussianMixture has
    no sample_weight), so the density approximates the MSM-reweighted
    equilibrium distribution, and sigma_L is the weighted std. Every frame is
    still scored. Without weights, every frame counts equally (per-frame mode).
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from sklearn.mixture import GaussianMixture


def _weighted_std(L, weights):
    w = weights / weights.sum()
    mean_w = np.sum(w * L)
    return np.sqrt(np.sum(w * (L - mean_w) ** 2))


def _weighted_resample(X, weights, size, random_state):
    rs = np.random.RandomState(random_state)
    p = weights / weights.sum()
    idx = rs.choice(X.shape[0], size=size, replace=True, p=p)
    return X[idx]


def normalized_loglik(X, n_components, weights=None, resample_factor=5, random_state=0,
                      max_fit_frames=None, covariance_type="full", n_init=1):
    """Fit GMM(n_components) and return l_i = (L_i - min L) / std(L) for every frame."""
    if weights is not None:
        size = resample_factor * X.shape[0]
        if max_fit_frames:
            size = min(size, int(max_fit_frames))
        fit_X = _weighted_resample(X, weights, size=size, random_state=random_state)
    elif max_fit_frames and X.shape[0] > max_fit_frames:
        rs = np.random.RandomState(random_state)
        fit_X = X[rs.choice(X.shape[0], size=int(max_fit_frames), replace=False)]
    else:
        fit_X = X
    gmm = GaussianMixture(n_components=n_components, random_state=random_state,
                          covariance_type=covariance_type, n_init=n_init)
    gmm.fit(fit_X)
    L = gmm.score_samples(X)
    sigma = _weighted_std(L, weights) if weights is not None else L.std()
    if sigma == 0:
        raise ValueError(f"log-likelihoods have zero variance for n_components={n_components} "
                         "(degenerate fit)")
    return (L - L.min()) / sigma


def pwcs_score(feature_sets: Dict[str, np.ndarray], d_range=range(2, 9), weights=None,
               resample_factor=5, random_state=0, return_components=False, max_fit_frames=None,
               covariance_type="full", n_init=1, verbose=False):
    """PWCS for every frame. See the module docstring for the algorithm."""
    names = list(feature_sets)
    n_frames = feature_sets[names[0]].shape[0]
    for name in names:
        if feature_sets[name].shape[0] != n_frames:
            raise ValueError(f"feature set '{name}' has {feature_sets[name].shape[0]} frames, "
                             f"expected {n_frames}")
    if weights is not None:
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape[0] != n_frames:
            raise ValueError(f"weights has {weights.shape[0]} entries, expected {n_frames}")
        if weights.sum() <= 0:
            raise ValueError("weights sum to zero")

    components = {}
    acc = np.zeros(n_frames)
    n_models = 0
    for name in names:
        X = np.asarray(feature_sets[name], dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        for d in d_range:
            l = normalized_loglik(X, d, weights, resample_factor, random_state, max_fit_frames,
                                  covariance_type, n_init)
            if return_components:
                components[(name, d)] = {"l": l, "l_sorted": np.sort(l)}
            acc += l
            n_models += 1
            if verbose:
                print(f"  {name:>24s}  d={d}:  mean l = {l.mean():.3f}")
    S = acc / n_models
    return (S, components) if return_components else S
