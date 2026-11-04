"""Verbatim core of the original standalone pwcs.py (reference for tests)."""
import numpy as np
from sklearn.mixture import GaussianMixture


def _weighted_std(L, weights):
    w = weights / weights.sum()
    mean_w = np.sum(w * L)
    var_w = np.sum(w * (L - mean_w) ** 2)
    return np.sqrt(var_w)


def _weighted_resample(X, weights, size, random_state):
    rs = np.random.RandomState(random_state)
    p = weights / weights.sum()
    idx = rs.choice(X.shape[0], size=size, replace=True, p=p)
    return X[idx]


def _normalized_loglik(X, n_components, weights=None, resample_factor=5, random_state=0):
    if weights is not None:
        fit_X = _weighted_resample(X, weights, size=resample_factor * X.shape[0], random_state=random_state)
    else:
        fit_X = X
    gmm = GaussianMixture(n_components=n_components, random_state=random_state)
    gmm.fit(fit_X)
    L = gmm.score_samples(X)
    L_min = L.min()
    sigma_L = _weighted_std(L, weights) if weights is not None else L.std()
    return (L - L_min) / sigma_L


def pwcs_score(feature_sets, d_range=range(2, 9), weights=None, resample_factor=5, random_state=0):
    names = list(feature_sets)
    n_frames = feature_sets[names[0]].shape[0]
    L_list = [[] for _ in range(n_frames)]
    for name in names:
        X = feature_sets[name]
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        for d in d_range:
            l = _normalized_loglik(X, n_components=d, weights=weights,
                                   resample_factor=resample_factor, random_state=random_state)
            for i in range(n_frames):
                L_list[i].append(l[i])
    return np.array([np.mean(L_list[i]) for i in range(n_frames)])
