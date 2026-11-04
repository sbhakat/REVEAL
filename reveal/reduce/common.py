"""Shared helpers for the dimensionality-reduction methods."""
from __future__ import annotations

import csv
import json
import os
import pickle
from typing import List, Optional

import numpy as np

from ..io import Dataset
from .. import plotting


def column_stats(X: np.ndarray, chunk: int = 100_000):
    """Mean / std per column in float64 without copying X to float64."""
    n = X.shape[0]
    s = np.zeros(X.shape[1])
    ss = np.zeros(X.shape[1])
    for i in range(0, n, chunk):
        b = X[i:i + chunk].astype(np.float64)
        s += b.sum(0)
        ss += (b * b).sum(0)
    mean = s / n
    var = np.maximum(ss / n - mean ** 2, 0.0)
    return mean, np.sqrt(var)


def covariance(X: np.ndarray, mean: np.ndarray, scale: Optional[np.ndarray] = None,
               chunk: int = 50_000, ddof: int = 1):
    """Chunked covariance of (X - mean) * scale, float64 accumulation."""
    p = X.shape[1]
    C = np.zeros((p, p))
    for i in range(0, X.shape[0], chunk):
        b = X[i:i + chunk].astype(np.float64) - mean
        if scale is not None:
            b *= scale
        C += b.T @ b
    return C / (X.shape[0] - ddof)


def lagged_pairs(lengths: List[int], lag: int):
    """Global (t, t+lag) index pairs that stay inside one trajectory."""
    t0, t1 = [], []
    off = 0
    for n in lengths:
        if n > lag:
            idx = np.arange(off, off + n - lag)
            t0.append(idx)
            t1.append(idx + lag)
        off += n
    if not t0:
        raise ValueError(f"lag {lag} is longer than every trajectory")
    return np.concatenate(t0), np.concatenate(t1)


def comp_labels(method: str, n: int) -> List[str]:
    tag = {"fpca": "FPC", "sfa": "SF", "tae": "TAE"}.get(method, method.upper())
    return [f"{tag}{i + 1}" for i in range(n)]


def write_outputs(out_dir: str, method: str, src: Dataset, Y: np.ndarray,
                  W: Optional[np.ndarray], info: dict, top_k: int = 20,
                  weight_label: str = "weight", signed: bool = True, legacy_extra: Optional[dict] = None):
    """Write the standard REVEAL reduction outputs.

    projection.npz           REVEAL dataset (per-trajectory latent coordinates) -> input for msm / pwcs
    <method>_transformed.npy flat (n_frames, n_components) array (legacy layout)
    <method>_transformed.pkl {"data", "traj_lengths", ...}  (legacy layout)
    <method>_weights.npy     (n_components, n_features) feature weights
    <method>_weights.pkl     {"W", "feature_names", ...}
    <method>_top_features.csv / .png
    <method>_latent.png      1D free-energy profile or 2D landscape of the projection
    info.json                run parameters and diagnostics
    """
    os.makedirs(out_dir, exist_ok=True)
    Y = np.asarray(Y, dtype=np.float32)
    if Y.ndim == 1:
        Y = Y[:, None]
    labels = comp_labels(method, Y.shape[1])
    meta = {k: v for k, v in src.meta.items() if k in ("traj_files", "top_files", "frame_start", "frame_stride")}
    meta.update(kind="projection", method=method, source=src.meta.get("source"), **{
        k: v for k, v in info.items() if isinstance(v, (int, float, str, bool, list, type(None)))})
    proj = Dataset(src.split(Y), labels, meta)
    proj.save(os.path.join(out_dir, "projection.npz"))

    np.save(os.path.join(out_dir, f"{method}_transformed.npy"), Y)
    with open(os.path.join(out_dir, f"{method}_transformed.pkl"), "wb") as f:
        pickle.dump({"data": Y, "traj_lengths": src.lengths, "n_components": Y.shape[1],
                     "stride": src.frame_stride, **(legacy_extra or {})}, f)

    names = np.array(src.names())
    if W is not None:
        W = np.atleast_2d(np.asarray(W))
        np.save(os.path.join(out_dir, f"{method}_weights.npy"), W)
        with open(os.path.join(out_dir, f"{method}_weights.pkl"), "wb") as f:
            pickle.dump({"W": W, "feature_names": names, "n_components": W.shape[0],
                         "stride": src.frame_stride, **(legacy_extra or {})}, f)
        with open(os.path.join(out_dir, f"{method}_top_features.csv"), "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["component", "rank", "feature", weight_label, "abs"])
            for c in range(W.shape[0]):
                order = np.argsort(np.abs(W[c]))[::-1][:max(top_k, 50)]
                for r, j in enumerate(order, 1):
                    wr.writerow([labels[c], r, names[j], f"{W[c, j]:.6g}", f"{abs(W[c, j]):.6g}"])
        plotting.plot_top_features(W, names, os.path.join(out_dir, f"{method}_top_features.png"),
                                   top_k=top_k, comp_labels=labels, ylabel=weight_label, signed=signed)
        print(f"\nTop features ({weight_label}):")
        for c in range(min(W.shape[0], 4)):
            order = np.argsort(np.abs(W[c]))[::-1][:min(top_k, 10)]
            print(f"  {labels[c]}: " + ", ".join(f"{names[j]} ({W[c, j]:+.3g})" for j in order[:5]))

    plotting.plot_latent(Y[:, :2], os.path.join(out_dir, f"{method}_latent.png"), labels[:2],
                         title=f"{method.upper()} projection (unweighted)")
    with open(os.path.join(out_dir, "info.json"), "w") as f:
        json.dump({"method": method, "n_frames": src.n_frames, "n_traj": src.n_traj,
                   "n_features": src.n_dim, "frame_stride": src.frame_stride,
                   **{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in info.items()}},
                  f, indent=2, default=str)
    print(f"\nwrote {os.path.join(out_dir, 'projection.npz')} "
          f"({src.n_traj} trajs, {src.n_frames} frames, {Y.shape[1]} dims) + weights/plots")
    return proj
