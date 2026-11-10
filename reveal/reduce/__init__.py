"""Dimensionality-reduction toolbox: FPCA, SFA, TAE."""
from __future__ import annotations

import os
import pickle

import numpy as np

from ..io import Dataset
from .. import plotting
from .common import write_outputs

METHODS = ("fpca", "sfa", "tae")


def run_fpca(ds: Dataset, out_dir: str, n_components=2, backend="skfda", top_k=20):
    from .linear import fpca_native, fpca_skfda
    X = ds.concat(np.float32)
    print(f"FPCA ({backend}): {X.shape[0]} frames x {X.shape[1]} features -> {n_components} components")
    fn = fpca_skfda if backend == "skfda" else fpca_native
    Y, W, mean, extra = fn(X, n_components)
    evr = np.asarray(extra["explained_variance_ratio"])
    print("  explained variance ratio: " + ", ".join(f"{v:.4f}" for v in evr))
    info = {"n_components": n_components, "backend": backend,
            "explained_variance_ratio": evr.tolist(),
            "explained_variance": np.asarray(extra["explained_variance"]).tolist()}
    proj = write_outputs(out_dir, "fpca", ds, Y, W, info, top_k=top_k, weight_label="FPC weight",
                         legacy_extra={"mean": mean, "explained_variance_ratio": evr})
    return proj


def run_sfa(ds: Dataset, out_dir: str, n_components=2, tau=1, backend="sksfa",
            cross_boundaries=False, top_k=20):
    from .linear import sfa_native, sfa_sksfa
    print(f"SFA ({backend}): {ds.n_frames} frames x {ds.n_dim} features, tau={tau} "
          f"-> {n_components} slow features")
    if backend == "sksfa":
        Y, W, b, extra = sfa_sksfa(ds.trajs, n_components, tau, cross_boundaries=cross_boundaries)
        print(f"  sksfa route: {extra['sksfa_route']}")
    else:
        Y, W, b, extra = sfa_native(ds.trajs, n_components, tau, cross_boundaries=cross_boundaries)
    print("  delta values (slowness, lower = slower): "
          + ", ".join(f"{v:.4g}" for v in extra["delta_values"]))
    info = {"n_components": n_components, "tau": tau, "backend": backend,
            "cross_boundaries": cross_boundaries,
            **{k: (np.asarray(v).tolist() if isinstance(v, np.ndarray) else v) for k, v in extra.items()}}
    return write_outputs(out_dir, "sfa", ds, Y, W, info, top_k=top_k, weight_label="SF weight",
                         legacy_extra={"b": b, "tau": tau})


def run_tae(ds: Dataset, out_dir: str, n_components=2, lag=1, top_k=20, seed=0, tlcv_path=None, **kw):
    """TAE with mdml's timelaggedcv (see reduce/tlcv.py for the training procedure)."""
    from .tlcv import run_tlcv
    os.makedirs(out_dir, exist_ok=True)
    Z, W, info = run_tlcv(ds.trajs, ds.names(), out_dir, n_components, lag, seed=seed,
                          tlcv_path=tlcv_path, top_k=max(top_k, 30), **kw)
    # legacy output names, alongside REVEAL's standard projection.npz
    np.save(os.path.join(out_dir, "tae_latent.npy"), Z)
    with open(os.path.join(out_dir, "tae_latent.pkl"), "wb") as f:
        pickle.dump({"data": Z, "tau": lag, "n_cv": n_components, "stride": ds.frame_stride}, f)
    return write_outputs(out_dir, "tae", ds, Z, W, info, top_k=top_k, weight_label="kernel weight",
                         legacy_extra={"tau": lag, "n_cv": n_components, "n_active": info["n_active"],
                                       "target_n_feat": info["target_n_feat"]})
