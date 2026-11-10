"""
TAE = the time-lagged CV-autoencoder from mdml's `timelaggedcv` package.

This is the reference TAE training procedure (see tests/legacy_tae_gvec.py
for the frozen regression-test copy it reproduces bit-for-bit): same
timelaggedcv config.yaml layout, same CV_autoencoder construction, same
CVEstimator.fit_routine (pretrain -> sparsity rounds until each CV keeps
target_n_feat features -> unconstrained retrain), same predictions
(estimate_cv_from_dataset with mask noise switched off) and the same kernel
weights (net.get_kernel). It also writes CV_file{n}.yaml and the
correctly labelled Feature_weights_{n}_labeled.png.

The network is:
  one CV_NN per CV:   ReLU feature mask -> Linear(p, cv_width) -> Tanh ->
                      (cv_depth-1) x [Linear(cv_width, cv_width) -> Tanh] -> Linear(cv_width, 1)
  decoder:            Linear(n_cv, width) -> ELU -> n_res_blocks x [Linear(width, width) -> ELU]
                      -> Linear(width, p)
  loss:               L2 reconstruction of x_{t+tau} + VAMP-2 + CV range penalty + mask sparsity

Uses the copy of mdml's timelaggedcv bundled in reveal/_vendor (unchanged except
for relative self-imports; see reveal/_vendor/README.md). --tlcv-path DIR imports
an external timelaggedcv from DIR instead.

timelaggedcv works in the current directory (config.yaml, model files,
lightning_logs/), so training runs inside the output directory.
"""
from __future__ import annotations

import os
import pickle
import random
import sys

import numpy as np

# reference training settings (= defaults of `reveal reduce tae`)
TAE_GVEC_DEFAULTS = dict(
    target_n_feat=30, max_steps=1500, max_rounds=2, patience=2, step_lam=0.2, batch_size=1024,
    valid_ratio=0.2, lr=1e-3, weight_decay=0.1,
    cv_width=50, cv_depth=4, cv_acti="Tanh", width=200, residual=False, n_res_blocks=6, res_skip=1,
    sigma_bin=5, n_bins=100, sigma_feat_mask=1.0, vamp_loss="VAMP2", vamp_loss_weight=1.0,
    score_method="L2",
)


def import_timelaggedcv(tlcv_path=None):
    """The bundled copy (reveal._vendor.timelaggedcv), or an external one from --tlcv-path."""
    try:
        import torch  # noqa: F401
        import pytorch_lightning  # noqa: F401
    except ImportError as e:
        raise ImportError(f"TAE needs torch and pytorch_lightning (installed with REVEAL): {e}") from e
    if tlcv_path:
        import importlib
        sys.path.insert(0, os.path.abspath(tlcv_path))
        try:
            return importlib.import_module("timelaggedcv")
        except ImportError as e:
            raise ImportError(f"--tlcv-path {tlcv_path}: cannot import timelaggedcv from there ({e})") from e
    from .._vendor import timelaggedcv
    return timelaggedcv


def run_tlcv(trajs, feature_names, out_dir, n_components=2, lag=1, seed=0, tlcv_path=None,
             export_cv_files=True, top_k=30, **overrides):
    """Train the timelaggedcv TAE using the reference procedure (see module docstring).

    Returns (Z, W, info): latent CVs (n_frames, n_components), kernel weights
    (n_components, n_features), run information.
    """
    tlcv = import_timelaggedcv(tlcv_path)
    import importlib
    import torch
    from torch.utils.data import DataLoader, random_split
    CV_autoencoder, CVEstimator = tlcv.CV_autoencoder, tlcv.CVEstimator
    _ds = importlib.import_module(tlcv.__name__ + ".dataset")
    _util = importlib.import_module(tlcv.__name__ + ".util")
    CV_timelagged, CVfind = _ds.CV_timelagged, _ds.CVfind
    dump_yaml, load_yaml = _util.dump_yaml, _util.load_yaml

    unknown = set(overrides) - set(TAE_GVEC_DEFAULTS)
    if unknown:
        raise ValueError(f"unknown TAE options: {sorted(unknown)}")
    o = dict(TAE_GVEC_DEFAULTS, **{k: v for k, v in overrides.items() if v is not None})

    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    cache = os.path.join(out_dir, "features_cache.pkl")
    with open(cache, "wb") as f:   # same on-disk format as the reference implementation's feature cache
        pickle.dump([np.ascontiguousarray(t, dtype=np.float32) for t in trajs], f, protocol=4)
    n_feat = trajs[0].shape[1]
    feature_names = np.asarray(feature_names)

    if seed is not None:           # the reference implementation was unseeded; REVEAL seeds for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    cwd = os.getcwd()
    try:
        os.chdir(out_dir)
        # ---- build the timelaggedcv config -----------------------------------
        feat_list = [{"feature": "gvector", "name": str(n), "transform": "raw", "residues": []}
                     for n in feature_names]
        idx = np.arange(n_feat)
        dump_yaml({
            "CV_feature_dicts": [{"dihedrals": {"features": feat_list, "dihedral_list": []},
                                  "cv_feature_index": idx} for _ in range(n_components)],
            "decoder_dict": {"features": [], "n_atoms": 0, "cv_feature_index": idx, "atoms_decoder": []},
            "all_input_feature_dict": {},
            "weights_flag": False,
            "time_mode": "c_t",
            "max_time": 100000,
            "pickle_descriptors": [],
            "pickle_features": [cache],
            "skip": 1,
        }, "config.yaml")

        # ---- train -------------------------------------------------------------
        print(f"\n=== Training TAE (timelaggedcv {getattr(tlcv, '__file__', '')}): "
              f"tau={lag}, latent_dim={n_components} ===")
        dataset = CV_timelagged(lag)
        print(f"Number of data pairs: {len(dataset)}")
        dataset_weight = CVfind(dataset)
        net = CV_autoencoder(
            input_dims=dataset.input_dims, output_dim=dataset.output_dim,
            cv_width=o["cv_width"], cv_depth=o["cv_depth"], cv_acti=o["cv_acti"], width=o["width"],
            residual=o["residual"], n_res_blocks=o["n_res_blocks"], res_skip=o["res_skip"],
            acti=torch.nn.ELU(), decoder=True, score_method=o["score_method"], special_cv=False,
            special_rewiring=False, sigma_bin=o["sigma_bin"], n_bins=o["n_bins"],
            feat_indexes_input=dataset.feat_indexes_input, weights_feat=None,
            vamp_loss=o["vamp_loss"], sigma_feat_mask=o["sigma_feat_mask"])
        net.set_mean_std(dataset.means, dataset.stds)
        accelerator = "gpu" if torch.cuda.is_available() else "cpu"
        est = CVEstimator(net, accelerator, 1, lr=o["lr"], lr_decoder=None,
                          weight_decay=o["weight_decay"], vamp_loss=o["vamp_loss"],
                          vamp_loss_weight=o["vamp_loss_weight"],
                          precision_mode="32-true")   # VAMP loss needs full precision
        n_val = int(len(dataset) * o["valid_ratio"])
        train_data, val_data = random_split(dataset, [len(dataset) - n_val, n_val])
        loader_train = DataLoader(train_data, batch_size=o["batch_size"], shuffle=True)
        loader_val = DataLoader(val_data, batch_size=o["batch_size"] * 2, shuffle=False)
        est.fit_routine(target_n_feat=[o["target_n_feat"]] * n_components,
                        data_loader=loader_train, validation_loader=loader_val,
                        patience=o["patience"], step_lam=o["step_lam"],
                        max_steps=o["max_steps"], max_rounds=o["max_rounds"])
        net.switch_sigma_feat_mask(0)   # disable mask noise for clean final predictions
        Z = est.estimate_cv_from_dataset(dataset_weight, batchsize=10000)
        print(f"Predicted CV shape: {Z.shape}")
        W = np.stack([net.get_kernel(c) for c in range(n_components)])
        n_active = [int(net.get_active(c)) for c in range(n_components)]
        print(f"Active features per CV (target {o['target_n_feat']}): {n_active}")
        est.save("model_final.pt")

        # ---- export native CV_file{n}.yaml + labelled feature-weight plots -----
        if export_cv_files:
            lo, hi = Z.min(0), Z.max(0)
            rel = (hi - lo) * 0.1
            net.get_opt_feat("./", dataset, hi + rel, lo - rel)
            from .. import plotting
            for n in range(n_components):
                feats = load_yaml(f"CV_file{n}.yaml")["weights_feat"]
                names = [f_["name"] for f_ in feats]
                params = np.array([f_["parameter"] for f_ in feats])
                if len(params):
                    plotting.plot_top_features(params[None, :], names, f"Feature_weights_{n}_labeled.png",
                                               top_k=top_k, comp_labels=[f"CV{n}"], ylabel="weight")

        info = {"lag": lag, "n_components": n_components, "n_active": n_active, "n_pairs": len(dataset),
                "seed": seed, "timelaggedcv": getattr(tlcv, "__file__", None), **o}
        return np.asarray(Z, dtype=np.float64), W, info
    finally:
        os.chdir(cwd)
