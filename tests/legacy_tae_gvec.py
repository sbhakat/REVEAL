#!/usr/bin/env python
"""
Time-lagged autoencoder (TAE) on the RNA G-vector features, using the local
copy of `timelaggedcv` (vendored from
/Users/shrayvats/Documents/AlloTec/andreas/timelaggedcv) -- the same VAMP2
CV-autoencoder machinery used in Distances/ML-Distances/autoencoder_distances.py,
here applied to the precomputed G-vector dataset described in
Traning-data-RNA/README.md instead of re-derived contact distances.

For each (lag time, latent dimension) combination requested, this:
    1. Trains a sparse CV_autoencoder (VAMP2 objective + reconstruction +
       feature-mask sparsity) to find `n_cv` slow collective variables.
    2. Predicts the latent CV trajectory for every frame.
    3. Extracts the per-feature gate weight (net.get_kernel) for each CV,
       i.e. which G-vector features drive that slow coordinate.

Outputs per combination, written to --out-dir/tau{T}_dim{D}/:
    tae_latent.npy / .pkl    latent CV trajectory, shape (n_frames_total, n_cv)
    tae_weights.npy / .pkl   feature weights, shape (n_cv, n_features)
    tae_top_features.png     bar plot of top-k |weight| features per CV
    tae_latent_plot.png      1D histogram or 2D scatter of the latent space
    model_final.pt           trained estimator checkpoint

Usage:
    python tae_gvec.py --taus 1 10 50 --latent-dims 1 2 --stride 20
"""
import argparse
import glob
import json
import os
import pickle

import numpy as np


def load_and_cache_gvectors(data_dir, cache_dir, stride):
    """Loads+strides the G-vector trajectories once and caches them as a single
    pickle (list of per-trajectory arrays), which is the on-disk format
    timelaggedcv.dataset.get_features_pickle expects for `pickle_features`.
    Reused across all (tau, n_cv) runs so the (potentially large) cast to
    float32 arrays + pickling only happens once per --stride value.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.abspath(os.path.join(cache_dir, f"gvec_data_stride{stride}.pkl"))
    feature_names = np.load(
        os.path.join(data_dir, "rg_vectors_g_features.npy"), allow_pickle=True
    )
    if os.path.exists(cache_path):
        print(f"Using cached strided feature pickle: {cache_path}")
        return cache_path, feature_names

    part_files = sorted(glob.glob(os.path.join(data_dir, "rg_vectors_g_part*.npz")))
    if not part_files:
        raise FileNotFoundError(f"No rg_vectors_g_part*.npz found under {data_dir}")

    traj_list = []
    for part_path in part_files:
        part = np.load(part_path)
        traj_keys = sorted(k for k in part.files if k.startswith("traj_"))
        for key in traj_keys:
            traj_list.append(np.ascontiguousarray(part[key][::stride]))

    n_feat = traj_list[0].shape[1]
    assert feature_names.shape[0] == n_feat

    print(f"Writing strided feature cache ({len(traj_list)} trajs, "
          f"{sum(t.shape[0] for t in traj_list)} frames total) to {cache_path}")
    with open(cache_path, "wb") as f:
        pickle.dump(traj_list, f, protocol=4)
    return cache_path, feature_names


def build_config(n_feat, feature_names, n_cv, features_path, config_path="config.yaml"):
    """Builds a timelaggedcv config.yaml for the G-vector features.

    Every CV branch shares the same cv_feature_index (the full G-vector set),
    exactly as SFA's/FPCA's components both live over the same shared input
    space -- each CV_NN branch independently learns its own slow coordinate.

    The "dihedrals" key name is a quirk of
    timelaggedcv/dataset/utils.py:set_feature_mean_std, which hardcodes that
    key regardless of actual feature type (see
    Distances/ML-Distances/autoencoder_distances.py's build_config for the
    same workaround used there for contact distances). The per-feature dicts
    here just carry the G-vector feature name for interpretability; no atom
    indices are needed since CV_decoder_features is a plain MLP from the
    bottleneck to the full feature vector, not a physical/atom-based decoder.
    """
    from timelaggedcv.util import dump_yaml

    feat_list = [
        {"feature": "gvector", "name": str(name), "transform": "raw", "residues": []}
        for name in feature_names
    ]
    cv_feature_index = np.arange(n_feat)

    CV_feature_dicts = [
        {
            "dihedrals": {"features": feat_list, "dihedral_list": []},
            "cv_feature_index": cv_feature_index,
        }
        for _ in range(n_cv)
    ]

    decoder_dict = {
        "features": [],
        "n_atoms": 0,
        "cv_feature_index": cv_feature_index,
        "atoms_decoder": [],
    }

    config = {
        "CV_feature_dicts": CV_feature_dicts,
        "decoder_dict": decoder_dict,
        "all_input_feature_dict": {},
        "weights_flag": False,
        "time_mode": "c_t",
        "max_time": 100000,
        "pickle_descriptors": [],
        "pickle_features": [features_path],
        "skip": 1,
    }
    dump_yaml(config, config_path)


def run_one(tau, n_cv, features_path, feature_names, n_feat, args, base_out_dir):
    import torch
    from torch.utils.data import DataLoader, random_split
    from timelaggedcv import CV_autoencoder, CVEstimator
    from timelaggedcv.dataset import CV_timelagged, CVfind

    work_dir = os.path.abspath(os.path.join(base_out_dir, f"tau{tau}_dim{n_cv}"))
    os.makedirs(work_dir, exist_ok=True)
    cwd = os.getcwd()
    try:
        os.chdir(work_dir)
        print(f"\n=== Training TAE: tau={tau}, latent_dim={n_cv} (work_dir={work_dir}) ===")

        build_config(n_feat, feature_names, n_cv, features_path)

        dataset = CV_timelagged(tau)
        print(f"Number of data pairs: {len(dataset)}")
        dataset_weight = CVfind(dataset)

        net = CV_autoencoder(
            input_dims=dataset.input_dims,
            output_dim=dataset.output_dim,
            cv_width=50,
            cv_depth=4,
            cv_acti="Tanh",
            width=200,
            residual=False,
            n_res_blocks=6,
            res_skip=1,
            acti=torch.nn.ELU(),
            decoder=True,
            score_method="L2",
            special_cv=False,
            special_rewiring=False,
            sigma_bin=5,
            n_bins=100,
            feat_indexes_input=dataset.feat_indexes_input,
            weights_feat=None,
            vamp_loss="VAMP2",
            sigma_feat_mask=1.0,
        )
        net.set_mean_std(dataset.means, dataset.stds)

        accelerator = "gpu" if torch.cuda.is_available() else "cpu"
        estimator = CVEstimator(
            net, accelerator, 1,
            lr=args.lr, lr_decoder=None, weight_decay=args.weight_decay,
            vamp_loss="VAMP2", vamp_loss_weight=1.0,
            precision_mode="32-true",  # VAMP loss needs full precision
        )

        n_val = int(len(dataset) * args.valid_ratio)
        train_data, val_data = random_split(dataset, [len(dataset) - n_val, n_val])
        loader_train = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
        loader_val = DataLoader(val_data, batch_size=args.batch_size * 2, shuffle=False)

        estimator.fit_routine(
            target_n_feat=[args.target_n_feat] * n_cv,
            data_loader=loader_train,
            validation_loader=loader_val,
            patience=args.patience,
            step_lam=args.step_lam,
            max_steps=args.max_steps,
            max_rounds=args.max_rounds,
        )

        net.switch_sigma_feat_mask(0)  # disable mask noise for clean final predictions

        bott_all = estimator.estimate_cv_from_dataset(dataset_weight, batchsize=10000)
        print(f"Predicted CV shape: {bott_all.shape}")

        kernels = [net.get_kernel(cv_n) for cv_n in range(n_cv)]
        W = np.stack(kernels)  # (n_cv, n_feat)
        n_active = [net.get_active(cv_n) for cv_n in range(n_cv)]
        print(f"Active features per CV (target {args.target_n_feat}): {n_active}")

        np.save("tae_latent.npy", bott_all)
        with open("tae_latent.pkl", "wb") as f:
            pickle.dump({"data": bott_all, "tau": tau, "n_cv": n_cv,
                         "stride": args.stride}, f)

        np.save("tae_weights.npy", W)
        with open("tae_weights.pkl", "wb") as f:
            pickle.dump({"W": W, "feature_names": feature_names, "tau": tau,
                         "n_cv": n_cv, "n_active": n_active,
                         "target_n_feat": args.target_n_feat}, f)

        estimator.save("model_final.pt")

        plot_results(W, bott_all, feature_names, n_cv, args.top_k, tau)

        return {
            "tau": tau, "n_cv": n_cv, "n_active": n_active,
            "n_pairs": len(dataset), "work_dir": work_dir,
        }
    finally:
        os.chdir(cwd)


def plot_results(W, bott_all, feature_names, n_cv, top_k, tau):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(n_cv, 1, figsize=(12, 5 * n_cv), squeeze=False)
    for cv_n in range(n_cv):
        w = W[cv_n]
        k = min(top_k, (np.abs(w) > 0).sum() or top_k)
        top_idx = np.argsort(np.abs(w))[-k:][::-1]
        ax = axes[cv_n, 0]
        ax.bar(range(len(top_idx)), w[top_idx])
        ax.set_xticks(range(len(top_idx)))
        ax.set_xticklabels(feature_names[top_idx], rotation=90, fontsize=7)
        ax.set_ylabel(f"CV{cv_n + 1} kernel weight")
        ax.set_title(f"Top features for CV{cv_n + 1} (tau={tau})")
    fig.tight_layout()
    fig.savefig("tae_top_features.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    if n_cv == 1:
        ax.hist(bott_all[:, 0], bins=100)
        ax.set_xlabel("CV1")
        ax.set_ylabel("count")
        ax.set_title(f"1D latent distribution (tau={tau})")
    else:
        sc = ax.scatter(bott_all[:, 0], bott_all[:, 1], s=2, alpha=0.3,
                         c=np.arange(len(bott_all)), cmap="viridis")
        ax.set_xlabel("CV1")
        ax.set_ylabel("CV2")
        ax.set_title(f"Latent space, first 2 dims (tau={tau})")
        fig.colorbar(sc, ax=ax, label="frame index (concatenated)")
    fig.tight_layout()
    fig.savefig("tae_latent_plot.png", dpi=300)
    plt.close(fig)
    print("Saved tae_top_features.png and tae_latent_plot.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="../Traning-data-RNA")
    parser.add_argument("--cache-dir", default="./data")
    parser.add_argument("--out-dir", default="./results")
    parser.add_argument("--stride", type=int, default=20,
                         help="subsample frames within each trajectory before caching")
    parser.add_argument("--taus", type=int, nargs="+", default=[1, 10, 50],
                         help="lag times (in frames, after striding) to scan")
    parser.add_argument("--latent-dims", type=int, nargs="+", default=[1, 2],
                         help="latent CV dimensions to compare (1D vs 2D, etc.)")
    parser.add_argument("--target-n-feat", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--max-rounds", type=int, default=2)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--step-lam", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--valid-ratio", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    features_path, feature_names = load_and_cache_gvectors(
        args.data_dir, args.cache_dir, args.stride
    )
    n_feat = feature_names.shape[0]

    summary = []
    for tau in args.taus:
        for n_cv in args.latent_dims:
            result = run_one(tau, n_cv, features_path, feature_names, n_feat,
                              args, args.out_dir)
            summary.append(result)
            with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
                json.dump(summary, f, indent=2)

    print("\n=== Summary ===")
    for r in summary:
        print(r)
    print("TAE analysis complete!")


if __name__ == "__main__":
    main()
