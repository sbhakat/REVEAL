"""
REVEAL (RNA Ensemble Validation and Exploration with AI-guided Landscapes) command-line interface.

    reveal featurize   G-/R-vector features from trajectories (+ one global or per-traj topology)
    reveal reduce      fpca | sfa | tae on any feature dataset
    reveal msm         MSM on 1D/2D (or nD) variables -> per-frame equilibrium weights
    reveal pwcs        Population-Weighted Consensus Score on 1D/2D variables (per frame or MSM-weighted)
    reveal extract     write the highest/lowest-PWCS frames as PDB files
    reveal lengths     frames per trajectory (from trajectory files, a COLVAR, or a dataset) for --lengths
    reveal info        summarize any REVEAL / legacy dataset

Run `reveal <command> -h` for options.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

from . import __version__
from .io import (Dataset, align_strides, expand_paths, find_topology, find_trajectories,
                 load_dataset, pair_topologies, parse_spec, read_manifest)

FMT = argparse.ArgumentDefaultsHelpFormatter


# ==========================================================================
# featurize
# ==========================================================================
def add_featurize(sub):
    p = sub.add_parser("featurize", formatter_class=FMT,
                       help="compute G-vector / R-vector features from trajectories",
                       description="Compute per-frame nucleobase-pair G-vectors and R-vectors. "
                                   "Give trajectories (--traj or --traj-dir) and either ONE global "
                                   "topology compatible with every trajectory, or one topology per "
                                   "trajectory (--top a.pdb b.pdb ... / --manifest).")
    src = p.add_argument_group("inputs")
    src.add_argument("--traj", nargs="+", help="trajectory files or globs (sorted)")
    src.add_argument("--traj-dir", help="folder of trajectories (non-recursive, sorted by name)")
    src.add_argument("--traj-ext", default="xtc", help="extension used with --traj-dir")
    src.add_argument("--top", nargs="+",
                     help="one global topology, or one per trajectory (matched by file stem, else by order)")
    src.add_argument("--top-mode", choices=["auto", "global", "paired"], default="auto")
    src.add_argument("--manifest", help="text file with 'trajectory topology' per line")
    opt = p.add_argument_group("featurization")
    opt.add_argument("--features", nargs="+", choices=["g", "r"], default=["g", "r"])
    opt.add_argument("--stride", type=int, default=1, help="read every Nth frame")
    opt.add_argument("--r-cutoff", type=float, default=15.0, help="G-vector cutoff r_c in Angstrom")
    opt.add_argument("--selection", default=None,
                     help="MDTraj selection restricting residues, e.g. 'resid 0 to 30'")
    opt.add_argument("--chunk", type=int, default=1000, help="frames per read chunk (memory control)")
    out = p.add_argument_group("output")
    out.add_argument("--out-prefix", default="reveal_features",
                     help="writes <prefix>_g.npz / <prefix>_r.npz (+ _features.npy name lists)")
    out.add_argument("--compress", action="store_true", help="np.savez_compressed")
    out.add_argument("--names-only", action="store_true",
                     help="print/save feature names from the topology and exit")
    p.set_defaults(func=cmd_featurize)


def _resolve_traj_tops(args):
    if args.manifest:
        return read_manifest(args.manifest)
    if args.traj:
        trajs = expand_paths(args.traj)
    elif args.traj_dir:
        trajs = find_trajectories(args.traj_dir, args.traj_ext)
    else:
        sys.exit("error: give --traj, --traj-dir or --manifest")
    if args.top:
        tops = expand_paths(args.top)
    elif args.traj_dir:
        tops = [find_topology(args.traj_dir)]
        print(f"auto-detected global topology: {tops[0]}")
    else:
        sys.exit("error: give --top (one global topology or one per trajectory)")
    return trajs, pair_topologies(trajs, tops, args.top_mode)


def cmd_featurize(args):
    from .featurize import feature_names, featurize, ring_atom_indices
    import mdtraj as md

    trajs, tops = _resolve_traj_tops(args)
    mode = "global" if len(set(tops)) == 1 else "per-trajectory"
    print(f"{len(trajs)} trajector{'y' if len(trajs) == 1 else 'ies'}, {mode} topology, "
          f"stride {args.stride}, r_c = {args.r_cutoff} A")
    for t, p in zip(trajs, tops):
        print(f"  {t}  <-  {p}")

    if args.names_only:
        _, labels = ring_atom_indices(md.load_topology(tops[0]), args.selection)
        for k in args.features:
            names = feature_names(labels, k)
            path = f"{args.out_prefix}_{k}_features.npy"
            np.save(path, np.array(names))
            print(f"{k.upper()}: {len(names)} features -> {path}")
        return

    res = featurize(trajs, tops, kinds=args.features, r_cutoff=args.r_cutoff, stride=args.stride,
                    chunk=args.chunk, selection=args.selection)
    for k, ds in res.items():
        path = ds.save(f"{args.out_prefix}_{k}.npz", compress=args.compress)
        np.save(f"{args.out_prefix}_{k}_features.npy", np.array(ds.feature_names))
        print(f"wrote {k.upper()}-vectors -> {path}  ({ds.n_traj} trajs, {ds.n_frames} frames, "
              f"{ds.n_dim} dims)")


# ==========================================================================
# reduce
# ==========================================================================
def _add_input_opts(p, multi=False):
    g = p.add_argument_group("input")
    g.add_argument("--input", "-i", required=True, action="append" if multi else "store",
                   help="dataset spec  [name=]path[:cols]  (REVEAL/legacy .npz, glob over parts, "
                        ".pkl, or .npy with --lengths-from); cols like 0,1 or 0-3 or names"
                        + ("; repeat for several inputs" if multi else ""))
    g.add_argument("--stride", type=int, default=1, help="additional stride applied on load")
    g.add_argument("--lengths-from", help="dataset whose trajectory lengths split a flat .npy/.pkl/text file")
    g.add_argument("--lengths", nargs="+",
                   help="OPTIONAL. Frames per trajectory for a flat .npy/.pkl/text input, in trajectory "
                        "order: numbers, or a file written by `reveal lengths --out`")
    g.add_argument("--feature-names", help=".npy/.txt with feature names (overrides the file's)")
    g.add_argument("--frame-stride", type=int,
                   help="declare the stride of LEGACY inputs relative to the raw trajectories")
    return g


def add_reduce(sub):
    p = sub.add_parser("reduce", help="dimensionality reduction: fpca | sfa | tae",
                       description="Dimensionality reduction on a feature dataset (e.g. G-vectors).")
    rs = p.add_subparsers(dest="method", required=True)

    def common(q, lagged=False):
        _add_input_opts(q)
        o = q.add_argument_group("output")
        o.add_argument("--out", "-o", default=None, help="output directory (default: reveal_<method>)")
        o.add_argument("--top-k", type=int, default=20, help="features shown per component")
        q.add_argument("--n-components", "-n", type=int, nargs="+", default=[2],
                       help="latent dimension(s); several values -> one subfolder each")
        if lagged:
            q.add_argument("--lag", "--tau", dest="lag", type=int, nargs="+", default=[1],
                           help="lag time(s) in stored frames (after all striding)")

    q = rs.add_parser("fpca", formatter_class=FMT, help="functional PCA")
    common(q)
    q.add_argument("--backend", choices=["skfda", "native"], default="skfda",
                   help="skfda (scikit-fda FPCA) or native (NumPy re-implementation, same result)")

    q = rs.add_parser("sfa", formatter_class=FMT, help="slow feature analysis")
    common(q, lagged=True)
    q.add_argument("--backend", choices=["sksfa", "native"], default="sksfa",
                   help="sksfa (sklearn-sfa) or native (NumPy re-implementation)")
    q.add_argument("--cross-boundaries", action="store_true",
                   help="difference the concatenated frames across trajectory boundaries "
                        "(legacy behaviour)")

    q = rs.add_parser("tae", formatter_class=FMT,
                      help="time-lagged autoencoder (mdml's timelaggedcv)",
                      description="Time-lagged CV-autoencoder from mdml's timelaggedcv package.")
    common(q, lagged=True)
    t = q.add_argument_group("training")
    t.add_argument("--target-n-feat", type=int, default=30, help="active features to keep per CV")
    t.add_argument("--max-steps", type=int, default=1500)
    t.add_argument("--max-rounds", type=int, default=2, help="max sparsity rounds")
    t.add_argument("--patience", type=int, default=2)
    t.add_argument("--step-lam", type=float, default=0.2, help="initial sparsity penalty step")
    t.add_argument("--batch-size", type=int, default=1024)
    t.add_argument("--valid-ratio", type=float, default=0.2)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--weight-decay", type=float, default=0.1)
    t.add_argument("--seed", type=int, default=0,
                   help="seed torch/numpy/random before training (for reproducibility)")
    a = q.add_argument_group("architecture")
    a.add_argument("--cv-width", type=int, default=50, help="width of each CV network")
    a.add_argument("--cv-depth", type=int, default=4, help="layers of each CV network")
    a.add_argument("--width", type=int, default=200, help="decoder width")
    a.add_argument("--n-res-blocks", type=int, default=6, help="decoder blocks")
    a.add_argument("--residual", action="store_true", help="residual decoder blocks (off by default)")
    a.add_argument("--sigma-bin", type=float, default=5, help="decoder input noise (in bins of 100)")
    a.add_argument("--sigma-feat-mask", type=float, default=1.0, help="noise on masked-out features")
    a.add_argument("--vamp-loss", default="VAMP2", choices=["VAMP2", "VAMP1", "VAMPE"])
    a.add_argument("--vamp-loss-weight", type=float, default=1.0)
    q.add_argument("--tlcv-path", help="optional: use an external timelaggedcv from this folder instead of "
                                       "the copy bundled with REVEAL")

    p.set_defaults(func=cmd_reduce)


def _resolve_lengths(val):
    """--lengths given as numbers, or as one file (text with integers, or .npy)."""
    if not val:
        return None
    if len(val) == 1 and os.path.isfile(str(val[0])):
        f = str(val[0])
        arr = np.load(f) if f.endswith(".npy") else np.loadtxt(f, comments="#", ndmin=1)
        return [int(x) for x in np.ravel(arr)]
    try:
        return [int(x) for x in val]
    except ValueError:
        raise ValueError(f"--lengths: expected integers or an existing file, got {val[:3]}")


def _load_from_args(args, spec, stride=True):
    ds = load_dataset(spec, lengths_from=args.lengths_from, feature_names=args.feature_names,
                      frame_stride=getattr(args, "frame_stride", None),
                      lengths=_resolve_lengths(getattr(args, "lengths", None)))
    return ds.strided(args.stride) if stride else ds


def cmd_reduce(args):
    from . import reduce as R

    ds = _load_from_args(args, args.input)
    print(f"input: {args.input}\n" + "\n".join("  " + l for l in ds.summary().splitlines()))
    method = args.method
    base = args.out or f"reveal_{method}"
    lags = getattr(args, "lag", [None])
    combos = [(lag, n) for lag in lags for n in args.n_components]
    summary = []
    for lag, n in combos:
        if len(combos) == 1:
            out = base
        else:
            out = os.path.join(base, (f"tau{lag}_dim{n}" if lag is not None else f"dim{n}"))
        print(f"\n=== {method.upper()}  n_components={n}" + (f"  lag={lag}" if lag else "") + f"  -> {out}")
        if method == "fpca":
            R.run_fpca(ds, out, n, backend=args.backend, top_k=args.top_k)
        elif method == "sfa":
            R.run_sfa(ds, out, n, tau=lag, backend=args.backend,
                      cross_boundaries=args.cross_boundaries, top_k=args.top_k)
        elif method == "tae":
            R.run_tae(ds, out, n, lag, top_k=args.top_k, seed=args.seed, tlcv_path=args.tlcv_path,
                      target_n_feat=args.target_n_feat, max_steps=args.max_steps,
                      max_rounds=args.max_rounds, patience=args.patience, step_lam=args.step_lam,
                      batch_size=args.batch_size, valid_ratio=args.valid_ratio, lr=args.lr,
                      weight_decay=args.weight_decay, cv_width=args.cv_width, cv_depth=args.cv_depth,
                      width=args.width, n_res_blocks=args.n_res_blocks, residual=args.residual,
                      sigma_bin=args.sigma_bin, sigma_feat_mask=args.sigma_feat_mask,
                      vamp_loss=args.vamp_loss, vamp_loss_weight=args.vamp_loss_weight)
        with open(os.path.join(out, "info.json")) as f:
            summary.append(dict(json.load(f), out_dir=out))
    if len(combos) > 1:
        with open(os.path.join(base, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nwrote {os.path.join(base, 'summary.json')}")


# ==========================================================================
# msm
# ==========================================================================
def _msm_opts(g, prefix=""):
    d = (lambda s: f"--{prefix}{s}")
    g.add_argument(d("k"), type=int, default=200, help="k-means microstates")
    g.add_argument(d("lag"), type=int, default=10, help="MSM lag in stored frames")
    g.add_argument(d("ml"), action="store_true", help="maximum-likelihood MSM instead of Bayesian")
    g.add_argument(d("n-samples"), type=int, default=100, help="Bayesian MSM samples")
    g.add_argument(d("fit-stride"), type=int, default=1, help="fit k-means on every Nth frame")
    g.add_argument(d("seed"), type=int, default=1)
    g.add_argument(d("backend"), choices=["auto", "deeptime", "pyemma"], default="auto")


def add_msm(sub):
    p = sub.add_parser("msm", formatter_class=FMT,
                       help="build an MSM on 1D/2D variables and export per-frame weights",
                       description="k-means + (Bayesian) MSM on the given variables. Writes "
                                   "msm_weights.npz (per-frame equilibrium weights, usable by "
                                   "`reveal pwcs --weights`), implied timescales and landscapes.")
    _add_input_opts(p, multi=True)
    g = p.add_argument_group("MSM")
    _msm_opts(g)
    g.add_argument("--its-lags", type=int, nargs="+", default=[1, 2, 5, 10, 20, 50])
    g.add_argument("--nits", type=int, default=4)
    p.add_argument("--out", "-o", default="reveal_msm")
    p.set_defaults(func=cmd_msm)


def _name_columns(d, s):
    if s.name and d.feature_names is None:
        d.feature_names = [s.name] if d.n_dim == 1 else [f"{s.name}{j + 1}" for j in range(d.n_dim)]
    return d


def _load_many(args, specs, extra=()):
    """Load several specs, apply --stride, add already-built datasets (label, Dataset) from
    `extra` (e.g. weights), then align everything to a common frame_stride."""
    parsed = [parse_spec(s) for s in specs]
    dss = [_name_columns(_load_from_args(args, s), s) for s in parsed]
    labels = [s.name or os.path.basename(s.path) for s in parsed]
    for lab, d in extra:
        dss.append(d)
        labels.append(lab)
    return align_strides(dss, labels), labels, parsed


def _load_weights(args, var_spec):
    """Per-frame weights matched to the variable `var_spec` (one weight per frame of that
    variable as stored in its file). Returns a Dataset already strided like the variable."""
    from .io import META_KEY, check_weights, read_weight_arrays
    path = args.weights
    if path.endswith(".npz"):
        keys = np.load(path, allow_pickle=False).files
        if META_KEY in keys:                     # written by `reveal msm` (carries its frame_stride)
            ds = load_dataset(path)
            check_weights(ds.concat(np.float64)[:, 0], path)
            print(f"  weights: {path} (REVEAL file, frame_stride {ds.frame_stride})")
            return ds
    arrs = read_weight_arrays(path)
    w = np.concatenate(arrs)
    check_weights(w, path)                       # validate the whole file, before any striding
    raw = _load_from_args(args, var_spec, stride=False)     # variable as stored in its file
    sizes = [a.size for a in arrs]
    if (len(arrs) > 1 and raw.n_traj == 1 and sum(sizes) == raw.n_frames
            and not args.lengths and not args.lengths_from):
        # flat variable without --lengths, weights saved per trajectory: the weights define the
        # trajectory boundaries, so use them for every flat input
        args.lengths = [str(n) for n in sizes]
        print(f"  trajectory lengths taken from the per-trajectory weights in {path}: "
              f"{len(sizes)} trajectories ({sizes[0]}{', ...' if len(sizes) > 1 else ''})")
        raw = _load_from_args(args, var_spec, stride=False)
    strided = raw.strided(args.stride)
    if w.size == raw.n_frames:
        if len(arrs) > 1 and [a.size for a in arrs] != raw.lengths:
            raise ValueError(f"weights in {path} have per-trajectory lengths {[a.size for a in arrs][:5]}... "
                             f"but '{var_spec.path}' has {raw.lengths[:5]}...")
        ds = Dataset(raw.split(w[:, None]), ["weight"], dict(raw.meta)).strided(args.stride)
        how = "one per frame of the variable file" + (f", strided by {args.stride}" if args.stride > 1 else "")
    elif w.size == strided.n_frames:
        ds = Dataset(strided.split(w[:, None]), ["weight"], dict(strided.meta))
        how = "one per frame after --stride"
    else:
        raise ValueError(
            f"{path} has {w.size} weights, but '{var_spec.path}' has {raw.n_frames} frames "
            f"({strided.n_frames} after --stride {args.stride}). Weights must have one value per frame "
            "of the variable they were computed for (check stride, trajectory order, and --weights-for).")
    print(f"  weights: {path} -> {w.size} values ({how}) for '{var_spec.name or var_spec.path}'")
    return ds


def _concat_columns(dss):
    trajs = [np.concatenate([d.trajs[i] for d in dss], axis=1) for i in range(dss[0].n_traj)]
    names = sum((d.names() for d in dss), [])
    return Dataset(trajs, names, dict(dss[0].meta))


def cmd_msm(args):
    from .msm import build_msm, implied_timescales
    from . import plotting

    dss, labels, _ = _load_many(args, args.input)
    ds = _concat_columns(dss)
    print(ds.summary())
    os.makedirs(args.out, exist_ok=True)
    res = build_msm(ds.trajs, k=args.k, lag=args.lag, bayesian=not args.ml, n_samples=args.n_samples,
                    fit_stride=args.fit_stride, seed=args.seed, backend=args.backend)
    wds = Dataset(res.weights, ["msm_weight"], dict(ds.meta, kind="weights", method="msm", k=args.k,
                                                     lag=args.lag, bayesian=not args.ml,
                                                     variables=ds.names()))
    wds.save(os.path.join(args.out, "msm_weights.npz"))
    res.save(os.path.join(args.out, "msm.pkl"))
    np.savez(os.path.join(args.out, "dtrajs.npz"),
             **{f"traj_{i:02d}": d for i, d in enumerate(res.dtrajs)})

    lags = [l for l in args.its_lags if l < max(ds.lengths)]
    its = implied_timescales(res.dtrajs, lags, args.nits,
                             backend="pyemma" if res.backend == "pyemma" else "deeptime")
    plotting.plot_its(lags, its, os.path.join(args.out, "implied_timescales.png"), lag_used=args.lag)
    np.savez(os.path.join(args.out, "implied_timescales.npz"), lags=np.array(lags), timescales=its)
    Y = ds.concat(np.float64)
    if ds.n_dim <= 2:
        plotting.plot_fes_compare(Y, res.flat_weights(), os.path.join(args.out, "landscape.png"),
                                  ds.names())
    else:
        print("  (>2 variables: landscape plot uses the first two)")
        plotting.plot_fes_compare(Y[:, :2], res.flat_weights(), os.path.join(args.out, "landscape.png"),
                                  ds.names()[:2])
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump({"variables": ds.names(), "k": args.k, "lag": args.lag, "bayesian": not args.ml,
                   "backend": res.backend, "active_state_fraction": res.active_state_fraction,
                   "active_count_fraction": res.active_count_fraction,
                   "timescales": res.timescales.tolist(), "frame_stride": ds.frame_stride,
                   "its_lags": lags, "its": its.tolist()}, f, indent=2)
    print(f"wrote {args.out}/msm_weights.npz, msm.pkl, implied_timescales.png, landscape.png")


# ==========================================================================
# pwcs
# ==========================================================================
def add_pwcs(sub):
    p = sub.add_parser("pwcs", formatter_class=FMT,
                       help="Population-Weighted Consensus Score on 1D/2D variables",
                       description="PWCS over one or more 1D/2D variable sets (your own variables, CVs, "
                                   "projections). Per-frame by default; MSM-weighted with --msm "
                                   "(build an MSM here) or --weights (your own per-frame MSM weights "
                                   "for the variable, or msm_weights.npz from `reveal msm`).")
    g = _add_input_opts(p, multi=True)
    g.add_argument("--mode", choices=["joint", "separate", "both"], default="joint",
                   help="joint: each --input is one feature set (1D or 2D); separate: every column "
                        "is its own 1D set; both: joint + separate")
    g.add_argument("--traj-files", nargs="+", help="source trajectories for frame reports (legacy inputs)")
    g.add_argument("--traj-dir", help="folder whose sorted *.xtc are the source trajectories (legacy inputs)")
    s = p.add_argument_group("score")
    s.add_argument("--d-min", type=int, default=2, help="min GMM components")
    s.add_argument("--d-max", type=int, default=8, help="max GMM components (inclusive)")
    s.add_argument("--resample-factor", type=int, default=5,
                   help="weighted mode: GMMs are fit on a weighted resample of factor x n_frames")
    s.add_argument("--max-fit-frames", type=int, default=None,
                   help="cap the number of frames used to FIT each GMM (all frames are still scored)")
    s.add_argument("--covariance-type", default="full", choices=["full", "diag", "tied", "spherical"])
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--allow-high-dim", action="store_true",
                   help="allow feature sets with more than 2 dimensions")
    w = p.add_argument_group("weighting (default: per frame, unweighted)")
    w.add_argument("--msm", action="store_true", help="build an MSM and weight frames by it")
    w.add_argument("--msm-input", action="append",
                   help="variables to build the MSM on (default: all PWCS variables, concatenated)")
    _msm_opts(w, prefix="msm-")
    w.add_argument("--weights",
                   help="your per-frame weights (e.g. MSM stationary weights): msm_weights.npz from "
                        "`reveal msm`, or your own .npy / .npz (e.g. np.savez(f, *msm.trajectory_weights())) "
                        "/ .pkl / text file with one weight per frame of the variable")
    w.add_argument("--weights-for", default=None,
                   help="name (NAME=...) or index of the --input the weights belong to (default: the first)")
    o = p.add_argument_group("output")
    o.add_argument("--out", "-o", default="reveal_pwcs")
    o.add_argument("--top-n", type=int, default=5, help="lowest/highest frames to report")
    o.add_argument("--save-components", action="store_true",
                   help="save every per-model normalized log-likelihood")
    p.set_defaults(func=cmd_pwcs)


def cmd_pwcs(args):
    from .pwcs import pwcs_score
    from .msm import build_msm
    from . import plotting

    specs = list(args.input)
    extra = list(args.msm_input or []) if args.msm else []
    if args.msm and args.weights:
        sys.exit("error: use either --msm (build an MSM here) or --weights (your own weights), not both")
    w_extra = []
    if args.weights:
        parsed_in = [parse_spec(x) for x in specs]
        target = parsed_in[0]
        if args.weights_for is not None:
            names = [p_.name for p_ in parsed_in]
            if args.weights_for in names:
                target = parsed_in[names.index(args.weights_for)]
            elif args.weights_for.isdigit() and int(args.weights_for) < len(parsed_in):
                target = parsed_in[int(args.weights_for)]
            else:
                sys.exit(f"error: --weights-for '{args.weights_for}' matches no --input (names: {names})")
        w_extra = [("weights", _load_weights(args, target))]
    dss, labels, parsed = _load_many(args, specs + extra, extra=w_extra)
    in_dss = dss[:len(specs)]
    msm_dss = dss[len(specs):len(specs) + len(extra)]
    w_ds = dss[-1] if w_extra else None
    ref = in_dss[0]
    # provenance (source trajectories/topologies) from whichever input carries it
    for key in ("traj_files", "top_files", "frame_start"):
        if key not in ref.meta:
            for d in dss:
                if d.meta.get(key) is not None:
                    ref.meta[key] = d.meta[key]
                    break

    traj_files = args.traj_files or (find_trajectories(args.traj_dir) if args.traj_dir else None)
    if traj_files:
        ref.meta["traj_files"] = traj_files

    if ref.n_traj == 1 and not args.lengths and not args.lengths_from:
        if args.stride > 1:
            print(f"  note: no trajectory boundaries known (no --lengths): --stride {args.stride} is applied "
                  "to the whole array as one trajectory (variable and weights stay matched)")
        if args.msm:
            print("  note: no trajectory boundaries known (no --lengths): the MSM treats the data as one "
                  "continuous trajectory, so jumps between concatenated trajectories count as transitions. "
                  "Give --lengths (see `reveal lengths`) for a correct MSM.")

    # ---- feature sets ----------------------------------------------------
    feature_sets, set_cols = {}, {}
    for d, lab in zip(in_dss, labels[:len(specs)]):
        name = lab.replace(".npz", "")
        if args.mode in ("joint", "both"):
            if d.n_dim > 2 and not args.allow_high_dim:
                sys.exit(f"error: '{lab}' has {d.n_dim} dims; PWCS runs on 1D or 2D variables. "
                         f"Select columns (e.g. {lab}:0,1), use --mode separate, or --allow-high-dim.")
            key = name if args.mode == "joint" else f"{name}[{','.join(d.names())}]"
            feature_sets[key] = d.concat(np.float64)
            set_cols[key] = (d.concat(np.float64), d.names())
        if args.mode == "separate" or (args.mode == "both" and d.n_dim > 1):
            for j, cn in enumerate(d.names()):
                key = f"{name}:{cn}"
                col = d.concat(np.float64)[:, [j]]
                feature_sets[key] = col
                set_cols[key] = (col, [cn])
    print(f"\nPWCS over {ref.n_frames} frames ({ref.n_traj} trajs, frame_stride {ref.frame_stride}); "
          f"{len(feature_sets)} feature set(s) x GMM d = {args.d_min}..{args.d_max}")
    for k, v in feature_sets.items():
        print(f"  {k}: {v.shape[1]}D")

    # ---- weights ---------------------------------------------------------
    weights, wmode = None, ("per-frame (no weights): every frame counts equally; the GMM densities "
                            "are estimated from the frames in CV space")
    if args.msm:
        mds = _concat_columns(msm_dss) if msm_dss else _concat_columns(in_dss)
        print(f"\nbuilding MSM on {mds.names()} ...")
        res = build_msm(mds.trajs, k=args.msm_k, lag=args.msm_lag, bayesian=not args.msm_ml,
                        n_samples=args.msm_n_samples, fit_stride=args.msm_fit_stride,
                        seed=args.msm_seed, backend=args.msm_backend)
        weights = res.flat_weights()
        wmode = (f"MSM-weighted ({'Bayesian' if not args.msm_ml else 'ML'}, k={args.msm_k}, "
                 f"lag={args.msm_lag}, on {mds.names()})")
        os.makedirs(args.out, exist_ok=True)
        res.save(os.path.join(args.out, "msm.pkl"))
    elif args.weights:
        from .io import check_weights
        weights = w_ds.concat(np.float64)[:, 0]
        diag = check_weights(weights, args.weights)
        wmode = f"weights from {args.weights}"
        print(f"  weights used: {diag['n']} frames, {diag['n_zero']} with weight 0, "
              f"Kish effective sample size {diag['kish_ess']:.0f} ({100 * diag['kish_ess_fraction']:.1f}% of frames)")
    print(f"\nmode: {wmode}")

    S, comps = pwcs_score(feature_sets, d_range=range(args.d_min, args.d_max + 1), weights=weights,
                          resample_factor=args.resample_factor, random_state=args.seed,
                          return_components=True, max_fit_frames=args.max_fit_frames,
                          covariance_type=args.covariance_type, verbose=True)

    # ---- outputs ---------------------------------------------------------
    os.makedirs(args.out, exist_ok=True)
    meta = dict(ref.meta, kind="pwcs", weighting=wmode, feature_sets=list(feature_sets),
                d_range=[args.d_min, args.d_max])
    Dataset(ref.split(S), ["PWCS"], meta).save(os.path.join(args.out, "pwcs.npz"))
    np.savez(os.path.join(args.out, "pwcs_score.npz"), pwcs_score=S,
             **({"weights": weights} if weights is not None else {}))
    if args.save_components:
        np.savez(os.path.join(args.out, "pwcs_components.npz"),
                 **{f"{n}__d{d}": v["l"] for (n, d), v in comps.items()})

    ti, li, fi = ref.frame_table()
    files = ref.traj_files()
    with open(os.path.join(args.out, "pwcs_frames.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["global_index", "traj_index", "traj_key", "frame_stored", "frame_in_file",
                     "source_file", "pwcs"] + (["weight"] if weights is not None else []))
        for g in range(len(S)):
            row = [g, ti[g], f"traj_{ti[g]:02d}", li[g], fi[g],
                   os.path.basename(files[ti[g]]) if files and ti[g] < len(files) else "", f"{S[g]:.6f}"]
            if weights is not None:
                row.append(f"{weights[g]:.6g}")
            wr.writerow(row)

    for k, (Y, cn) in set_cols.items():
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in k)
        plotting.plot_pwcs(Y, S, os.path.join(args.out, f"pwcs_{safe}.png"), cn,
                           title=f"PWCS on {k} -- {wmode.split(' (')[0].split(':')[0]}", weights=weights)
    plotting.plot_pwcs_hist(S, os.path.join(args.out, "pwcs_hist.png"), weights)

    order = np.argsort(S)

    def describe(g):
        src = os.path.basename(files[ti[g]]) if files and ti[g] < len(files) else f"traj_{ti[g]:02d}"
        return f"score {S[g]:+.3f}  {src}  frame {fi[g]}  (stored frame {li[g]})"

    print(f"\nPWCS: mean {S.mean():.3f}  std {S.std():.3f}  min {S.min():.3f}  max {S.max():.3f}")
    print(f"lowest {args.top_n} (least supported):")
    for g in order[:args.top_n]:
        print("  " + describe(g))
    print(f"highest {args.top_n} (most consistently supported):")
    for g in order[::-1][:args.top_n]:
        print("  " + describe(g))
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump({"weighting": wmode, "feature_sets": list(feature_sets),
                   "d_range": [args.d_min, args.d_max], "n_frames": int(len(S)),
                   "frame_stride": ref.frame_stride, "mean": float(S.mean()), "std": float(S.std()),
                   "lowest": [int(g) for g in order[:args.top_n]],
                   "highest": [int(g) for g in order[::-1][:args.top_n]]}, f, indent=2)
    print(f"\nwrote {args.out}/pwcs.npz, pwcs_frames.csv, plots, summary.json")


# ==========================================================================
# extract / info
# ==========================================================================
def add_extract(sub):
    p = sub.add_parser("extract", formatter_class=FMT,
                       help="write highest/lowest-PWCS frames as PDB files")
    p.add_argument("--pwcs", required=True, help="pwcs.npz written by `reveal pwcs`")
    p.add_argument("--top-n", type=int, default=5, help="highest-scoring frames to write")
    p.add_argument("--bottom-n", type=int, default=5, help="lowest-scoring frames to write")
    p.add_argument("--traj-files", nargs="+", help="override source trajectories (sorted order)")
    p.add_argument("--top", nargs="+", help="override topology (one global or one per trajectory)")
    p.add_argument("--atom-selection", default=None, help="MDTraj selection for written atoms")
    p.add_argument("--out", "-o", default="reveal_frames")
    p.set_defaults(func=cmd_extract)


def cmd_extract(args):
    import mdtraj as md
    ds = load_dataset(args.pwcs)
    S = ds.concat()[:, 0]
    files = expand_paths(args.traj_files) if args.traj_files else ds.traj_files()
    tops = expand_paths(args.top) if args.top else ds.meta.get("top_files")
    if not files or not tops:
        sys.exit("error: source trajectories/topologies unknown -- pass --traj-files and --top")
    tops = pair_topologies(files, tops)
    ti, li, fi = ds.frame_table()
    order = np.argsort(S)
    picks = [("high", r, g) for r, g in enumerate(order[::-1][:args.top_n], 1)] + \
            [("low", r, g) for r, g in enumerate(order[:args.bottom_n], 1)]
    os.makedirs(args.out, exist_ok=True)
    for tag, r, g in picks:
        t = md.load_frame(files[ti[g]], int(fi[g]), top=tops[ti[g]])
        if args.atom_selection:
            t = t.atom_slice(t.topology.select(args.atom_selection))
        name = f"{tag}{r:02d}_pwcs{S[g]:+.3f}_traj{ti[g]:02d}_frame{fi[g]}.pdb"
        t.save_pdb(os.path.join(args.out, name))
        print(f"  {name}")
    print(f"wrote {len(picks)} PDB file(s) to {args.out}/")


def add_lengths(sub):
    p = sub.add_parser("lengths", formatter_class=FMT,
                       help="frames per trajectory, for --lengths",
                       description="Print (and optionally save) the number of frames per trajectory, "
                                   "to use with --lengths. Sources: trajectory files (counted from the "
                                   "files, optionally after --stride), a concatenated PLUMED COLVAR "
                                   "(split where the time column restarts), or an existing dataset.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--traj", nargs="+", help="trajectory files or globs, in the order you concatenated them")
    src.add_argument("--traj-dir", help="folder of trajectories (sorted by name, non-recursive)")
    src.add_argument("--colvar", help="concatenated COLVAR / text file whose time column restarts per trajectory")
    src.add_argument("--from", dest="from_ds", help="dataset (.npz with traj_XX keys, REVEAL output, ...)")
    p.add_argument("--traj-ext", default="xtc", help="extension used with --traj-dir")
    p.add_argument("--top", help="topology, only needed for formats that cannot be counted without one")
    p.add_argument("--stride", type=int, default=1,
                   help="report frames per trajectory after reading every Nth frame (ceil(n / N))")
    p.add_argument("--time-col", type=int, default=0, help="time column for --colvar")
    p.add_argument("--out", "-o", help="save to a file (.txt or .npy) usable as `--lengths FILE`")
    p.set_defaults(func=cmd_lengths)


def _count_frames(path, top=None):
    import mdtraj as md
    try:
        with md.open(path) as fh:
            return len(fh)
    except Exception:
        pass
    kw = {"top": top} if top else {}
    try:
        return sum(c.n_frames for c in md.iterload(path, chunk=10000, **kw))
    except Exception as e:
        raise ValueError(f"cannot count frames in {path} ({e}); pass --top for this format")


def cmd_lengths(args):
    if args.from_ds:
        lengths = load_dataset(args.from_ds).lengths
        rows = [(f"traj_{i:02d}", n) for i, n in enumerate(lengths)]
        src = args.from_ds
    elif args.colvar:
        from .io import _load_text
        t = _load_text(args.colvar)[:, args.time_col]
        cuts = np.flatnonzero(np.diff(t) < 0) + 1
        lengths = np.diff(np.r_[0, cuts, len(t)]).astype(int).tolist()
        rows = [(f"segment {i} (t = {t[a]:g} ... {t[a + n - 1]:g})", n)
                for i, (a, n) in enumerate(zip(np.r_[0, cuts], lengths))]
        src = f"{args.colvar} (time column {args.time_col} restarts {len(cuts)} time(s))"
        if len(lengths) == 1:
            print("note: the time column never decreases, so only one trajectory was found. If the file holds "
                  "several continued runs, their boundaries cannot be recovered from time; count frames from "
                  "the trajectory files instead.")
    else:
        files = expand_paths(args.traj) if args.traj else find_trajectories(args.traj_dir, args.traj_ext)
        raw = [_count_frames(f, args.top) for f in files]
        s = max(1, args.stride)
        lengths = [-(-n // s) for n in raw]
        rows = [(f"{f}  ({n} frames{f', stride {s}' if s > 1 else ''})", m)
                for f, n, m in zip(files, raw, lengths)]
        src = f"{len(files)} trajectory file(s)"
    if args.stride > 1 and not (args.traj or args.traj_dir):
        print("note: --stride only applies to --traj / --traj-dir")
    print(f"frames per trajectory from {src}:")
    for name, n in rows:
        print(f"  {n:>9d}  {name}")
    print(f"total {int(sum(lengths))} frames in {len(lengths)} trajectories")
    if args.out:
        if args.out.endswith(".npy"):
            np.save(args.out, np.array(lengths, dtype=int))
        else:
            with open(args.out, "w") as f:
                f.write("\n".join(str(n) for n in lengths) + "\n")
        print(f"wrote {args.out}  (use: --lengths {args.out})")
    print("\n--lengths " + " ".join(str(n) for n in lengths))


def add_info(sub):
    p = sub.add_parser("info", help="summarize a dataset")
    p.add_argument("input", nargs="+")
    p.set_defaults(func=lambda a: [print(f"{s}\n" + "\n".join("  " + l for l in
                                                               load_dataset(s).summary().splitlines()))
                                   for s in a.input])


# ==========================================================================
def build_parser():
    p = argparse.ArgumentParser(prog="reveal", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=f"REVEAL {__version__}")
    sub = p.add_subparsers(dest="command", required=True)
    add_featurize(sub)
    add_reduce(sub)
    add_msm(sub)
    add_pwcs(sub)
    add_extract(sub)
    add_lengths(sub)
    add_info(sub)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except (ValueError, FileNotFoundError, KeyError, IndexError, ImportError) as e:
        if os.environ.get("REVEAL_DEBUG"):
            raise
        sys.exit(f"error: {e}\n(set REVEAL_DEBUG=1 for the full traceback)")


if __name__ == "__main__":
    main()
