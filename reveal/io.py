"""
REVEAL data layer.

Every step of REVEAL reads and writes the same on-disk format, a
"REVEAL dataset" .npz:

    traj_00, traj_01, ...   per-trajectory arrays, shape (n_frames_i, n_dim)
    feature_names           (n_dim,) str array            [optional]
    __meta__                JSON string with provenance   [optional]

Keys are ordered by *sorted position* of the source trajectory files, the
same convention as cv1_cv2.npz / rg_vectors_*.npz, so trajectory index N
refers to the same file (and frame-for-frame the same conformation) in
every REVEAL output built from the same trajectory list.

The important provenance field is ``meta["frame_stride"]``: the cumulative
stride between consecutive stored frames relative to the original
trajectory files (featurize --stride x reduce --stride x pwcs --stride).
It is what lets REVEAL map any stored frame back to (file, frame_in_file),
and what lets PWCS align inputs that were produced at different strides.

Loaders also accept legacy inputs from the standalone scripts:
    * any .npz with traj_* keys (cv1_cv2.npz, rg_vectors_g.npz, ...)
    * glob patterns over multi-part files (e.g. "rg_vectors_g_part*.npz")
    * .pkl dicts with {"data", "traj_lengths"} (fpca/sfa_transformed.pkl)
    * .pkl lists of per-trajectory arrays (the TAE feature cache)
    * .npy arrays, split per trajectory with lengths taken from another dataset
"""
from __future__ import annotations

import glob
import json
import os
import pickle
import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

META_KEY = "__meta__"
RESERVED_KEYS = {META_KEY, "feature_names", "residues", "pairs", "pwcs_score"}
TRAJ_KEY_RE = re.compile(r"^traj_(\d+)$")


# --------------------------------------------------------------------------
# Dataset container
# --------------------------------------------------------------------------
@dataclass
class Dataset:
    trajs: List[np.ndarray]
    feature_names: Optional[List[str]] = None
    meta: dict = field(default_factory=dict)
    extras: dict = field(default_factory=dict)  # residues, pairs, ...

    def __post_init__(self):
        self.trajs = [np.asarray(t) if np.asarray(t).ndim == 2 else np.asarray(t).reshape(-1, 1)
                      for t in self.trajs]
        if not self.trajs:
            raise ValueError("dataset has no trajectories")
        dims = {t.shape[1] for t in self.trajs}
        if len(dims) != 1:
            raise ValueError(f"trajectories have inconsistent feature dims: {sorted(dims)}")
        if self.feature_names is not None:
            self.feature_names = [str(n) for n in self.feature_names]
            if len(self.feature_names) != self.n_dim:
                raise ValueError(f"{len(self.feature_names)} feature names for {self.n_dim} columns")
        self.meta.setdefault("frame_stride", 1)

    # ---- shape helpers ------------------------------------------------
    @property
    def n_traj(self) -> int:
        return len(self.trajs)

    @property
    def n_dim(self) -> int:
        return self.trajs[0].shape[1]

    @property
    def lengths(self) -> List[int]:
        return [t.shape[0] for t in self.trajs]

    @property
    def n_frames(self) -> int:
        return int(sum(self.lengths))

    @property
    def frame_stride(self) -> int:
        return int(self.meta.get("frame_stride", 1))

    def names(self) -> List[str]:
        return self.feature_names or [f"dim{i}" for i in range(self.n_dim)]

    def concat(self, dtype=None) -> np.ndarray:
        X = np.concatenate(self.trajs, axis=0)
        return X.astype(dtype, copy=False) if dtype is not None else X

    def split(self, flat: np.ndarray) -> List[np.ndarray]:
        """Split a concatenated (n_frames, ...) array back into per-trajectory pieces."""
        if flat.shape[0] != self.n_frames:
            raise ValueError(f"array has {flat.shape[0]} rows, dataset has {self.n_frames} frames")
        bounds = np.cumsum([0] + self.lengths)
        return [flat[bounds[i]:bounds[i + 1]] for i in range(self.n_traj)]

    # ---- transforms (return new datasets) -----------------------------
    def strided(self, stride: int) -> "Dataset":
        stride = int(stride)
        if stride <= 1:
            return self
        meta = dict(self.meta, frame_stride=self.frame_stride * stride)
        return Dataset([t[::stride] for t in self.trajs], self.feature_names, meta, dict(self.extras))

    def select(self, cols: Optional[Sequence[int]]) -> "Dataset":
        if cols is None:
            return self
        cols = list(cols)
        bad = [c for c in cols if c < 0 or c >= self.n_dim]
        if bad:
            raise IndexError(f"column(s) {bad} out of range for {self.n_dim}-dim dataset")
        names = [self.names()[c] for c in cols]
        return Dataset([t[:, cols] for t in self.trajs], names, dict(self.meta), dict(self.extras))

    def frame_table(self):
        """(traj_index, local_index, frame_in_original_file) for every stored frame."""
        ti = np.concatenate([np.full(n, i) for i, n in enumerate(self.lengths)])
        li = np.concatenate([np.arange(n) for n in self.lengths])
        start = int(self.meta.get("frame_start", 0))
        return ti, li, start + li * self.frame_stride

    def traj_files(self) -> Optional[List[str]]:
        f = self.meta.get("traj_files")
        return list(f) if f else None

    # ---- I/O ------------------------------------------------------------
    def save(self, path: str, compress: bool = False) -> str:
        path = path if path.endswith(".npz") else path + ".npz"
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        payload = {f"traj_{i:02d}": t for i, t in enumerate(self.trajs)}
        if self.feature_names is not None:
            payload["feature_names"] = np.array(self.feature_names)
        for k, v in self.extras.items():
            payload[k] = np.asarray(v)
        payload[META_KEY] = np.array(json.dumps(_jsonable(self.meta)))
        (np.savez_compressed if compress else np.savez)(path, **payload)
        return path

    def summary(self) -> str:
        L = self.lengths
        s = [f"{self.n_traj} trajectories, {self.n_frames} frames total, {self.n_dim} dims",
             f"frames/traj: min {min(L)}, max {max(L)}; frame_stride = {self.frame_stride}"]
        if self.feature_names:
            shown = ", ".join(self.feature_names[:6]) + (" ..." if self.n_dim > 6 else "")
            s.append(f"features: {shown}")
        for k in ("kind", "method", "source"):
            if k in self.meta:
                s.append(f"{k}: {self.meta[k]}")
        return "\n".join(s)


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


# --------------------------------------------------------------------------
# Input specs:  [name=]path[:cols]
# --------------------------------------------------------------------------
@dataclass
class InputSpec:
    path: str
    cols: Optional[List[int]] = None
    col_names: Optional[List[str]] = None
    name: Optional[str] = None


_COLS_RE = re.compile(r"^[\w.+\-]+(,[\w.+\-]+)*$")


def parse_cols(text: str):
    """'0,1' / '0-3' / '0,2-4' -> [ints];  'CV1,CV2' -> names."""
    ints, names = [], []
    for tok in text.split(","):
        tok = tok.strip()
        if re.fullmatch(r"\d+-\d+", tok):
            a, b = map(int, tok.split("-"))
            ints.extend(range(a, b + 1))
        elif tok.isdigit():
            ints.append(int(tok))
        else:
            names.append(tok)
    if ints and names:
        raise ValueError(f"mix of column indices and names in '{text}'")
    return (ints or None), (names or None)


def parse_spec(spec: str) -> InputSpec:
    name = None
    m = re.match(r"^([A-Za-z_][\w\-]*)=(.+)$", spec)
    if m and not os.path.exists(spec):
        name, spec = m.group(1), m.group(2)
    path, cols, col_names = spec, None, None
    if ":" in spec and not os.path.exists(spec):
        head, tail = spec.rsplit(":", 1)
        if _COLS_RE.match(tail):
            path = head
            cols, col_names = parse_cols(tail)
    return InputSpec(path=path, cols=cols, col_names=col_names, name=name)


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def _natural_key(k: str):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", k)]


TEXT_EXTS = (".txt", ".dat", ".csv", ".xvg", ".colvar")


def _load_text(path):
    """Whitespace/comma-separated numbers; '#' and '@' lines (PLUMED COLVAR, GROMACS xvg) skipped."""
    delim = "," if path.endswith(".csv") else None
    try:
        return np.loadtxt(path, comments=("#", "@"), delimiter=delim, ndmin=2)
    except ValueError:                     # header row without '#'
        return np.loadtxt(path, comments=("#", "@"), delimiter=delim, ndmin=2, skiprows=1)


def _traj_keys(files) -> List[str]:
    keys = [k for k in files if TRAJ_KEY_RE.match(k)]
    return sorted(keys, key=lambda k: int(TRAJ_KEY_RE.match(k).group(1)))


def _load_npz(path):
    d = np.load(path, allow_pickle=False)
    keys = _traj_keys(d.files)
    if not keys:
        # no traj_XX keys: every other array is one trajectory, in natural key order
        # (np.savez("w.npz", *list_of_arrays) -> arr_0, arr_1, ..., arr_10 in that order)
        keys = sorted((k for k in d.files if k not in RESERVED_KEYS and d[k].ndim >= 1),
                      key=_natural_key)
        if not keys:
            raise ValueError(f"{path}: no arrays found (keys: {d.files})")
    trajs = [d[k] for k in keys]
    names = [str(x) for x in d["feature_names"]] if "feature_names" in d.files else None
    meta = json.loads(str(d[META_KEY])) if META_KEY in d.files else {}
    extras = {k: d[k] for k in ("residues", "pairs") if k in d.files}
    return trajs, names, meta, extras


def _auto_feature_names(path_pattern: str, n_dim: int):
    """Legacy helper: rg_vectors_g_part*.npz -> rg_vectors_g_features.npy if present."""
    d = os.path.dirname(os.path.abspath(path_pattern)) or "."
    base = os.path.basename(path_pattern)
    stem = re.sub(r"(_part[\*\d]*)?\.npz$", "", base).replace("*", "")
    for cand in (os.path.join(d, f"{stem}_features.npy"),):
        if os.path.exists(cand):
            arr = np.load(cand, allow_pickle=True)
            if arr.shape[0] == n_dim:
                return [str(x) for x in arr]
    return None


def load_dataset(spec, lengths_from: Optional[str] = None, feature_names: Optional[str] = None,
                 frame_stride: Optional[int] = None, traj_files=None,
                 lengths: Optional[Sequence[int]] = None) -> Dataset:
    """Load any supported input into a Dataset and apply its column selection.

    Parameters
    ----------
    spec : str or InputSpec
        "[name=]path[:cols]". path may be a glob over multi-part npz files.
    lengths_from : str, optional
        Dataset whose per-trajectory lengths are used to split a flat .npy/.pkl/text array.
    lengths : list of int, optional
        Frames per trajectory for a flat array (alternative to lengths_from).
    feature_names : str, optional
        .npy/.txt file of names overriding whatever the file carries.
    frame_stride : int, optional
        Declare the stride of a legacy file relative to the raw trajectories
        (REVEAL files carry this in their metadata already).
    traj_files : list of str, optional
        Declare the source trajectory files of a legacy file (for frame reports).
    """
    if isinstance(spec, str):
        spec = parse_spec(spec)
    path = spec.path
    names = None
    meta: dict = {}
    extras: dict = {}

    if any(ch in path for ch in "*?["):
        parts = sorted(glob.glob(path))
        if not parts:
            raise FileNotFoundError(f"no files match '{path}'")
        trajs = []
        for p in parts:
            t, n, m, e = _load_npz(p)
            trajs.extend(t)
            names = names or n
            meta = meta or m
            extras = extras or e
        meta = dict(meta, source=parts)
    elif path.endswith(".npz"):
        trajs, names, meta, extras = _load_npz(path)
        meta.setdefault("source", path)
    elif path.endswith(".pkl"):
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, dict) and "data" in obj:
            flat = np.asarray(obj["data"])
            lens = obj.get("traj_lengths")
            if lens is None and lengths:
                lens = list(lengths)
            if lens is None and lengths_from:
                lens = load_dataset(lengths_from).lengths
            trajs = _split_flat(flat, lens)
            if "stride" in obj:
                meta["frame_stride"] = int(obj["stride"])
            names = obj.get("feature_names")
        elif isinstance(obj, (list, tuple)):
            trajs = [np.asarray(t) for t in obj]
        else:
            raise ValueError(f"{path}: unrecognised pickle content ({type(obj)})")
        meta.setdefault("source", path)
    elif path.endswith(".npy") or path.endswith(TEXT_EXTS):
        flat = np.load(path) if path.endswith(".npy") else _load_text(path)
        lens = list(lengths) if lengths else (load_dataset(lengths_from).lengths if lengths_from else None)
        trajs = _split_flat(flat, lens)
        meta.setdefault("source", path)
    else:
        raise ValueError(f"unsupported input '{path}' (expected .npz, .npy, .pkl, a text file "
                         f"({', '.join(TEXT_EXTS)}) or a glob)")

    if names is None:
        names = _auto_feature_names(path, np.asarray(trajs[0]).reshape(len(trajs[0]), -1).shape[1])
    if feature_names:
        names = _read_names(feature_names)
    if frame_stride:
        meta["frame_stride"] = int(frame_stride)
    if traj_files:
        meta["traj_files"] = list(traj_files)

    ds = Dataset(list(trajs), names, meta, extras)
    cols = spec.cols
    if spec.col_names:
        lookup = {n: i for i, n in enumerate(ds.names())}
        missing = [n for n in spec.col_names if n not in lookup]
        if missing:
            raise KeyError(f"columns {missing} not found; available: {ds.names()[:20]}")
        cols = [lookup[n] for n in spec.col_names]
    return ds.select(cols)


def _split_flat(flat, lens):
    flat = np.asarray(flat)
    if flat.ndim == 1:
        flat = flat[:, None]
    if lens is None:
        return [flat]
    if int(np.sum(lens)) != flat.shape[0]:
        raise ValueError(f"lengths sum to {int(np.sum(lens))} but array has {flat.shape[0]} rows")
    bounds = np.cumsum([0] + list(lens))
    return [flat[bounds[i]:bounds[i + 1]] for i in range(len(lens))]


def _read_names(path):
    if path.endswith(".npy"):
        return [str(x) for x in np.load(path, allow_pickle=True)]
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def align_strides(datasets: Sequence[Dataset], labels: Sequence[str]) -> List[Dataset]:
    """Subsample finer-strided datasets so all share the coarsest frame_stride.

    E.g. CVs saved for every frame (stride 1) + an FPCA projection computed
    at stride 10 -> CVs are subsampled [::10] so both cover the same frames.
    """
    target = max(d.frame_stride for d in datasets)
    out = []
    for d, lab in zip(datasets, labels):
        if d.frame_stride == target:
            out.append(d)
            continue
        if target % d.frame_stride:
            raise ValueError(f"cannot align '{lab}' (frame_stride {d.frame_stride}) "
                             f"to frame_stride {target}: not a divisor")
        factor = target // d.frame_stride
        print(f"  aligning '{lab}': frame_stride {d.frame_stride} -> {target} (subsampling by {factor})")
        out.append(d.strided(factor))
    ref = out[0]
    for d, lab in zip(out[1:], labels[1:]):
        if d.lengths != ref.lengths:
            raise ValueError(
                f"'{lab}' has per-trajectory lengths {d.lengths[:4]}... but '{labels[0]}' has "
                f"{ref.lengths[:4]}... -- inputs must cover the same trajectories and frames "
                "(check --stride / --frame-stride).")
    return out


# --------------------------------------------------------------------------
# Trajectory / topology discovery (featurize)
# --------------------------------------------------------------------------
TRAJ_EXTS = (".xtc", ".dcd", ".trr", ".nc", ".netcdf", ".h5", ".lh5", ".pdb", ".gro", ".binpos", ".mdcrd")
TOP_EXTS = (".pdb", ".gro", ".prmtop", ".parm7", ".psf", ".mol2", ".h5")


def expand_paths(items: Optional[Sequence[str]]) -> List[str]:
    out = []
    for it in items or []:
        hits = sorted(glob.glob(it)) if any(ch in it for ch in "*?[") else [it]
        if not hits:
            raise FileNotFoundError(f"no files match '{it}'")
        out.extend(hits)
    return out


def find_trajectories(traj_dir: str, ext: str = "xtc") -> List[str]:
    """Non-recursive, sorted -- same convention as cv1_mdtraj.find_xtc_files."""
    ext = ext.lstrip(".")
    files = sorted(glob.glob(os.path.join(traj_dir, f"*.{ext}")))
    if not files:
        raise FileNotFoundError(f"no *.{ext} files in '{traj_dir}'")
    return files


def find_topology(data_dir: str) -> str:
    """Prefer .pdb, then .gro, then .prmtop/.parm7 -- same as cv1_mdtraj.find_topology."""
    for ext in (".pdb", ".gro", ".prmtop", ".parm7"):
        hits = sorted(glob.glob(os.path.join(data_dir, f"*{ext}")))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"no topology (.pdb/.gro/.prmtop) in '{data_dir}'")


def pair_topologies(trajs: List[str], tops: List[str], mode: str = "auto") -> List[str]:
    """Return one topology path per trajectory.

    mode
      global : exactly one topology used for every trajectory
      paired : len(tops) == len(trajs); matched by file stem when every stem
               matches, otherwise by sorted order
      auto   : global if one topology, paired if counts match
    """
    if not tops:
        raise ValueError("no topology given")
    if mode == "global" or (mode == "auto" and len(tops) == 1):
        if len(tops) != 1:
            raise ValueError(f"--top-mode global needs exactly one topology, got {len(tops)}")
        return [tops[0]] * len(trajs)
    if len(tops) != len(trajs):
        raise ValueError(f"{len(tops)} topologies for {len(trajs)} trajectories: give one global "
                         "topology or exactly one per trajectory (or use --manifest)")
    stem = lambda p: os.path.splitext(os.path.basename(p))[0]
    by_stem = {stem(t): t for t in tops}
    if all(stem(t) in by_stem for t in trajs):
        return [by_stem[stem(t)] for t in trajs]
    return list(tops)


def read_manifest(path: str):
    """Text file, one 'trajectory topology' pair per line ('#' comments allowed)."""
    trajs, tops = [], []
    base = os.path.dirname(os.path.abspath(path))
    with open(path) as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(f"manifest line needs 'trajectory topology': {line!r}")
            t, p = (x if os.path.isabs(x) else os.path.join(base, x) for x in parts)
            trajs.append(t)
            tops.append(p)
    return trajs, tops


# --------------------------------------------------------------------------
# Per-frame weights (e.g. MSM stationary weights) supplied by the user
# --------------------------------------------------------------------------
def read_weight_arrays(path: str) -> List[np.ndarray]:
    """Read per-frame weights from a file the user made themselves.

    Accepted: .npy (flat), text (.txt/.dat/.csv/.xvg, one value per line, or the last column),
    .npz (traj_XX keys, or any keys in natural order, e.g. np.savez("w.npz", *msm.trajectory_weights())),
    .pkl (list of per-trajectory arrays, a flat array, or a dict with key "weights").
    Returns a list of 1-D arrays (one per trajectory, or a single flat array).
    """
    if path.endswith(".npz"):
        d = np.load(path, allow_pickle=False)
        keys = _traj_keys(d.files) or sorted((k for k in d.files if k not in RESERVED_KEYS),
                                             key=_natural_key)
        arrs = [d[k] for k in keys]
    elif path.endswith(".npy"):
        arrs = [np.load(path)]
    elif path.endswith(".pkl"):
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, dict):
            obj = obj.get("weights", obj.get("data"))
        arrs = list(obj) if isinstance(obj, (list, tuple)) else [obj]
    elif path.endswith(TEXT_EXTS):
        a = _load_text(path)
        arrs = [a[:, -1]]
    else:
        raise ValueError(f"unsupported weights file '{path}'")
    out = []
    for a in arrs:
        a = np.asarray(a, dtype=np.float64)
        if a.ndim == 2 and 1 in a.shape:
            a = a.ravel()
        if a.ndim != 1:
            raise ValueError(f"weights in '{path}' must be 1-D per trajectory, got shape {a.shape}")
        out.append(a)
    return out


def check_weights(w: np.ndarray, label: str = "weights") -> dict:
    """Validate per-frame weights and return diagnostics (Kish effective sample size)."""
    w = np.asarray(w, dtype=np.float64)
    if not np.all(np.isfinite(w)):
        raise ValueError(f"{label}: contains NaN or inf")
    if np.any(w < 0):
        raise ValueError(f"{label}: contains negative values (min {w.min():.3g})")
    if w.sum() <= 0:
        raise ValueError(f"{label}: all weights are zero")
    ess = w.sum() ** 2 / np.sum(w ** 2)
    return {"n": int(w.size), "n_zero": int((w == 0).sum()), "sum": float(w.sum()),
            "kish_ess": float(ess), "kish_ess_fraction": float(ess / w.size)}
