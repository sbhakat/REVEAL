# REVEAL

**R**NA **E**nsemble **V**alidation and **E**xploration with **A**I-guided **L**andscapes.

REVEAL turns a set of MD trajectories into features, low-dimensional variables, MSM weights and
per-frame Population-Weighted Consensus Scores (PWCS), one step at a time from the command line.

```
trajectories + PDB(s) ──► featurize ──► reduce (FPCA | SFA | TAE) ──► msm ──► pwcs ──► extract
                           G/R-vectors        1D/2D projections                weights  scores    PDBs
```

Every step reads and writes the same `.npz` layout (`traj_00, traj_01, …` in sorted trajectory order,
plus feature names and provenance), so any output can feed any later step. Legacy files from earlier,
non-public versions of this pipeline (plain `.npz`/`.pkl`/`.npy` CV, feature, and latent-variable
dumps) load directly, with no conversion step.

## Install

One command installs REVEAL and everything it needs:

```bash
pip install ./REVEAL          # or, for development: pip install -e "./REVEAL[test]"
reveal --version
```

This installs numpy, scipy, scikit-learn, matplotlib, mdtraj, deeptime (MSM), scikit-fda (FPCA),
torch and pytorch_lightning (TAE), and addict, pandas, ruamel.yaml and h5py. mdml does **not** need
to be installed separately: the two parts of it that REVEAL uses are bundled inside the package.

**Bundled from mdml** (in `reveal/_vendor/`; details in `reveal/_vendor/README.md`):

| bundled | what it is | license | used by |
|---|---|---|---|
| `sksfa` | sklearn-sfa 0.1.6 with mdml's patch adding `fit(X, tau)` | BSD-style, © 2020 Merlin Schüler and sklearn-sfa contributors | `reveal reduce sfa` |
| `timelaggedcv` | mdml's time-lagged CV-autoencoder | mdml repository license (GNU LGPL v2.1) | `reveal reduce tae` |

Both are unchanged from mdml except six import lines (made relative so they run from inside REVEAL).
Because they live under `reveal._vendor`, they never clash with a separately installed sklearn-sfa
or mdml, and REVEAL always uses its own copies. To use a different timelaggedcv, pass
`reveal reduce tae --tlcv-path DIR`.

## 1. Featurize: G-vectors and R-vectors

```bash
# one global topology, compatible with every .xtc
reveal featurize --traj-dir Data --top Data/prot-masses_rank_01.pdb --stride 10 --out-prefix out/rg

# one PDB per trajectory (matched by file stem, otherwise by sorted order)
reveal featurize --traj Data/*.xtc --top Data/PDBs/*.pdb --stride 10 --out-prefix out/rg

# explicit pairs, one "trajectory topology" per line
reveal featurize --manifest pairs.txt --stride 10 --out-prefix out/rg
```

* Ring center = mean of N1, C2, N3, C4, C5, C6 (shared by A/G/C/U). `R_ij = c_j − c_i`, and
  `G_ij = [sin(γ|r|) r̂, 1 + cos(γ|r|)]` for `|r| < r_c`, with `γ = π/r_c` (Bottaro et al. 2014).
  Only i<j pairs are stored. `--r-cutoff` defaults to 15 Å, the value calibrated against HIV-1 TAR.
* `--stride N` keeps every Nth frame. Frames are read in chunks (`--chunk`) and only ring atoms are
  read from disk, so long trajectories never have to fit in memory.
* With per-trajectory topologies, ring atoms are resolved separately for each PDB, so atom order
  can differ between PDBs. Residue sequences must match, otherwise REVEAL stops with an error.
* `--selection "resSeq 20 to 26"` restricts residues. `--features g` writes only G-vectors.
* Outputs: `out/rg_g.npz`, `out/rg_r.npz` and `out/rg_{g,r}_features.npy` (column names like
  `G_A35-U36_sin_x`).

## 2. Reduce: FPCA, SFA, TAE

```bash
reveal reduce fpca     -i out/rg_g.npz -n 2                  -o out/fpca
reveal reduce sfa      -i out/rg_g.npz -n 2 --tau 1          -o out/sfa
reveal reduce tae      -i out/rg_g.npz -n 1 2 --lag 1 10  -o out/tae        # scan -> out/tae/tau*_dim*/
```

| method   | what it does | backends |
|----------|--------------|----------|
| FPCA     | Functional PCA over the feature-index grid; no lag time | **`skfda`** (default, scikit-fda `FPCA`) or `native` (NumPy, identical results) |
| SFA      | Slow feature analysis at lag τ | **`sksfa`** (default; bundled sklearn-sfa with mdml's tau patch; see below) or `native` (NumPy) |
| TAE      | mdml's time-lagged CV-autoencoder (`timelaggedcv`) | bundled mdml `timelaggedcv` |

Each run writes `projection.npz` (input for `msm` and `pwcs`), `<method>_weights.npy/.pkl`
(n_components × n_features), `<method>_top_features.csv/.png`, `<method>_latent.png`,
`<method>_transformed.npy/.pkl` (legacy layout) and `info.json`.

FPCA has no lag time. It is a static decomposition over all frames, so it takes no `--lag`/`--tau`.
For some data shapes skfda's internal scikit-learn PCA picks a randomized SVD, so REVEAL seeds
NumPy (seed 0) before the skfda fit to make FPCA reproducible run to run.
Only the time-lagged methods (SFA, TAE) take `--lag`/`--tau` (alias), in stored frames
after all striding. `-n` (all methods) and `--lag` (time-lagged methods) accept several values, and
each combination gets its own subfolder (`dim2/` for FPCA, `tau10_dim2/` otherwise).

**SFA with sksfa.** REVEAL uses the bundled copy of mdml's patched sklearn-sfa. Every sksfa build
(PyPI, GitHub, and mdml's patched copy) takes
differences over whatever array it is given. Applied to concatenated trajectories, that pairs the
last frame of one trajectory with the first frame of the next. REVEAL therefore picks the sksfa
route by case:

| case | sksfa call | boundaries |
|---|---|---|
| τ = 1, equal-length trajectories | `fit(X)` with `batch_size` = trajectory length (sksfa's standard method) | respected |
| τ > 1 or unequal lengths | sksfa's whitening (PCA `whiten=True`, drop directions with explained-variance ratio ≤ 1e-15), then `partial()` once per lag-τ sub-series `x[o::τ]` of each trajectory, then sksfa's generalized eigensolver | respected, exact lag τ |
| `--cross-boundaries` | `fit(X, tau)` on concatenated frames, with mdml's sksfa | crossed (legacy) |

All routes reproduce native SFA on full-rank data, and `--cross-boundaries` reproduces a direct
`sksfa.SFA(n).fit(X, tau=τ)` call exactly. The route used is printed and saved in `info.json`.
(mdml's patch only changes the non-batch path, so `fit(..., batch_size=...)` there still uses lag 1.
REVEAL never combines `batch_size` with τ > 1.)

**TAE (mdml timelaggedcv).** REVEAL writes a timelaggedcv `config.yaml` and builds a
`CV_autoencoder`:
- one network per CV: ReLU feature mask → Linear(p, 50) → Tanh → 3 × [Linear(50, 50) → Tanh] → Linear(50, 1)
- decoder: Linear(n_cv, 200) → ELU → 6 × [Linear(200, 200) → ELU] → Linear(200, p)
- loss: L2 reconstruction of x_{t+τ} + VAMP-2 + CV range penalty + mask sparsity

Training uses `CVEstimator.fit_routine`: pretrain, then sparsity rounds until each CV keeps
`--target-n-feat` features, then an unconstrained retrain. Predictions use mask noise off, and
feature weights are the mask kernels (`net.get_kernel`). Defaults (target 30 features, 1500 steps,
2 rounds, patience 2, λ step 0.2, batch 1024, lr 1e-3, weight decay 0.1) match the reference
implementation this reproduces (see `tests/legacy_tae_gvec.py` and `test_tae_matches_tae_gvec` in
the test suite). Every architecture value can be changed (`--cv-width`, `--cv-depth`, `--width`,
`--n-res-blocks`, `--residual`, `--sigma-bin`, `--sigma-feat-mask`, `--vamp-loss`,
`--vamp-loss-weight`). The reference implementation was unseeded; REVEAL seeds
torch/NumPy/random (`--seed`, default 0).

Outputs: `projection.npz`, plus legacy-format copies (`tae_latent.npy/.pkl`, `tae_weights.npy/.pkl`)
and timelaggedcv's own files (`model_final.pt`, `model_full.pt`, `model_final_best.pt`,
`temp_path_model_constraint.pt`, `bottleneck_scatter.png`, `lightning_logs/`),
and `CV_file{n}.yaml` / `Feature_weights_{n}_labeled.png` (the per-feature mask weights).

## 3. MSM weights (optional)

```bash
reveal msm -i FPCA=out/fpca/projection.npz --k 200 --lag 10 --its-lags 1 2 5 10 20 50 -o out/msm_fpca
```

MSM engine: **deeptime** (tested with 0.4.5). REVEAL calls `deeptime.clustering.KMeans`
(k-means++), `deeptime.markov.TransitionCountEstimator` (`effective` counting for Bayesian,
`sliding` for `--ml`), keeps the largest connected set, and fits `deeptime.markov.msm.BayesianMSM`
(default) or `MaximumLikelihoodMSM(reversible=True)`. Per-frame weights come from
`compute_trajectory_weights` (for the Bayesian MSM, from its maximum-likelihood prior model).
If deeptime is not installed, REVEAL falls back to the equivalent pyemma calls
(`cluster_kmeans`, `bayesian_markov_model`, `trajectory_weights`). This fallback path has no
automated test coverage: pyemma's build tooling uses `configparser` APIs removed in Python 3.12,
and (once that is patched) fails to compile against newer deeptime releases, so it cannot be
installed in this project's Python >= 3.12 test environment.

It writes `msm_weights.npz` (per-frame weights w_t = π_s(t) / N_s(t),
summing to 1), `implied_timescales.png`, `landscape.png` (raw vs MSM-reweighted), `msm.pkl` and
`dtrajs.npz`. Frames outside the largest connected set get weight 0.

## 4. PWCS on 1D or 2D variables

For each feature set (a 1D or 2D variable) and each GMM size d in `--d-min…--d-max` (default 2–8):
fit a GMM, score every frame (L_i), normalize to (L_i − min L)/std(L), then average over all
(set, d) models. The result is numerically identical to the reference implementation this
reproduces (preserved verbatim in `tests/legacy_pwcs.py`).

```bash
# per-frame (no MSM): every frame counts equally
reveal pwcs -i CV=cv1_cv2.npz -o out/pwcs

# MSM-weighted, MSM built inline (default: on all PWCS variables; or --msm-input)
reveal pwcs -i CV=cv1_cv2.npz --msm --msm-k 200 --msm-lag 10 -o out/pwcs_msm

# MSM-weighted with weights from step 3
reveal pwcs -i CV=cv1_cv2.npz -i FPCA=out/fpca/projection.npz --weights out/msm/msm_weights.npz -o out/pwcs_w

# any column(s) of any dataset: path:cols
reveal pwcs -i TAE=out/tae/tau10_dim2/projection.npz:0 -i CV=cv1_cv2.npz:0,1 -o out/pwcs_mix

# MSM-weighted PWCS using FPCA
reveal pwcs -i FPCA=out/fpca/projection.npz --weights out/msm_fpca/msm_weights.npz -o out/pwcs_fpca 
```

* **Feature sets.** `--mode joint` (default) makes each `-i` one set, 1D or 2D. `--mode separate`
  makes every column its own 1D set; this is what the original reference implementation did with CV1_pivot and
  CV1_deltaD. `--mode both` does both. Sets with more than 2 dimensions are refused unless you pass
  `--allow-high-dim`.
* **Weighting.** Without `--msm`/`--weights`, every frame counts equally. With weights, each GMM is
  fit on a weighted resample (`--resample-factor`, default 5 × n_frames, capped by
  `--max-fit-frames`), σ_L is the weighted std, and every frame is still scored.
* **Strides.** Inputs at different strides are aligned automatically. A CV file holding every frame
  is subsampled to match a projection computed at `--stride 10`. Legacy files have no stride
  metadata, so declare it with `--frame-stride`.
* **Outputs.** `pwcs.npz` (per trajectory), `pwcs_score.npz` (flat, legacy), `pwcs_frames.csv`
  (traj, source file, frame in file, score, weight), one plot per set (2D plane or 1D profile
  colored by PWCS), `pwcs_hist.png`, `summary.json`, and `--save-components` for every per-model l_i.

## 5. PWCS on your own variable with your own MSM weights

PWCS can score any 1D or 2D variable you computed yourself (a CV, a distance, a reaction coordinate,
an embedding), weighted by MSM weights you computed yourself for that variable (e.g. with pyemma,
deeptime or msmbuilder). REVEAL does not need to build the MSM.

**The minimum is a CV file and, optionally, a weights file. `--lengths` and `--stride` are optional.**

```bash
# CV + its MSM weights (one weight per frame of the CV file)
reveal pwcs -i RC=my_cv.npy --weights my_msm_weights.npy -o out/pwcs_rc

# CV only: no weights -> every frame counts equally and the density in each GMM is
# estimated from the frames themselves in CV space
reveal pwcs -i RC=my_cv.npy -o out/pwcs_rc_frames
```

Without `--weights` (and without `--msm`), every frame has weight 1: each GMM is fit directly to the
frames' positions in CV space, so the density reflects how often the simulation visited each region.
With `--weights`, each frame counts in proportion to its MSM equilibrium weight instead.

The other options are for when you need them:

```bash
# 2D CV from a PLUMED COLVAR (columns 1 and 2; column 0 is time), every 10th frame
reveal pwcs -i "RC=COLVAR:1,2" --stride 10 --weights my_msm_weights.npz -o out/pwcs_colvar

# trajectory boundaries from a file written by `reveal lengths` (see section 6)
reveal pwcs -i RC=my_cv.npy --lengths lengths.txt --stride 10 --weights my_msm_weights.npy -o out/pwcs_rc

# several variables scored together; the weights belong to the one named RC
reveal pwcs -i CV=cv1_cv2.npz -i RC=my_cv.npy --weights my_msm_weights.npy --weights-for RC -o out/pwcs_mix
```

**The variable** (`-i NAME=path[:cols]`) can be:
- `.npz` with `traj_00, traj_01, …` keys, or any keys (taken in natural order: `arr_0, arr_1, …, arr_10`)
- `.npy`, shape (n_frames,) or (n_frames, n_cols)
- a text file (`.txt`, `.dat`, `.csv`, `.xvg`, `.colvar`; lines starting with `#` or `@` are skipped)
- a `.pkl` list of per-trajectory arrays

Pick columns with `:cols` (`:1`, `:1,2`, `:0-1`). Each `-i` must be 1D or 2D (or use `--mode separate`).

**Trajectory boundaries (`--lengths`) are optional.** Without them, a flat `.npy` or text CV is
treated as one long trajectory. PWCS scores frames independently, so with no `--stride` this makes no
difference at all. With `--stride N`, REVEAL takes every Nth frame of the whole array instead of
every Nth frame of each trajectory, which is a slightly different subset of frames. It applies the
same selection to the CV and its weights, so they stay matched frame for frame either way.
Boundaries are found automatically when the CV is an `.npz` with
one array per trajectory, or when the weights are saved per trajectory
(`np.savez(f, *msm.trajectory_weights())`). In that second case REVEAL takes the lengths from the
weights file. Give `--lengths` (numbers, or a file from `reveal lengths`) when you want:
- `--stride` applied per trajectory (frame 0 of each trajectory, as mdtraj strides files), e.g. to
  line the CV up with other inputs or REVEAL outputs;
- `--msm` (REVEAL builds the MSM): otherwise jumps between concatenated trajectories count as
  transitions, and REVEAL prints a note saying so;
- per-trajectory frame numbers in `pwcs_frames.csv` and `reveal extract`.

**The weights** (`--weights`) are one non-negative number per frame of the variable they were
computed for: the first `-i`, or the one named by `--weights-for NAME`. Accepted formats:

| what you have | save it as |
|---|---|
| pyemma: `msm.trajectory_weights()` (list, one array per trajectory) | `np.savez("w.npz", *msm.trajectory_weights())` or `np.save("w.npy", np.concatenate(msm.trajectory_weights()))` |
| deeptime: `msm.compute_trajectory_weights(dtrajs)` | same as above |
| one weight per line | `w.txt` / `w.dat` (the last column is used) |
| a `.pkl` | a list of per-trajectory arrays, a flat array, or a dict with key `"weights"` |
| `reveal msm` output | `msm_weights.npz` (carries its own stride; aligned automatically) |

How REVEAL matches them to the variable:
- If there is one weight per frame **of the variable file**, REVEAL splits them by the variable's
  trajectories and applies the same `--stride` to both, so variable and weights always refer to the
  same frames.
- If there is one weight per frame **after `--stride`**, they are used as given.
- Any other count stops with an error giving both numbers. For per-trajectory weights (`.npz` /
  list), the per-trajectory lengths must also match.
- Weights are checked on the whole file, before striding: NaN, inf, negative values and an all-zero
  file are errors. Scale doesn't matter (they are normalized). REVEAL prints the number of
  zero-weight frames and the Kish effective sample size (Σw)²/Σw².
- REVEAL can only check counts. It cannot tell whether your MSM was built on the same frames in the
  same trajectory order, so keep the variable and the `dtrajs` behind the weights from the same
  trajectory list and stride.

With weights, each GMM is fit on a weighted resample of the frames (`--resample-factor`, default
5 × n_frames, capped by `--max-fit-frames`), σ_L is the weighted standard deviation, and every frame is
still scored. This is the same algorithm as the reference implementation's MSM-weighted mode
(`tests/legacy_pwcs.py`). The weights actually used
are saved in `pwcs_score.npz` (`weights`) and listed per frame in `pwcs_frames.csv`.

## 6. Finding `--lengths`: `reveal lengths`

A flat array of CV values does not record where one trajectory ended and the next began, and
guessing from jumps in the values is unreliable (a real conformational change looks like a jump).
The boundaries have to come from where the data was made. `reveal lengths` reads them from there and
prints a ready-to-paste `--lengths …` line. `--out lengths.txt` saves them for `--lengths lengths.txt`.

```bash
# from the trajectory files the CV was computed from (same order you concatenated them)
reveal lengths --traj Data/*.xtc                     # frames per file (read from the file index)
reveal lengths --traj Data/*.xtc --stride 10 -o lengths.txt   # if the CV was computed with stride 10
reveal lengths --traj-dir Data --traj-ext dcd        # other formats; --top if a format needs one

# from a concatenated PLUMED COLVAR whose time column restarts for each trajectory
reveal lengths --colvar COLVAR --time-col 0 -o lengths.txt

# from an existing dataset (an .npz with one array per trajectory, any REVEAL output)
reveal lengths --from cv1_cv2.npz
```

- `--stride N` reports ceil(n / N) frames per file, which is what `md.load(..., stride=N)` returns.
- `--colvar` splits wherever time decreases. If the file holds continued runs whose time keeps
  increasing, their boundaries cannot be recovered from time. REVEAL says so; count frames from the
  trajectory files instead.
- If you built the MSM yourself, `[len(d) for d in dtrajs]` (pyemma, deeptime) gives the same lengths
  your weights correspond to.
- REVEAL checks that the lengths add up to the number of frames, and that per-trajectory weights
  agree with them. It cannot check the order, which must be the order the trajectories were
  concatenated in.

## 7. Extract structures

```bash
reveal extract --pwcs out/pwcs_msm/pwcs.npz --top-n 5 --bottom-n 5 --atom-selection "not water" -o out/frames
```

Writes PDBs named like `low01_pwcs+4.089_traj07_frame4950.pdb`. Frame numbers refer to the
original trajectory file, with all strides undone.

## Utilities

* `reveal info FILE…` summarizes any dataset.
* `reveal lengths …` gives frames per trajectory for `--lengths` (section 6).
* `reveal featurize --names-only` saves feature names without reading trajectories.
* `REVEAL_DEBUG=1` shows full tracebacks.
* `examples/run_pipeline.sh` is the complete TAR-style pipeline.
* `pytest tests/` runs the test suite (builds a synthetic 12-nt system with a slow loop flip).

## Validation

`pytest tests/` builds a synthetic 12-nt system with a slow loop flip and checks:

* G/R-vectors match the reference Bottaro/Herron formulas to float32 precision, with chunked +
  strided reading and per-trajectory PDBs whose atom order differs.
* PWCS matches the original reference implementation (preserved verbatim in
  `tests/legacy_pwcs.py`), per-frame and weighted.
* FPCA (skfda backend) equals a direct skfda `FPCA` call on the same data.
* SFA: every sksfa route equals native SFA on full-rank data (τ = 1 and 3, equal and unequal
  lengths, cross-boundaries), and `--cross-boundaries` equals a direct `SFA(n).fit(X, tau=3)` call
  with the bundled mdml sksfa.
* The bundled sksfa has the `tau` patch, and TAE imports the bundled timelaggedcv.
* TAE: `reveal reduce tae` gives bit-identical CVs and kernel weights to the reference
  implementation (preserved verbatim in `tests/legacy_tae_gvec.py`) with the same seed. If mdml's
  own timelaggedcv is installed, the reference script runs on it; otherwise on the bundled copy.
* msm → pwcs `--weights` gives the same scores as pwcs `--msm`. Stride alignment and `extract` work.
* PWCS on a user variable (flat `.npy` + `--lengths`, text file) with user weights (flat `.npy`,
  `arr_*` `.npz`, text, already-strided) equals `pwcs_score` called directly on the same frames and
  weights. Wrong weight counts and negative weights are rejected.
* Minimal use: CV + weights, and CV only (no weights), with no `--lengths` and no `--stride`, equal a
  direct `pwcs_score` call. Lengths taken from per-trajectory weights equal an explicit `--lengths` run.
* `reveal lengths` counts frames from `.xtc`/`.dcd` files (with and without `--stride`), splits a
  COLVAR at time restarts, reads datasets, and its `--out` file works as `--lengths FILE`.
