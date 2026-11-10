#!/usr/bin/env bash
# REVEAL step-by-step pipeline for an RNA MD dataset (HIV-1 TAR layout:
# Data/*.xtc + one shared topology). Edit paths/parameters, then run steps
# one at a time or all together.
set -euo pipefail

DATA=Data                # folder with trajfit_*.xtc
TOP=Data/prot-masses_rank_01.pdb
OUT=reveal_out
STRIDE=10                # read every 10th frame (10 ps/frame -> 100 ps)

# ---- 1. featurize: G- and R-vectors -------------------------------------
# one global topology for all trajectories:
reveal featurize --traj-dir "$DATA" --top "$TOP" --stride $STRIDE --out-prefix $OUT/rg
# ...or one topology per trajectory (matched by file stem, else by order):
#   reveal featurize --traj Data/*.xtc --top Data/PDBs/*.pdb --stride $STRIDE --out-prefix $OUT/rg
# ...or explicit pairs:
#   reveal featurize --manifest pairs.txt --stride $STRIDE --out-prefix $OUT/rg

# ---- 2. dimensionality reduction on G-vectors ---------------------------
reveal reduce fpca     -i $OUT/rg_g.npz -n 2                 -o $OUT/fpca         # skfda (default), no lag
reveal reduce sfa      -i $OUT/rg_g.npz -n 2 --tau 1         -o $OUT/sfa          # sksfa (default)
# legacy behaviour (differences across trajectory boundaries):
#   reveal reduce sfa -i $OUT/rg_g.npz -n 2 --tau 10 --cross-boundaries -o $OUT/sfa_legacy
reveal reduce tae      -i $OUT/rg_g.npz -n 1 2 --lag 1 10 50 -o $OUT/tae   # mdml's timelaggedcv; scan several lags/dims -> tau*_dim*/

# ---- 3. MSM on a 1D/2D variable -> per-frame weights ---------------------
reveal msm -i TAE=$OUT/tae/tau10_dim2/projection.npz --k 200 --lag 10 -o $OUT/msm_tae

# ---- 4. PWCS on 1D/2D variables ------------------------------------------
# per frame (unweighted), CV1/CV2 file saved for every frame -> auto-aligned to stride 10:
reveal pwcs -i CV=cv1_cv2.npz --frame-stride 1 -i FPCA=$OUT/fpca/projection.npz \
            -i SFA=$OUT/sfa/projection.npz -o $OUT/pwcs_perframe
# MSM-weighted, MSM built inline on the CVs:
reveal pwcs -i CV=cv1_cv2.npz -i TAE=$OUT/tae/tau10_dim2/projection.npz \
            --msm --msm-input CV=cv1_cv2.npz --msm-k 200 --msm-lag 10 -o $OUT/pwcs_msm
# MSM-weighted with weights from step 3; each CV column as its own 1D set:
reveal pwcs -i CV=cv1_cv2.npz --mode separate --weights $OUT/msm_tae/msm_weights.npz \
            -o $OUT/pwcs_cv_sep

# ---- 5. structures behind the scores -------------------------------------
reveal extract --pwcs $OUT/pwcs_msm/pwcs.npz --top-n 5 --bottom-n 5 --atom-selection "not water" \
               -o $OUT/pwcs_msm/frames
