"""Shared plotting helpers (matplotlib, Agg backend, PNG output)."""
from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 300, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 11})


# --------------------------------------------------------------------------
# free energy estimates
# --------------------------------------------------------------------------
def fes_1d(x, weights=None, bins=100, range_=None):
    counts, edges = np.histogram(x, bins=bins, weights=weights, range=range_)
    centers = 0.5 * (edges[:-1] + edges[1:])
    p = counts / counts.sum() if counts.sum() > 0 else counts
    with np.errstate(divide="ignore"):
        F = -np.log(p)
    finite = np.isfinite(F)
    F = F - F[finite].min() if finite.any() else F
    F[~finite] = np.nan
    return centers, F


def fes_2d(x, y, weights=None, bins=80, range_=None):
    H, xe, ye = np.histogram2d(x, y, bins=bins, weights=weights, range=range_)
    p = H / H.sum() if H.sum() > 0 else H
    with np.errstate(divide="ignore"):
        F = -np.log(p)
    finite = np.isfinite(F)
    F = F - F[finite].min() if finite.any() else F
    F[~finite] = np.nan
    return 0.5 * (xe[:-1] + xe[1:]), 0.5 * (ye[:-1] + ye[1:]), F


def _contour_fes(ax, xc, yc, F, vmax=None, cmap="nipy_spectral"):
    vmax = vmax if vmax is not None else np.nanpercentile(F, 99) if np.isfinite(F).any() else 1
    levels = np.linspace(0, max(vmax, 1e-6), 25)
    cs = ax.contourf(xc, yc, np.clip(F.T, 0, vmax), levels=levels, cmap=cmap)
    return cs


# --------------------------------------------------------------------------
# reduction outputs
# --------------------------------------------------------------------------
def plot_top_features(W, names, out_png, top_k=20, comp_labels=None, ylabel="weight", signed=True):
    W = np.atleast_2d(W)
    names = np.asarray(names)
    n = W.shape[0]
    comp_labels = comp_labels or [f"C{i + 1}" for i in range(n)]
    fig, axes = plt.subplots(n, 1, figsize=(max(8, 0.42 * top_k + 2), 3.6 * n), squeeze=False)
    for i in range(n):
        w = W[i]
        k = min(top_k, int((np.abs(w) > 0).sum()) or top_k, w.size)
        idx = np.argsort(np.abs(w))[-k:][::-1]
        vals = w[idx] if signed else np.abs(w[idx])
        ax = axes[i, 0]
        ax.bar(range(k), vals, color=["#3b6ea8" if v >= 0 else "#c2533b" for v in vals])
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(range(k))
        ax.set_xticklabels(names[idx], rotation=90, fontsize=7)
        ax.set_ylabel(f"{comp_labels[i]} {ylabel}")
        ax.set_title(f"Top {k} features for {comp_labels[i]}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def plot_latent(Y, out_png, labels=None, title=None, weights=None):
    Y = np.asarray(Y)
    if Y.ndim == 1:
        Y = Y[:, None]
    labels = labels or [f"C{i + 1}" for i in range(Y.shape[1])]
    if Y.shape[1] == 1:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        c, F = fes_1d(Y[:, 0], weights)
        ax.plot(c, F, color="#3b6ea8", lw=2)
        ax.set_xlabel(labels[0])
        ax.set_ylabel("free energy (kT)")
    else:
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        xc, yc, F = fes_2d(Y[:, 0], Y[:, 1], weights)
        cs = _contour_fes(ax, xc, yc, F)
        fig.colorbar(cs, ax=ax, label="free energy (kT)")
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
    if title:
        ax.set_title(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def plot_training(history, out_png, title=None):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(history["train"], label="train")
    if history.get("val"):
        ax.plot(history["val"], label="validation")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend()
    if title:
        ax.set_title(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


# --------------------------------------------------------------------------
# MSM outputs
# --------------------------------------------------------------------------
def plot_its(lags, its, out_png, dt_label="lag (stored frames)", lag_used=None):
    its = np.asarray(its, dtype=float)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for k in range(its.shape[1]):
        ax.plot(lags, its[:, k], "o-", lw=1.6, ms=4, label=f"t{k + 2}")
    ax.fill_between(lags, 0.1, lags, color="0.85", zorder=0)
    ax.plot(lags, lags, "k--", lw=0.8)
    if lag_used is not None:
        ax.axvline(lag_used, color="#c2533b", lw=1, ls=":")
    ax.set_yscale("log")
    ax.set_xlabel(dt_label)
    ax.set_ylabel("implied timescale (stored frames)")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def plot_fes_compare(Y, weights, out_png, labels=None, title=None):
    """Raw vs MSM-reweighted landscape, 1D (overlaid) or 2D (side by side)."""
    Y = np.asarray(Y)
    if Y.ndim == 1:
        Y = Y[:, None]
    labels = labels or [f"C{i + 1}" for i in range(Y.shape[1])]
    if Y.shape[1] == 1:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        c, F = fes_1d(Y[:, 0])
        ax.plot(c, F, "--", color="0.5", label="raw (per frame)")
        c, F = fes_1d(Y[:, 0], weights)
        ax.plot(c, F, color="#3b6ea8", lw=2, label="MSM reweighted")
        ax.set_xlabel(labels[0])
        ax.set_ylabel("free energy (kT)")
        ax.legend()
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
        rng = [[Y[:, 0].min(), Y[:, 0].max()], [Y[:, 1].min(), Y[:, 1].max()]]
        for ax, w, t in ((axes[0], None, "raw (per frame)"), (axes[1], weights, "MSM reweighted")):
            xc, yc, F = fes_2d(Y[:, 0], Y[:, 1], w, range_=rng)
            cs = _contour_fes(ax, xc, yc, F, vmax=10)
            ax.set_title(t, fontsize=11)
            ax.set_xlabel(labels[0])
        axes[0].set_ylabel(labels[1])
        fig.colorbar(cs, ax=axes, label="free energy (kT)")
    if title:
        fig.suptitle(title, fontsize=11)
    if Y.shape[1] == 1:
        fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# PWCS outputs
# --------------------------------------------------------------------------
def plot_pwcs(Y, S, out_png, labels=None, title=None, weights=None):
    """2D: CV plane coloured by PWCS (low scores drawn on top).
    1D: PWCS vs the variable, over the (weighted) free-energy profile."""
    Y = np.asarray(Y)
    if Y.ndim == 1:
        Y = Y[:, None]
    labels = labels or [f"C{i + 1}" for i in range(Y.shape[1])]
    order = np.argsort(S)[::-1]  # high first, so low-score outliers end up on top
    if Y.shape[1] >= 2:
        fig, ax = plt.subplots(figsize=(7.5, 6))
        sc = ax.scatter(Y[order, 0], Y[order, 1], c=S[order], cmap="viridis", s=4, alpha=0.7,
                        rasterized=True)
        fig.colorbar(sc, ax=ax, label="PWCS")
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
    else:
        fig, ax = plt.subplots(figsize=(7.5, 5))
        sc = ax.scatter(Y[order, 0], S[order], c=S[order], cmap="viridis", s=4, alpha=0.6,
                        rasterized=True)
        ax.set_xlabel(labels[0])
        ax.set_ylabel("PWCS")
        ax2 = ax.twinx()
        c, F = fes_1d(Y[:, 0], weights)
        ax2.plot(c, F, color="#c2533b", lw=1.5, label="free energy")
        ax2.set_ylabel("free energy (kT)", color="#c2533b")
        ax2.spines["right"].set_visible(True)
    if title:
        ax.set_title(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def plot_pwcs_hist(S, out_png, weights=None):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.hist(S, bins=100, color="#3b6ea8", alpha=0.75, label="per frame", density=True)
    if weights is not None:
        ax.hist(S, bins=100, weights=weights, histtype="step", color="#c2533b", lw=1.5,
                label="MSM weighted", density=True)
        ax.legend()
    ax.set_xlabel("PWCS")
    ax.set_ylabel("density")
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
