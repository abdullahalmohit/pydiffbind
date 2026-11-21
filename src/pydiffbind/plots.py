"""Plotting helpers (matplotlib)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _log_norm(norm_counts: pd.DataFrame) -> pd.DataFrame:
    return np.log2(norm_counts + 1)


def plot_pca(norm_counts: pd.DataFrame, meta: pd.DataFrame, color_by: str = "Condition", ax=None):
    from sklearn.decomposition import PCA
    x = _log_norm(norm_counts).T
    x = x.loc[:, x.var() > 0]
    pca = PCA(n_components=2).fit(x)
    pc = pca.transform(x)
    ax = ax or plt.subplots(figsize=(5, 4.5))[1]
    groups = meta.loc[x.index, color_by].astype(str)
    for g in sorted(groups.unique()):
        idx = (groups == g).to_numpy()
        ax.scatter(pc[idx, 0], pc[idx, 1], label=g, s=60)
    for name, (a, b) in zip(x.index, pc):
        ax.annotate(name, (a, b), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%})")
    ax.legend(title=color_by)
    ax.set_title("PCA of normalized counts")
    return ax


def plot_correlation(norm_counts: pd.DataFrame, ax=None):
    corr = _log_norm(norm_counts).corr()
    ax = ax or plt.subplots(figsize=(5.5, 4.5))[1]
    im = ax.imshow(corr.values, cmap="viridis")
    ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.index, fontsize=7)
    ax.figure.colorbar(im, ax=ax, label="Pearson r")
    ax.set_title("Sample correlation")
    return ax


def _scatter_updown(ax, x, y, up, down, other):
    """Grey = not significant, red = up (higher in group1), blue = down (lower in group1)."""
    ax.scatter(x[other], y[other], s=4, c="lightgrey", alpha=.6, label="Not significant", rasterized=True)
    ax.scatter(x[down], y[down], s=7, c="royalblue", label=f"Down ({down.sum()})", rasterized=True)
    ax.scatter(x[up], y[up], s=7, c="red", label=f"Up ({up.sum()})", rasterized=True)


def plot_ma(res: pd.DataFrame, fdr: float = 0.05, lfc: float = 0.0, ax=None):
    r = res.dropna(subset=["padj"])
    sig = (r["padj"] < fdr) & (r["log2FoldChange"].abs() >= lfc)
    up, down = sig & (r["log2FoldChange"] > 0), sig & (r["log2FoldChange"] < 0)
    ax = ax or plt.subplots(figsize=(5.5, 4.5))[1]
    x = np.log10(r["baseMean"] + 1)
    _scatter_updown(ax, x, r["log2FoldChange"], up, down, ~sig)
    ax.axhline(0, c="k", lw=.5)
    ax.set_xlabel("log10(mean normalized count + 1)"); ax.set_ylabel("log2 fold change")
    ax.set_title(f"MA plot (FDR<{fdr})")
    ax.legend(markerscale=2, frameon=False, fontsize=8)
    return ax


def plot_volcano(res: pd.DataFrame, fdr: float = 0.05, lfc: float = 0.0, ax=None):
    r = res.dropna(subset=["padj"])
    sig = (r["padj"] < fdr) & (r["log2FoldChange"].abs() >= lfc)
    up, down = sig & (r["log2FoldChange"] > 0), sig & (r["log2FoldChange"] < 0)
    ax = ax or plt.subplots(figsize=(5.5, 4.5))[1]
    y = -np.log10(r["pvalue"].clip(lower=1e-300))
    _scatter_updown(ax, r["log2FoldChange"], y, up, down, ~sig)
    ax.axvline(0, c="k", lw=.5)
    ax.set_xlabel("log2 fold change"); ax.set_ylabel("-log10(p)")
    ax.set_title(f"Volcano plot (FDR<{fdr}" + (f", |log2FC|>={lfc})" if lfc else ")"))
    ax.legend(markerscale=2, frameon=False, fontsize=8)
    return ax
