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


_FEATURE_COLORS = {"Promoter": "#d62728", "5UTR": "#ff7f0e", "3UTR": "#ffbb78", "UTR": "#f0a030",
                   "Exon": "#2ca02c", "Intron": "#1f77b4", "Downstream": "#9467bd",
                   "Intergenic": "#bdbdbd", "Unmatched_chrom": "#555555"}


def plot_feature_distribution(up: pd.DataFrame, down: pd.DataFrame, all_tested: pd.DataFrame, ax=None):
    """Stacked horizontal bars: % of regions per genomic feature for Up / Down / All tested."""
    groups = {f"Up (n={len(up)})": up, f"Down (n={len(down)})": down,
              f"All tested (n={len(all_tested)})": all_tested}
    groups = {k: v for k, v in groups.items() if len(v)}
    pct = pd.DataFrame({k: v["feature"].value_counts(normalize=True) * 100 for k, v in groups.items()}).fillna(0)
    ax = ax or plt.subplots(figsize=(7, 2.8 + 0.2 * len(groups)))[1]
    left = np.zeros(len(pct.columns))
    for feat in [f for f in list(_FEATURE_COLORS) if f in pct.index]:
        ax.barh(pct.columns, pct.loc[feat].to_numpy(), left=left, color=_FEATURE_COLORS[feat], label=feat)
        left += pct.loc[feat].to_numpy()
    ax.set_xlim(0, 100); ax.invert_yaxis()
    ax.set_xlabel("% of regions (feature at region centre)")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False, fontsize=8)
    ax.set_title("Genomic feature distribution")
    return ax


def plot_tss_distance(up: pd.DataFrame, down: pd.DataFrame, window: int = 50000, bins: int = 50, ax=None):
    """Histogram of distance to the nearest TSS (negative = upstream) for gained (red) and lost (blue)."""
    ax = ax or plt.subplots(figsize=(5.5, 4))[1]
    edges = np.linspace(-window, window, bins + 1)
    for df, c, lab in [(up, "red", "Up"), (down, "royalblue", "Down")]:
        if len(df):
            d = df["distance_to_tss"].dropna()
            ax.hist(d[d.abs() <= window], bins=edges, alpha=.6, color=c, label=f"{lab} (n={len(df)})")
    ax.axvline(0, c="k", lw=.5)
    ax.set_xlabel("Distance to nearest TSS (bp)"); ax.set_ylabel("Regions")
    ax.set_title(f"Distance to TSS (within ±{window // 1000} kb)")
    ax.legend(frameon=False)
    return ax
