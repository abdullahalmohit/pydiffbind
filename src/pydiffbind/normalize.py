"""Normalization: compute per-sample size factors."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _geo_center(sf: np.ndarray) -> np.ndarray:
    return sf / np.exp(np.mean(np.log(sf)))


def size_factors_lib(libs: pd.Series) -> pd.Series:
    """Library-size normalization."""
    return pd.Series(_geo_center(libs.to_numpy(float)), index=libs.index)


def size_factors_rle(counts: pd.DataFrame) -> pd.Series:
    """DESeq2 median-of-ratios (RLE) on a region x sample matrix."""
    x = counts.to_numpy(float)
    keep = (x > 0).all(axis=1)
    if keep.sum() < 10:
        raise ValueError("Too few regions with non-zero counts in all samples for RLE normalization.")
    logx = np.log(x[keep])
    ref = logx.mean(axis=1, keepdims=True)
    sf = np.exp(np.median(logx - ref, axis=0))
    return pd.Series(_geo_center(sf), index=counts.columns)


def size_factors_tmm(counts: pd.DataFrame, libs: pd.Series | None = None,
                     logratio_trim: float = 0.3, sum_trim: float = 0.05) -> pd.Series:
    """Trimmed mean of M-values (Robinson & Oshlack 2010), reference = sample closest to upper quartile."""
    x = counts.to_numpy(float)
    lib = x.sum(axis=0) if libs is None else libs.to_numpy(float)
    uq = np.array([np.quantile(c[c > 0], 0.75) / l if (c > 0).any() else 0 for c, l in zip(x.T, lib)])
    ref = int(np.argmin(np.abs(uq - np.mean(uq))))
    f = np.ones(x.shape[1])
    for k in range(x.shape[1]):
        if k == ref:
            continue
        o, r = x[:, k], x[:, ref]
        ok = (o > 0) & (r > 0)
        o, r = o[ok], r[ok]
        no, nr = lib[k], lib[ref]
        m = np.log2((o / no) / (r / nr))
        a = 0.5 * np.log2((o / no) * (r / nr))
        ml = (m >= np.quantile(m, logratio_trim)) & (m <= np.quantile(m, 1 - logratio_trim))
        al = (a >= np.quantile(a, sum_trim)) & (a <= np.quantile(a, 1 - sum_trim))
        keep = ml & al
        w = (no - o) / no / o + (nr - r) / nr / r
        f[k] = 2 ** (np.sum(m[keep] / w[keep]) / np.sum(1 / w[keep]))
    f = f / np.exp(np.mean(np.log(f)))
    # TMM factors adjust library sizes: effective lib = lib * f; size factor = lib * f / mean
    return pd.Series(_geo_center(lib * f), index=counts.columns)


def compute_size_factors(method: str, counts: pd.DataFrame, libs: pd.Series,
                         bin_counts: pd.DataFrame | None = None,
                         bin_libs: pd.Series | None = None) -> pd.Series:
    """Dispatch on method: 'lib', 'rle', 'tmm', 'background' (RLE on genome-wide bins)."""
    method = method.lower()
    if method == "lib":
        return size_factors_lib(libs)
    if method == "rle":
        return size_factors_rle(counts)
    if method == "tmm":
        return size_factors_tmm(counts)
    if method == "background":
        if bin_counts is None:
            raise ValueError("Background normalization needs bin counts; call dba.count(background=True).")
        return size_factors_rle(bin_counts + 1)
    raise ValueError(f"Unknown normalization method: {method!r} (use lib, rle, tmm, background)")
