"""Peak reading and consensus peakset construction."""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd


def read_peaks(path: str, fmt: str = "bed") -> pd.DataFrame:
    """Read a peak file. Returns columns Chromosome, Start, End, Summit (absolute coord).

    For narrowPeak the summit is start + column 10; otherwise the interval midpoint.
    """
    df = pd.read_csv(path, sep="\t", header=None, comment="#", compression="infer")
    out = pd.DataFrame({"Chromosome": df[0].astype(str), "Start": df[1].astype(int), "End": df[2].astype(int)})
    if fmt.lower() == "narrowpeak" and df.shape[1] >= 10 and (df[9] >= 0).all():
        out["Summit"] = out["Start"] + df[9].astype(int)
    else:
        out["Summit"] = (out["Start"] + out["End"]) // 2
    return out


def build_consensus(
    peaksets: List[pd.DataFrame],
    min_overlap: int = 2,
    summits: Optional[int] = 200,
    chrom_sizes: Optional[dict] = None,
) -> pd.DataFrame:
    """Merge overlapping peaks across samples into a consensus peakset.

    Parameters
    ----------
    peaksets : one DataFrame per sample (from :func:`read_peaks`).
    min_overlap : keep merged regions supported by at least this many samples.
    summits : if an int, re-centre each region on the mean summit and use
        ``summit +/- summits`` (fixed width, as DiffBind does). If None/0 keep merged extents.
    chrom_sizes : optional dict used to clip regions at chromosome ends.
    """
    frames = []
    for i, ps in enumerate(peaksets):
        f = ps[["Chromosome", "Start", "End", "Summit"]].copy()
        f["sample"] = i
        frames.append(f)
    allp = pd.concat(frames, ignore_index=True).sort_values(["Chromosome", "Start", "End"], kind="mergesort")
    allp = allp.reset_index(drop=True)

    # cluster overlapping intervals: new cluster when start >= running max end within a chromosome
    prev_max_end = allp.groupby("Chromosome")["End"].transform(lambda s: s.cummax().shift(1))
    new = prev_max_end.isna() | (allp["Start"] >= prev_max_end)
    allp["cluster"] = new.cumsum()

    g = allp.groupby("cluster")
    reg = g.agg(
        Chromosome=("Chromosome", "first"),
        Start=("Start", "min"),
        End=("End", "max"),
        Summit=("Summit", "mean"),
        n_samples=("sample", "nunique"),
    )
    reg = reg[reg["n_samples"] >= min_overlap].copy()

    if summits:
        c = reg["Summit"].round().astype(int)
        reg["Start"] = (c - summits).clip(lower=0)
        reg["End"] = c + summits
    if chrom_sizes:
        sizes = reg["Chromosome"].map(chrom_sizes)
        reg["End"] = np.minimum(reg["End"], sizes.fillna(reg["End"])).astype(int)
    reg = reg.drop(columns="Summit").sort_values(["Chromosome", "Start"]).reset_index(drop=True)
    reg["PeakID"] = reg["Chromosome"] + ":" + reg["Start"].astype(str) + "-" + reg["End"].astype(str)
    return reg


def filter_blacklist(regions: pd.DataFrame, blacklist_bed: str) -> pd.DataFrame:
    """Remove regions overlapping any interval of a blacklist BED file."""
    bl = pd.read_csv(blacklist_bed, sep="\t", header=None, usecols=[0, 1, 2], names=["c", "s", "e"])
    bl["c"] = bl["c"].astype(str)
    keep = np.ones(len(regions), dtype=bool)
    for chrom, sub in regions.groupby("Chromosome"):
        b = bl[bl["c"] == chrom].sort_values("s")
        if b.empty:
            continue
        bs, be = b["s"].to_numpy(), b["e"].to_numpy()
        be_cummax = np.maximum.accumulate(be)
        # an interval i overlaps iff some blacklist start < end and the running max end > start
        idx = np.searchsorted(bs, sub["End"].to_numpy(), side="left")  # number of bl starts < End
        has = idx > 0
        maxend = np.where(has, be_cummax[np.clip(idx - 1, 0, None)], -1)
        overlap = has & (maxend > sub["Start"].to_numpy())
        keep[sub.index[overlap]] = False
    return regions[keep].reset_index(drop=True)
