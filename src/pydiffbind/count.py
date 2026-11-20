"""Read counting with pysam."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import pysam


def _is_paired(bam: pysam.AlignmentFile, n: int = 2000) -> bool:
    for i, r in enumerate(bam.fetch(until_eof=True)):
        if r.is_paired:
            return True
        if i >= n:
            break
    return False


def _count_intervals(
    bam_path: str,
    chroms: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    min_mapq: int,
    fragment_size: int,
    remove_dups: bool,
) -> Tuple[np.ndarray, int]:
    """Count fragments overlapping each interval. Returns (counts, total mapped reads)."""
    counts = np.zeros(len(starts), dtype=np.int64)
    with pysam.AlignmentFile(bam_path, "rb") as bam:
        paired = _is_paired(bam)
        refs = set(bam.references)
        lengths = dict(zip(bam.references, bam.lengths))
        pad = 1000 if paired else fragment_size
        for i, (c, s, e) in enumerate(zip(chroms, starts, ends)):
            if c not in refs:
                continue
            n = 0
            lo, hi = max(0, s - pad), min(lengths[c], e + pad)
            for r in bam.fetch(c, lo, hi):
                if r.is_unmapped or r.is_secondary or r.is_supplementary or r.is_qcfail:
                    continue
                if remove_dups and r.is_duplicate:
                    continue
                if r.mapping_quality < min_mapq:
                    continue
                if paired:
                    # count each proper pair once, via its leftmost mate
                    if not r.is_proper_pair or r.template_length <= 0:
                        continue
                    fs, fe = r.reference_start, r.reference_start + r.template_length
                else:
                    if r.is_reverse:
                        fe = r.reference_end
                        fs = fe - fragment_size
                    else:
                        fs = r.reference_start
                        fe = fs + fragment_size
                if fs < e and fe > s:
                    n += 1
            counts[i] = n
        total = int(bam.mapped)
    return counts, total


def _worker(args):
    return _count_intervals(*args)


def count_bam_set(
    bam_paths: Dict[str, str],
    regions: pd.DataFrame,
    min_mapq: int = 10,
    fragment_size: int = 200,
    remove_dups: bool = True,
    threads: int = 1,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Count reads in ``regions`` for every BAM. Returns (counts[region x sample], lib_sizes)."""
    chroms = regions["Chromosome"].to_numpy()
    starts = regions["Start"].to_numpy()
    ends = regions["End"].to_numpy()
    jobs = [(p, chroms, starts, ends, min_mapq, fragment_size, remove_dups) for p in bam_paths.values()]
    if threads > 1:
        with ProcessPoolExecutor(max_workers=threads) as ex:
            res = list(ex.map(_worker, jobs))
    else:
        res = [_worker(j) for j in jobs]
    counts = pd.DataFrame({k: r[0] for k, r in zip(bam_paths, res)}, index=regions["PeakID"].to_numpy())
    libs = pd.Series({k: r[1] for k, r in zip(bam_paths, res)}, name="LibSize")
    return counts, libs


def subtract_control(
    counts: pd.DataFrame,
    libs: pd.Series,
    ctrl_counts: pd.DataFrame,
    ctrl_libs: pd.Series,
    sample_to_ctrl: Dict[str, str],
) -> pd.DataFrame:
    """Subtract library-size-scaled control counts (floor of 1), like DiffBind's control subtraction."""
    out = counts.copy()
    for s, c in sample_to_ctrl.items():
        if c is None or c not in ctrl_counts.columns:
            continue
        scaled = ctrl_counts[c] * (libs[s] / ctrl_libs[c])
        out[s] = np.maximum(np.round(counts[s] - scaled), 1).astype(int)
    return out


def genome_bins(bam_path: str, bin_size: int = 15000) -> pd.DataFrame:
    """Tile the genome (as defined by a BAM header) in fixed-size bins."""
    rows = []
    with pysam.AlignmentFile(bam_path, "rb") as bam:
        for c, L in zip(bam.references, bam.lengths):
            for s in range(0, L, bin_size):
                rows.append((c, s, min(s + bin_size, L)))
    df = pd.DataFrame(rows, columns=["Chromosome", "Start", "End"])
    df["PeakID"] = df["Chromosome"] + ":" + df["Start"].astype(str) + "-" + df["End"].astype(str)
    return df
