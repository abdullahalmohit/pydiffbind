"""Differential analysis via PyDESeq2."""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd


def run_deseq2(
    counts: pd.DataFrame,
    meta: pd.DataFrame,
    factor: str,
    group1: str,
    group2: str,
    covariates: Optional[List[str]] = None,
    size_factors: Optional[pd.Series] = None,
    min_count: int = 1,
    n_cpus: int = 1,
) -> pd.DataFrame:
    """Fit a negative-binomial model and test ``group1`` vs ``group2`` (log2FC = group1 / group2).

    counts : region x sample integer matrix.  meta : sample-indexed DataFrame.
    size_factors : precomputed factors (sample-indexed) to override DESeq2's own normalization.
    """
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats

    covariates = covariates or []
    samples = meta.index[meta[factor].isin([group1, group2])]
    if (meta.loc[samples, factor] == group1).sum() < 2 or (meta.loc[samples, factor] == group2).sum() < 2:
        raise ValueError("Each group needs at least 2 replicates for a differential test.")

    c = counts[samples]
    c = c[c.sum(axis=1) >= min_count]            # drop empty regions
    design = "~" + "+".join(covariates + [factor])
    m = meta.loc[samples, [factor] + covariates].copy()

    dds = DeseqDataSet(counts=c.T.astype(int), metadata=m, design=design, quiet=True, n_cpus=n_cpus)
    if size_factors is None:
        dds.deseq2()
    else:
        sf = size_factors.loc[samples].to_numpy(float)
        sf = sf / np.exp(np.mean(np.log(sf)))
        dds.obs["size_factors"] = sf
        dds.layers["normed_counts"] = dds.X / sf[:, None]
        dds.var["_normed_means"] = dds.layers["normed_counts"].mean(axis=0)
        dds.fit_genewise_dispersions()
        dds.fit_dispersion_trend()
        dds.fit_dispersion_prior()
        dds.fit_MAP_dispersions()
        dds.fit_LFC()
        dds.calculate_cooks()
        if dds.refit_cooks:
            dds.refit()

    stats = DeseqStats(dds, contrast=[factor, group1, group2], quiet=True)
    stats.summary()
    res = stats.results_df.copy()

    norm = pd.DataFrame(dds.layers["normed_counts"], index=dds.obs_names, columns=dds.var_names)
    res[f"Conc_{group1}"] = norm.loc[m.index[m[factor] == group1]].mean(axis=0)
    res[f"Conc_{group2}"] = norm.loc[m.index[m[factor] == group2]].mean(axis=0)
    res.index.name = "PeakID"
    return res
