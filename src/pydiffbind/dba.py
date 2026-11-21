"""Main user-facing class."""
from __future__ import annotations

import os
from typing import List, Optional, Union

import pandas as pd

from . import analyze, count as _count, normalize as _norm, peaks as _peaks, plots, samplesheet
from .annotate import Annotation


class DBA:
    """Differential binding analysis object (analogous to a DiffBind DBA object).

    Example
    -------
    >>> dba = DBA("samples.csv")
    >>> dba.count(summits=250, min_overlap=2, threads=4)
    >>> dba.normalize("rle")
    >>> dba.analyze("Condition", "treated", "control")
    >>> sites = dba.report(fdr=0.05)
    """

    def __init__(self, sheet: Union[str, pd.DataFrame], check_files: bool = True):
        self.sheet = samplesheet.load_samplesheet(sheet, check_files=check_files)
        self.meta = self.sheet.set_index("SampleID")
        self.peaksets: Optional[List[pd.DataFrame]] = None
        self.consensus: Optional[pd.DataFrame] = None
        self.counts: Optional[pd.DataFrame] = None
        self.libs: Optional[pd.Series] = None
        self.bin_counts: Optional[pd.DataFrame] = None
        self.size_factors: Optional[pd.Series] = None
        self.norm_method: Optional[str] = None
        self.results: Optional[pd.DataFrame] = None
        self.annotation: Optional[pd.DataFrame] = None
        self._ann: Optional[Annotation] = None
        self.contrast: Optional[tuple] = None
        self._covariates: List[str] = []

    def __repr__(self) -> str:
        s = f"DBA: {len(self.sheet)} samples"
        if self.consensus is not None:
            s += f", {len(self.consensus)} consensus sites"
        if self.size_factors is not None:
            s += f", normalized ({self.norm_method})"
        if self.annotation is not None:
            s += ", annotated"
        if self.results is not None:
            s += f", contrast {self.contrast}"
        return s

    # ------------------------------------------------------------------ peaks
    def load_peaks(self) -> None:
        self.peaksets = [_peaks.read_peaks(r.Peaks, r.PeakFormat) for r in self.sheet.itertuples()]

    def build_consensus(self, min_overlap: int = 2, summits: Optional[int] = 200,
                        blacklist: Optional[str] = None) -> pd.DataFrame:
        if self.peaksets is None:
            self.load_peaks()
        self.consensus = _peaks.build_consensus(self.peaksets, min_overlap=min_overlap, summits=summits)
        if blacklist:
            self.consensus = _peaks.filter_blacklist(self.consensus, blacklist)
        return self.consensus

    # ------------------------------------------------------------------ count
    def count(self, summits: Optional[int] = 200, min_overlap: int = 2, min_mapq: int = 10,
              fragment_size: int = 200, remove_dups: bool = True, blacklist: Optional[str] = None,
              subtract_control: bool = False, background: bool = False, bin_size: int = 15000,
              threads: int = 1) -> pd.DataFrame:
        """Build the consensus peakset and count reads in it for every sample.

        background=True additionally counts reads in genome-wide bins, which enables
        ``normalize('background')``. subtract_control=True uses the ``bamControl`` column.
        """
        self.build_consensus(min_overlap=min_overlap, summits=summits, blacklist=blacklist)
        if self.consensus.empty:
            raise ValueError("Consensus peakset is empty; lower min_overlap or check peak files.")
        kw = dict(min_mapq=min_mapq, fragment_size=fragment_size, remove_dups=remove_dups, threads=threads)
        bams = dict(zip(self.sheet["SampleID"], self.sheet["bamReads"]))
        self.counts, self.libs = _count.count_bam_set(bams, self.consensus, **kw)

        if subtract_control:
            ctrl = {s: c for s, c in zip(self.sheet["SampleID"], self.sheet["bamControl"]) if pd.notna(c)}
            if not ctrl:
                raise ValueError("subtract_control=True but no 'bamControl' values in the sample sheet.")
            uniq = {os.path.basename(p): p for p in ctrl.values()}
            cc, cl = _count.count_bam_set(uniq, self.consensus, **kw)
            self.counts = _count.subtract_control(
                self.counts, self.libs, cc, cl, {s: os.path.basename(c) for s, c in ctrl.items()})

        if background:
            bins = _count.genome_bins(self.sheet["bamReads"].iloc[0], bin_size)
            self.bin_counts, _ = _count.count_bam_set(bams, bins, **kw)
        self.size_factors = None
        self.results = None
        self.annotation = None
        return self.counts

    # -------------------------------------------------------------- normalize
    def normalize(self, method: str = "rle") -> pd.Series:
        """Compute size factors: 'rle' (DESeq2 default), 'lib', 'tmm' or 'background'."""
        self._require_counts()
        self.size_factors = _norm.compute_size_factors(method, self.counts, self.libs, self.bin_counts)
        self.norm_method = method
        return self.size_factors

    @property
    def norm_counts(self) -> pd.DataFrame:
        self._require_counts()
        if self.size_factors is None:
            self.normalize("rle")
        return self.counts / self.size_factors

    # ---------------------------------------------------------------- analyze
    def analyze(self, factor: str = "Condition", group1: Optional[str] = None, group2: Optional[str] = None,
                covariates: Optional[List[str]] = None, n_cpus: int = 1) -> pd.DataFrame:
        """Differential test group1 vs group2 on a sample-sheet column (log2FC = group1/group2)."""
        self._require_counts()
        if factor not in self.meta.columns:
            raise ValueError(f"Column {factor!r} not in sample sheet.")
        levels = list(pd.unique(self.meta[factor]))
        if group1 is None or group2 is None:
            if len(levels) != 2:
                raise ValueError(f"{factor} has {len(levels)} levels {levels}; specify group1 and group2.")
            group1, group2 = levels
        if self.size_factors is None:
            self.normalize("rle")
        self.results = analyze.run_deseq2(
            self.counts, self.meta, factor, group1, group2, covariates=covariates,
            size_factors=self.size_factors, n_cpus=n_cpus)
        self.contrast = (factor, group1, group2)
        self._covariates = covariates or []
        return self.results

    def report(self, fdr: float = 0.05, lfc: float = 0.0, direction: Optional[str] = None) -> pd.DataFrame:
        """Significant sites with coordinates. direction: None, 'up' (group1 higher) or 'down'."""
        if self.results is None:
            raise RuntimeError("Run analyze() first.")
        r = self.full_results()
        r = r[(r["padj"] < fdr) & (r["log2FoldChange"].abs() >= lfc)]
        if direction == "up":
            r = r[r["log2FoldChange"] > 0]
        elif direction == "down":
            r = r[r["log2FoldChange"] < 0]
        cols = ["Chromosome", "Start", "End"] + [c for c in r.columns if c not in ("Chromosome", "Start", "End")]
        return r[cols].sort_values("padj")

    def to_bed(self, path: str, **report_kw) -> None:
        r = self.report(**report_kw)
        r[["Chromosome", "Start", "End"]].assign(name=r.index, score=r["log2FoldChange"].round(3)) \
            .to_csv(path, sep="\t", header=False, index=False)


    # ------------------------------------------------------------- annotation
    def annotate(self, gtf: Union[str, Annotation], promoter=(-2000, 500), downstream: int = 1000,
                 symbols: Optional[str] = None, biotypes: Optional[List[str]] = None,
                 id_attr: Optional[str] = None, name_attr: Optional[str] = None,
                 chrom_map: Optional[str] = None) -> pd.DataFrame:
        """Annotate all consensus regions using a user-supplied GTF/GFF3 (any species).

        promoter : (upstream, downstream) of the TSS, e.g. (-2000, 500).
        symbols : optional 2-column file (gene_id, symbol), useful for plants.
        biotypes : optional filter, e.g. ["protein_coding"].
        id_attr / name_attr : GTF attributes for gene ID / symbol (NCBI RefSeq: id_attr="locus_tag").
        chrom_map : optional 2-column file mapping annotation chromosome names to BAM names.
        Adds genes, nearest gene, distance to TSS and feature type to ``report()``.
        """
        if self.consensus is None:
            raise RuntimeError("Run count() first.")
        self._ann = gtf if isinstance(gtf, Annotation) else Annotation.from_file(gtf, symbols, biotypes, id_attr, name_attr, chrom_map)
        self.annotation = self._ann.annotate(self.consensus, promoter=promoter, downstream=downstream)
        return self.annotation

    def full_results(self) -> pd.DataFrame:
        """All tested regions: statistics + coordinates (+ annotation if available)."""
        self._require_results()
        r = self.results.join(self.consensus.set_index("PeakID")[["Chromosome", "Start", "End"]])
        if self.annotation is not None:
            r = r.join(self.annotation)
        return r

    @staticmethod
    def _split_genes(values) -> List[str]:
        out = set()
        for v in values:
            if isinstance(v, str) and v:
                out.update(x for x in v.split(";") if x)
        return sorted(out)

    def _gene_col(self, by: str, names: bool) -> str:
        if self.annotation is None:
            raise RuntimeError("Run annotate(gtf=...) first.")
        if by not in ("overlap", "nearest"):
            raise ValueError("by must be 'overlap' or 'nearest'")
        return {("overlap", False): "overlapping_genes", ("overlap", True): "overlapping_gene_names",
                ("nearest", False): "nearest_gene", ("nearest", True): "nearest_gene_name"}[(by, bool(names))]

    def gene_lists(self, fdr: float = 0.05, lfc: float = 0.0, by: str = "overlap",
                   names: bool = False) -> dict:
        """Gene lists for enrichment: {'up': [...], 'down': [...], 'all': [...]}.

        by='overlap' (default; best for broad marks like H3K27me3) uses every gene whose body or
        promoter overlaps a differential region; by='nearest' uses the nearest TSS only.
        names=True returns symbols instead of gene IDs.
        """
        col = self._gene_col(by, names)
        rep = self.report(fdr=fdr, lfc=lfc)
        up = self._split_genes(rep.loc[rep["log2FoldChange"] > 0, col])
        down = self._split_genes(rep.loc[rep["log2FoldChange"] < 0, col])
        return {"up": up, "down": down, "all": sorted(set(up) | set(down))}

    def tested_genes(self, by: str = "overlap", names: bool = False) -> List[str]:
        """Background gene universe for enrichment (genes touched by any tested region)."""
        col = self._gene_col(by, names)
        tested = self.annotation.loc[self.results.index, col]
        return self._split_genes(tested)

    def save_gene_lists(self, outdir: str, prefix: str = "", **kw) -> None:
        """Write up/down/background gene lists as one-per-line text files."""
        os.makedirs(outdir, exist_ok=True)
        gl = self.gene_lists(**kw)
        by, names = kw.get("by", "overlap"), kw.get("names", False)
        for k, fname in [("up", "up_genes.txt"), ("down", "down_genes.txt")]:
            with open(os.path.join(outdir, prefix + fname), "w") as f:
                f.write("\n".join(gl[k]) + ("\n" if gl[k] else ""))
        bg = self.tested_genes(by=by, names=names)
        with open(os.path.join(outdir, prefix + "background_genes.txt"), "w") as f:
            f.write("\n".join(bg) + ("\n" if bg else ""))

    def plot_annotation(self, fdr: float = 0.05, lfc: float = 0.0, ax=None):
        """Genomic feature distribution of gained / lost / all tested regions."""
        rep = self.report(fdr=fdr, lfc=lfc)
        return plots.plot_feature_distribution(
            rep[rep["log2FoldChange"] > 0], rep[rep["log2FoldChange"] < 0],
            self.annotation.loc[self.results.index], ax)

    def plot_tss_distance(self, fdr: float = 0.05, lfc: float = 0.0, ax=None):
        rep = self.report(fdr=fdr, lfc=lfc)
        return plots.plot_tss_distance(rep[rep["log2FoldChange"] > 0], rep[rep["log2FoldChange"] < 0], ax=ax)

    # ------------------------------------------------------------------ plots
    def plot_pca(self, color_by: str = "Condition", ax=None):
        return plots.plot_pca(self.norm_counts, self.meta, color_by, ax)

    def plot_correlation(self, ax=None):
        return plots.plot_correlation(self.norm_counts, ax)

    def plot_ma(self, fdr: float = 0.05, lfc: float = 0.0, ax=None):
        self._require_results()
        return plots.plot_ma(self.results, fdr, lfc, ax)

    def plot_volcano(self, fdr: float = 0.05, lfc: float = 0.0, ax=None):
        self._require_results()
        return plots.plot_volcano(self.results, fdr, lfc, ax)

    # --------------------------------------------------------------- helpers
    def _require_counts(self):
        if self.counts is None:
            raise RuntimeError("Run count() first.")

    def _require_results(self):
        if self.results is None:
            raise RuntimeError("Run analyze() first.")
