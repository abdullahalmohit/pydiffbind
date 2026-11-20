# pydiffbind

Differential binding analysis for ChIP-seq / ATAC-seq in Python, modelled on R/Bioconductor **DiffBind**.
Peaks + BAMs in, statistically differential sites out (DESeq2 via PyDESeq2).

## Install

```bash
pip install .            # from this directory
# or for development
pip install -e ".[dev]"
```

## Sample sheet (CSV)

| SampleID | Condition | Replicate | bamReads | Peaks | bamControl (optional) |
|---|---|---|---|---|---|
| t1 | treated | 1 | t1.bam | t1.narrowPeak | input1.bam |
| t2 | treated | 2 | t2.bam | t2.narrowPeak | input1.bam |
| c1 | control | 1 | c1.bam | c1.narrowPeak | input1.bam |
| c2 | control | 2 | c2.bam | c2.narrowPeak | input1.bam |

BAMs must be coordinate-sorted and indexed (`.bai`). Peak files: narrowPeak (summits used) or BED.
Extra columns (Tissue, Factor, batch...) can be used as covariates.

## Python API

```python
from pydiffbind import DBA

dba = DBA("samples.csv")
dba.count(summits=250, min_overlap=2, threads=4)   # consensus peaks + read counts
dba.normalize("rle")                                # 'rle' | 'lib' | 'tmm' | 'background'
dba.analyze("Condition", "treated", "control")      # log2FC = treated / control
sites = dba.report(fdr=0.05)                        # DataFrame of significant sites
dba.to_bed("gained.bed", fdr=0.05, direction="up")

dba.plot_pca(); dba.plot_correlation(); dba.plot_ma(); dba.plot_volcano()
```

Useful options of `count()`: `summits` (re-centre peaks, fixed width), `blacklist="blacklist.bed"`,
`subtract_control=True`, `background=True` (needed for `normalize("background")`, use when global
binding changes), `min_mapq`, `fragment_size` (single-end extension), `remove_dups`.
Add covariates (e.g. batch): `dba.analyze("Condition", "treated", "control", covariates=["Batch"])`.

## Command line

```bash
pydiffbind samples.csv --group1 treated --group2 control --summits 250 \
    --normalize rle --threads 4 -o results/
```
Writes `raw_counts.csv`, `normalized_counts.csv`, `all_results.csv`, `significant.csv` and PCA/correlation/MA/volcano PNGs.

## Notes / differences from DiffBind

- Counting uses pysam in pure Python; for very large datasets pre-count with `featureCounts` and
  extend `count.py`. Parallelism is per-BAM (`threads`).
- Paired-end BAMs are auto-detected and counted per fragment; single-end reads are extended to `fragment_size`.
- Control subtraction scales input counts by library size and floors at 1.
- Only DESeq2 is supported as the test engine. Not yet implemented: peak annotation, edgeR, Venn plots, blocking designs beyond covariates.

## Tests

```bash
pytest   # simulates BAMs with known differential sites and checks they are recovered
```
