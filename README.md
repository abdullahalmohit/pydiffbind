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

## Annotation (any species, user-supplied GTF/GFF3)

```python
dba.analyze("Condition", "FIE", "Control")
dba.annotate("TAIR10.gtf.gz", promoter=(-2000, 500))   # GTF or GFF3, plain or .gz
sites = dba.report(fdr=0.05)          # adds feature, overlapping_genes, nearest_gene, distance_to_tss
genes = dba.gene_lists(fdr=0.05)      # {'up': [...], 'down': [...], 'all': [...]}
bg    = dba.tested_genes()            # background universe for GO/KEGG enrichment
dba.save_gene_lists("results/")       # up_genes.txt, down_genes.txt, background_genes.txt
dba.plot_annotation(); dba.plot_tss_distance()
```

- Works with any species because the annotation comes from your file (human, mouse, Arabidopsis, rice...).
  Ensembl/GENCODE GTF, Ensembl Plants / TAIR / Araport11 / RAP-DB / MSU GFF3 are supported.
- Chromosome names are matched leniently (`chr1` = `Chr1` = `1` = `chr01`; `M`/`Mt`, `C`/`Pt`).
- `overlapping_genes` lists *every* gene whose body or promoter overlaps the region. Use this for broad
  marks (H3K27me3), where a domain spans several genes. `nearest_gene` is based on the nearest TSS.
- `feature` is the genomic feature at the region centre: Promoter > 5UTR > 3UTR > Exon > Intron > Downstream > Intergenic.
- Plant annotations often lack gene symbols: pass `symbols="id_to_symbol.tsv"` (two columns: gene_id, symbol).
- `biotypes=["protein_coding"]` restricts to selected gene biotypes.
- `distance_to_tss` is signed: negative = upstream of the TSS in the gene's orientation.

## Command line

```bash
pydiffbind samples.csv --group1 treated --group2 control --summits 250 \
    --normalize rle --threads 4 -o results/
```
With annotation:

```bash
pydiffbind samples.csv --group1 FIE --group2 Control --summits 0 --normalize background \
    --gtf TAIR10.gtf.gz --promoter-up 2000 --promoter-down 500 -o results/
```
Also writes gene lists (IDs and `names_` symbols), feature-distribution and TSS-distance plots.

Writes `raw_counts.csv`, `normalized_counts.csv`, `all_results.csv`, `significant.csv` and PCA/correlation/MA/volcano PNGs.

## Notes / differences from DiffBind

- Counting uses pysam in pure Python; for very large datasets pre-count with `featureCounts` and
  extend `count.py`. Parallelism is per-BAM (`threads`).
- Paired-end BAMs are auto-detected and counted per fragment; single-end reads are extended to `fragment_size`.
- Control subtraction scales input counts by library size and floors at 1.
- Only DESeq2 is supported as the test engine. Not yet implemented: edgeR, Venn plots, blocking designs beyond covariates.

## Tests

```bash
pytest   # simulates BAMs with known differential sites and checks they are recovered
```
