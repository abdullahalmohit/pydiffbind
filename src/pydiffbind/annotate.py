"""Annotate genomic regions with genes and genomic features from a user-supplied GTF or GFF3.

Works for any species (human, mouse, Arabidopsis, rice, ...) because everything comes from the
annotation file the user provides. Coordinates: regions are 0-based half-open (BED-like);
GTF/GFF3 are converted from 1-based inclusive on load.
"""
from __future__ import annotations

import re
import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

FEATURE_ORDER = ["Promoter", "5UTR", "3UTR", "UTR", "Exon", "Intron", "Downstream", "Intergenic"]

_COLS = ["chrom", "source", "feature", "start", "end", "score", "strand", "phase", "attr"]
_UTR5 = {"five_prime_utr", "5utr", "five_prime_utr"}
_UTR3 = {"three_prime_utr", "3utr"}
_UTR = {"utr"}
_DROP = {"cds", "start_codon", "stop_codon", "selenocysteine"}


def _chrom_key(c) -> str:
    """Normalize chromosome names so 'chr1', 'Chr1', '1' and 'CHR01' compare equal-ish; M/Mt, C/Pt unified."""
    s = str(c).lower()
    s = re.sub(r"^(chromosome|chr)[_]?", "", s)
    s = re.sub(r"^0+(\d)", r"\1", s)   # chr01 -> 1
    return {"m": "mt", "c": "pt", "pltd": "pt"}.get(s, s)


def _attr(series: pd.Series, key: str, gtf: bool) -> pd.Series:
    pat = rf'(?:^|;)\s*{key} "([^"]*)"' if gtf else rf"(?:^|;)\s*{key}=([^;]*)"
    return series.str.extract(pat, expand=False)


def _utr_kind(feature_lower: pd.Series) -> pd.Series:
    kind = pd.Series(np.nan, index=feature_lower.index, dtype=object)
    kind[feature_lower.isin(_UTR5)] = "5UTR"
    kind[feature_lower.isin(_UTR3)] = "3UTR"
    kind[feature_lower.isin(_UTR)] = "UTR"
    return kind


def _union(starts: np.ndarray, ends: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Merge intervals into sorted, non-overlapping ones."""
    if len(starts) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    o = np.argsort(starts, kind="mergesort")
    s, e = starts[o], ends[o]
    cm = np.maximum.accumulate(e)
    new = np.ones(len(s), dtype=bool)
    new[1:] = s[1:] > cm[:-1]
    first = np.nonzero(new)[0]
    last = np.r_[first[1:] - 1, len(s) - 1]
    return s[first], cm[last]


def _in_union(pos: np.ndarray, us: np.ndarray, ue: np.ndarray) -> np.ndarray:
    if len(us) == 0:
        return np.zeros(len(pos), dtype=bool)
    i = np.searchsorted(us, pos, side="right") - 1
    return (i >= 0) & (pos < ue[np.clip(i, 0, None)])


class Annotation:
    """Gene models parsed from a GTF/GFF3 file.

    Attributes: ``genes`` (gene_id, gene_name, biotype, chrom, start, end, strand),
    ``exons`` and ``utrs`` (chrom, start, end, gene_id [, kind]).
    """

    def __init__(self, genes: pd.DataFrame, exons: pd.DataFrame, utrs: pd.DataFrame):
        self.genes = genes.reset_index(drop=True)
        self.exons = exons.reset_index(drop=True)
        self.utrs = utrs.reset_index(drop=True)

    def __repr__(self) -> str:
        return f"Annotation: {len(self.genes)} genes, {len(self.exons)} exons, {len(self.utrs)} UTR segments"

    # ------------------------------------------------------------------ loading
    @classmethod
    def from_file(cls, path: str, symbols: Optional[str] = None,
                  biotypes: Optional[Sequence[str]] = None) -> "Annotation":
        df = pd.read_csv(path, sep="\t", header=None, comment="#", names=_COLS,
                         usecols=[0, 2, 3, 4, 6, 8], dtype={"chrom": str, "attr": str},
                         quoting=3, compression="infer")
        if df.empty:
            raise ValueError(f"No annotation rows read from {path}")
        gtf = bool(re.search(r'\w+ "', df["attr"].iloc[0]))
        df["f"] = df["feature"].str.lower()
        df["start"] = df["start"].astype(np.int64) - 1          # -> 0-based
        df["end"] = df["end"].astype(np.int64)
        df["strand"] = df["strand"].where(df["strand"].isin(["+", "-"]), "+")

        genes, exons, utrs = (_parse_gtf if gtf else _parse_gff3)(df)
        genes = genes.drop_duplicates("gene_id")
        genes["gene_name"] = genes["gene_name"].fillna(genes["gene_id"])

        if symbols:
            m = pd.read_csv(symbols, sep=None, engine="python", header=None, usecols=[0, 1], names=["id", "sym"])
            m = m[~m["id"].astype(str).str.lower().isin(["gene_id", "gene", "id", "geneid"])]
            sym = dict(zip(m["id"].astype(str), m["sym"].astype(str)))
            genes["gene_name"] = genes["gene_id"].map(sym).fillna(genes["gene_name"])
        if biotypes:
            keep = genes["biotype"].isin(biotypes)
            genes = genes[keep]
            exons = exons[exons["gene_id"].isin(genes["gene_id"])]
            utrs = utrs[utrs["gene_id"].isin(genes["gene_id"])]
        if genes.empty:
            raise ValueError("No genes found in the annotation file (check the file and biotype filter).")
        return cls(genes, exons, utrs)

    # --------------------------------------------------------------- annotating
    def annotate(self, regions: pd.DataFrame, promoter: Tuple[int, int] = (-2000, 500),
                 downstream: int = 1000) -> pd.DataFrame:
        """Annotate regions (needs Chromosome, Start, End, PeakID).

        promoter : (upstream, downstream) offsets relative to the TSS, e.g. (-2000, 500).
        Returns a DataFrame indexed by PeakID with: feature (at the region centre),
        overlapping_genes / overlapping_gene_names (every gene whose body or promoter touches the
        region; best for broad domains), nearest_gene*, distance_to_tss (negative = upstream).
        """
        up, down = -int(promoter[0]), int(promoter[1])
        n = len(regions)
        g_all = self.genes
        chrom_map = {_chrom_key(c): c for c in g_all["chrom"].unique()}
        rc = regions["Chromosome"].map(lambda c: chrom_map.get(_chrom_key(c))).to_numpy(object)
        matched = np.array([c is not None for c in rc])
        if not matched.any():
            raise ValueError(
                "No chromosome names in the regions match the annotation. "
                f"Regions look like {list(regions['Chromosome'].unique()[:3])}, "
                f"annotation like {list(g_all['chrom'].unique()[:3])}.")
        if not matched.all():
            bad = sorted(set(regions["Chromosome"][~matched]))
            warnings.warn(f"{(~matched).sum()} regions on chromosomes absent from the annotation: {bad[:5]}")

        S = regions["Start"].to_numpy(np.int64)
        E = regions["End"].to_numpy(np.int64)
        feature = np.array(["Unmatched_chrom" if not m else "Intergenic" for m in matched], dtype=object)
        ov_ids = np.array([""] * n, dtype=object)
        ov_names = np.array([""] * n, dtype=object)
        n_ov = np.zeros(n, dtype=int)
        nn_id = np.array([""] * n, dtype=object)
        nn_name = np.array([""] * n, dtype=object)
        nn_bio = np.array([""] * n, dtype=object)
        nn_str = np.array([""] * n, dtype=object)
        dist = np.full(n, np.nan)

        for chrom in {c for c in rc if c is not None}:
            pos = np.nonzero(rc == chrom)[0]
            g = g_all[g_all["chrom"] == chrom]
            gs, ge = g["start"].to_numpy(), g["end"].to_numpy()
            minus = (g["strand"] == "-").to_numpy()
            tss = np.where(minus, ge - 1, gs)
            ps = np.where(minus, tss - down, tss - up).clip(min=0)
            pe = np.where(minus, tss + up + 1, tss + down + 1)
            ds_s = np.where(minus, gs - downstream, ge).clip(min=0)
            ds_e = np.where(minus, gs, ge + downstream)
            ext_s = np.minimum(gs, ps)
            ext_e = np.maximum(ge, pe)
            ids, names = g["gene_id"].to_numpy(), g["gene_name"].to_numpy()
            bios, strands = g["biotype"].fillna("").to_numpy(), g["strand"].to_numpy()

            s, e = S[pos], E[pos]
            center = (s + e) // 2

            # feature at region centre, lowest priority first so higher priorities overwrite
            f = np.array(["Intergenic"] * len(pos), dtype=object)
            f[_in_union(center, *_union(ds_s, ds_e))] = "Downstream"
            f[_in_union(center, *_union(gs, ge))] = "Intron"
            ex = self.exons[self.exons["chrom"] == chrom]
            f[_in_union(center, *_union(ex["start"].to_numpy(), ex["end"].to_numpy()))] = "Exon"
            ut = self.utrs[self.utrs["chrom"] == chrom]
            for kind in ["UTR", "3UTR", "5UTR"]:
                u = ut[ut["kind"] == kind]
                f[_in_union(center, *_union(u["start"].to_numpy(), u["end"].to_numpy()))] = kind
            f[_in_union(center, *_union(ps, pe))] = "Promoter"
            feature[pos] = f

            # all genes whose body/promoter overlaps the region (chunked broadcasting)
            for a in range(0, len(pos), 1000):
                sl = slice(a, a + 1000)
                m = (ext_s[None, :] < e[sl, None]) & (ext_e[None, :] > s[sl, None])
                for k, row in zip(pos[sl], m):
                    hit = np.nonzero(row)[0]
                    if len(hit):
                        n_ov[k] = len(hit)
                        ov_ids[k] = ";".join(ids[hit])
                        ov_names[k] = ";".join(names[hit])

            # nearest TSS
            order = np.argsort(tss, kind="mergesort")
            ts = tss[order]
            i = np.searchsorted(ts, center)
            lo, hi = np.clip(i - 1, 0, len(ts) - 1), np.clip(i, 0, len(ts) - 1)
            pick = np.where(np.abs(ts[lo] - center) <= np.abs(ts[hi] - center), lo, hi)
            gi = order[pick]
            t = tss[gi]
            raw = np.where(t < s, s - t, np.where(t >= e, (e - 1) - t, 0))
            dist[pos] = raw * np.where(strands[gi] == "-", -1, 1)
            nn_id[pos], nn_name[pos], nn_bio[pos], nn_str[pos] = ids[gi], names[gi], bios[gi], strands[gi]

        return pd.DataFrame({
            "feature": feature, "overlapping_genes": ov_ids, "overlapping_gene_names": ov_names,
            "n_overlapping_genes": n_ov, "nearest_gene": nn_id, "nearest_gene_name": nn_name,
            "nearest_gene_biotype": nn_bio, "nearest_gene_strand": nn_str, "distance_to_tss": dist,
        }, index=regions["PeakID"].to_numpy())


# ---------------------------------------------------------------------- parsers
def _parse_gtf(df: pd.DataFrame):
    a = df["attr"]
    rows = df[df["f"].isin({"gene", "exon"} | _UTR5 | _UTR3 | _UTR)].copy()
    rows["gene_id"] = _attr(rows["attr"], "gene_id", True)
    rows = rows.dropna(subset=["gene_id"])

    g = rows[rows["f"] == "gene"].copy()
    if g.empty:   # GTF without gene lines: derive from exons/UTRs
        src = rows.copy()
        agg = src.groupby("gene_id").agg(chrom=("chrom", "first"), start=("start", "min"),
                                         end=("end", "max"), strand=("strand", "first")).reset_index()
        first = src.drop_duplicates("gene_id").set_index("gene_id")["attr"]
        agg["_attr"] = agg["gene_id"].map(first)
        g = agg
        g["attr"] = g["_attr"]
    g["gene_name"] = _attr(g["attr"], "gene_name", True)
    bio = _attr(g["attr"], "gene_biotype", True)
    for k in ["gene_type", "biotype"]:
        bio = bio.fillna(_attr(g["attr"], k, True))
    g["biotype"] = bio
    genes = g[["gene_id", "gene_name", "biotype", "chrom", "start", "end", "strand"]]

    exons = rows.loc[rows["f"] == "exon", ["chrom", "start", "end", "gene_id"]]
    u = rows[rows["f"].isin(_UTR5 | _UTR3 | _UTR)].copy()
    u["kind"] = _utr_kind(u["f"])
    return genes, exons, u[["chrom", "start", "end", "gene_id", "kind"]]


def _parse_gff3(df: pd.DataFrame):
    d = df[~df["f"].isin(_DROP)].copy()
    d["ID"] = _attr(d["attr"], "ID", False)
    d["Parent"] = _attr(d["attr"], "Parent", False)
    strip_g = lambda s: s.str.replace(r"^gene:", "", regex=True)
    strip_t = lambda s: s.str.replace(r"^transcript:", "", regex=True)

    is_gene = d["f"].str.endswith("gene")
    g = d[is_gene].copy()
    g["gene_id"] = strip_g(g["ID"])
    g = g.dropna(subset=["gene_id"])
    name = _attr(g["attr"], "Name", False)
    for k in ["gene_name", "symbol"]:
        name = name.fillna(_attr(g["attr"], k, False))
    g["gene_name"] = name
    bio = _attr(g["attr"], "biotype", False)
    for k in ["gene_biotype", "gene_type"]:
        bio = bio.fillna(_attr(g["attr"], k, False))
    g["biotype"] = bio
    genes = g[["gene_id", "gene_name", "biotype", "chrom", "start", "end", "strand"]]
    gene_set = set(genes["gene_id"])

    is_exon = d["f"] == "exon"
    is_utr = d["f"].isin(_UTR5 | _UTR3 | _UTR)
    t = d[~is_gene & ~is_exon & ~is_utr & d["Parent"].notna() & d["ID"].notna()]
    t_parent = strip_g(t["Parent"].str.split(",").str[0])
    t = t[t_parent.isin(gene_set)]
    tx2gene = dict(zip(strip_t(t["ID"]), strip_g(t["Parent"].str.split(",").str[0])))

    def to_gene(parent: pd.Series) -> pd.Series:
        p = strip_t(parent.str.split(",").str[0])
        gid = p.map(tx2gene)
        return gid.fillna(p.where(strip_g(p).isin(gene_set)).pipe(strip_g))

    ex = d[is_exon].copy()
    ex["gene_id"] = to_gene(ex["Parent"].fillna(""))
    ex = ex.dropna(subset=["gene_id"])
    ut = d[is_utr].copy()
    ut["gene_id"] = to_gene(ut["Parent"].fillna(""))
    ut = ut.dropna(subset=["gene_id"])
    ut["kind"] = _utr_kind(ut["f"])
    return genes, ex[["chrom", "start", "end", "gene_id"]], ut[["chrom", "start", "end", "gene_id", "kind"]]
