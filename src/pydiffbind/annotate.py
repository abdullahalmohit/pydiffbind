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

    def __init__(self, genes: pd.DataFrame, exons: pd.DataFrame, utrs: pd.DataFrame,
                 cds: Optional[pd.DataFrame] = None):
        self.genes = genes.reset_index(drop=True)
        self.exons = exons.reset_index(drop=True)
        self.utrs = utrs.reset_index(drop=True)
        self.cds = (cds if cds is not None else
                    pd.DataFrame(columns=["chrom", "start", "end", "gene_id", "tx_id", "strand"])).reset_index(drop=True)
        self._models_cache = None

    def __repr__(self) -> str:
        return f"Annotation: {len(self.genes)} genes, {len(self.exons)} exons, {len(self.utrs)} UTR segments"

    # ------------------------------------------------------------------ loading
    @classmethod
    def from_file(cls, path: str, symbols: Optional[str] = None,
                  biotypes: Optional[Sequence[str]] = None, id_attr: Optional[str] = None,
                  name_attr: Optional[str] = None, chrom_map: Optional[str] = None) -> "Annotation":
        """Parse a GTF/GFF3 (plain or .gz).

        id_attr   : attribute to use as the gene ID (e.g. 'locus_tag' for NCBI RefSeq to get AT1G01010
                    instead of the symbol). Default: gene_id (GTF) / ID (GFF3).
        name_attr : attribute to use as the gene name/symbol. Default: gene_name, gene or Name.
        chrom_map : optional 2-column file (annotation_name<TAB>bam_name), e.g. NC_003070.9<TAB>Chr1.
        """
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

        genes, exons, utrs, cds = (_parse_gtf if gtf else _parse_gff3)(df, id_attr, name_attr)
        genes = genes.drop_duplicates("gene_id")
        genes["gene_name"] = genes["gene_name"].fillna(genes["gene_id"])

        if chrom_map:
            cm = pd.read_csv(chrom_map, sep=None, engine="python", header=None, usecols=[0, 1],
                             names=["a", "b"], dtype=str)
            cmap = dict(zip(cm["a"], cm["b"]))
            for t in (genes, exons, utrs, cds):
                t["chrom"] = t["chrom"].map(lambda c: cmap.get(c, c))
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
            cds = cds[cds["gene_id"].isin(genes["gene_id"])]
        if genes.empty:
            raise ValueError("No genes found in the annotation file (check the file and biotype filter).")
        return cls(genes, exons, utrs, cds)

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


    # ------------------------------------------- transcript-level (ChIPseeker-style)
    def _models(self) -> dict:
        """Build transcript / exon / intron / UTR tables (cached)."""
        if self._models_cache is not None:
            return self._models_cache
        ex = self.exons.copy()
        if ex.empty:
            raise ValueError("The annotation has no exon records; cannot build transcript models.")
        ex["key"] = ex["gene_id"].astype(str) + "\x1f" + ex["tx_id"].astype(str)
        ex = ex.sort_values(["key", "start"], kind="mergesort").reset_index(drop=True)
        tx = ex.groupby("key", sort=False).agg(
            chrom=("chrom", "first"), start=("start", "min"), end=("end", "max"),
            strand=("strand", "first"), gene_id=("gene_id", "first"), tx_id=("tx_id", "first"),
            n_exons=("start", "size")).reset_index()
        tx["txi"] = np.arange(len(tx))
        txi = dict(zip(tx["key"], tx["txi"]))
        ex["txi"] = ex["key"].map(txi)
        pos = ex.groupby("key").cumcount().to_numpy()
        n = ex["key"].map(dict(zip(tx["key"], tx["n_exons"]))).to_numpy()
        minus = (ex["strand"] == "-").to_numpy()
        ex["rank"] = np.where(minus, n - pos, pos + 1)
        ex["total"] = n

        # introns: gaps between consecutive exons of a transcript
        nxt = ex.groupby("key")["start"].shift(-1)
        gap = nxt.notna() & (nxt > ex["end"])
        it = ex.loc[gap, ["chrom", "end", "txi", "strand"]].rename(columns={"end": "start"}).copy()
        it["end"] = nxt[gap].astype(np.int64).to_numpy()
        ipos = it.groupby("txi").cumcount().to_numpy()
        itot = it.groupby("txi")["start"].transform("size").to_numpy()
        it["rank"] = np.where((it["strand"] == "-").to_numpy(), itot - ipos, ipos + 1)
        it["total"] = itot

        # UTRs: explicit records if the file has them, otherwise inferred from CDS
        if (self.utrs["kind"].isin(["5UTR", "3UTR"])).any():
            u = self.utrs[self.utrs["kind"].isin(["5UTR", "3UTR"])].copy()
            u["key"] = u["gene_id"].astype(str) + "\x1f" + u["tx_id"].astype(str)
            u["txi"] = u["key"].map(txi)
            u5 = u[u["kind"] == "5UTR"][["chrom", "start", "end", "txi"]]
            u3 = u[u["kind"] == "3UTR"][["chrom", "start", "end", "txi"]]
        else:
            u5 = u3 = pd.DataFrame(columns=["chrom", "start", "end", "txi"])
            if not self.cds.empty:
                c = self.cds.copy()
                c["key"] = c["gene_id"].astype(str) + "\x1f" + c["tx_id"].astype(str)
                cb = c.groupby("key").agg(cmin=("start", "min"), cmax=("end", "max"))
                e2 = ex.join(cb, on="key").dropna(subset=["cmin"])
                left = e2[e2["start"] < e2["cmin"]].copy()
                left["end"] = np.minimum(left["end"], left["cmin"])
                right = e2[e2["end"] > e2["cmax"]].copy()
                right["start"] = np.maximum(right["start"], right["cmax"])
                lm = (left["strand"] == "-").to_numpy()
                rm = (right["strand"] == "-").to_numpy()
                cols = ["chrom", "start", "end", "txi"]
                u5 = pd.concat([left.loc[~lm, cols], right.loc[rm, cols]])
                u3 = pd.concat([left.loc[lm, cols], right.loc[~rm, cols]])

        chroms = list(pd.unique(pd.concat([self.genes["chrom"], tx["chrom"]])))
        self._models_cache = {"tx": tx, "ex": ex, "it": it, "u5": u5, "u3": u3,
                              "chrom_index": {c: i + 1 for i, c in enumerate(chroms)}}
        return self._models_cache

    def annotate_chipseeker(self, regions: pd.DataFrame, tss_region: Tuple[int, int] = (-3000, 3000),
                            downstream: int = 300) -> pd.DataFrame:
        """ChIPseeker-style transcript-level annotation of regions (needs Chromosome, Start, End, PeakID).

        The nearest transcript TSS defines geneId/transcriptId/geneStart/geneEnd/geneStrand and
        distanceToTSS (to the nearest region edge; negative = upstream of the TSS). The ``annotation``
        label uses whole-region overlap with priority Promoter > 5' UTR > 3' UTR > Exon > Intron >
        Downstream > Distal Intergenic, e.g. ``Exon (NM_179615.2/AT2G08986, exon 43 of 46)``.
        """
        m = self._models()
        tx, ex, it = m["tx"], m["ex"], m["it"]
        n = len(regions)
        tx_chroms = {_chrom_key(c): c for c in tx["chrom"].unique()}
        rc = regions["Chromosome"].map(lambda c: tx_chroms.get(_chrom_key(c))).to_numpy(object)
        if all(c is None for c in rc):
            raise ValueError("No chromosome names in the regions match the annotation. "
                             f"Regions look like {list(regions['Chromosome'].unique()[:3])}, "
                             f"annotation like {list(tx['chrom'].unique()[:3])}.")
        S = regions["Start"].to_numpy(np.int64)
        E = regions["End"].to_numpy(np.int64)

        label = np.array(["Distal Intergenic"] * n, dtype=object)
        out = {k: np.array([None] * n, dtype=object) for k in
               ["geneChr", "geneStart", "geneEnd", "geneLength", "geneStrand", "geneId", "transcriptId"]}
        dist = np.full(n, np.nan)
        lo_t, hi_t = tss_region

        for chrom in {c for c in rc if c is not None}:
            pos = np.nonzero(rc == chrom)[0]
            s, e = S[pos], E[pos]
            tmask = (tx["chrom"] == chrom).to_numpy()
            tpos = np.nonzero(tmask)[0]
            t_start, t_end = tx["start"].to_numpy()[tpos], tx["end"].to_numpy()[tpos]
            t_minus = (tx["strand"] == "-").to_numpy()[tpos]
            tss = np.where(t_minus, t_end - 1, t_start)

            # nearest TSS (distance to region edge, signed by gene orientation)
            order = np.argsort(tss, kind="mergesort")
            idx, raw = _nearest_tss(s, e, tss[order])
            loc = order[idx]                       # index within this chromosome's transcripts
            d = raw * np.where(t_minus[loc], -1, 1)
            gi = tpos[loc]                         # global transcript index
            dist[pos] = d
            out["geneChr"][pos] = m["chrom_index"][chrom]
            out["geneStart"][pos] = t_start[loc] + 1
            out["geneEnd"][pos] = t_end[loc]
            out["geneLength"][pos] = t_end[loc] - t_start[loc]
            out["geneStrand"][pos] = np.where(t_minus[loc], 2, 1)
            out["geneId"][pos] = tx["gene_id"].to_numpy()[gi]
            out["transcriptId"][pos] = tx["tx_id"].to_numpy()[gi]

            lab = np.array(["Distal Intergenic"] * len(pos), dtype=object)
            # downstream windows
            ws = np.where(t_minus, t_start - downstream, t_end).clip(min=0)
            we = np.where(t_minus, t_start, t_end + downstream)
            lab[_overlaps_union(s, e, *_union(ws, we))] = f"Downstream (<={downstream}bp)"
            # introns / exons with ranks
            for tbl, kind in ((it, "Intron"), (ex, "Exon")):
                f = tbl[tbl["chrom"] == chrom]
                if len(f):
                    best = _best_overlap(s, e, f["start"].to_numpy(), f["end"].to_numpy())
                    ok = best >= 0
                    if ok.any():
                        fi = f.iloc[best[ok]]
                        tidx = fi["txi"].to_numpy()
                        txt = [f"{kind} ({tx['tx_id'].iat[ti]}/{tx['gene_id'].iat[ti]}, {kind.lower()} {int(r)} of {int(t)})"
                               for ti, r, t in zip(tidx, fi["rank"], fi["total"])]
                        lab[np.nonzero(ok)[0]] = txt
            for key, name in (("u3", "3' UTR"), ("u5", "5' UTR")):
                u = m[key]
                u = u[u["chrom"] == chrom]
                if len(u):
                    lab[_overlaps_union(s, e, *_union(u["start"].to_numpy(np.int64), u["end"].to_numpy(np.int64)))] = name
            # promoter by distance of the nearest TSS
            ad = np.abs(d)
            k = np.maximum(1, np.ceil(ad / 1000.0)).astype(int)
            prom = (d >= lo_t) & (d <= hi_t)
            plabel = np.where(k == 1, "Promoter (<=1kb)", [f"Promoter ({x - 1}-{x}kb)" for x in k])
            lab[prom] = plabel[prom]
            label[pos] = lab

        res = pd.DataFrame({"annotation": label, **out, "distanceToTSS": dist},
                           index=regions["PeakID"].to_numpy())
        res["distanceToTSS"] = res["distanceToTSS"].astype("Int64")
        return res


def _nearest_tss(s: np.ndarray, e: np.ndarray, ts: np.ndarray):
    """Nearest TSS to each [s, e) region (0 if inside). ts must be sorted. Returns (index into ts, raw distance).

    raw > 0 means the region lies downstream (to the right) of the TSS, raw < 0 upstream (to the left),
    in genomic coordinates; the caller flips the sign for minus-strand transcripts.
    """
    n = len(ts)
    center = (s + e) // 2
    lo = np.searchsorted(ts, s, "left")
    hi = np.searchsorted(ts, e, "left")
    inside = hi > lo
    top = np.maximum(hi - 1, lo)
    j = np.clip(np.clip(np.searchsorted(ts, center, "left"), lo, top), 0, n - 1)
    k = np.clip(np.clip(j - 1, lo, top), 0, n - 1)
    pick_in = np.where(np.abs(ts[j] - center) <= np.abs(ts[k] - center), j, k)
    big = np.iinfo(np.int64).max // 4
    L, R = np.clip(lo - 1, 0, n - 1), np.clip(lo, 0, n - 1)
    dl = np.where(lo - 1 >= 0, s - ts[L], big)
    dr = np.where(lo < n, ts[R] - (e - 1), big)
    pick_out = np.where(dl <= dr, L, R)
    idx = np.where(inside, pick_in, pick_out)
    t = ts[idx]
    raw = np.where(inside, 0, np.where(t < s, s - t, (e - 1) - t))
    return idx, raw


def _overlaps_union(s: np.ndarray, e: np.ndarray, us: np.ndarray, ue: np.ndarray) -> np.ndarray:
    """True where [s, e) overlaps any interval of a merged union."""
    if len(us) == 0:
        return np.zeros(len(s), dtype=bool)
    i = np.searchsorted(us, e, side="left") - 1
    return (i >= 0) & (ue[np.clip(i, 0, None)] > s)


def _best_overlap(s: np.ndarray, e: np.ndarray, fs: np.ndarray, fe: np.ndarray) -> np.ndarray:
    """Index of the feature with the largest overlap for each region, -1 if none (chunked)."""
    out = np.full(len(s), -1, dtype=np.int64)
    step = max(1, int(4e6 // max(1, len(fs))))
    for a in range(0, len(s), step):
        ss, ee = s[a:a + step, None], e[a:a + step, None]
        ov = np.minimum(ee, fe[None, :]) - np.maximum(ss, fs[None, :])
        best = ov.argmax(axis=1)
        ok = ov[np.arange(len(best)), best] > 0
        out[a:a + step] = np.where(ok, best, -1)
    return out


# ---------------------------------------------------------------------- parsers
_TXCOLS = ["chrom", "start", "end", "gene_id", "tx_id", "strand"]


def _clean_tx(s: pd.Series) -> pd.Series:
    return s.fillna("NA").replace("", "NA")


def _parse_gtf(df: pd.DataFrame, id_attr=None, name_attr=None):
    rows = df[df["f"].isin({"gene", "exon", "cds"} | _UTR5 | _UTR3 | _UTR)].copy()
    gid = _attr(rows["attr"], "gene_id", True)
    if id_attr and id_attr != "gene_id":
        gid = _attr(rows["attr"], id_attr, True).fillna(gid)
    rows["gene_id"] = gid
    rows = rows.dropna(subset=["gene_id"])
    rows["tx_id"] = _clean_tx(_attr(rows["attr"], "transcript_id", True))

    g = rows[rows["f"] == "gene"].copy()
    if g.empty:   # GTF without gene lines: derive from exons/CDS/UTRs
        agg = rows.groupby("gene_id").agg(chrom=("chrom", "first"), start=("start", "min"),
                                          end=("end", "max"), strand=("strand", "first")).reset_index()
        agg["attr"] = agg["gene_id"].map(rows.drop_duplicates("gene_id").set_index("gene_id")["attr"])
        g = agg
    name = _attr(g["attr"], name_attr, True) if name_attr else pd.Series(np.nan, index=g.index, dtype=object)
    for k in ["gene_name", "gene", "gene_symbol"]:
        name = name.fillna(_attr(g["attr"], k, True))
    g["gene_name"] = name
    bio = _attr(g["attr"], "gene_biotype", True)
    for k in ["gene_type", "biotype"]:
        bio = bio.fillna(_attr(g["attr"], k, True))
    g["biotype"] = bio
    genes = g[["gene_id", "gene_name", "biotype", "chrom", "start", "end", "strand"]]

    exons = rows.loc[rows["f"] == "exon", _TXCOLS]
    cds = rows.loc[rows["f"] == "cds", _TXCOLS]
    u = rows[rows["f"].isin(_UTR5 | _UTR3 | _UTR)].copy()
    u["kind"] = _utr_kind(u["f"])
    return genes, exons, u[_TXCOLS + ["kind"]], cds


def _parse_gff3(df: pd.DataFrame, id_attr=None, name_attr=None):
    d = df[~df["f"].isin(_DROP)].copy()
    d["ID"] = _attr(d["attr"], "ID", False)
    d["Parent"] = _attr(d["attr"], "Parent", False)
    strip_g = lambda s: s.str.replace(r"^gene[:-]", "", regex=True)
    strip_t = lambda s: s.str.replace(r"^(transcript:|rna-)", "", regex=True)

    is_gene = d["f"].str.endswith("gene")
    g = d[is_gene].copy()
    g["gene_id"] = strip_g(g["ID"])
    g = g.dropna(subset=["gene_id"])
    name = _attr(g["attr"], name_attr, False) if name_attr else _attr(g["attr"], "Name", False)
    for k in ["gene_name", "gene", "symbol"]:
        name = name.fillna(_attr(g["attr"], k, False))
    g["gene_name"] = name
    bio = _attr(g["attr"], "biotype", False)
    for k in ["gene_biotype", "gene_type"]:
        bio = bio.fillna(_attr(g["attr"], k, False))
    g["biotype"] = bio
    genes = g[["gene_id", "gene_name", "biotype", "chrom", "start", "end", "strand"]]
    gene_set = set(genes["gene_id"])

    is_exon = d["f"] == "exon"
    is_cds = d["f"] == "cds"
    is_utr = d["f"].isin(_UTR5 | _UTR3 | _UTR)
    t = d[~is_gene & ~is_exon & ~is_cds & ~is_utr & d["Parent"].notna() & d["ID"].notna()]
    t_parent = strip_g(t["Parent"].str.split(",").str[0])
    t = t[t_parent.isin(gene_set)]
    tx2gene = dict(zip(strip_t(t["ID"]), strip_g(t["Parent"].str.split(",").str[0])))

    def with_gene(x: pd.DataFrame) -> pd.DataFrame:
        x = x.copy()
        p = strip_t(x["Parent"].fillna("").str.split(",").str[0])
        gid = p.map(tx2gene)
        direct = strip_g(p).where(strip_g(p).isin(gene_set))     # parent is the gene itself
        x["tx_id"] = np.where(gid.notna(), p, "NA")
        x["gene_id"] = gid.fillna(direct)
        return x.dropna(subset=["gene_id"])

    ex, cd, ut = with_gene(d[is_exon]), with_gene(d[is_cds]), with_gene(d[is_utr])
    ut["kind"] = _utr_kind(ut["f"])
    ex, cd, ut = ex[_TXCOLS].copy(), cd[_TXCOLS].copy(), ut[_TXCOLS + ["kind"]].copy()
    if id_attr and id_attr != "ID":      # e.g. locus_tag: swap gene IDs everywhere
        new = _attr(g["attr"], id_attr, False)
        remap = dict(zip(g["gene_id"], new.fillna(g["gene_id"])))
        genes = genes.copy()
        genes["gene_id"] = genes["gene_id"].map(remap)
        for tbl in (ex, cd, ut):
            tbl["gene_id"] = tbl["gene_id"].map(remap).fillna(tbl["gene_id"])
    return genes, ex, ut, cd
