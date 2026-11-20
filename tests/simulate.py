"""Simulate small BAMs + narrowPeak files with known differential sites."""
import os
import numpy as np
import pandas as pd
import pysam


def simulate(outdir, n_sites=300, n_diff=60, reps=3, seed=0):
    rng = np.random.default_rng(seed)
    os.makedirs(outdir, exist_ok=True)
    L = 2_000_000
    header = {"HD": {"VN": "1.0", "SO": "coordinate"}, "SQ": [{"SN": "chr1", "LN": L}]}
    centers = np.sort(rng.choice(np.arange(5000, L - 5000, 2000), n_sites, replace=False))
    diff = np.zeros(n_sites, bool); diff[:n_diff] = True
    rng.shuffle(diff)
    base = rng.uniform(40, 120, n_sites)
    rows = []
    for cond in ["treated", "control"]:
        for r in range(reps):
            sid = f"{cond}_{r+1}"
            mu = base.copy()
            if cond == "treated":
                mu[diff] *= 4
            reads = []
            for c, m in zip(centers, mu):
                n = rng.poisson(m * rng.uniform(.8, 1.2))
                pos = rng.normal(c, 120, n).astype(int)
                reads += [(max(0, p - 50), rng.random() < .5) for p in pos]
            reads += [(int(p), rng.random() < .5) for p in rng.integers(0, L - 100, 3000)]  # background
            reads.sort()
            bam = f"{outdir}/{sid}.bam"
            with pysam.AlignmentFile(bam, "wb", header=header) as f:
                for i, (p, rev) in enumerate(reads):
                    a = pysam.AlignedSegment()
                    a.query_name = f"r{i}"; a.query_sequence = "A" * 50
                    a.flag = 16 if rev else 0
                    a.reference_id = 0; a.reference_start = int(min(p, L - 60))
                    a.mapping_quality = 60; a.cigartuples = [(0, 50)]
                    a.query_qualities = pysam.qualitystring_to_array("I" * 50)
                    f.write(a)
            pysam.index(bam)
            pk = f"{outdir}/{sid}.narrowPeak"
            keep = rng.random(n_sites) < .9
            with open(pk, "w") as f:
                for c, k in zip(centers, keep):
                    if k:
                        j = int(rng.integers(-30, 30))
                        f.write(f"chr1\t{c-300+j}\t{c+300+j}\tp\t0\t.\t0\t0\t0\t300\n")
            rows.append(dict(SampleID=sid, Condition=cond, Replicate=r + 1, bamReads=bam, Peaks=pk))
    sheet = f"{outdir}/samples.csv"
    pd.DataFrame(rows).to_csv(sheet, index=False)
    return sheet, centers, diff
