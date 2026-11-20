"""Command-line interface: ``pydiffbind samples.csv --group1 treated --group2 control -o out/``"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .dba import DBA


def main(argv=None):
    p = argparse.ArgumentParser(prog="pydiffbind", description="Differential binding analysis")
    p.add_argument("samplesheet")
    p.add_argument("-o", "--outdir", default="pydiffbind_out")
    p.add_argument("--factor", default="Condition")
    p.add_argument("--group1"); p.add_argument("--group2")
    p.add_argument("--covariates", nargs="*", default=[])
    p.add_argument("--summits", type=int, default=200, help="half-width around summit; 0 keeps merged extents")
    p.add_argument("--min-overlap", type=int, default=2)
    p.add_argument("--normalize", default="rle", choices=["rle", "lib", "tmm", "background"])
    p.add_argument("--blacklist")
    p.add_argument("--subtract-control", action="store_true")
    p.add_argument("--fragment-size", type=int, default=200)
    p.add_argument("--min-mapq", type=int, default=10)
    p.add_argument("--fdr", type=float, default=0.05)
    p.add_argument("--threads", type=int, default=1)
    a = p.parse_args(argv)

    os.makedirs(a.outdir, exist_ok=True)
    dba = DBA(a.samplesheet)
    dba.count(summits=a.summits, min_overlap=a.min_overlap, min_mapq=a.min_mapq,
              fragment_size=a.fragment_size, blacklist=a.blacklist,
              subtract_control=a.subtract_control, background=a.normalize == "background",
              threads=a.threads)
    dba.normalize(a.normalize)
    dba.analyze(a.factor, a.group1, a.group2, covariates=a.covariates, n_cpus=a.threads)

    dba.counts.to_csv(f"{a.outdir}/raw_counts.csv")
    dba.norm_counts.to_csv(f"{a.outdir}/normalized_counts.csv")
    dba.results.join(dba.consensus.set_index("PeakID")[["Chromosome", "Start", "End"]]) \
        .to_csv(f"{a.outdir}/all_results.csv")
    dba.report(fdr=a.fdr).to_csv(f"{a.outdir}/significant.csv")
    for name, fn in [("pca", dba.plot_pca), ("correlation", dba.plot_correlation),
                     ("ma", dba.plot_ma), ("volcano", dba.plot_volcano)]:
        fn(); plt.tight_layout(); plt.savefig(f"{a.outdir}/{name}.png", dpi=150); plt.close()
    print(f"{dba}\n{len(dba.report(fdr=a.fdr))} significant sites at FDR<{a.fdr}; results in {a.outdir}/")


if __name__ == "__main__":
    main()
