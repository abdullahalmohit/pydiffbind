import numpy as np
import pandas as pd
from simulate import simulate
from pydiffbind import DBA


def test_annotated_pipeline(tmp_path):
    sheet, centers, diff = simulate(str(tmp_path))
    # one gene around each simulated site; gene i is 'differential' if diff[i]
    lines = []
    for i, c in enumerate(centers):
        lines.append(f'chr1\tsrc\tgene\t{c-800}\t{c+1200}\t.\t{"+-"[i % 2]}\t.\tgene_id "g{i}"; gene_name "Gene{i}";')
        lines.append(f'chr1\tsrc\texon\t{c-800}\t{c+1200}\t.\t{"+-"[i % 2]}\t.\tgene_id "g{i}"; transcript_id "t{i}";')
    gtf = tmp_path / "anno.gtf"; gtf.write_text("\n".join(lines) + "\n")

    dba = DBA(sheet)
    dba.count(summits=250, fragment_size=150)
    dba.normalize("rle"); dba.analyze("Condition", "treated", "control")
    ann = dba.annotate(str(gtf), promoter=(-100, 100))  # sites are only 2 kb apart
    assert len(ann) == len(dba.consensus)
    rep = dba.report(fdr=0.05)
    assert {"feature", "overlapping_genes", "nearest_gene", "distance_to_tss"} <= set(rep.columns)

    gl = dba.gene_lists(fdr=0.05)
    up_idx = [int(g[1:]) for g in gl["up"]]
    assert len(up_idx) >= 30 and diff[up_idx].mean() > 0.9           # up genes are the truly differential ones
    assert set(gl["all"]) == set(gl["up"]) | set(gl["down"])
    assert set(gl["up"]) <= set(dba.tested_genes())
    assert dba.gene_lists(names=True)["up"][0].startswith("Gene")

    dba.save_gene_lists(str(tmp_path / "out"))
    assert (tmp_path / "out" / "up_genes.txt").read_text().strip()
    import matplotlib; matplotlib.use("Agg")
    dba.plot_annotation(); dba.plot_tss_distance()
