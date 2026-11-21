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


def test_chipseeker_export(tmp_path):
    sheet, centers, diff = simulate(str(tmp_path))
    lines = []
    for i, c in enumerate(centers):
        st = "+-"[i % 2]
        lines += [f'chr1\tsrc\tgene\t{c-800}\t{c+1200}\t.\t{st}\t.\tgene_id "AT1G{i:05d}";',
                  f'chr1\tsrc\texon\t{c-800}\t{c-300}\t.\t{st}\t.\tgene_id "AT1G{i:05d}"; transcript_id "NM_{i}.1";',
                  f'chr1\tsrc\texon\t{c+700}\t{c+1200}\t.\t{st}\t.\tgene_id "AT1G{i:05d}"; transcript_id "NM_{i}.1";']
    gtf = tmp_path / "a.gtf"; gtf.write_text("\n".join(lines) + "\n")
    dba = DBA(sheet)
    dba.count(summits=250, fragment_size=150)
    dba.normalize("rle"); dba.analyze("Condition", "treated", "control")
    dba.annotate(str(gtf))
    out = tmp_path / "annotated.tsv"
    t = dba.export_chipseeker(str(out), fdr=0.05)
    expected = ["seqnames", "start", "end", "width", "strand", "Conc", "Conc_treated", "Conc_control", "Fold",
                "p.value", "FDR", "annotation", "geneChr", "geneStart", "geneEnd", "geneLength", "geneStrand",
                "geneId", "transcriptId", "distanceToTSS", "TAIR"]
    back = pd.read_csv(out, sep="\t")
    assert list(back.columns) == expected and len(back) == len(t) >= 30
    assert (back["width"] == back["end"] - back["start"] + 1).all() and (back["strand"] == "*").all()
    assert back["FDR"].is_monotonic_increasing and (back["TAIR"] == back["geneId"]).all()
    assert back["annotation"].str.contains(r"^(Promoter|Exon|Intron|Distal|5'|3'|Downstream)").all()
    assert set(back["geneStrand"]) <= {1, 2}
    assert dba._make_names("Amir-FIE") == "Amir.FIE"
    assert set(dba.gene_lists(by="chipseeker")["up"]) <= set(back["geneId"]) | set(dba.annotation["geneId"])
