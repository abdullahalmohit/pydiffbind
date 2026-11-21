import pandas as pd
from pydiffbind.annotate import Annotation

C = "NC_000001.1"
GTF = "\n".join([
    f'{C}\tRefSeq\tgene\t1001\t2600\t.\t+\t.\tgene_id "G1"; gene_name "Alpha";',
    f'{C}\tRefSeq\texon\t1001\t1500\t.\t+\t.\tgene_id "G1"; transcript_id "T1";',
    f'{C}\tRefSeq\texon\t2001\t2600\t.\t+\t.\tgene_id "G1"; transcript_id "T1";',
    f'{C}\tRefSeq\tCDS\t1301\t1500\t.\t+\t0\tgene_id "G1"; transcript_id "T1";',
    f'{C}\tRefSeq\tCDS\t2001\t2300\t.\t+\t0\tgene_id "G1"; transcript_id "T1";',
    f'{C}\tRefSeq\tgene\t5001\t6400\t.\t-\t.\tgene_id "G2"; gene_name "Beta";',
    f'{C}\tRefSeq\texon\t6001\t6400\t.\t-\t.\tgene_id "G2"; transcript_id "T2";',
    f'{C}\tRefSeq\texon\t5001\t5300\t.\t-\t.\tgene_id "G2"; transcript_id "T2";',
    f'{C}\tRefSeq\tgene\t20001\t20500\t.\t+\t.\tgene_id "G3";',
    f'{C}\tRefSeq\texon\t20001\t20500\t.\t+\t.\tgene_id "G3"; transcript_id "";',   # NA transcript id
]) + "\n"


def regs(rows):
    df = pd.DataFrame(rows, columns=["Start", "End"])
    df["Chromosome"] = C
    df["PeakID"] = [f"r{i}" for i in range(len(df))]
    return df


def test_labels_and_distances(tmp_path):
    p = tmp_path / "genomic.gtf"; p.write_text(GTF)
    a = Annotation.from_file(str(p))
    r = a.annotate_chipseeker(regs([
        (950, 1050),     # TSS inside              -> Promoter (<=1kb), distance 0
        (1150, 1250),    # 5' UTR (inferred from CDS)
        (1350, 1450),    # coding exon 1
        (1700, 1800),    # intron
        (2400, 2500),    # 3' UTR
        (2650, 2800),    # downstream
        (5100, 5200),    # minus-strand exon, rank 2 of 2 (counted from the TSS end)
        (20300, 20400),  # gene without transcript id -> NA
        (10000, 10100),  # intergenic
    ]), tss_region=(-100, 100), downstream=300)
    assert list(r["annotation"]) == [
        "Promoter (<=1kb)", "5' UTR", "Exon (T1/G1, exon 1 of 2)", "Intron (T1/G1, intron 1 of 1)",
        "3' UTR", "Downstream (<=300bp)", "Exon (T2/G2, exon 2 of 2)", "Exon (NA/G3, exon 1 of 1)", "Distal Intergenic"]
    assert r.loc["r0", "distanceToTSS"] == 0 and r.loc["r0", "geneId"] == "G1"
    assert r.loc["r0", "geneStart"] == 1001 and r.loc["r0", "geneEnd"] == 2600 and r.loc["r0", "geneLength"] == 1600
    assert r.loc["r0", "geneStrand"] == 1 and r.loc["r0", "geneChr"] == 1
    assert r.loc["r6", "geneId"] == "G2" and r.loc["r6", "geneStrand"] == 2 and r.loc["r6", "transcriptId"] == "T2" and r.loc["r7", "transcriptId"] == "NA"
    assert r.loc["r6", "distanceToTSS"] == 1200       # minus strand: region is downstream of the TSS (6399)
    assert r.loc["r8", "geneId"] == "G2" and r.loc["r8", "distanceToTSS"] == -3601  # upstream of G2's TSS


def test_promoter_bins_default_region(tmp_path):
    p = tmp_path / "genomic.gtf"; p.write_text(GTF)
    a = Annotation.from_file(str(p))
    r = a.annotate_chipseeker(regs([(1500, 1600), (2500, 2560)]))   # 500 and 1500 bp from TSS 1000
    assert r.loc["r0", "annotation"] == "Promoter (<=1kb)" and r.loc["r0", "distanceToTSS"] == 500
    assert r.loc["r1", "annotation"] == "Promoter (1-2kb)" and r.loc["r1", "distanceToTSS"] == 1500
