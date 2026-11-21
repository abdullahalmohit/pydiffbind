import pandas as pd
import pytest
from pydiffbind.annotate import Annotation

GTF = """#comment
chr1\tsrc\tgene\t10001\t14000\t.\t+\t.\tgene_id "G1"; gene_name "Alpha"; gene_biotype "protein_coding";
chr1\tsrc\texon\t10001\t10500\t.\t+\t.\tgene_id "G1"; transcript_id "T1";
chr1\tsrc\texon\t13001\t14000\t.\t+\t.\tgene_id "G1"; transcript_id "T1";
chr1\tsrc\tgene\t30001\t32000\t.\t-\t.\tgene_id "G2"; gene_name "Beta"; gene_biotype "lncRNA";
chr1\tsrc\texon\t30001\t32000\t.\t-\t.\tgene_id "G2"; transcript_id "T2";
"""

GFF3 = """##gff-version 3
Chr1\tsrc\tgene\t10001\t14000\t.\t+\t.\tID=gene:G1;Name=Alpha;biotype=protein_coding
Chr1\tsrc\tmRNA\t10001\t14000\t.\t+\t.\tID=transcript:G1.1;Parent=gene:G1
Chr1\tsrc\texon\t10001\t10500\t.\t+\t.\tParent=transcript:G1.1
Chr1\tsrc\tfive_prime_UTR\t10001\t10300\t.\t+\t.\tParent=transcript:G1.1
Chr1\tsrc\texon\t13001\t14000\t.\t+\t.\tParent=transcript:G1.1
Chr1\tsrc\tgene\t30001\t32000\t.\t-\t.\tID=gene:G2;Name=Beta;biotype=lncRNA
Chr1\tsrc\tmRNA\t30001\t32000\t.\t-\t.\tID=transcript:G2.1;Parent=gene:G2
Chr1\tsrc\texon\t30001\t32000\t.\t-\t.\tParent=transcript:G2.1
"""


def regions(chrom):
    rows = [  # start, end -> expected feature at centre
        (10100, 10300),   # + gene exon (GFF3: 5UTR)
        (11000, 11200),   # intron
        (8500, 8700),     # promoter of G1 (TSS 10000, window 8000..10500)
        (14100, 14300),   # downstream of G1
        (20000, 20200),   # intergenic
        (32000, 32400),   # promoter of G2 (minus strand: window beyond end)
    ]
    df = pd.DataFrame(rows, columns=["Start", "End"])
    df["Chromosome"] = chrom
    df["PeakID"] = [f"r{i}" for i in range(len(df))]
    return df


def test_gtf(tmp_path):
    p = tmp_path / "a.gtf"; p.write_text(GTF)
    a = Annotation.from_file(str(p))
    r = a.annotate(regions("chr1"), promoter=(-2000, 100))
    assert list(r["feature"]) == ["Exon", "Intron", "Promoter", "Downstream", "Intergenic", "Promoter"]
    assert r.loc["r0", "overlapping_genes"] == "G1" and r.loc["r0", "overlapping_gene_names"] == "Alpha"
    assert r.loc["r4", "overlapping_genes"] == "" and r.loc["r4", "n_overlapping_genes"] == 0
    assert r.loc["r5", "overlapping_genes"] == "G2"      # promoter overlap counts
    assert r.loc["r2", "nearest_gene"] == "G1" and r.loc["r2", "distance_to_tss"] < 0   # upstream
    assert r.loc["r4", "nearest_gene"] == "G1" or r.loc["r4", "nearest_gene"] == "G2"


def test_gff3_and_chrom_names(tmp_path):
    p = tmp_path / "a.gff3"; p.write_text(GFF3)
    a = Annotation.from_file(str(p))
    assert set(a.genes["gene_id"]) == {"G1", "G2"}
    assert len(a.exons) == 3 and (a.utrs["kind"] == "5UTR").sum() == 1
    r = a.annotate(regions("chr1"), promoter=(-2000, 100))     # regions 'chr1' vs annotation 'Chr1'
    assert r.loc["r0", "feature"] == "5UTR" and r.loc["r1", "feature"] == "Intron"
    assert r.loc["r0", "overlapping_gene_names"] == "Alpha"


def test_symbols_biotype_and_errors(tmp_path):
    p = tmp_path / "a.gtf"; p.write_text(GTF)
    s = tmp_path / "sym.tsv"; s.write_text("gene_id\tsymbol\nG1\tMYGENE\n")
    a = Annotation.from_file(str(p), symbols=str(s), biotypes=["protein_coding"])
    assert list(a.genes["gene_name"]) == ["MYGENE"]
    with pytest.raises(ValueError):
        a.annotate(regions("scaffold_x"))


NCBI_GTF = (
    'NC_003070.9\tBestRefSeq\tgene\t10001\t14000\t.\t+\t.\tgene_id "NAC001"; transcript_id ""; db_xref "GeneID:1"; '
    'gene "NAC001"; gene_biotype "protein_coding"; locus_tag "AT1G01010";\n'
    'NC_003070.9\tBestRefSeq\texon\t10001\t14000\t.\t+\t.\tgene_id "NAC001"; transcript_id "NM_1"; gene "NAC001"; '
    'locus_tag "AT1G01010";\n'
)


def test_ncbi_style_gtf_with_locus_tag_and_chrom_map(tmp_path):
    p = tmp_path / "genomic.gtf"; p.write_text(NCBI_GTF)
    cm = tmp_path / "map.tsv"; cm.write_text("NC_003070.9\tChr1\n")
    a = Annotation.from_file(str(p), id_attr="locus_tag", chrom_map=str(cm))
    assert list(a.genes["gene_id"]) == ["AT1G01010"] and list(a.genes["gene_name"]) == ["NAC001"]
    r = a.annotate(regions("Chr1"), promoter=(-2000, 100))
    assert r.loc["r0", "overlapping_genes"] == "AT1G01010" and r.loc["r0", "overlapping_gene_names"] == "NAC001"
    # default (no id_attr) keeps the symbol as the ID
    b = Annotation.from_file(str(p))
    assert list(b.genes["gene_id"]) == ["NAC001"]


def test_ncbi_gff3_locus_tag(tmp_path):
    g = ("NC_003070.9\tRefSeq\tgene\t10001\t14000\t.\t+\t.\tID=gene-AT1G01010;Name=NAC001;locus_tag=AT1G01010\n"
         "NC_003070.9\tRefSeq\tmRNA\t10001\t14000\t.\t+\t.\tID=rna-NM_1;Parent=gene-AT1G01010\n"
         "NC_003070.9\tRefSeq\texon\t10001\t14000\t.\t+\t.\tID=exon-NM_1-1;Parent=rna-NM_1\n")
    p = tmp_path / "genomic.gff"; p.write_text(g)
    a = Annotation.from_file(str(p), id_attr="locus_tag")
    assert list(a.genes["gene_id"]) == ["AT1G01010"] and len(a.exons) == 1
    assert a.exons["gene_id"].iloc[0] == "AT1G01010"
