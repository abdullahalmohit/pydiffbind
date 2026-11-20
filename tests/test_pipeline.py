import numpy as np
from simulate import simulate
from pydiffbind import DBA


def test_end_to_end(tmp_path):
    sheet, centers, diff = simulate(str(tmp_path))
    dba = DBA(sheet)
    dba.count(summits=250, min_overlap=2, fragment_size=150, background=True, bin_size=50000)
    assert len(dba.consensus) > 200
    for m in ["rle", "lib", "tmm", "background"]:
        sf = dba.normalize(m)
        assert (sf > 0).all()
    dba.normalize("rle")
    res = dba.analyze("Condition", "treated", "control")
    sig = dba.report(fdr=0.05, direction="up")
    mid = ((sig["Start"] + sig["End"]) // 2).to_numpy()
    hit = np.abs(mid[:, None] - centers[None, :]).argmin(axis=1)
    true_pos = diff[hit].mean()
    assert len(sig) >= 30 and true_pos > 0.9
    assert res.loc[sig.index, "log2FoldChange"].median() > 1.5
