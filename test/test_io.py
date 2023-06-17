from AnalysisG.IO import UpROOT

root1 = "./samples/sample1/smpl1.root"
root2 = "./samples/sample1/smpl2.root"


def test_uproot():
    io = UpROOT([root1, root2])
    io.Trees = ["nominal", "nominal-1"]
    io.Branches = ["children_index", "hello"]
    io.Leaves = ["children_index", "nothing"]
    io.ScanKeys

    root1_, root2_ = list(io.Keys)
    assert "nominal-1" in io.Keys[root1_]["missed"]["TREE"]
    assert "nominal-1" in io.Keys[root2_]["missed"]["TREE"]

    assert "hello" in io.Keys[root1_]["missed"]["BRANCH"]
    assert "hello" in io.Keys[root2_]["missed"]["BRANCH"]

    assert "nothing" in io.Keys[root1_]["missed"]["LEAF"]
    assert "nothing" in io.Keys[root2_]["missed"]["LEAF"]

    io = UpROOT([root1, root2])
    io.Trees = ["nominal"]
    io.Branches = ["children_index"]
    io.Leaves = ["met_met"]
    assert len(io) == 165

    io = UpROOT([root1, root2])
    io.Trees = ["nominal", "truth"]
    io.Leaves = ["weight_mc", "weight_pileup", "met_phi"]

    for i in io:
        assert "nominal/weight_mc" in i
        assert "truth/weight_mc" in i
        assert "nominal/weight_pileup" in i
        assert "truth/weight_pileup" in i
        assert "nominal/met_phi" in i
        assert "truth/met_phi" not in i


if __name__ == "__main__":
    test_uproot()
