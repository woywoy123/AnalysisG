from AnalysisG.IO import IO

root1 = "./samples/sample1/smpl1.root"
root2 = "./samples/sample1/smpl2.root"

def test_reading_root():
    io = IO()
    io.Files = [root1, root2]
    io.Trees = ["nominal", "nominal-1"]
    io.Branches = ["children_index", "hello"]
    io.Leaves = ["children_phi", "nothing"]

    root1_, root2_ = list(io.Keys)
    assert "nominal-1" in io.Keys[root1_]["missed"]["Trees"]
    assert "nominal-1" in io.Keys[root2_]["missed"]["Trees"]

    assert "hello" in io.Keys[root1_]["missed"]["Branches"]
    assert "hello" in io.Keys[root2_]["missed"]["Branches"]

    assert "nothing" in io.Keys[root1_]["missed"]["Leaves"]
    assert "nothing" in io.Keys[root2_]["missed"]["Leaves"]
    del io

    io = IO([root1, root2])
    io.Trees = ["nominal"]
    io.Branches = ["children_index"]
    io.Leaves = ["met_met"]
    assert len(io) == 165
    del io

    for k in range(10):
        io = IO([root1, root2])
        io.Trees = ["nominal", "truth"]
        io.Leaves = ["weight_pileup", "weight_mc", "met_phi"]
        io.ScanKeys()
        len_nom, len_truth = 0, 0
        for i in io:
            if b"truth.weight_mc.weight_mc" in i:
                assert b"truth.weight_mc.weight_mc" in i
                assert b"truth.weight_pileup.weight_pileup" in i
                len_truth += 1
            if b"nominal.weight_mc.weight_mc" in i:
                assert b"nominal.weight_mc.weight_mc" in i
                assert b"nominal.weight_pileup.weight_pileup" in i
                assert b"nominal.met_phi.met_phi" in i
                len_nom += 1
        assert len_truth == 2000
        assert len_nom == 165
        del io

if __name__ == "__main__":
    test_reading_root()
