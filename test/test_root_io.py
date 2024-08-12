from AnalysisG.core.io import IO

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
            print(i)
        assert len_truth == 2000
        assert len_nom == 165
        del io

# breaks with python3.12 due to ansible removing key_file input field
#def test_pyami():
#
#    smpl = IO("samples/dilepton/*")
#    smpl.Trees = ["nominal"]
#    smpl.Leaves = ["weight_mc"]
#    smpl.EnablePyAMI = False
#    assert len(smpl) == 1098
#    meta = smpl.MetaData()
#    print(meta)
#    print(list(meta.values())[0].dsid)
#

    #assert len(meta) == 1
    #f = next(iter(meta))
    #assert meta[f].isMC
    #assert len(meta[f].Files) == 1

    #data = "mc16_13TeV.312446.MadGraphPythia8EvtGen_noallhad_ttH_tttt_m1000"
    #assert meta[f].crossSection >= 0
    #assert data in meta[f].DatasetName

    #assert len([i for i in smpl]) > 0
    #for i in smpl:
    #    assert data in i["MetaData"].DatasetName

#def test_ami_injection():
#    smpl = IO(["samples/dilepton/"])
#    smpl.Trees = ["nominal"]
#    smpl.Leaves = ["weight_mc", "eventNumber"]
#    for i in smpl:
#        meta = i["MetaData"]
#        evnt = i["EventIndex"]
#        evnt_nr = i["nominal/eventNumber"]
#        assert evnt_nr == x["eventNumber"][evnt]
#        assert "dilepton" not in meta.IndexToSample(i["nominal/eventNumber"])
#        lst.append(1)
#    assert len(lst) == len(smpl)

if __name__ == "__main__":
    test_reading_root()
#    test_pyami()
#    test_ami_injection()

