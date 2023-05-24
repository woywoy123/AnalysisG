from AnalysisG.IO import UpROOT

def test_pyami():
    d = "samples/dilepton/DAOD_TOPQ1.21955717._000001.root"
    smpl = UpROOT("samples/dilepton/")
    meta = smpl.GetAmiMeta
    assert len(meta) == 1
    f = next(iter(meta))
    assert meta[f].isMC
    assert len(meta[f].Files) == 1
    assert meta[f].cross_section != None
    assert meta[f].DatasetName  
    smpl.Trees = ["nominal"]
    smpl.Leaves = ["weight_mc"]
    
    for i in smpl:
        print(i) 





if __name__ == "__main__":
    test_pyami()    
