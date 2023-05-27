from AnalysisG.IO import UpROOT

def test_pyami():
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

def test_ami_injection():
    smpl = UpROOT("samples/dilepton/")
    smpl.Trees = ["nominal"]
    smpl.Leaves = ["weight_mc", "eventNumber"]

    lst = [] 
    for i in smpl:
        meta = i["MetaData"]
        assert "dilepton" not in meta.GetDAOD(i["nominal/eventNumber"])[0]

if __name__ == "__main__":
    #test_pyami()    
    test_ami_injection()
    pass
