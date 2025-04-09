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
        assert len_truth == 2000
        assert len_nom == 165
        del io

def test_random():
    import time 
    spl = "./big-dr0.4.root"
    io = IO()
    io.Verbose = False
    io.Files = [spl]
    io.Trees = ["nominal_Loose"]
    io.Leaves = ["phystru_partontruthlabel", "phystru_top_index", "phystru_index", "phystru_type", "physdet_partontruthlabel", "physdet_top_index"]
    for i in io:
        del i["filename"]
        for l in i:
            print(l.decode("utf-8").split(".")[-1], i[l])
        print("___")
        time.sleep(0.1)

def test_pyami():

    smpl = IO(root1)
    smpl.MetaCachePath = "./meta_cache"
    smpl.Trees = ["nominal"]
    smpl.Leaves = ["weight_mc"]
    smpl.EnablePyAMI = True
    smpl.Keys

    meta = smpl.MetaData()
    meta = list(meta.values())[0]
    print(meta.dsid)
    print(meta.amitag)
    print(meta.logicalDatasetName)
    print(meta.nFiles)
    print(meta.totalEvents)
    print(meta.totalSize)
    print(meta.dataType)
    print(meta.prodsysStatus)
    print(meta.completion)
    print(meta.ecmEnergy)
    print(meta.generators)
    print(meta.isMC)
    print(meta.derivationFormat)
    print(meta.eventNumber)
    print(meta.genFiltEff)
    print(meta.beam_energy)
    print(meta.crossSection)
    print(meta.crossSection_mean)
    print(meta.run_number)
    print(meta.datasetNumber)
    print(meta.identifier)
    print(meta.version)
    print(meta.PDF)
    print(meta.AtlasRelease)
    print(meta.principalPhysicsGroup)
    print(meta.physicsShort)
    print(meta.generatorName)
    print(meta.geometryVersion)
    print(meta.conditionsTag)
    print(meta.generatorTune)
    print(meta.amiStatus)
    print(meta.beamType)
    print(meta.productionStep)
    print(meta.projectName)
    print(meta.statsAlgorithm)
    print(meta.genFilterNames)
    print(meta.file_type)
    print(meta.DatasetName)
    print(meta.event_index)
    print(meta.keywords)
    print(meta.weights)
    print(meta.keyword)
    print(meta.found)
    print(meta.Files)
    print(meta.fileGUID)
    print(meta.events)
    print(meta)
    print(meta.fileSize)
    print(meta.sample_name)

if __name__ == "__main__":
    test_reading_root()
#    test_pyami()
#    test_random()

