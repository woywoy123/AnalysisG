from Closure import IO
from Closure import Event
from Closure import DataLoader
from Closure import Optimizer
from Closure import Metrics
from Closure import Exporter
from Closure import TopBuilder
from Closure import DelphesImplementation
from Functions.Event import CacheGenerators


def Test(F, **kargs):
    from Functions.IO.Files import WriteDirectory
    import sys
    import traceback
    test_dir = WriteDirectory()
    name = F.__name__
    var = F.__code__.co_varnames
    result = ""
    CallerDir = traceback.format_stack()[0].split("Test(")[1].split("." +name)[0]
    keys = list(set(list(var)).intersection(set(list(kargs))))
    driver = {} 
    for i in keys:
        driver[i] = kargs[i]
    try:
        if F(**driver):
            result = "(+) Passed: " + name + "\n"
        else:
            result = "(-) Failed: " + name + "\n"
    except:
        e = str(sys.exc_info()[1])
        traceback.print_tb(sys.exc_info()[2])
        result = "(-) Failed: " + name + "\n" + e
    
    print(result)
    test_dir.WriteTextFile(result, "_TestResults/" + CallerDir, name)


if __name__ == "__main__":
    GeneralDir = "/CERN/CustomAnalysisTopOutput/"
   
    ## ===== Test IO ===== ##
    #Test(IO.TestDir, di = GeneralDir)
    #Test(IO.TestReadSingleFile, dir_f = GeneralDir + "t/MCa.root") 
    #Test(IO.TestReadFile, di = GeneralDir)
    #Test(IO.TestFileConvertArray, di = GeneralDir)
    #Test(IO.TestHDF5ReadAndWriteParticle)
    #Test(IO.TestHDF5ReadAndWriteEvent, di = GeneralDir + "tttt/OldSample/1500_GeV/MCe/QU_0.root", Cache = True)

    ## ===== Test Cache ==== ##
    #Test(CacheGenerators.BuildCacheDirectory, Name = "tttt")
    #CacheGenerators.Generate_Cache(GeneralDir + "tttt/OldSample/1500_GeV/MCe/QU_0.root", Compiler = "tttt")
    #Test(CacheGenerators.BuildCacheDirectory, Name = "t")
    #CacheGenerators.Generate_Cache(GeneralDir + "t/MCa", Compiler = "t")

    ## ===== Test of EventGenerator ===== ##
    #Test(Event.TestEvents, di = GeneralDir + "tttt/OldSample/1500_GeV/MCe/QU_0.root")
    #Test(Event.TestParticleAssignment, di = GeneralDir + "tttt/OldSample/1500_GeV/MCe/QU_0.root")
    #Test(Event.TestSignalMultipleFile, di = GeneralDir + "tttt/OldSample/1500_GeV/MCe/")
    #Test(Event.TestSignalDirectory, di = GeneralDir + "t/MCa/")
    #Test(DelphesImplementation.TestDelphes, FileDir = "/CERN/Delphes/tag_1_delphes_events.root")

    ## ====== Test of DataLoader ====== ##
    #CacheGenerators.Generate_Cache(GeneralDir + "tttt/OldSample/1500_GeV/MCe/QU_0.root", Stop = 100, Compiler = "DataLoaderTest", Outdir = "_Pickle")
    #CacheGenerators.Generate_Cache(GeneralDir + "tttt/OldSample/1500_GeV/MCe/QU_1.root", Stop = 100, Compiler = "DataLoaderTest_1", Outdir = "_Pickle")
    #CacheGenerators.Generate_Cache(GeneralDir + "t/MCa", Stop = 100, Compiler = "DataLoaderTest_2", Outdir = "_Pickle") 

    #Test(DataLoader.TestEventGraph, Name = "DataLoaderTest/DataLoaderTest.pkl", Level = "TruthTops")
    #Test(DataLoader.TestEventGraph, Name = "DataLoaderTest/DataLoaderTest.pkl", Level = "TruthTopChildren")
    #Test(DataLoader.TestEventGraph, Name = "DataLoaderTest/DataLoaderTest.pkl", Level = "DetectorParticles")
    #
    #Test(DataLoader.TestDataLoader, Name = "DataLoaderTest/DataLoaderTest.pkl", Level = "TruthTops")
    #Test(DataLoader.TestDataLoader, Name = "DataLoaderTest/DataLoaderTest.pkl", Level = "TruthTopChildren")
    #Test(DataLoader.TestDataLoader, Name = "DataLoaderTest/DataLoaderTest.pkl", Level = "DetectorParticles")

    #Test(DataLoader.TestDataLoaderMixing, Files = ["DataLoaderTest", "DataLoaderTest_1", "DataLoaderTest_2"], Level = "TruthTops")

    ## ====== Test of Optimizer/Metrics ====== ##
    #Test(Optimizer.TestOptimizerGraph, Files = ["DataLoaderTest", "DataLoaderTest_1", "DataLoaderTest_2"], Level = "TruthTopChildren", Name = "GraphTest", CreateCache = True)
    #Test(Metrics.TestReadTraining, modelname = "GraphTest")

    #Test(Optimizer.TestOptimizerNode, Files = ["DataLoaderTest", "DataLoaderTest_1", "DataLoaderTest_2"], Level = "TruthTopChildren", Name = "NodeTest", CreateCache = True)
    #Test(Metrics.TestReadTraining, modelname = "NodeTest")

    #Test(Optimizer.TestOptimizerEdge, Files = ["DataLoaderTest", "DataLoaderTest_1", "DataLoaderTest_2"], Level = "TruthTopChildren", Name = "EdgeTest", CreateCache = True)
    #Test(Metrics.TestReadTraining, modelname = "EdgeTest")

    #Test(Optimizer.TestOptimizerCombined, Files = ["DataLoaderTest", "DataLoaderTest_1", "DataLoaderTest_2"], Level = "TruthTopChildren", Name = "CombinedTest", CreateCache = True)
    #Test(Metrics.TestReadTraining, modelname = "CombinedTest")
    #
    ## ======== Test Model/Data Exporting ======= #
    #Test(Exporter.TestModelExport, Files = ["DataLoaderTest"], Name = "ExportModel", Level = "TruthTopChildren", CreateCache = True)
    #Test(Exporter.TestEventGeneratorExport, File = GeneralDir + "t/MCa", Name = "EventGeneratorExport", CreateCache = True)
    #Test(Exporter.TestDataLoaderExport, Files = [GeneralDir + "tttt/OldSample/1500_GeV/MCe/QU_0.root", GeneralDir + "t/MCa"], CreateCache = True)
    #Test(Exporter.TestEventGeneratorWithDataLoader, Files = [GeneralDir + "tttt/OldSample/1500_GeV/MCe/QU_0.root", GeneralDir + "t/MCa"], CreateCache = True)
    #Test(TopBuilder.TestBuilder, Files = [GeneralDir + "tttt/OldSample/1500_GeV/MCe/QU_0.root", GeneralDir + "t/MCa"], CreateCache = True)



