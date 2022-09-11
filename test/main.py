import sys
sys.path.append("../")
import IO
import Event
import DataLoader
import Optimizer
import TopBuilder
import Exporter
from AnalysisTopGNN.Generators import CacheGenerators

#import EventImplementations
#import FeatureTester
#import ModelProofOfConcept
#import Analysis

def Test(F, **kargs):
    from AnalysisTopGNN.IO import WriteDirectory
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
        exit()
    print(result)
    test_dir.WriteTextFile(result, "_TestResults/" + CallerDir +  "/", name)


if __name__ == "__main__":
    GeneralDir = "/CERN/CustomAnalysisTopOutputTest/"
    
    ## ===== Test IO ===== #
    #Test(IO.TestReadROOTNominal, file = GeneralDir + "/tttt/QU_0.root")
    #Test(IO.TestReadROOTDelphes, file = "/CERN/Delphes/tag_1_delphes_events.root")
    #Test(IO.TestHDF5ReadAndWriteParticle)
    #Test(IO.TestHDF5ReadAndWriteEvent, di = GeneralDir + "tttt/QU_0.root", Cache = True)

    ## ===== Test of EventGenerator ===== ##
    #Test(Event.TestEvents, di = GeneralDir + "tttt/QU_0.root")
    #Test(Event.TestSignalMultipleFile, di = GeneralDir + "tttt/")
    #Test(Event.TestSignalDirectory, di = GeneralDir + "t/")

    ## ====== Test of DataLoader ====== ##
    #CacheGenerators.Generate_Cache(GeneralDir + "tttt/QU_0.root", Stop = 100, Compiler = "DataLoaderTest", Outdir = "_Pickle")
    #CacheGenerators.Generate_Cache(GeneralDir + "tttt/QU_1.root", Stop = 100, Compiler = "DataLoaderTest_1", Outdir = "_Pickle")
    #CacheGenerators.Generate_Cache(GeneralDir + "t/", Stop = 100, Compiler = "DataLoaderTest_2", Outdir = "_Pickle") 
    
    #Test(DataLoader.TestEventGraph, Name = "DataLoaderTest.pkl", Level = "TruthTops")
    #Test(DataLoader.TestEventGraph, Name = "DataLoaderTest.pkl", Level = "TruthTopChildren")
    #Test(DataLoader.TestEventGraph, Name = "DataLoaderTest.pkl", Level = "DetectorParticles")
    #
    #Test(DataLoader.TestDataLoader, Name = "DataLoaderTest.pkl", Level = "TruthTops")
    #Test(DataLoader.TestDataLoader, Name = "DataLoaderTest.pkl", Level = "TruthTopChildren")
    #Test(DataLoader.TestDataLoader, Name = "DataLoaderTest.pkl", Level = "DetectorParticles")

    #Test(DataLoader.TestDataLoaderMixing, Files = ["DataLoaderTest", "DataLoaderTest_1", "DataLoaderTest_2"], Level = "TruthTops")

    ## ====== Test of Optimizer/Metrics ====== ##
    #Test(Optimizer.TestOptimizerGraph, Files = ["DataLoaderTest", "DataLoaderTest_1", "DataLoaderTest_2"], Level = "TruthTopChildren", Name = "GraphTest", CreateCache = True)
    #Test(Optimizer.TestReadTraining, modelname = "GraphTest")

    #Test(Optimizer.TestOptimizerNode, Files = ["DataLoaderTest", "DataLoaderTest_1", "DataLoaderTest_2"], Level = "TruthTopChildren", Name = "NodeTest", CreateCache = True)
    #Test(Optimizer.TestReadTraining, modelname = "NodeTest")

    #Test(Optimizer.TestOptimizerEdge, Files = ["DataLoaderTest", "DataLoaderTest_1", "DataLoaderTest_2"], Level = "TruthTopChildren", Name = "EdgeTest", CreateCache = True)
    #Test(Optimizer.TestReadTraining, modelname = "EdgeTest")

    #Test(Optimizer.TestOptimizerCombined, Files = ["DataLoaderTest", "DataLoaderTest_1", "DataLoaderTest_2"], Level = "TruthTopChildren", Name = "CombinedTest", CreateCache = True)
    #Test(Optimizer.TestReadTraining, modelname = "CombinedTest")
 
    ## ======== Test Model/Data Exporting ======= #
    #Test(Exporter.TestModelExport, Files = ["DataLoaderTest"], Name = "ExportModel", Level = "TruthTopChildren", CreateCache = True)
    #Test(Exporter.TestEventGeneratorExport, File = GeneralDir + "t/", Name = "EventGeneratorExport", CreateCache = True)
    #Test(Exporter.TestDataLoaderExport, Files = [GeneralDir + "tttt/QU_0.root", GeneralDir + "t/"], CreateCache = True)
    #Test(Exporter.TestEventGeneratorWithDataLoader, Files = [GeneralDir + "tttt/QU_0.root", GeneralDir + "t/"], CreateCache = True)
    Test(TopBuilder.TestBuilder, Files = [GeneralDir + "tttt/QU_0.root", GeneralDir + "t/"], CreateCache = False)

    # ===== Test Event Implementations ===== #
    #Test(EventImplementations.TestExperiemental, FileDir = "/CERN/CustomAnalysisTopOutputSameSignDilepton/Merger/QU_0.root")



    # ========== Test Entire Aggregation of Framework =========== #
    #Files = { 
    #            "4-Tops" :  GeneralDir + "tttt/MCe/", 
    #            "SingleTop" : GeneralDir + "t/MCa/", 
    #            "ttbar" : GeneralDir + "ttbar/MCa/QU_0.root", 
    #            "Zmumu" : GeneralDir + "Zmumu/MCd/"
    #        }
    #Test(Analysis.TestUnificationEventGenerator, FileDir = GeneralDir, Files = Files)
    #Test(Analysis.TestUnificationDataLoader)
    #Test(Analysis.TestUnificationOptimizer)
    #Test(Analysis.TestUnificationSubmission)

    ## ========= Presentation Plots ============ #
    #Test(Presentation1.CreatePlots, FileDir = GeneralDir + "tttt/MCe/", CreateCache = True)

