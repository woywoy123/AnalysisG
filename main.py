from Closure.IO import TestDir, TestReadSingleFile, TestReadFile, TestFileConvertArray
from Closure.Event import TestEvents, TestParticleAssignment, TestSignalMultipleFile, TestSignalDirectory, TestAnomalousStatistics
from Closure.Plotting import TestTops, TestResonance, TestBackGroundProcesses, TestGNNMonitor, KinematicsPlotting, TopologicalComplexityMassPlot, TestDataSamples, TestWorkingExample4TopsComplexity
from Closure.DataLoader import TestEventGraphs, TestDataLoader, TestDataLoaderTrainingValidationTest, TestEventNodeEdgeFeatures
from Closure.GNN import SimpleFourTops, TestInvMassGNN_Children_Edge, TestInvMassGNN_Children_Node, TestPathNetGNN_Children_Edge, TestPathNetGNN_Children_Node, TestInvMassGNN_TruthJets, TestPathNetGNN_TruthJets, TestInvMassGNN_Tops_Edge, TestInvMassGNN_Tops_Node, TestInvMassGNN_Children_NoLep_Edge, TestInvMassGNN_Children_NoLep_Node
from Closure.Models import TestEdgeConvModel, TestGCNModel, TestInvMassGNN, TestPathNet
from Closure.TruthMatchingAnalysisTop import TestTopShapes, Test_ttbar, Test_tttt, Test_SingleTop


def Passed(F, name):
    if F:
        print("(+)Passed: "+name)
    else:
        print("(-)Failed: "+name)
        exit()


dir = "/home/tnom6927/Downloads/user.pgadow.310845.MGPy8EG.DAOD_TOPQ1.e7058_s3126_r10724_p3980.bsm4t-21.2.164-1-0-mc16e_output_root/user.pgadow.24765302._000001.output.root"
def Generate_Cache(di, Stop = -1, SingleThread = False, Compiler = "EventGenerator", Custom = False):
    from Functions.Event.EventGenerator import EventGenerator
    from Functions.IO.IO import PickleObject
    ev = EventGenerator(di, Stop = Stop)
    ev.SpawnEvents(Custom)
    ev.CompileEvent(SingleThread = SingleThread)
    PickleObject(ev, Compiler)

if __name__ == '__main__':
    #Generate_Cache(dir, Stop = -1, SingleThread = False, Compiler = "SignalSample.pkl")
    #Generate_Cache("/CERN/Grid/Samples/NAF/2021-05-05-2cRC-all/mc16a/postProcessed_ttbar_PhPy8_Total.root", Stop = 150000, SingleThread = True, Compiler = "ttbar.pkl")
    #Generate_Cache("/home/tnom6927/Downloads/SimpleTTBAR/Out_0/output.root", Compiler = "CustomSignalSample.pkl", Custom = True)
    #Generate_Cache("/CERN/CustomAnalysisTopOutput/tttt/tttt.root", Stop = 5000, Compiler = "tttt.pkl", Custom = True)
    #Generate_Cache("/CERN/CustomAnalysisTopOutput/ttbar/", Stop = 100, SingleThread = False, Compiler = "ttbar.pkl", Custom = True)
    #Generate_Cache("/CERN/CustomAnalysisTopOutput/t/SingleTop_S_Channel.root", Stop = 100, Compiler = "SingleTop_S.pkl", Custom = True)
    
    ## ====== Test of IO 
    #Passed(TestDir(), "TestDir")
    #Passed(TestReadSingleFile(), "TestReadSingleFile")
    #Passed(TestReadFile(), "TestReadFile")
    #Passed(TestFileConvertArray(), "TestFileConvertArray")
    #Passed(TestSignalMultipleFile(), "TestSignalMultipleFile")
    #Passed(TestSignalDirectory(), "TestSignalDirectory")

    ## ====== Test of EventGenerator 
    Generate_Cache(dir)
    Passed(TestEvents(), "TestEvents")
    Passed(TestParticleAssignment(), "TestParticleAssignment")
    Passed(TestAnomalousStatistics(), "TestAnomalousStatistics") 
    Passed(TestTops(), "TestTop")
    Passed(TestResonance(), "TestResonance")
    Passed(TestBackGroundProcesses(), "TestBackGroundProcesses")

    ## ====== Test of DataLoader
    #Passed(TestEventGraphs(), "TestEventGraphs")
    #Passed(TestDataLoader(), "TestDataLoader")
    #Passed(TestDataLoaderTrainingValidationTest(), "TestDataLoaderTrainingValidationTest")
    #Passed(TestEventNodeEdgeFeatures(), "TestEventNodeEdgeFeatures")
    
    # ====== Test of Optimizer
    #Passed(SimpleFourTops(), "SimpleFourTops")

    # ====== Test of Plotting 
    #Passed(TestGNNMonitor(), "TestGNNMonitor")
    #Passed(KinematicsPlotting(), "KinematicsPlotting")
    #Passed(TopologicalComplexityMassPlot(), "TopologicalComplexityMassPlot")
    #Passed(TestDataSamples(), "TestDataSamples")
    #Passed(TestWorkingExample4TopsComplexity(), "TestWorkingExample4TopsComplexity")
    
    # ====== Test of GNN Model implementations 
    #Passed(TestEdgeConvModel(), "TestEdgeConvModel")
    #Passed(TestGCNModel(), "TestGCNModel")
    #Passed(TestInvMassGNN(), "TestInvMassGNN")
    #Passed(TestPathNet(), "TestPathNet") 

    # ====== Evaluation of Models ======== #
    #Passed(TestInvMassGNN_Tops_Edge(), "TestInvMassGNN_Tops_Edge")
    #Passed(TestInvMassGNN_Tops_Node(), "TestInvMassGNN_Tops_Node")

    #Passed(TestInvMassGNN_Children_Edge(), "TestInvMassGNN_Children_Edge")
    #Passed(TestInvMassGNN_Children_Node(), "TestInvMassGNN_Children_Node")

    #Passed(TestInvMassGNN_Children_NoLep_Edge(), "TestInvMassGNN_Children_Edge")
    #Passed(TestInvMassGNN_Children_NoLep_Node(), "TestInvMassGNN_Children_Node")

    Passed(TestInvMassGNN_TruthJets(), "TestInvMassGNN_TruthJets") 

    #Passed(TestPathNetGNN_Children_Edge(), "TestPathNetGNN_Children_Edge") 
    #Passed(TestPathNetGNN_Children_Node(), "TestPathNetGNN_Children_Node") 
    #Passed(TestPathNetGNN_TruthJets(), "TestPathNetGNN_TruthJets") 


    # ====== Truth Debugging Stuff ======== #
    #Passed(TestTopShapes(), "TestTopShapes")
    #Passed(Test_tttt(), "Test_tttt")
    #Passed(Test_ttbar(), "Test_ttbar")
    #Passed(Test_SingleTop(), "Test_SingleTop")




