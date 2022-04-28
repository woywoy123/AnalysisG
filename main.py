from Closure.IO import TestDir, TestReadSingleFile, TestReadFile, TestFileConvertArray
from Closure.Event import TestEvents, TestParticleAssignment, TestSignalMultipleFile, TestSignalDirectory
from Closure.Plotting import TestTops, TestResonance, TestBackGroundProcesses, TestGNNMonitor, KinematicsPlotting, TopologicalComplexityMassPlot, TestDataSamples, TestWorkingExample4TopsComplexity
from Closure.DataLoader import TestEventGraphs, TestDataLoader, TestDataLoaderTrainingValidationTest, TestEventNodeEdgeFeatures
from Closure.GNN import SimpleFourTops, TestInvMassGNN_Children_Edge, TestInvMassGNN_Children_Node, TestPathNetGNN_Children_Edge, TestPathNetGNN_Children_Node, TestInvMassGNN_TruthJets, TestPathNetGNN_TruthJets, TestInvMassGNN_Tops_Edge, TestInvMassGNN_Tops_Node, GenerateTemplate, TestPathNetGNN_Tops_Edge
from Closure.Models import TestEdgeConvModel, TestGCNModel, TestInvMassGNN, TestPathNet, TestJetMergingTagging
from Closure.TruthMatchingAnalysisTop import Test_SimilarityCustomOriginalMethods, Test_SimilarityCustomOriginalMethods_Plot
from Closure.Benchmarking import Combinatorials, LorentzVectorBenchmark
from Closure.ParticlePropertiesMonteCarlo import JetMergingFrequency, JetMergingFrequencyFraction, JetMergingFrequencyFractionPlot, FragmentationOfTriplets, FragmentationOfTripletsScanning, TopJetTrajectory
import os


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

def Generate_Cache_Batches(di, Stop = -1, SingleThread = False, Compiler = "EventGenerator", Custom = False, CustomDirectory = "_Cache"):
    from Functions.Event.EventGenerator import EventGenerator
    from Functions.IO.IO import PickleObject
    from Functions.IO.Files import WriteDirectory, Directories 
    import os

    def Compile(File, Name):
        ev = EventGenerator(File, Stop = Stop)
        ev.SpawnEvents(Custom)
        ev.CompileEvent(SingleThread = SingleThread)
        PickleObject(ev, Name)
    
    if CustomDirectory != "_Cache":
        d = CustomDirectory + "/" + Compiler + "_Cache/"
        k = ""
        for i in d.split("/"):
            k += "/"+i
            try:
                os.mkdir(k)   
            except FileExistsError:
                pass
            except PermissionError:
                pass
        os.chdir(CustomDirectory + "/" + Compiler + "_Cache/")   
    else:
        MKDIR = WriteDirectory()
        MKDIR.MakeDir(CustomDirectory + "/" + Compiler + "_Cache")

    d = Directories(di)
    Files = d.ListFilesInDir(di)
    for f in Files:
        Compile(di + "/" + f, CustomDirectory + "/" + Compiler + "_Cache/"+f.replace(".root", ".pkl"))



if __name__ == '__main__':
    #x = os.getcwd().split("/")
    #Generate_Cache(dir, Stop = -1, SingleThread = False, Compiler = "SignalSample.pkl")
    #Generate_Cache("/CERN/Grid/Samples/NAF/2021-05-05-2cRC-all/mc16a/postProcessed_ttbar_PhPy8_Total.root", Stop = 150000, SingleThread = True, Compiler = "ttbar.pkl")
    #Generate_Cache("/CERN/CustomAnalysisTopOutput/TMP_DELETE_AFTER/output.root", Stop = 100, SingleThread = True, Compiler = "CustomSignalSample.pkl", Custom = True)
    #Generate_Cache("/CERN/CustomAnalysisTopOutput/tttt/OldSample/1500_GeV/MCe.root", Stop = 1000, SingleThread = True, Compiler = "CustomSignalSample.pkl", Custom = True)
    #Generate_Cache("/CERN/CustomAnalysisTopOutput/ttbar/MCa/", Stop = -1, SingleThread = True, Compiler = "ttbar.pkl", Custom = True)
    #Generate_Cache("/CERN/CustomAnalysisTopOutput/t/MCa/", Stop = -1, Compiler = "SingleTop_S.pkl", Custom = True)
   
    #for i in ["a", "d", "e"]:
    #    for j in range(6):
    #        Energy = ["1000", "1250", "1500", "2000", "2500", "3000"]
    #        Generate_Cache_Batches("/CERN/CustomAnalysisTopOutput/tttt/NewSample/" + Energy[j] + "_GeV/MC" + i, Stop = -1, SingleThread = False, Compiler = "CustomSample_tttt_" + Energy[j] + "_MC_" + i, Custom = True)

    #Generate_Cache_Batches("/CERN/CustomAnalysisTopOutput/t/MCa/", Stop = -1, SingleThread = True, Compiler = "CustomSample_t", Custom = True)
    #Generate_Cache_Batches("/CERN/CustomAnalysisTopOutput/ttbar/MCa/", Stop = -1, SingleThread = False, Compiler = "CustomSample_ttbar", Custom = True)
    #Generate_Cache_Batches("/CERN/CustomAnalysisTopOutput/tttt/OldSample/1500_GeV/MCe/", SingleThread = True, Stop = -1, Compiler = "CustomSample_tttt", Custom = True)
    
    ## ====== Test of IO 
    #Passed(TestDir(), "TestDir")
    #Passed(TestReadSingleFile(), "TestReadSingleFile")
    #Passed(TestReadFile(), "TestReadFile")
    #Passed(TestFileConvertArray(), "TestFileConvertArray")
    #Passed(TestSignalMultipleFile(), "TestSignalMultipleFile")
    #Passed(TestSignalDirectory(), "TestSignalDirectory")

    ## ====== Test of EventGenerator 
    #Generate_Cache(dir)
    #Passed(TestEvents(), "TestEvents")
    #Passed(TestParticleAssignment(), "TestParticleAssignment")
    #Passed(TestTops(), "TestTop")
    #Passed(TestResonance(), "TestResonance")
    #Passed(TestBackGroundProcesses(), "TestBackGroundProcesses")

    ### ====== Test of DataLoader
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
    # ---> Passed(TestJetMergingTagging(), "TestJetMergingTagging")

    # ====== Custom Code Benchmarks ======== #
    #Passed(Combinatorials(), "Combinatorials")
    #Passed(LorentzVectorBenchmark(), "LorentzVectorBenchmark")
   
    # ====== Monte Carlo Particle Properties ====== #
    #Passed(JetMergingFrequency("_Cache/CustomSample_tttt_Cache"), "JetMergingFrequency")
    #Passed(JetMergingFrequencyFraction("_Cache/CustomSample_tttt_Cache"), "JetMergingFrequencyFraction")
    #Passed(JetMergingFrequencyFractionPlot("_Pickles/Fraction_of_Jets_Tops_100.0.pkl"), "JetMergingFrequencyFractionPlot")
    #Passed(FragmentationOfTriplets("_Cache/CustomSample_tttt_Cache"), "FragmentationOfTriplets")
    #Passed(FragmentationOfTripletsScanning("_Cache/CustomSample_tttt_Cache", "_Pickles/Fragmentation_Triplets100.0.pkl"), "FragmentationOfTripletsScannning")
    #Passed(TopJetTrajectory("_Cache/CustomSample_tttt_Cache"), "TopJetTrajectory")

    # ====== Evaluation of Models ======== #
    #GenerateTemplate(Tree = "TruthTops")
    #Passed(TestInvMassGNN_Tops_Edge(), "TestInvMassGNN_Tops_Edge")
    #Passed(TestPathNetGNN_Tops_Edge(), "PathNetGNN_Tops_Edge")
    #Passed(TestInvMassGNN_Tops_Node(), "TestInvMassGNN_Tops_Node")
    
    GenerateTemplate(SignalSample = "CustomSignalSample.pkl", Level = "TruthJetsLep")
    #Passed(TestInvMassGNN_Children_Edge(), "TestInvMassGNN_Children_Edge")
    #Passed(TestInvMassGNN_Children_Node(), "TestInvMassGNN_Children_Node")
    
    Passed(TestPathNetGNN_Children_Edge(), "TestPathNetGNN_Children_Edge") 
    #Passed(TestPathNetGNN_Children_Node(), "TestPathNetGNN_Children_Node") 

    #GenerateTemplate(SignalSample = "tttt.pkl", Tree = "TruthJetsLep", Additional_Samples = ["ttbar.pkl", "SingleTop_S.pkl"], OutputName = "LoaderSignalSample.pkl")
    #Passed(TestInvMassGNN_TruthJets(), "TestInvMassGNN_TruthJets") 
    #Passed(TestPathNetGNN_TruthJets(), "TestPathNetGNN_TruthJets") 

    
    # ====== Truth Debugging Stuff ======== #
    #i = ["1000", "1250", "1500", "2000", "2500", "3000"]
    #for k in i: 
    #li = ["tttt", "ttbar"]
    #for i in li:
    #    Passed(Test_SimilarityCustomOriginalMethods(i), "Test_SimilarityCustomOriginalMethods")
    #    Passed(Test_SimilarityCustomOriginalMethods_Plot(i), "Test_SimilarityCustomOriginalMethods_Plot")
    #pass

