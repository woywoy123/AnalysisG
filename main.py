from Closure.IO import TestDir, TestReadSingleFile, TestReadFile, TestFileConvertArray
from Closure.Event import TestEvents, TestParticleAssignment, TestSignalMultipleFile, TestSignalDirectory, TestAnomalousStatistics
from Closure.Plotting import TestTops, TestResonance, TestBackGroundProcesses, TestGNNMonitor
from Closure.DataLoader import TestEventGraphs, TestDataLoader, TestDataLoaderTrainingValidationTest, TestEventNodeEdgeFeatures
from Closure.GNN import SimpleFourTops
from Closure.Models import TestEdgeConvModel

def Passed(F, name):
    if F:
        print("(+)Passed: "+name)
    else:
        print("(-)Failed: "+name)


dir = "/home/tnom6927/Downloads/user.pgadow.310845.MGPy8EG.DAOD_TOPQ1.e7058_s3126_r10724_p3980.bsm4t-21.2.164-1-0-mc16e_output_root/user.pgadow.24765302._000001.output.root"
def Generate_Cache(di, Stop = -1, SingleThread = False, Compiler = "EventGenerator"):
    from Functions.Event.Event import EventGenerator
    from Functions.IO.IO import PickleObject
    ev = EventGenerator(di, Stop = Stop)
    ev.SpawnEvents()
    ev.CompileEvent(SingleThread = SingleThread)
    PickleObject(ev, Compiler)

if __name__ == '__main__':
    # ====== Test of IO 
    #Passed(TestDir(), "TestDir")
    #Passed(TestReadSingleFile(), "TestReadSingleFile")
    #Passed(TestReadFile(), "TestReadFile")
    #Passed(TestFileConvertArray(), "TestFileConvertArray")
    #Passed(TestSignalMultipleFile(), "TestSignalMultipleFile")
    #Passed(TestSignalDirectory(), "TestSignalDirectory")

    # ====== Test of EventGenerator 
    #Generate_Cache(dir)
    #Passed(TestEvents(), "TestEvents")
    #Passed(TestParticleAssignment(), "TestParticleAssignment")
    #Passed(TestAnomalousStatistics(), "TestAnomalousStatistics") # <--- Need to checkout detector particle matching 
    # ====== Test of DataLoader
    #Generate_Cache(dir, Stop = 1000, SingleThread = True, Compiler = "SignalSample.pkl")
    #Generate_Cache("/CERN/Grid/Samples/NAF/2021-05-05-2cRC-all/mc16a/postProcessed_ttbar_PhPy8_Total.root", Stop = 1000, SingleThread = True, Compiler = "ttbar.pkl")
    #Passed(TestEventGraphs(), "TestEventGraphs")
    #Passed(TestDataLoader(), "TestDataLoader")
    #Passed(TestDataLoaderTrainingValidationTest(), "TestDataLoaderTrainingValidationTest")
    #Passed(TestEventNodeEdgeFeatures(), "TestEventNodeEdgeFeatures")
    
    # ====== Test of Optimizer
    #Passed(SimpleFourTops(), "SimpleFourTops")

    # ====== Test of Plotting 
    #Passed(TestGNNMonitor(), "TestGNNMonitor")

    # ====== Test of GNN Model implementations 
    Passed(TestEdgeConvModel(), "TestEdgeConvModel")
        


