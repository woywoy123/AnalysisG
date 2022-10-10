from Templates.TruthJetFeatures import TruthJetsFeatures
from AnalysisTopGNN.Submission import Condor
from AnalysisTopGNN.Events import EventGraphTruthTopChildren, Event, EventGraphTruthJetLepton
from AnalysisTopGNN.Generators import Analysis
from TrivialModels import GraphNN
from TestModelCondor.Model import BasicBaseLineTruthJet

def TestCustom(GeneralDir):

    def Test(ev):
        return int(len(ev.TruthTops) == 4)

    def EventGen(Dir, Name, Cache1, Cache2):
        Ana = Analysis()
        Ana.InputSample(Name, Dir)
        Ana.EventCache = Cache1
        Ana.DataCache = Cache2
        Ana.Event = Event
        Ana.Tree = "nominal"
        Ana.Threads = 10
        Ana.EventEnd = 100
        Ana.DumpHDF5 = True
        Ana.ProjectName = "TMPProject"
        Ana.EventGraph = EventGraphTruthTopChildren
        Ana.AddGraphFeature("Signal", Test)
        Ana.AddGraphTruth("Signal", Test)
        Ana.Launch()

    Ev = False
    EventGen(GeneralDir + "/t", "SingleTop", Ev, False)
    EventGen(GeneralDir + "/ttbar", "ttbar", Ev, False)
    EventGen(GeneralDir + "/tttt", "Signal", Ev, False)
    EventGen(GeneralDir + "/Zmumu", "Zmumu", Ev, False)
    
    Ev = True
    EventGen(GeneralDir + "/t", "SingleTop", False, Ev)
    EventGen(GeneralDir + "/ttbar", "ttbar", False, Ev)
    EventGen(GeneralDir + "/tttt", "Signal", False, Ev)
    EventGen(GeneralDir + "/Zmumu", "Zmumu", False, Ev)

    Ana = Analysis()
    Ana.InputSample("SingleTop")
    Ana.InputSample("ttbar")
    Ana.InputSample("Signal")
    Ana.InputSample("Zmumu")
    Ana.EventCache = False
    Ana.DataCache = False
    Ana.DumpHDF5 = False
    Ana.MergeSamples = True
    Ana.GenerateTrainingSample = True
    Ana.Threads = 10
    Ana.ProjectName = "TMPProject"
    Ana.Model = GraphNN()
    Ana.Launch()

def TestSubmission():

    nfs = "/CERN/"
    out = "/home/tnom6927/Dokumente/Project/Analysis/bsm4tops-gnn-analysis/AnalysisTopGNN/test"

    # Job for creating samples
    A1 = Analysis()
    A1.Threads = 4
    A1.EventCache = True
    A1.Event = Event
    TruthJetsFeatures(A1)
    A1.InputSample("ttbar", nfs + "CustomAnalysisTopOutputTest/ttbar")


    A2 = Analysis()
    A2.Threads = 4
    A2.EventCache = True
    A2.Event = Event
    TruthJetsFeatures(A2)
    A2.InputSample("zmumu", nfs + "CustomAnalysisTopOutputTest/Zmumu")

    
    A3 = Analysis()
    A3.Threads = 4
    A3.EventCache = True
    A3.Event = Event
    TruthJetsFeatures(A3)
    A3.InputSample("t", nfs + "CustomAnalysisTopOutputTest/t")

    A4 = Analysis()
    A4.Threads = 4
    A4.EventCache = True
    A4.Event = Event
    TruthJetsFeatures(A4)
    A4.InputSample("tttt", nfs + "CustomAnalysisTopOutputTest/tttt")


    # Job for creating Dataloader
    D1 = Analysis()
    D1.DataCache = True
    D1.DumpHDF5 = True
    D1.Threads = 12
    D1.EventGraph = EventGraphTruthJetLepton
    D1.InputSample("t")
    TruthJetsFeatures(D1)

    # Job for creating Dataloader
    D2 = Analysis()
    D2.DataCache = True
    D2.DumpHDF5 = True
    D2.Threads = 12
    D2.EventGraph = EventGraphTruthJetLepton
    D2.InputSample("tttt")
    TruthJetsFeatures(D2)

    # Job for creating TrainingSample
    T2 = Analysis()
    T2.InputSample("tttt")
    T2.InputSample("t")
    T2.MergeSamples = True
    T2.GenerateTrainingSample = True


    # Job for optimization
    Op = Analysis()
    Op.ProjectName = "TopEvaluation"
    Op.Device = "cuda"
    Op.TrainWithoutCache = True
    Op.LearningRate = 0.0001
    Op.WeightDecay = 0.0001
    Op.kFold = 2
    Op.Epochs = 3
    Op.BatchSize = 20
    Op.RunName = "BasicBaseLineTruthJet"
    Op.ONNX_Export = True
    Op.TorchScript_Export = True
    Op.Model = BasicBaseLineTruthJet()

    T = Condor()
    T.DisableEventCache = True
    T.DisableDataCache = False
    T.OutputDirectory = out
    T.Tree = "nominal"
    T.ProjectName = "TopEvaluation"
    T.AddJob("ttbar", A1, "10GB", "1h")
    T.AddJob("Zmumu", A2, "10GB", "1h")
    T.AddJob("t", A3, "10GB", "1h")
    T.AddJob("tttt", A4, "10GB", "1h")

    T.AddJob("tData", D1, "10GB", "1h", ["t", "Zmumu"])
    T.AddJob("ZmumuData", D2, "10GB", "1h", ["t", "Zmumu"])
    T.AddJob("DataTraining", T2, "10GB", "1h", ["tData", "ZmumuData"])

    T.AddJob("TruthJet", Op, "10GB", "1h", ["DataTraining"])

    T.LocalDryRun() 
    T.DumpCondorJobs() 
 

if __name__ == '__main__':
    Dir = "/CERN/Delphes/"
    #TestCustom("/CERN/CustomAnalysisTopOutputTest/")
    TestSubmission()
