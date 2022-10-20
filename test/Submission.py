from AnalysisTopGNN.Events import EventGraphTruthTopChildren, Event
from AnalysisTopGNN.Generators import Analysis
from AnalysisTopGNN.Submission import Condor
from Templates.EventFeatureTemplate import ApplyFeatures


def TestAnalysis(GeneralDir):

    def Test(ev):
        return int(len(ev.TruthTops) == 4)

    def EventGen(Dir, Name):
        Ana = Analysis()
        Ana.ProjectName = "TMPProject"
        Ana.InputSample(Name, Dir)
        Ana.EventCache = True
        Ana.Event = Event
        Ana.Threads = 3
        Ana.EventStop = 9
        Ana.DumpHDF5 = False
        Ana.DumpPickle = True
        Ana.Launch()
        return Ana

    def DataGen(Name):
        Ana = Analysis()
        Ana.ProjectName = "TMPProject"
        Ana.InputSample(Name)
        Ana.DataCache = True
        Ana.EventGraph = EventGraphTruthTopChildren
        Ana.AddGraphFeature(Test)
        Ana.Threads = 10
        Ana.EventStop = None
        Ana.DumpHDF5 = True
        Ana.DumpPickle = False
        Ana.Launch()
        return Ana
    
    ev = EventGen(GeneralDir + "/t", "SingleTop")
    ev += EventGen(GeneralDir + "/ttbar", "ttbar")
    ev += EventGen(GeneralDir + "/tttt", "Signal")
    ev += EventGen(GeneralDir + "/Zmumu", "Zmumu")
    ev += EventGen([GeneralDir + "/t", GeneralDir + "/ttbar"], "Combined")
    
    gr = DataGen("SingleTop")
    gr += DataGen("ttbar")
    gr += DataGen("Signal")
    gr += DataGen("Zmumu")
        
    Objects0 = {}
    for i in ev:
        Objects0[i.Filename] = i
    
    Objects1 = {}
    for i in gr:
        Objects1[i.Filename] = i

    print(len(Objects1), len(Objects0))


    for i in Objects0:
        if i not in Objects1:
            print(i, Objects0[i])
            return False
            continue
        del Objects1[i]
        Objects0[i] = True
    
    return len(Objects1) == sum([0 if Objects0[i] else 1 for i in Objects0 ])


def TestCondorDumping(GeneralDir):
    nfs = GeneralDir
    out = "/home/tnom6927/Dokumente/Project/Analysis/bsm4tops-gnn-analysis/AnalysisTopGNN/test"

    T = Condor()
    T.SkipEventCache = False
    T.SkipDataCache = False
    T.OutputDirectory = out
    T.Tree = "nominal"
    T.ProjectName = "TopEvaluation"

    ## Job for creating samples
    #A1 = Analysis()
    #A1.Threads = 4
    #A1.EventCache = True
    #A1.DumpPickle = True
    #A1.Event = Event
    #A1.InputSample("ttbar", nfs + "ttbar")
    #T.AddJob("ttbar", A1, "10GB", "1h")

    #A2 = Analysis()
    #A2.Threads = 4
    #A2.EventCache = True
    #A2.DumpPickle = True
    #A2.Event = Event
    #A2.InputSample("zmumu", nfs + "Zmumu")
    #T.AddJob("Zmumu", A2, "10GB", "1h")
 
    A3 = Analysis()
    A3.Threads = 4
    A3.EventCache = True
    A3.DumpPickle = True
    A3.Event = Event
    A3.InputSample("t", nfs + "t")
    T.AddJob("t", A3, "10GB", "1h")

    #A4 = Analysis()
    #A4.Threads = 4
    #A4.EventCache = True
    #A4.DumpPickle = True
    #A4.Event = Event
    #A4.InputSample("tttt", nfs + "tttt")
    #T.AddJob("tttt", A4, "10GB", "1h")

    # Job for creating Dataloader
    D1 = Analysis()
    D1.DataCache = True
    D1.DumpHDF5 = True
    D1.Threads = 12
    D1.EventGraph = EventGraphTruthTopChildren
    D1.InputSample("t")
    ApplyFeatures(D1, "TruthChildren")
    T.AddJob("t_Data", D1, "10GB", "1h", "t")

    # Job for creating Dataloader
    #D2 = Analysis()
    #D2.DataCache = True
    #D2.DumpHDF5 = True
    #D2.Threads = 12
    #D2.EventGraph = EventGraphTruthJetLepton
    #D2.InputSample("tttt")
    #TruthJetsFeatures(D2)

    # Job for creating TrainingSample
    #T2 = Analysis()
    #T2.InputSample("tttt")
    #T2.InputSample("t")
    #T2.MergeSamples = True
    #T2.GenerateTrainingSample = True


    # Job for optimization
    #Op = Analysis()
    #Op.ProjectName = "TopEvaluation"
    #Op.Device = "cuda"
    #Op.TrainWithoutCache = True
    #Op.LearningRate = 0.0001
    #Op.WeightDecay = 0.0001
    #Op.kFold = 2
    #Op.Epochs = 3
    #Op.BatchSize = 20
    #Op.RunName = "BasicBaseLineTruthJet"
    #Op.ONNX_Export = True
    #Op.TorchScript_Export = True
    #Op.Model = BasicBaseLineTruthJet()



    #T.AddJob("tData", D1, "10GB", "1h", ["t", "Zmumu"])
    #T.AddJob("ZmumuData", D2, "10GB", "1h", ["t", "Zmumu"])
    #T.AddJob("DataTraining", T2, "10GB", "1h", ["tData", "ZmumuData"])

    #T.AddJob("TruthJet", Op, "10GB", "1h", ["DataTraining"])

    #T.LocalDryRun() 
    T.DumpCondorJobs() 
 
