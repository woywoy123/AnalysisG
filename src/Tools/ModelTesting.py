from AnalysisTopGNN.Generators import Generate_Cache_Batches, GenerateDataLoader
from AnalysisTopGNN.IO import WriteDirectory, Directories, UnpickleObject, PickleObject
from AnalysisTopGNN.Events import *
import inspect 

def AddFeature(Prefix, dic):
    return {Prefix + "_" + i : dic[i] for i in dic} 

def CreateWorkspace(Files, DataFeatures, Cache, Stop, ProcessName, Level, selfloop = False):
    CallerName = inspect.stack()[1].function
    Outdir = "_Cache/" + CallerName
    
    if Cache:
        x = WriteDirectory()
        x.MakeDir(Outdir)
    
    Out = []
    for i, j in zip(Files, ProcessName):
        Out += Generate_Cache_Batches(i, Stop = Stop, Compiler = j, OutDirectory = Outdir, CreateCache = Cache)
   
    if Cache:
        DL = GenerateDataLoader()
        if Level == "TruthTops":
            DL.EventGraph = EventGraphTruthTops
        elif Level == "TruthTopChildren":
            DL.EventGraph = EventGraphTruthTopChildren
        elif Level == "DetectorParticles":
            DL.EventGraph = EventGraphDetector
        elif Level == "TruthJetLepton":
            DL.EventGraph = EventGraphTruthJetLepton

        DL.SetDevice("cuda")
        for key, fx in DataFeatures.items():
            if "GT_" == key[0:3]:
                DL.AddGraphTruth(key[3:], fx)

            if "NT_" == key[0:3]:
                DL.AddNodeTruth(key[3:], fx)

            if "ET_" == key[0:3]:
                DL.AddEdgeTruth(key[3:], fx)
   

            if "GF_" == key[0:3]:
                DL.AddGraphFeature(key[3:], fx)

            if "NF_" == key[0:3]:
                DL.AddNodeFeature(key[3:], fx)

            if "EF_" == key[0:3]:
                DL.AddEdgeFeature(key[3:], fx)
 

            if "GP_" == key[0:3]:
                DL.AddGraphPreprocessing(key[3:], fx)

            if "NP_" == key[0:3]:
                DL.AddNodePreprocessing(key[3:], fx)

            if "EP_" == key[0:3]:
                DL.AddEdgePreprocessing(key[3:], fx)

        for i in Out:
            ev = UnpickleObject(i)
            DL.AddSample(ev, "nominal", selfloop, True)
        DL.MakeTrainingSample(10)
        PickleObject(DL, "DataLoader", Outdir)
    return UnpickleObject("DataLoader", Outdir)

def OptimizerTemplate(DataLoader, Model):
    from AnalysisTopGNN.Generators import Optimizer

    Op = Optimizer(DataLoader)
    Op.Verbose = False
    Op.ONNX_Export = False
    Op.TorchScript_Export = False
    Op.Model = Model
    Op.DefineOptimizer()
    N_Nodes = list(Op.TrainingSample)
    N_Nodes.sort(reverse = True)
    Op.Sample = Op.TrainingSample[N_Nodes[0]][0]
    Op.InitializeModel()
    Op.GetTruthFlags(Op.EdgeAttribute, "E")
    Op.GetTruthFlags(Op.NodeAttribute, "N")
    Op.GetTruthFlags(Op.GraphAttribute, "G")
    return Op

def KillCondition(Variable, TestIndex, Optimizer, Samples, Iterations, sleep = -1, batched = 1):
    import torch
    import time
    from torch_geometric.loader import DataLoader
    
    def Classification(truth, model):
        return int(torch.sum(torch.eq(truth[0], model[0]))) == len(truth[0])
    
    def Regression(truth, model):
        tru = sum([float(k) for k in truth[0]])
        dif = sum([abs(float(k-j)) for k, j in zip(truth[0], model[0].t())])
        return abs(round(dif/tru, 5)) <= 1e-4
    
    Passed = False
    for k in range(Iterations):
        Optimizer.Debug = "Loss"
        if isinstance(Samples, dict):
            Sample = []
            for inpt in Samples: 
                for i in DataLoader(Samples[inpt], batch_size = batched, shuffle = False):
                    Optimizer.Train(i)
        else:
            for i in DataLoader(Samples, batch_size = batched, shuffle = False):
                Optimizer.Train(i)

        if k/TestIndex - int(k/TestIndex) == 0:
            Optimizer.Debug = True
            truth, model = Optimizer.Train(i)
            
            Pass = 0
            for key, cl in Variable.items():
                if cl == "C":
                    Passed = Classification(truth[key], model[key])
                if cl == "R":
                    Passed = Regression(truth[key], model[key])
                if Passed:
                    Pass += 1
            if sleep > 0:
                time.sleep(sleep)
            if Pass == len(list(Variable)):
                return True
    return Passed





