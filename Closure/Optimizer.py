from Functions.IO.IO import PickleObject, UnpickleObject
from Functions.Event.DataLoader import GenerateDataLoader
import Functions.FeatureTemplates.EdgeFeatures as ef
import Functions.FeatureTemplates.NodeFeatures as nf
import Functions.FeatureTemplates.GraphFeatures as gf
from Functions.GNN.Optimizer import Optimizer
from Functions.GNN.TrivialModels import GraphNN, NodeConv, EdgeConv, CombinedConv

def TestOptimizerGraph(Files, Level, Name, CreateCache):
    
    if CreateCache:
        DL = GenerateDataLoader()
        DL.AddGraphFeature("Signal", gf.Signal)
        DL.AddGraphTruth("Signal", gf.Signal)
        DL.SetDevice("cuda")
        for i in Files:
            ev = UnpickleObject(i + "/" + i)
            DL.AddSample(ev, "nominal", Level)
        DL.MakeTrainingSample(50)
        PickleObject(DL, "TestOptimizerGraph")
    DL = UnpickleObject("TestOptimizerGraph")


    op = Optimizer(DL)
    op.VerboseLevel = 1
    op.RunName = Name
    op.Epochs = 10
    op.kFold = 2
    op.Model = GraphNN()
    op.DefineOptimizer()
    op.KFoldTraining()

    return True

def TestOptimizerNode(Files, Level, Name, CreateCache):

    if CreateCache: 
        DL = GenerateDataLoader()
        DL.AddNodeFeature("x", nf.Signal)
        DL.AddNodeFeature("Sig", nf.Signal)
        DL.AddNodeTruth("x", nf.Signal)
        DL.SetDevice("cuda")
        for i in Files:
            ev = UnpickleObject(i + "/" + i)
            DL.AddSample(ev, "nominal", Level, True, False)
        DL.MakeTrainingSample(50)
        PickleObject(DL, Name)
    DL = UnpickleObject(Name)

    op = Optimizer(DL)
    op.VerboseLevel = 1
    op.BatchSize = 1
    op.kFold = 4
    op.RunName = Name
    op.Model = NodeConv(2, 2)
    op.KFoldTraining()

    return True

def TestOptimizerEdge(Files, Level, Name, CreateCache):

    if CreateCache: 
        DL = GenerateDataLoader()
        DL.AddEdgeFeature("x", ef.Signal)
        DL.AddEdgeTruth("x", ef.Signal)
        DL.SetDevice("cuda")
        for i in Files:
            ev = UnpickleObject(i + "/" + i)
            DL.AddSample(ev, "nominal", Level, False, True)
        DL.MakeTrainingSample(20)
        PickleObject(DL, Name)
    DL = UnpickleObject(Name)

    op = Optimizer(DL)
    op.VerboseLevel = 1
    op.BatchSize = 10
    op.kFold = 10
    op.Epochs= 20
    op.RunName = Name
    op.Model = EdgeConv(1, 1)
    op.KFoldTraining()

    return True

def TestOptimizerCombined(Files, Level, Name, CreateCache):
    from Closure.GenericFunctions import CreateDataLoaderComplete
    DL = CreateDataLoaderComplete(Files, Level, Name, CreateCache)
    op = Optimizer(DL)
    op.VerboseLevel = 2
    op.BatchSize = 100
    op.LearningRate = 0.0001
    op.WeightDecay = 0.0001
    op.kFold = 10
    op.Epochs= 20
    op.RunName = Name
    op.Model = CombinedConv()
    op.KFoldTraining()

    return True
