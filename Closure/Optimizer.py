from Functions.IO.IO import PickleObject, UnpickleObject
from Functions.Event.DataLoader import GenerateDataLoader
import Functions.FeatureTemplates.EdgeFeatures as ef
import Functions.FeatureTemplates.NodeFeatures as nf
import Functions.FeatureTemplates.GraphFeatures as gf
from Functions.GNN.Optimizer import Optimizer
from Functions.GNN.TrivialModels import GraphNN

def TestOptimizerGraph(Files, Level, Name):
    
    DL = GenerateDataLoader()
    DL.AddGraphFeature("Signal", gf.Resonance)
    DL.AddGraphTruth("Signal", gf.Resonance)
    DL.SetDevice("cuda")
    for i in Files:
        ev = UnpickleObject(i + "/" + i)
        DL.AddSample(ev, "nominal", Level)
    DL.MakeTrainingSample(0)
    
    op = Optimizer(DL)
    op.RunName = Name
    op.Epochs = 10
    op.kFold = 2
    op.Model = GraphNN()
    op.DefineOptimizer()
    op.KFoldTraining()

    return True

def TestOptimizerNode(Files, Level):
    DL = GenerateDataLoader()
    DL.AddGraphFeature("Signal", gf.Resonance)
    DL.AddGraphTruth("Signal", gf.Resonance)
    DL.SetDevice("cuda")
    for i in Files:
        ev = UnpickleObject(i + "/" + i)
        DL.AddSample(ev, "nominal", Level)
    DL.MakeTrainingSample(0)
    PickleObject("TestOptimizerNode")
    DL = UnpickleObject("testOptimizerNode")

    op = Optimizer(DL)
    op.Model = GraphNN()
    op.DefineOptimizer()
    op.KFoldTraining()

    return True
