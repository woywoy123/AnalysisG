from Functions.IO.Exporter import ExportToDataScience
from Functions.Event.DataLoader import GenerateDataLoader
from Functions.GNN.Optimizer import Optimizer
from Functions.IO.IO import UnpickleObject, PickleObject
from Functions.GNN.TrivialModels import *

import Functions.FeatureTemplates.GraphFeatures as gf
import Functions.FeatureTemplates.NodeFeatures as nf

def TestModelExport(Files, Level, Name, CreateCache):

    if CreateCache:
        DL = GenerateDataLoader()
        DL.AddNodeFeature("x", nf.Signal)
        DL.AddNodeFeature("Sig", nf.Signal)
        DL.AddNodeTruth("x", nf.Signal)
        DL.SetDevice("cuda")
        for i in Files:
            ev = UnpickleObject(i + "/" + i)
            DL.AddSample(ev, "nominal", Level)
            break
        DL.MakeTrainingSample(0)
        PickleObject(DL, "TestOptimizerGraph")
    DL = UnpickleObject("TestOptimizerGraph")
    Op = Optimizer()
    Op.RunName = Name 
    Op.kFold = 10
    Op.Epochs = 20
    Op.ONNX_Export = True
    Op.TorchScript_Export = True
    Op.ReadInDataLoader(DL)
    Op.Model = NodeConv(2, 2)
    Op.KFoldTraining()

    return True

def TestEventGeneratorExport(Files, Name):
    for i in Files:
        ev = UnpickleObject(i + "/" + i)
        break
    Exp = ExportToDataScience()
    Exp.ExportEventGenerator(ev)



    return True
