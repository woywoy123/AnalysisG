from Functions.IO.Exporter import ExportToDataScience
from Functions.Event.DataLoader import GenerateDataLoader
from Functions.Event.EventGenerator import EventGenerator
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

def TestEventGeneratorExport(File, Name, CreateCache):
    from Closure.GenericFunctions import CompareObjects

    if CreateCache:
        ev = EventGenerator(File, Stop = 5)
        ev.SpawnEvents()
        ev.CompileEvent(SingleThread = False)
        PickleObject(ev, Name) 
    ev = UnpickleObject(Name)

    Exp = ExportToDataScience()
    Exp.ExportEventGenerator(ev, Name = "TestEventGeneratorExport_Events")
    ev_event = Exp.ImportEventGenerator(Name = "TestEventGeneratorExport_Events")
    for i in ev_event.Events:
        for tree in ev_event.Events[i]:
            ev_p = ev_event.Events[i][tree][0] 
            ev_t = ev.Events[i][tree]
            CompareObjects(ev_t, ev_p)
    return True


def TestDataLoaderExport(Files, CreateCache):
    from Closure.GenericFunctions import CompareObjects, CreateDataLoaderComplete
    DL = CreateDataLoaderComplete(Files, "TruthTopChildren", "TestDataLoaderExport", CreateCache)
    
    Exp = ExportToDataScience()
    Exp.ExportDataGenerator(DL, Name = "TestDataLoaderExportHDF5")
    
    Exp = ExportToDataScience()
    obj = Exp.ImportDataGenerator(Name = "TestDataLoaderExportHDF5")

    DL = CreateDataLoaderComplete(Files, "TruthTopChildren", "TestDataLoaderExport", CreateCache)
    CompareObjects(DL, obj)


    return True

