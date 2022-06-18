from Functions.IO.Exporter import ExportToDataScience
from Functions.Event.DataLoader import GenerateDataLoader
from Functions.Event.EventGenerator import EventGenerator
from Functions.GNN.Optimizer import Optimizer
from Functions.IO.IO import UnpickleObject, PickleObject
from Functions.GNN.TrivialModels import *

import Closure.FeatureTemplates.EdgeFeatures as ef
import Closure.FeatureTemplates.NodeFeatures as nf
import Closure.FeatureTemplates.GraphFeatures as gf
from Functions.Event.Implementations.EventGraphs import EventGraphTruthTops, EventGraphTruthTopChildren, EventGraphDetector

def TestModelExport(Files, Level, Name, CreateCache):
    if CreateCache:
        DL = GenerateDataLoader()

        if Level == "TruthTops":
            DL.EventGraph = EventGraphTruthTops
        elif Level == "TruthTopChildren":
            DL.EventGraph = EventGraphTruthTopChildren
        elif Level == "DetectorParticles":
            DL.EventGraph = EventGraphDetector
        DL.AddNodeFeature("x", nf.Signal)
        DL.AddNodeFeature("Sig", nf.Signal)
        DL.AddNodeTruth("x", nf.Signal)
        DL.SetDevice("cuda")
        for i in Files:
            ev = UnpickleObject(i)
            DL.AddSample(ev, "nominal")
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
    ev = UnpickleObject(Name)
    for i in ev_event.Events:
        for tree in ev_event.Events[i]:
            ev_p = ev_event.Events[i][tree] 
            ev_t = ev.Events[i][tree]
            CompareObjects(ev_t, ev_p)
    return True


def TestDataLoaderExport(Files, CreateCache):
    from Closure.GenericFunctions import CreateEventGeneratorComplete, CreateDataLoaderComplete, CompareObjects
    
    it = 10
    ev_O = CreateEventGeneratorComplete(it, Files, ["tttt", "t"], CreateCache, "TestDataLoaderExport")
    DL = CreateDataLoaderComplete(["tttt", "t"], "TruthTopChildren", "TestDataLoaderExport", CreateCache, NameOfCaller = "TestDataLoaderExport" )
    
    Exp = ExportToDataScience()
    Exp.ExportDataGenerator(DL, Name = "TestDataLoaderExportHDF5")
    
    Exp = ExportToDataScience()
    obj = Exp.ImportDataGenerator(Name = "TestDataLoaderExportHDF5")

    DL = CreateDataLoaderComplete(["tttt", "t"], "TruthTopChildren", "TestDataLoaderExport", CreateCache, NameOfCaller = "TestDataLoaderExport" )   
    CompareObjects(DL, obj)
    return True

def TestEventGeneratorWithDataLoader(Files, CreateCache):
    from Closure.GenericFunctions import CreateEventGeneratorComplete, CreateDataLoaderComplete, CompareObjects
    from Functions.Event.DataLoader import GenerateDataLoader
    from Functions.GNN.Optimizer import Optimizer
    from Functions.GNN.TrivialModels import CombinedConv
        
    it = 10
    ev_O = CreateEventGeneratorComplete(it, Files, ["tttt", "t"], CreateCache, "TestEventGeneratorWithDataLoader")

    Exp = ExportToDataScience() 
    Exp.ExportEventGenerator(ev_O[0], Name = "ExpEventGenerator_1", OutDirectory = "_Pickle/TestEventGeneratorWithDataLoader")
    Exp.ExportEventGenerator(ev_O[1], Name = "ExpEventGenerator_2", OutDirectory = "_Pickle/TestEventGeneratorWithDataLoader")
   
    Exp_In = ExportToDataScience()
    ev_R1 = Exp_In.ImportEventGenerator(Name = "ExpEventGenerator_1", InputDirectory = "_Pickle/TestEventGeneratorWithDataLoader")
    ev_R2 = Exp_In.ImportEventGenerator(Name = "ExpEventGenerator_2", InputDirectory = "_Pickle/TestEventGeneratorWithDataLoader")

    ev_P = CreateEventGeneratorComplete(it, Files, ["tttt", "t"], False, "TestEventGeneratorWithDataLoader")
    CompareObjects(ev_R1, ev_P[0])
    CompareObjects(ev_R2, ev_P[1])
    
    DL = GenerateDataLoader()
    DL.AddEdgeFeature("dr", ef.d_r)
    DL.AddEdgeFeature("mass", ef.mass)       
    DL.AddEdgeFeature("signal", ef.Signal)
    DL.AddNodeFeature("eta", nf.eta)
    DL.AddNodeFeature("pt", nf.pt)       
    DL.AddNodeFeature("phi", nf.phi)      
    DL.AddNodeFeature("energy", nf.energy)
    DL.AddNodeFeature("signal", nf.Signal)
    DL.AddGraphFeature("mu", gf.Mu)
    DL.AddGraphFeature("m_phi", gf.MissingPhi)       
    DL.AddGraphFeature("m_et", gf.MissingET)      
    DL.AddGraphFeature("signal", gf.Signal)       
    DL.AddEdgeTruth("Topology", ef.Signal)
    DL.AddNodeTruth("NodeSignal", nf.Signal)
    DL.AddGraphTruth("GraphMuActual", gf.MuActual)
    DL.AddGraphTruth("GraphEt", gf.MissingET)
    DL.AddGraphTruth("GraphPhi", gf.MissingPhi)
    DL.AddGraphTruth("GraphSignal", gf.Signal)
    DL.SetDevice("cuda")

    DL.EventGraph = EventGraphTruthTopChildren

    DL.AddSample(ev_R1, "nominal", True, True)
    DL.AddSample(ev_R2, "nominal", True, True)

    DLO = GenerateDataLoader()
    DLO.AddEdgeFeature("dr", ef.d_r)
    DLO.AddEdgeFeature("mass", ef.mass)       
    DLO.AddEdgeFeature("signal", ef.Signal)
    DLO.AddNodeFeature("eta", nf.eta)
    DLO.AddNodeFeature("pt", nf.pt)       
    DLO.AddNodeFeature("phi", nf.phi)      
    DLO.AddNodeFeature("energy", nf.energy)
    DLO.AddNodeFeature("signal", nf.Signal)
    DLO.AddGraphFeature("mu", gf.Mu)
    DLO.AddGraphFeature("m_phi", gf.MissingPhi)       
    DLO.AddGraphFeature("m_et", gf.MissingET)      
    DLO.AddGraphFeature("signal", gf.Signal)       
    DLO.AddEdgeTruth("Topology", ef.Signal)
    DLO.AddNodeTruth("NodeSignal", nf.Signal)
    DLO.AddGraphTruth("GraphMuActual", gf.MuActual)
    DLO.AddGraphTruth("GraphEt", gf.MissingET)
    DLO.AddGraphTruth("GraphPhi", gf.MissingPhi)
    DLO.AddGraphTruth("GraphSignal", gf.Signal)
    DLO.SetDevice("cuda")

    DLO.EventGraph = EventGraphTruthTopChildren

    DLO.AddSample(ev_P[0], "nominal", True, True)
    DLO.AddSample(ev_P[1], "nominal", True, True)

    assert DL.__dict__.keys() == DLO.__dict__.keys()

    CompareObjects(DLO, DL)

    DL.MakeTrainingSample(0)
    op = Optimizer(DL)
    op.VerboseLevel = 1
    op.Model = CombinedConv()
    op.KFoldTraining()
    return True
