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
    def Recursive(inp1, inp2):
        if isinstance(inp1, dict):
            if len(inp1) != len(inp2):
                exit()

            for key in inp1:
                Recursive(inp1[key], inp2[str(key)])
        elif isinstance(inp1, list):
            for key1, key2 in zip(inp1, inp2):
                Recursive(key1, key2)
        else:
            if inp1 != inp2:
                if str(inp1) == inp2:
                    return 
                print([inp1, inp2], inp1 == inp2, type(inp2) == type(inp1))
                #exit()
    def CompareObjects(in1, in2):
        key_t = set(list(in1.__dict__.keys()))
        key_r = set(list(in2.__dict__.keys()))
        d = key_t ^ key_r
        
        if len(list(d)) != 0:
            print("!!!!!!!!!!!!!!!!Variable difference: ", d)

        for i, j in zip(list(key_t), list(key_r)):
            print("----> ", i, j)
            Recursive(in1.__dict__[i], in2.__dict__[j])

    if CreateCache:
        ev = EventGenerator(File, Stop = 5)
        ev.SpawnEvents()
        ev.CompileEvent(SingleThread = False)
        PickleObject(ev, Name) 
    ev = UnpickleObject(Name)

    #Exp = ExportToDataScience()
    #Exp.ExportEventGenerator(ev, Name = "TestEventGeneratorExport", DumpName = "EventLoader")
    #ev_restore = Exp.ImportEventGenerator(Name = "TestEventGeneratorExport", DumpName = "EventLoader")
    #CompareObjects(ev, ev_restore)


    Exp = ExportToDataScience()
    Exp.ExportEventGenerator(ev, Name = "TestEventGeneratorExport_Events")
    ev_event = Exp.ImportEventGenerator(Name = "TestEventGeneratorExport_Events")
    for i in ev_event:
        if "EventGenerator" in str(i):
            continue
        
        ev_t = ev.Events[i.iter][i.Tree]
        print(ev_t.iter, i.iter)
        print(ev_t.mu, i.mu)
        break
        CompareObjects(ev_t, i)



        break







    return True
