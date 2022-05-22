from Functions.IO.Exporter import ExportToDataScience
from Functions.Event.DataLoader import GenerateDataLoader
from Functions.GNN.Optimizer import Optimizer
from Functions.IO.IO import UnpickleObject, PickleObject
import Functions.FeatureTemplates.GraphFeatures as gf


def TestModelExport(Files, Level, Name, CreateCache):

    if CreateCache:
        DL = GenerateDataLoader()
        DL.AddGraphFeature("Signal", gf.Signal)
        DL.AddGraphTruth("Signal", gf.Signal)
        DL.SetDevice("cuda")
        for i in Files:
            ev = UnpickleObject(i + "/" + i)
            DL.AddSample(ev, "nominal", Level)
            break
        DL.MakeTrainingSample(0)
        PickleObject(DL, "TestOptimizerGraph")
    DL = UnpickleObject("TestOptimizerGraph")
    Op = Optimizer()
    Op.DataLoaderInstance = DL













    return True
