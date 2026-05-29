from training.template import *
from training.atomics import *

class PipeLine:
    def __init__(self):
        self.base      = BaseCfg()
        self.sample    = SampleCfg()
        self.training  = TrainingCfg()
        self.cosmetic  = CosmeticCfg()
        self.models    = {}

    def ConfigModel(self, name, model, optim = None):
        self.models[name] = ModelCfg(model, optim)
        return self.models[name]

    def __base__(self):
        ana = Analysis()
        SetEnvironment(ana, self.base, "OutputPath")
        SetEnvironment(ana, self.base, "DebugMode")
        SetEnvironment(ana, self.base, "Threads")
        SetEnvironment(ana, self.base, "IntraThreads")
        return ana

    def __training__(self, ana):
        SetEnvironment(ana, self.training, "Epochs")
        SetEnvironment(ana, self.training, "BatchSize")
       
        SetEnvironment(ana, self.training, "kFold")
        SetEnvironment(ana, self.training, "kFolds")
        SetEnvironment(ana, self.training, "TrainSize")
        SetEnvironment(ana, self.training, "TrainingDataset")

    def __graphs__(self, ana):
        SetEnvironment(ana, self.sample.graph, "GraphCache")
        SetEnvironment(ana, self.sample.graph, "BuildCache")
    
    def __cosmetic__(self, ana):
        SetEnvironment(ana, self.cosmetic, "nBins")
        SetEnvironment(ana, self.cosmetic, "MaxRange")

        SetEnvironment(ana, self.cosmetic, "VarPt")
        SetEnvironment(ana, self.cosmetic, "VarEta")
        SetEnvironment(ana, self.cosmetic, "VarPhi")
        SetEnvironment(ana, self.cosmetic, "VarEnergy")

        SetEnvironment(ana, self.cosmetic, "Targets")
        SetEnvironment(ana, self.cosmetic, "SetLogY")

    def Build(self):
        for i in self.sample: 
            ana = self.__base__()
            ana.AddSamples(i._path, i._dset)

            ev = i.event.Event()
            if ev is None: break
            else: ana.AddEvent(ev, i._dset)

            gr = i.graph.Graph()
            if gr is None: pass
            else: ana.AddGraph(gr, i._dset)
    
            SetEnvironment(ana, i.graph, "GraphCache")
            SetEnvironment(ana, i.graph, "BuildCache")
            ana.Start()
        
        ana = self.__base__()
        self.__graphs__(ana)
        self.__training__(ana)
        self.__cosmetic__(ana)
        for i in self.models: 
            self.models[i].__compile__()
            ana.AddModel(self.models[i].model, self.models[i].optimizer.optim, i)
        ana.Start()
 
