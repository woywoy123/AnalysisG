import AnalysisG
from AnalysisG.metrics import AccuracyMetric
from AnalysisG.core import TH1F
import pickle
import pathlib

class Data:
    def __init__(self, inst = None):
        if inst is None: return 
        self.RawTruth    = inst.RawTruth      
        self.AdjTruth    = inst.AdjTruth        
        self.RawNominal  = inst.RawNominal      
        self.AdjNominal  = inst.AdjNominal      
        self.RawUnmasked = inst.RawUnmasked     
        self.AdjUnmasked = inst.AdjUnmasked     
        self.RawMasked   = inst.RawMasked       
        self.AdjMasked   = inst.AdjMasked       

        self.ModelName = inst.ModelName
        self.ModeName  = inst.ModeName
        self.kFold = inst.kFold
        self.Epoch = inst.Epoch
        pth  = "/run/media/tnom6927/1.44.1-64570/pckl/" + self.ModelName
        pth += "/" + self.ModeName  
        pth += "/epoch-" + str(self.Epoch)
        pathlib.Path(pth).mkdir(parents = True, exist_ok = True)
        pickle.dump(self, open(pth + "/kfold-" + str(self.kFold) + ".pkl", "wb"))
        print("+>", pth + "/kfold-" + str(self.kFold) + ".pkl")

    def __hash__(self):
        cfg = self.ModelName
        cfg += self.ModeName
        cfg += str(self.Epoch)
        cfg += str(self.kFold)
        return hash(cfg)

    def __eq__(self, obj):
        return hash(obj) == hash(self)


#base_dir   = "/CERN/thesis-data/gnn-model/"
#base_model = "Grift"

br = "/home/tnom6927/scratch/"
#br = "/run/media/tnom6927/1.44.1-64570"

class Metric(AccuracyMetric):
    def __init__(self): AccuracyMetric.__init__(self)
    def Postprocessing(self):
        for i in self.training:   
            for k in self.training[i]: Data(k)
        for i in self.validation:   
            for k in self.validation[i]: Data(k)
        for i in self.evaluation:   
            for k in self.evaluation[i]: Data(k)
        self.training   = {} 
        self.validation = {}
        self.evaluation = {}


mdlx = ["MRK-1", "MRK-2", "MRK-3", "MRK-4", "MRK-5", "MRK-6", "MRK-7", "MRK-8"]

am = Metric()
am.InterpretROOT(br, [i+1 for i in range(200)], [i+1 for i in range(10)], "k-", ["MRK-8"], ["validation"])

am = Metric()
am.InterpretROOT(br, [i+1 for i in range(200)], [i+1 for i in range(10)], "k-", ["MRK-8"], ["training"])
