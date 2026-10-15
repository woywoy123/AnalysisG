from AnalysisG.metrics import AccuracyMetric
from AnalysisG.core import Tools
from tqdm import tqdm
from utils import * 

class Data:

    def __init__(self, inst = None):
        if inst is None: return 
        self.ModelName = inst.ModelName
        self.ModeName  = inst.ModeName

        self.RawTruth = inst.RawTruth      
        self.AdjTruth = inst.AdjTruth        

        self.RawNominal = inst.RawNominal      
        self.AdjNominal = inst.AdjNominal      

        self.RawUnmasked = inst.RawUnmasked     
        self.AdjUnmasked = inst.AdjUnmasked   

        self.RawMasked = inst.RawMasked       
        self.AdjMasked = inst.AdjMasked       

        self.kFold = inst.kFold
        self.Epoch = inst.Epoch

        pth  = "/run/media/tnom6927/1.44.1-64570/pckl/" + self.ModelName
        pth += "/" + self.ModeName + "/epoch-" + str(self.Epoch)
        mk_pth(pth)
        svpkl(self, pth + "/kfold-" + str(self.kFold) + ".pkl")

    def __hash__(self):
        cfg = self.ModelName
        cfg += self.ModeName
        cfg += str(self.Epoch)
        cfg += str(self.kFold)
        return hash(cfg)

    def __eq__(self, obj):
        return hash(obj) == hash(self)

class Metric(AccuracyMetric):

    def __init__(self, tr, va, ev): 
        AccuracyMetric.__init__(self, tr, va, ev)

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



