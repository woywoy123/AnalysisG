import AnalysisG
from   AnalysisG.core import Tools
from   AnalysisG.metrics import AccuracyMetric
from  tqdm import tqdm
from  utils import * 

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
        mk_pth(pth)
        svpkl(self, pth + "/kfold-" + str(self.kFold) + ".pkl")
        print("+>", pth + "/kfold-" + str(self.kFold) + ".pkl")

    def __hash__(self):
        cfg = self.ModelName
        cfg += self.ModeName
        cfg += str(self.Epoch)
        cfg += str(self.kFold)
        return hash(cfg)

    def __eq__(self, obj):
        return hash(obj) == hash(self)

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



class Process:

    def __init__(self, key, idp, data):
        self.var  = idp
        self.prc   = to_str(key)
        self.mode  = None
        self.model = None
        self.kfold = None
        self.epoch = None
        self._term = None

        data = get_dk(data, key)
        if len(data) == 0: data = {}
        self.weight = get_dk(data, "weight")
        self.masses = get_dk(data, "masses")
        self.error  = get_dk(data, "error" )  
        self.ntops  = get_dk(data, "ntops" ) 
        self.ntrus  = get_dk(data, "ntrus" ) 
        self.mlp    = get_dk(data, "mlp"   )   
        self.matrix = get_dk(data, "matrix")

    def __contains__(self, key):
        return key in str(self)


    def __str__(self):
        if self._term is not None: return self._term
        dx  = mk_str("Model", self.model)
        dx += mk_str("Epoch", self.epoch)
        dx += mk_str("kFold", self.kfold)
        dx += mk_str("Mode" , self.mode )
        dx += mk_str("prc"  , self.prc  )
        dx += mk_str("Var"  , self.var  )
        self._term = dx
        return str(self)

class Atomic(Tools):

    def __init__(self, pkls, fdx):
        self.pkls = pkls + ".pkl"
        self.fdx  = fdx
        self.avil = self.is_file(self.pkls)
        self.lx   = 0

    @property
    def load(self):
        if not self.avil: return []
        loaded = depkl(self.pkls)
        xi = [i for i in list(loaded.__dict__) if "Adj" in i or "Raw" in i]
        prck = unq_lst((sms_lst([list(loaded.__dict__[i]) for i in xi])))
        data = [Process(i, k, loaded.__dict__[k]) for i in prck for k in xi] 
        for i in data: 
            i.model = loaded.ModelName
            i.mode  = loaded.ModeName
            i.kfold = loaded.kFold
            i.epoch = loaded.Epoch
        self.lx = len(data)
        loaded = None
        return data

class Dataloader:

    def __init__(self, mdl, kfold, epoch, base_dir):
        self.base_dir = base_dir
        self.models = mdl
        self.kfolds = kfold
        self.epochs = epoch
        self.lsts   = pth_fm(self.base_dir, self.models)
        self.lsts   = sms_lst([pth_fm(i, ["training", "validation"]) for i in self.lsts])
        self.lsts   = sms_lst([pth_fm(i, self.epochs) for i in self.lsts])
        self.lsts   = sms_lst([pth_fm(i, self.kfolds) for i in self.lsts])
        self.fdx    = [Atomic(self.lsts[i], i) for i in range(len(self.lsts))]

    def __iter__(self):
        self.chx = None
        self.idx = len(self.fdx)
        self.itx = iter(tqdm(self.fdx))
        return self

    def __next__(self):
        if self.chx is None: 
            self.chx = next(self.itx)
            self.chx = self.chx.load
        try: return self.chx.pop()
        except IndexError: self.chx = None
        return next(self) 
