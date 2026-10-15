from AnalysisG.metrics import AccuracyMetric
from AnalysisG.metrics import PDax
from AnalysisG.core import Tools

from utils import *
from base  import *
import time 

class Atomic(PDax):

    def __init__(self, pkls, k):
        def fx(l, p): return p.replace("<#>", l)
        pkls = [fx(i, pkls) for i in ["training", "validation"]] 
        PDax.__init__(self, pkls, k)

    def loader(self):
        loaded  = self.load("RawTruth"   , "AdjTruth"  )
        loaded *= self.load("RawNominal" , "AdjNominal")
        loaded *= self.load("RawUnmasked", "AdjUnmasked")
        loaded *= self.load("RawMasked"  , "AdjMasked"  )
        return self.data()

class Dataloader:

    def __init__(self, mdl, kfold, epoch, base_dir):
        lsts = pth_fm(base_dir, mdl)
        lsts = sms_lst([pth_fm(i, ["<#>"]) for i in lsts])
        lsts = sms_lst([pth_fm(i, epoch) for i in lsts])
        self.lsts = sms_lst([pth_fm(i, kfold) for i in lsts])

    def __iter__(self):
        self.fdx  = [Atomic(self.lsts[i], i) for i in range(len(self.lsts))]
        self.itx = iter(self.fdx)
        return self

    def __next__(self):
        self.chx = next(self.itx).loader()
        if len(self.chx): return self.chx[0]
        raise StopIteration
