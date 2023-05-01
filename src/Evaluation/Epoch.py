from AnalysisG.IO import PickleObject
from time import time 

class Epoch:

    def __init__(self):
        self.Epoch = None
        self.RunName = None
        self.train = None
        self._train = {}
        self._val = {}
        self._timer = []

        self.o_model = {}
        self.i_model = {}

    @property 
    def start(self):
        self._t1 = time()

    @property 
    def end(self):
        self._timer.append(time() - self._t1)

    @property
    def init(self):
        self._train = {i[2:] : {"pred" : [], "truth" : [], "acc" : [], "loss" : []} for i in self.o_model}
        self._val   = {i[2:] : {"pred" : [], "truth" : [], "acc" : [], "loss" : []} for i in self.o_model}       
        self._train["nodes"] = []
        self._val["nodes"] = []

    def Collect(self, truth, pred, loss):
        lst = self._train if self.train else self._val 
        nodes = truth.num_nodes.tolist()
        if isinstance(nodes, list): lst["nodes"] += nodes
        else: lst["nodes"] += [nodes]
        for o, t in zip(self.o_model, self.o_model.values()):
            feat = o[2:]
            lst[feat]["pred"] += getattr(pred, t).tolist()
            lst[feat]["truth"] += getattr(truth, t).tolist()
            lst[feat]["acc"] += [loss[feat]["acc"].tolist()]
            lst[feat]["loss"] += [loss[feat]["loss"].tolist()]

    @property
    def dump(self):
        PickleObject(self, self.OutputDirectory + "/" + self.RunName + "/" + str(self.Epoch) + "/Stats")
