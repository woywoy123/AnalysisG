import AnalysisG
from AnalysisG.core import Analysis
from AnalysisG.graphs.bsm_4tops import *
from AnalysisG.events.bsm_4tops import *
from AnalysisG.core.lossfx import *
from AnalysisG.metrics import *
from AnalysisG.models import *
import random
import urllib.request
import socket
import json


dxf = random.randint(1, 1000000)
dxf = str(hash(dxf))

ev = BSM4Tops()
gr = GraphTruthJets()
base_dir =  None
idx = socket.gethostname()
addr = f"http://xyz:2301/?task_id={idx + dxf}"

class task:
    def __init__(self):
        self.epoch = None
        self.fold  = None
        self.mode  = None
        self.model = None

    def get_task(self):
        rep = urllib.request.urlopen(addr)
        data = json.loads(rep.read().decode("utf-8"))
        self.fold  = str(data["kfold"])
        self.epoch = "epoch-" + str(data["epoch"])
        self.model = data["model"]
        self.mode  = data["mode"]
        self.run   = data["status"] == "run"
        return self

    def get_pth(self):
        k  = self.model + "/state/"
        k += self.epoch + "/kfold-" + self.fold + "_model.pt"
        return k

    def get_key(self):
        return self.model + "::" + self.epoch + "::k-" + self.fold

    def get_vars(self):
        kx =  ["pt", "eta", "phi", "energy", "is_lep", "is_b", "index"]
        kin  = ["data::node::" + i for i in kx]
        kin += [
            "data::edge::index",
            "data::graph::num_jets", 
            "data::graph::num_leps",
            "batch::node::index", 
            "batch::graph::index",
            "truth::graph::ntops", 
            "prediction::extra::ntops_score", 
            "prediction::extra::top_edge_score",
            "truth::edge::top_edge"
        ]
        return [self.model + "::" + i for i in kin]


def generate_model(mrk, dev):
    gn = Grift()
    gn.name = mrk
    gn.o_edge  = {"top_edge" : "CrossEntropyLoss", "res_edge" : "CrossEntropyLoss"}
    gn.o_graph = {"ntops"    : "CrossEntropyLoss", "signal"   : "CrossEntropyLoss"}
    gn.i_node  = ["pt", "eta", "phi", "energy", "charge"]
    gn.i_graph = ["met", "phi"]
    gn.device = "cuda:" + str(dev)
    return gn

def accuracy_metric(step, dev):
    tsk = [task().get_task() for i in range(step)]
    tsk = [i for i in tsk if i.run]
    mkx = {i.model : [] for i in tsk}
    mtr = {i : AccuracyMetric() for i in mkx}
    for i in tsk: mkx[i.model].append(i)
    for i in mtr: mtr[i].RunNames  = {j.get_key() : base_dir + "/" + j.get_pth() for j in mkx[i]}
    for i in mtr: mtr[i].Variables = mkx[i][0].get_vars()
    for i in mtr: mtr[i] = [mtr[i], generate_model(i, dev), mkx[i][0].mode]
    return mtr


ana = Analysis()
ana.GraphCache      = "/dev/shm/Graphs/graph_jets_detector_lep"
ana.TrainingDataset = "/dev/shm/Graphs/GraphDetectorLep_train.h5"
ana.Threads = 32
step = 5

sc = []
tkx = []
for k in range(4):
    mx1 = accuracy_metric(step, k)
    for t in mx1: ana.AddMetric(mx1[t][0], mx1[t][1])
    sc.append(mx1)
    tkx += [mx1[t][-1]]

mode = list({i : None for i in tkx})[0]
ana.BatchSize  = 20
ana.Validation = mode == "validation"
ana.Evaluation = mode == "evaluation"
ana.Training   = mode == "training"
ana.Start()


