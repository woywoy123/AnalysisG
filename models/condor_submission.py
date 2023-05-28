from AnalysisG.Events import GraphChildren, GraphTruthJet, GraphDetector
from hyperparams_scan import Opt, sched, btch
from AnalysisG.Templates import ApplyFeatures
from AnalysisG.Submission import Condor
from AnalysisG.Events import Event
from AnalysisG import Analysis 
import os

Mode = "TruthChildren"
GNN = "BasicGraphNeuralNetwork" 
Names = {
          "other" : "other", 
          "t"     : "t", 
          "tt"    : "tt", 
          "ttbar" : "ttbar", 
          "ttH"   : "ttH", 
          "SM4t"  : "tttt", 
          "ttX"   : "ttX", 
          "ttXll" : "ttXll", 
          "ttXqq" : "ttXqq", 
          "ttZ-1000" : "ttZ-1000", 
          "V"     : "V", 
          "Vll"   : "Vll", 
          "Vqq"   : "Vqq"
}

kFolds = 10

if   Mode == "TruthChildren": gr = GraphChildren
elif Mode == "TruthJets": gr = GraphTruthJets
elif Mode == "Jets": gr = GraphDetector
else: print("failure"); exit()

if GNN == "BasicGraphNeuralNetwork": from BasicGraphNeuralNetwork.model import BasicGraphNeuralNetwork as model
else: print("failure"); exit()


def EventGen(name = None, daod = ""):
    Ana = Analysis()
    Ana.Event = Event 
    Ana.chnk = 100
    Ana.Threads = 12
    Ana.InputSample(name, os.environ["Samples"] + daod)
    return Ana

def DataGen(name = None):
    Ana = Analysis()
    Ana.InputSample(name)
    Ana.EventGraph = gr
    ApplyFeatures(Ana, Mode) 
    Ana.TrainingSize = 90
    return Ana
   
def Optim(op, op_para, sc, sc_para, batch, k, kF):
    Ana = Analysis()
    Ana.Device = "cuda"
    Ana.kFold = k
    Ana.kFolds = kF
    Ana.ContinueTraining = True
    Ana.Optimizer = op
    Ana.Scheduler = sc
    Ana.OptimizerParams = op_para
    Ana.SchedulerParams = sc_para 
    Ana.EnableReconstruction = True 
    Ana.BatchSize = batch 
    Ana.Model = model
    return Ana 
 
Sub = Condor()
Sub.EventCache = True 
Sub.DataCache = True
Sub.ProjectName = "Project_" + Mode
Sub.Verbose = 3

all_jbs = []
for name in Names:
    evnt = EventGen(name, Names[name])
    jb_ev_ = "evnt_" + name
    Sub.AddJob(jb_ev_, evnt, "8GB", "4h")
     
    data = DataGen(name)
    jb_da_ = "data_" + name
    Sub.AddJob(jb_da_, data, "8GB", "4h", waitfor = [jb_ev_])

    all_jbs.append(jb_da_)

op_it, sc_it, b_it = iter(Opt), iter(sched), iter(btch)

for _ in range(len(Opt)):
    n, n_, b_ = next(op_it), next(sc_it), next(b_it)
    name = "-".join([n, n_, b_])

    op, op_par = Opt[n]
    sc, sc_par = sched[n_]
    b = btch[b_] 
    
    for k in range(kFolds):
        op_ana = Optim(op, op_par, sc, sc_par, b, k+1, kFolds)  
        op_ana.RunName = name
        Sub.AddJob(name + "kF-" + str(k+1), op_ana, "8GB", "8h", waitfor = all_jbs)
Sub.PythonVenv = "$PythonGNN"
Sub.DumpCondorJobs

