from AnalysisG.IO import nTupler, PickleObject
from AnalysisG.Submission import Condor
from AnalysisG.Events import SSML
from AnalysisG import Analysis
from double_leptonic import DiLeptonic
import os

smpls = os.environ["Samples"] + "../SSML_MC/"

def EventGen(path, name):
    ana = Analysis()
    ana.Event = SSML
    ana.EventCache = True
    ana.InputSample(name, path)
    ana.Threads = 12
    ana.Chunks = 1000
#    ana.EventStop = 1000
    return ana

def SelectionGen(name):
    sel = DiLeptonic()
    sel.__params__ = {"btagger" : "gn2_btag_85", "truth" : "detector"}
    ana = Analysis()
    ana.AddSelection(sel)
    ana.InputSample(name)
    ana.EventName = "SSML"
    ana.Event = SSML
    ana.Threads = 48
    ana.Chunks = 10000
    return ana

con = Condor()
con.PythonVenv = "GNN"
con.ProjectName = "Dilepton"

smpl = con.ListFilesInDir(smpls, ".root")
smpl = [i + "/" + j for i in smpl for j in smpl[i]]
for x in smpl:
    name = x.split("/")[-1].split(".root")[0]
    con.AddJob(name, EventGen(x, name), memory = "12GB", time = "6h")
    con.AddJob("sel-" + name, SelectionGen(name), memory = "12GB", time = "6h", waitfor = [name])
con.LocalRun()

