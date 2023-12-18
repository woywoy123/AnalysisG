from ObjectDefinitions.selection import ExampleSelection
from AnalysisG.Submission import Condor
from AnalysisG.Events import SSML
from AnalysisG import Analysis
from AnalysisG.IO import nTupler, PickleObject

samples = "/nfs/dust/atlas/user/<...>/Samples/lisa/mc16a/2lss3lge1DL1r/"

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
    ana = Analysis()
    ana.AddSelection(ExampleSelection)
    ana.InputSample(name)
    ana.EventName = "SSML"
    ana.Threads = 12
    ana.Chunks = 1000
    return ana

con = Condor()
con.PythonVenv = "/nfs/dust/atlas/user/<...>/AnalysisG/setup-scripts/source_this.sh"
con.ProjectName = "ExampleProject"

sample_list = []
dic = con.ListFilesInDir(samples, ".root")
for i in dic: sample_list += [i + "/" + j for j in dic[i]]

for x in sample_list:
    name = x.split("/")[-1].split(".root")[0]
    con.AddJob(name, EventGen(x, name), memory = "12GB", time = "6h")
    con.AddJob("sel-" + name, SelectionGen(name), memory = "12GB", time = "6h") #, waitfor = [name])
#con.LocalRun()
con.SubmitToCondor()
exit()

nt = nTupler()
nt.ProjectName = "ExampleProject"
nt.This("ExampleSelection", "nominal_Loose")
nt.Threads = 12
x = nt.merged()
PickleObject(x, "example_selection")


