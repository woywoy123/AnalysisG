from AnalysisG import Analysis
from AnalysisG.Tools import Tools
from AnalysisG.IO import PickleObject
from AnalysisG.Templates import ApplyFeatures
from AnalysisG.Events import *
import os

smpls= os.environ["Samples"]
print(smpls)
x = Tools()
x = [k for k in x.lsFiles(smpls) if "ttZ-" in k][:3]

ana = Analysis()
ana.ProjectName = "smpl"
#ana.InputSample(None, x)
#ana.Event = Event
ana.EventName = "Event"
#ana.EventCache = True
ana.DataCache = True
#ana.Graph = GraphChildren
ana.GraphName = "GraphChildren" #"GraphTruthJet"
#ApplyFeatures(ana, "TruthChildren")
lists = []
for i in ana:
    x = i.release_graph()
    i = x.__data__().clone()
    lists.append(i)
    if len(lists) < 100: continue
    break
PickleObject(lists, "GraphTruthChildren") #"GraphTruthJet")
