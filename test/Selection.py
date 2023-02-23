from ExampleSelection import Example
from AnalysisTopGNN import Analysis
from AnalysisTopGNN.Events import Event

def TestSelection(Files):
    AnaE = Analysis()
    AnaE.ProjectName = "Project"
    AnaE.InputSample("bsm-4t", "/".join(Files[0].split("/")[:-1]))
    AnaE.InputSample("t", Files[1])
    AnaE.AddSelection("Example", Example)
    AnaE.AddSelection("Example2", Example())
    AnaE.Event = Event
    AnaE.EventCache = True
    AnaE.DumpPickle = True
    AnaE.Threads = 12
    AnaE.Launch()
 


    return True
