from AnalysisTopGNN.Generators import EventGenerator, GraphGenerator
from AnalysisTopGNN.Events import Event, EventGraphTruthTopChildren
from AnalysisTopGNN.IO import PickleObject, UnpickleObject
from AnalysisTopGNN.Tools import Tools

def Test(a):
    return 1

def TestEventGraph(Files):
   
    #Ev = EventGenerator(Files)
    #Ev.EventStart = 0
    #Ev.EventStop = 100
    #Ev.Event = Event
    #Ev.SpawnEvents()
    #Ev.CompileEvent()
    #PickleObject(Ev, "TMP")
    Ev = UnpickleObject("TMP")

    Gr = GraphGenerator()
    Gr += Ev
    Gr.EventGraph = EventGraphTruthTopChildren
    Gr.AddGraphFeature(Test)
    Gr.TestFeatures(10)
    Gr.EventStart = 1
    Gr.EventStop = 10 
    Gr.CompileEventGraph()

    for i in Gr:
        print(i)
    return True 
