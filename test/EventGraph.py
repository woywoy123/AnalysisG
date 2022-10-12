from AnalysisTopGNN.Generators import EventGenerator, GraphGenerator
from AnalysisTopGNN.Events import Event, EventGraphTruthTopChildren
from AnalysisTopGNN.IO import PickleObject, UnpickleObject
from AnalysisTopGNN.Tools import Tools



def Test(a):
    return 1

def TestEventGraph(Files):
   
    #Ev = EventGenerator(Files)
    #Ev.EventStart = 0
    #Ev.EventStop = 3000
    #Ev.Event = Event
    #Ev.SpawnEvents()
    #Ev.CompileEvent()

    #PickleObject(Ev, "TMP2")
    Ev = UnpickleObject("TMP2")
 
    Gr = GraphGenerator()
    Gr.EventGraph = EventGraphTruthTopChildren
    Gr.ImportTracer(Ev.Tracer)
    Gr.AddGraphFeature(Test)
    Gr.TestFeatures(100)
    #Gr.CompileEventGraph()
