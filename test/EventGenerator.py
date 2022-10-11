from AnalysisTopGNN.Generators import EventGenerator 
from AnalysisTopGNN.Events import Event
def TestEventGenerator(Files):


    EvtGen = EventGenerator(Files)
    EvtGen.EventStop = 3000
    EvtGen.EventStart = 10
    EvtGen.Event = Event
    EvtGen.SpawnEvents()
    Tracer = EvtGen.Tracer
    print(Tracer.EventInfo)
    print(Tracer.Events)







    return True
