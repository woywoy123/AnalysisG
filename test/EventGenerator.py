from AnalysisTopGNN.Generators import EventGenerator 
from AnalysisTopGNN.Events import Event
from AnalysisTopGNN.IO import PickleObject, UnpickleObject
from AnalysisTopGNN.Samples import SampleTracer

def TestEventGenerator(Files):
    EvtGen = EventGenerator(Files)
    EvtGen.EventStop = 2999
    EvtGen.EventStart = 1
    EvtGen.Event = Event
    EvtGen.SpawnEvents()
  
    exit()
    passedEvents = False
    for i in Tracer.Events:
        if i == EvtGen.EventStart:
            passedEvents = True
        if i == EvtGen.EventStop and passedEvents:
            passedEvents = True
        if passedEvents == False:
            passedEvents = False
    print("Passed (Consistent Number of Events)")
    EvtGen.CompileEvent()
    
    Tr = SampleTracer(EvtGen.Tracer) 
    for i in EvtGen:
        Tr.IndexToHash(i.EventIndex)
        Tr.IndexToEvent(i.EventIndex)
        Tr.IndexToROOT(i.EventIndex)
    
    print("Passed Tracing")
    return passedEvents
