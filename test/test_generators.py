from AnalysisTopGNN.Generators import EventGenerator, GraphGenerator
from AnalysisTopGNN.Events import Event, EventGraphChildren

smpl = "./TestCaseFiles/Sample/"
Files = {smpl + "tttt" : ["output.root"], smpl + "tttt_m400" : ["smpl1.root", "smpl2.root", "smpl3.root"]}
def test_eventgenerator():
    EvtGen = EventGenerator(Files)
    EvtGen.EventStop = 3
    EvtGen.EventStart = 1
    EvtGen.Event = Event
    EvtGen.SpawnEvents()
   
    assert len(EvtGen) != 0
    for event in EvtGen:
        assert event == EvtGen[event.Filename]

def _fx(a):
    return 1

def test_eventgraph():
    Ev = EventGenerator(Files)
    Ev.Event = Event 
    Ev.EventStop = 100
    Ev.SpawnEvents()
    
    Gr = GraphGenerator()
    Gr += Ev
    Gr.EventGraph = EventGraphChildren
    Gr.AddGraphFeature(_fx)
    Gr.TestFeatures(len(Ev))
    Gr.CompileEventGraph()
   
    assert len(Gr) == len(Ev)
    for i in Gr:
        assert i.Compiled
        assert Gr.HashToROOT(i.Filename) != None

