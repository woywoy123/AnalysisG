from AnalysisG.Generators import EventGenerator, GraphGenerator
from AnalysisG.Events.Events.Event import Event
from AnalysisG.Events.Graphs.EventGraphs import GraphChildren

smpl = "./samples/"
def test_event_generator():
    root1 = "./samples/sample1/smpl1.root"

    EvtGen = EventGenerator({root1: []})
    EvtGen.EventStop = 50
    EvtGen.EventStart = 10
    EvtGen.Event = Event
    EvtGen.Threads = 1
    EvtGen.MakeEvents
    lst = {}
    for i in EvtGen:
        lst[i.hash] = i
    assert len(lst) == 40
    
    EvtGen_ = EventGenerator({root1: []})
    EvtGen_.EventStop = 50
    EvtGen_.EventStart = 10
    EvtGen_.Event = Event
    EvtGen_.Threads = 2
    EvtGen_.MakeEvents
    lst_ = {}
    for i in EvtGen_:
        lst_[i.hash] = i
    assert len(lst_) == len(lst)
    for i in lst_:
        ev_, ev = lst_[i], lst[i]
        sum([t_ == t for t_, t in zip(ev_.Tops, ev.Tops)]) == len(ev.Tops)
        for t_, t in zip(ev_.Tops, ev.Tops):
            for c_, c in zip(t_.Children, t.Children): assert c_ == c
        assert len(ev_.TopChildren) == len(ev.TopChildren)
        for t_, t in zip(ev_.TopChildren, ev.TopChildren): assert t_ == t
        
        assert len(ev_.TruthJets) == len(ev.TruthJets)
        for tj_, tj in zip(ev_.TruthJets, ev.TruthJets): assert tj_ == tj

        assert len(ev_.DetectorObjects) == len(ev.DetectorObjects)
        for tj_, tj in zip(ev_.DetectorObjects, ev.DetectorObjects): assert tj_ == tj


def test_event_generator_more():
    Files = {
                smpl + "sample1" : ["smpl1.root"], 
                smpl + "sample2" : ["smpl1.root", "smpl2.root", "smpl3.root"]
    }

    EvtGen = EventGenerator(Files)
    EvtGen.EventStop = 1000
    EvtGen.EventStart = 50
    EvtGen.Event = Event
    EvtGen.MakeEvents
    assert len(EvtGen) != 0
    for event in EvtGen: assert event == EvtGen[event.hash]


def test_eventgraph():
    Files = {
                smpl + "sample1" : ["smpl1.root"], 
                #smpl + "sample2" : ["smpl1.root", "smpl2.root", "smpl3.root"]
    }

    def _fx(a):
        return 1



    Ev = EventGenerator(Files)
    Ev.Event = Event 
    Ev.EventStop = 100
    Ev.MakeEvents
    
    Gr = GraphGenerator(Ev)
    Gr.EventGraph = GraphChildren
    Gr.AddGraphFeature(_fx)
    Gr.Threads = 1
    Gr.MakeGraphs
    #Gr.TestFeatures(len(Ev))
    #Gr.CompileEventGraph()
   
    #assert len(Gr) == len(Ev)
    #for i in Gr:
    #    assert i.Compiled
    #    assert Gr.HashToROOT(i.Filename) != None


if __name__ == "__main__":
    #test_event_generator()
    #test_event_generator_more()
    test_eventgraph()
