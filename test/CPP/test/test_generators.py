from AnalysisG.Generators import EventGenerator, GraphGenerator, SelectionGenerator, Analysis
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


def _fx(a): return 1

def test_eventgraph():
    Files = {
                smpl + "sample1" : ["smpl1.root"], 
                smpl + "sample2" : ["smpl1.root", "smpl2.root", "smpl3.root"]
    }

    Ev = EventGenerator(Files)
    Ev.Event = Event
    Ev.Threads = 1
    Ev.EventStop = 100
    Ev.MakeEvents
    
    for i in Ev:
        assert i.Event 
        assert i.hash
        assert i.met

    Gr = GraphGenerator(Ev)
    Gr.EventGraph = GraphChildren
    Gr.AddGraphFeature(_fx)
    Gr.Threads = 2
    Gr.Device = "cuda"
    Gr.MakeGraphs
   
    assert len(Gr) == len(Ev)
    for i in Gr:
        assert i.Graph 
        assert i.hash 
        assert i.i >= 0
        assert i.weight 
        assert i.G__fx

def test_selection_generator():
    Files = {
                smpl + "sample1" : ["smpl1.root"], 
                smpl + "sample2" : ["smpl1.root", "smpl2.root", "smpl3.root"]
    }

    from examples.ExampleSelection import Example, Example2

    Ev = EventGenerator(Files)
    Ev.Event = Event
    Ev.Threads = 1
    Ev.EventStop = 100
    Ev.MakeEvents

    sel = SelectionGenerator(Ev)
    sel += Ev
    sel.Threads = 2
    sel.AddSelection("Example", Example)
    sel.AddSelection("Example2", Example2)
    sel.MakeSelection

    assert sel.result["Example2"].CutFlow["Success->Example"] == len(Ev)
    assert len(sel.result["Example2"].TimeStats) == len(Ev)
    assert len(Ev)*4 == len(sel.result["Example2"].Top["Truth"])

def test_Analysis():
    Sample1 = {smpl + "sample1" : ["smpl1.root"]}
    Sample2 = smpl + "sample2"
    
    Ana = Analysis()
    Ana.ProjectName = "_Test"
    Ana.InputSample("Sample1", Sample1)
    Ana.InputSample("Sample2", Sample2)
    Ana.PurgeCache = False
    Ana.OutputDirectory = "../test/"
    Ana.EventStop = 100
    Ana.Launch


if __name__ == "__main__":
    #test_event_generator()
    #test_event_generator_more()
    #test_eventgraph()
    #test_selection_generator() 
    test_Analysis()
    pass
