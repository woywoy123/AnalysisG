from AnalysisG.Events.Graphs.EventGraphs import GraphChildren
from AnalysisG.Events.Events.Event import Event

from AnalysisG.Generators import EventGenerator
from AnalysisG.Generators import GraphGenerator
#from AnalysisG.Generators import SelectionGenerator
from AnalysisG.Generators import Analysis
from conftest import clean_dir

smpl = "./samples/"
Files = {
    smpl + "sample1": ["smpl1.root"],
    smpl + "sample2": ["smpl1.root", "smpl2.root", "smpl3.root"],
}


def _fx(a):
    return 1

def test_eventgraph():
    Ev = EventGenerator(Files)
    Ev.Event = Event
    Ev.Threads = 1
    Ev.EventStop = 100
    Ev.MakeEvents()

    for i in Ev:
        assert i.Event
        assert i.hash
        assert i.met

    Gr = GraphGenerator(Ev)
    Gr.EventGraph = GraphChildren
    Gr.AddGraphFeature(_fx)
    Gr.Threads = 2
    Gr.Device = "cuda"
    Gr.MakeGraphs()

    assert len(Gr) == len(Ev)
    for i in Gr:
        assert i.Graph
        assert i.hash
        assert i.i >= 0
        assert i.weight
        assert i.G__fx
    clean_dir()

def test_event_generator_merge():
    f = list(Files)
    File0 = {f[0]: [Files[f[0]][0]]}
    File1 = {f[1]: [Files[f[1]][1]]}

    _Files = {}
    _Files.update(File0)
    _Files.update(File1)

    ev0 = EventGenerator(File0)
    ev0.Event = Event
    ev0.MakeEvents()

    ev1 = EventGenerator(File1)
    ev1.Event = Event
    ev1.MakeEvents()

    combined = EventGenerator(_Files)
    combined.Event = Event
    combined.MakeEvents()

    Object0 = {}
    for i in ev0: Object0[i.hash] = i
    Object1 = {}
    for i in ev1: Object1[i.hash] = i

    ObjectSum = {}
    for i in combined: ObjectSum[i.hash] = i

    assert len(ObjectSum) == len(Object0) + len(Object1)
    assert len(combined) == len(ev0) + len(ev1)

    for i in Object0: assert ObjectSum[i] == Object0[i]
    for i in Object1: assert ObjectSum[i] == Object1[i]

    combined = ev0 + ev1
    ObjectSum = {}
    for i in combined: ObjectSum[i.hash] = i
    for i in Object0: assert ObjectSum[i] == Object0[i]
    for i in Object1: assert ObjectSum[i] == Object1[i]

def test_analysis_data_nocache():
    AnaE = _template()
    AnaE.AddGraphFeature(_fx)
    AnaE.Event = Event
    AnaE.EventGraph = GraphChildren
    AnaE.Launch()

    assert len([i for i in AnaE if i.Graph]) != 0

def test_analysis_data_nocache_nolaunch():
    AnaE = _template()
    AnaE.AddGraphFeature(_fx)
    AnaE.Event = Event
    AnaE.EventGraph = GraphChildren

    assert len([i for i in AnaE if i.Graph]) != 0

def test_analysis_data_cache():
    AnaE = _template()
    AnaE.DataCache = True
    AnaE.AddGraphFeature(_fx)
    AnaE.EventGraph = GraphChildren
    AnaE.Event = Event
    AnaE.Launch()

    assert len([i for i in AnaE]) != 0

    AnaE = _template()
    AnaE.DataCache = True
    AnaE.Launch()

    assert len([i for i in AnaE if i.Graph]) != 0

    clean_dir()


def test_analysis_data_cache_diff_sample():
    Ana1 = _template(
        {
            "Name": "Sample2",
            "SampleDirectory": smpl + "sample2/" + Files[smpl + "sample2"][1],
        }
    )
    Ana1.Event = Event
    Ana1.EventGraph = GraphChildren
    Ana1.AddGraphFeature(_fx)
    Ana1.DataCache = True
    Ana1.Launch()

    assert len([i for i in Ana1 if i.Graph]) != 0

    Ana2 = _template(
        {
            "Name": "Sample1",
            "SampleDirectory": smpl + "sample1/" + Files[smpl + "sample1"][0],
        }
    )
    Ana2.Event = Event
    Ana2.EventGraph = GraphChildren
    Ana2.AddGraphFeature(_fx)
    Ana2.DataCache = True
    Ana2.Launch()

    assert len([i for i in Ana2 if i.Graph]) != 0

    AnaE = _template()
    AnaE.DataCache = True
    AnaE.Launch()

    AnaS = Ana2 + Ana1
    assert len(AnaE) != 0
    assert len(AnaS) != 0
    assert len([i for i in AnaS if i.hash in AnaE]) == len(AnaE)

    clean_dir()


def test_analysis_data_event_cache_diff_sample():
    Ana1 = _template(
        {
            "Name": "Sample2",
            "SampleDirectory": smpl + "sample2/" + Files[smpl + "sample2"][1],
        }
    )
    Ana1.Event = Event
    Ana1.EventCache = True
    Ana1.Launch()

    assert len([i for i in Ana1 if i.Event]) != 0

    Ana1 = _template({"Name": "Sample2"})
    Ana1.EventGraph = GraphChildren
    Ana1.AddGraphFeature(_fx)
    Ana1.DataCache = True

    assert len([i for i in Ana1 if i.Graph and i.G__fx[0][0] == 1]) != 0

    Ana2 = _template(
        {
            "Name": "Sample1",
            "SampleDirectory": smpl + "sample1/" + Files[smpl + "sample1"][0],
        }
    )
    Ana2.Event = Event
    Ana2.EventCache = True
    Ana2.Launch()

    assert len([i for i in Ana2]) != 0

    Ana2 = _template({"Name": "Sample1"})
    Ana2.EventGraph = GraphChildren
    Ana2.AddGraphFeature(_fx)
    Ana2.DataCache = True
    Ana2.Launch()

    assert len([i for i in Ana2 if i.Graph]) != 0

    clean_dir()


if __name__ == "__main__":
    #test_event_generator()
    #test_event_generator_more()
    test_event_generator_merge()
    #test_eventgraph()
    #test_Analysis()
    #test_analysis_event_nocache()
    #test_analysis_event_nocache_nolaunch()
    #test_analysis_event_cache()
    #test_analysis_event_cache_diff_sample()
    #test_analysis_data_nocache()
    #test_analysis_data_nocache_nolaunch()
    #test_analysis_data_cache()
    #test_analysis_data_cache_diff_sample()
    #test_analysis_data_event_cache_diff_sample()
