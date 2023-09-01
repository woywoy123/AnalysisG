from AnalysisG.Generators import GraphGenerator
from AnalysisG.Generators import EventGenerator
from examples.Graph import DataGraph
from examples.Event import EventEx
from conftest import clean_dir
from AnalysisG.Tools import Code

smpl = "./samples/"
Files = {
    smpl + "sample1": ["smpl1.root"],
    smpl + "sample2": ["smpl1.root", "smpl2.root", "smpl3.root"],
}


def fx_edge(a, b):
    return (a + b).Mass

def fx_node(a):
    return a.Mass

def fx_graph(ev):
    return ev.NotAFeature

def fx_mev(ev):
    return ev.met

def fx_custom_topology(a, b):
    return a.FromResonance == b.FromResonance == 1

def fx_pmu(a):
    return [a.pt, a.eta, a.phi, a.e]

def fx_prefilter(ev):
    return ev.nJets > 2

def test_graph_generator():
    root1 = "./samples/sample1/smpl1.root"
    root1 = "/home/tnom6927/Downloads/samples/Dilepton/ttZ-1000/"
    EvtGen = EventGenerator(root1)
    EvtGen.Event = EventEx
    EvtGen.Threads = 12
    EvtGen.Chunks = 1000
    EvtGen.MakeEvents()

    gr = DataGraph()
    gr.AddGraphFeature(fx_graph, "Failed")
    gr.AddGraphFeature(fx_mev, "mev")
    gr.AddNodeFeature(fx_pmu, "pmu")
    gr.AddEdgeFeature(fx_edge, "Mass")
    gr.AddEdgeTruthFeature(fx_custom_topology)
    gr.SetTopology(fx_custom_topology)
    gr.AddPreSelection(fx_prefilter)

    GrGen = GraphGenerator(EvtGen)
    GrGen.Graph = gr
    GrGen.Threads = 10
    GrGen.Chunks = 1000
    GrGen.MakeGraphs()

    assert len(GrGen.ShowLength) == 2
    assert "nominal/DataGraph" in GrGen.ShowLength
    assert "nominal/EventEx" in GrGen.ShowLength
    assert "nominal/DataGraph" not in EvtGen.ShowLength

    leng = GrGen.ShowLength
    assert leng["nominal/DataGraph"] == leng["nominal/EventEx"]

    GrGen = GraphGenerator(EvtGen)
    GrGen.Graph = gr
    GrGen.Threads = 1
    GrGen.Chunks = 10
    GrGen.EventStop = 10
    GrGen.MakeGraphs()
    assert len(GrGen.ShowLength) == 2
    assert "nominal/DataGraph" in GrGen.ShowLength
    assert "nominal/EventEx" in GrGen.ShowLength
    assert "nominal/DataGraph" not in EvtGen.ShowLength
    leng = GrGen.ShowLength
    assert leng["nominal/DataGraph"] == 10

    GrGen = GraphGenerator(EvtGen)
    GrGen.Graph = gr
    GrGen.Threads = 1
    GrGen.Chunks = 10
    GrGen.MakeGraphs()
    GrGen.GraphName = "DataGraph"
    GrGen.EventName = "EventEx"

    x = []
    for i in GrGen:
        assert i.edge_index is not None
        x.append(i)
        assert isinstance(i.Topology, list)
        assert len(i.Tops) != 0
        for t in i.Particles:
            assert t.px != 0
            assert t.pt != 0
            assert t.pt is not None
        assert "DAOD" in i.ROOT
        assert len(i.Errors) == 1
        assert i.N_pmu is not None
        assert i.GraphName == "DataGraph"
    assert len(x) == GrGen.ShowLength["nominal/EventEx"]

    for i in range(10):
        Gr = GraphGenerator(GrGen)
        Gr.Graph = gr
        Gr.Threads = 2
        Gr.Chunks = 100
        Gr.MakeGraphs()
        GrGen += Gr
    data, event = list(GrGen.ShowLength.values())
    assert data == event





if __name__ == "__main__":
    test_graph_generator()
