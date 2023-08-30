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
    EvtGen = EventGenerator({root1: []})
    EvtGen.EventStop = 50
    EvtGen.EventStart = 10
    EvtGen.Event = EventEx
    EvtGen.Threads = 1
    EvtGen.MakeEvents()

    gr = DataGraph()
    gr.AddGraphFeature(fx_graph, "Failed")
    gr.AddGraphFeature(fx_mev, "mev")
    gr.AddNodeFeature(fx_pmu, "pmu")
    gr.AddEdgeFeature(fx_edge, "Mass")
    gr.AddEdgeTruthFeature(fx_custom_topology)
    gr.AddPreSelection(fx_prefilter)

    GrGen = GraphGenerator(EvtGen)
    GrGen.Graph = gr
    GrGen.Threads = 1
    GrGen.MakeGraphs()

if __name__ == "__main__":
    test_graph_generator()
