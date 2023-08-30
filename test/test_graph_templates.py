from AnalysisG.Generators import EventGenerator
from examples.Graph import DataGraph
from examples.Event import EventEx
from AnalysisG.IO import UpROOT
import torch

smpl = "./samples/"
Files = {
    smpl + "sample1": ["smpl1.root"],
    smpl + "sample2": ["smpl1.root", "smpl2.root", "smpl3.root"],
}

def template():
    EvtGen = EventGenerator(Files)
    EvtGen.EventStop = 50
    EvtGen.EventStart = 10
    EvtGen.Event = EventEx
    EvtGen.Threads = 1
    EvtGen.MakeEvents()
    return EvtGen

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


def test_graph_topology():
    ev = template()
    for event in ev:
        x = DataGraph(event)
        x.SetTopology()
        lt = len(event.Tops)
        assert len(x.Topology) == lt**2
        x.self_loops = False
        x.SetTopology()
        assert len(x.Topology) == lt**2 - lt

        cust = [[1, 2], [0, 1]]
        x.SetTopology(cust)
        res = x.Topology
        for t in cust: assert t in res

        x.self_loops = False
        x.SetTopology()
        x.SetTopology(fx_custom_topology)
        topo = x.Topology
        res_ = []
        for t in event.Tops:
            index = x.ParticleToIndex(t)
            assert index > -1
            if not t.FromResonance: continue
            res_.append((t, index))
        pairs = [index for t, index in res_]
        assert pairs in topo
        assert [pairs[0], pairs[0]] not in topo

        x.self_loops = True
        x.SetTopology()
        x.SetTopology(fx_custom_topology)
        topo = x.Topology
        res_ = []
        for t in event.Tops:
            index = x.ParticleToIndex(t)
            assert index > -1
            if not t.FromResonance: continue
            res_.append((t, index))
        pairs = [index for t, index in res_]
        assert pairs in topo
        assert [pairs[0], pairs[0]] in topo

def test_graph_features():
    ev = template()
    for event in ev:
        x = DataGraph(event)
        x.SetTopology([[1, 0], [1, 2]])
        x.AddGraphFeature(fx_graph, "Failed")
        x.AddGraphFeature(fx_mev, "mev")
        x.AddNodeFeature(fx_pmu, "pmu")
        x.AddEdgeFeature(fx_edge, "Mass")
        x.AddEdgeTruthFeature(fx_custom_topology)
        x.AddPreSelection(fx_prefilter)
        x.Build()
        if x.SkipGraph: continue
        tops_1p = [event.Tops[1], event.Tops[0]]
        tops_2p = [event.Tops[1], event.Tops[2]]
        topo = x.edge_index
        assert len(x.Errors) == 1
        assert x.G_mev is not None
        assert x.N_pmu is not None
        assert x.E_Mass is not None
        assert x.E_T_fx_custom_topology is not None

        x1 = sum(tops_1p).Mass
        x2 = sum(tops_2p).Mass
        t = abs(x.E_Mass - torch.tensor([[x1], [x2]])) < 1
        assert sum(t) == 2
        assert "DAOD" in x.ROOT

if __name__ == "__main__":
    #test_graph_topology()
    test_graph_features()
    pass
