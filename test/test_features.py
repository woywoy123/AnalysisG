from AnalysisG.Events import Event
from AnalysisG.Events import GraphTops, GraphChildren
from AnalysisG.Templates import ApplyFeatures
from AnalysisG.Model import ModelWrapper
from AnalysisG import Analysis

smpl = "./samples/"
Files = {
            smpl + "sample1" : ["smpl1.root"], 
            smpl + "sample2" : ["smpl1.root", "smpl2.root", "smpl3.root"]
}

def _MakeSample():
    Ana = Analysis()
    Ana.ProjectName = "_Project"
    Ana.InputSample(None, Files)
    Ana.Event = Event
    Ana.EventCache = True
    Ana.Launch 
    return Ana

def _MakeGraph(Ana, Graph, mode):
    ApplyFeatures(Ana, mode) 
    Ana.EventGraph = Graph
    Ana.DataCache = True
    Ana.Launch
    return Ana

def test_truth_top():
    Ana = _MakeSample()
    res_mass = []
    for i in Ana:
        res = sum([k for k in i.Tops if k.FromRes == 1]).Mass/1000
        res_mass.append(res)

    Ana = _MakeGraph(Ana, GraphTops, "TruthTops")
    M = ModelWrapper()  

    it = 0 
    for i in Ana:
        i = i.clone()
        x = M.MassEdgeFeature(i, i.E_T_res_edge.view(-1))
        diff = 100*abs(res_mass[it] - x.item()) / res_mass[it]
        assert diff < 1
       
        x = M.MassNodeFeature(i, i.N_T_res_node.view(-1))
        diff = 100*abs(res_mass[it] - x.item()) / res_mass[it]
        assert diff < 1

        assert i.G_T_ntops
        assert i.G_phi
        assert i.G_met

        it += 1

def test_truth_children():
    Ana = _MakeSample()
    res_mass = []
    top_mass = []
    edge_lep = []
    n_nus = []
    for i in Ana:
        tops = i.Tops
        children = [c for t in tops for c in t.Children if t.FromRes == 1]
        to = children[0].Parent + children[1].Parent
        res_mass.append(sum(children).Mass/1000)
        top_mass.append([sum(t.Children).Mass/1000 for t in tops])
        _leps = [int((c1.is_b + c2.is_b)*(c1.is_lep + c2.is_lep)*(c1.Parent[0] == c2.Parent[0]) > 0) for c1 in i.TopChildren for c2 in i.TopChildren]
        edge_lep.append(_leps)
        n_nus.append(sum([c.is_nu for c in i.TopChildren]))

    Ana = _MakeGraph(Ana, GraphChildren, "TruthChildren")
    M = ModelWrapper()  

    it = 0
    for i in Ana:
        i = i.clone()
        assert i.G_T_signal.item() == 1
        assert i.G_T_ntops.item() == 4
        assert i.G_T_n_nu.item() == n_nus[it]

        x = M.MassEdgeFeature(i, i.E_T_res_edge.view(-1))
        diff = 100*abs(res_mass[it] - x.item())/res_mass[it]
        assert diff < 1

        x = M.MassEdgeFeature(i, i.E_T_top_edge.view(-1))
        x = x.tolist()
        x.sort()
        top_mass[it].sort()
        diff = sum([abs(top_mass[it][g] - x[g])/top_mass[it][g] >= 0.1 for g in range(len(x))])
        assert diff < 1
        assert len(top_mass[it]) == len(x)

        _leps = i.E_T_lep_edge.view(-1).tolist()
        assert len(edge_lep[it]) == len(_leps)
        assert sum([k == j for k, j in zip(_leps, edge_lep[it])]) == len(edge_lep[it])
        it+=1





if __name__ == "__main__":

    #Ana = Analysis()
    #Ana.rm("_Project")
    #test_truth_top()
    test_truth_children()    

