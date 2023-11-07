from AnalysisG.Plotting import TLine
from AnalysisG.IO import PickleObject, UnpickleObject
import pyc.Graph.Cartesian as graph
from random import random
from time import time
import statistics
import torch

def py_aggregation(edge_index, pred, pmu, incl_zero = False):
    cls_ = torch.max(pred)
    output = {}
    p = pmu.size(0)
    for i in torch.arange(cls_+1):
        if i == 0 and not incl_zero: continue
        sel = int(i.item())
        pair_m = (torch.ones(p, p, dtype = torch.long)*-1).to(device = "cuda")

        e_i, e_j = edge_index
        # Make sure to add the self loops
        pair_m[e_i, e_i] = e_i

        # now remove the self loops (avoid double counting)
        msk_self = e_i != e_j

        # remove the edges based on the prediction 
        msk = (pred.view(-1) == i)*msk_self
        e_i, e_j = edge_index[:, msk.view(-1)]
        pair_m[e_i, e_j] = e_j
        pair_m, _ = pair_m.sort(-1, True)

        # aggregate the incoming edges
        pmu_i = torch.zeros_like(pmu)
        for this, j in zip(range(p), pair_m): pmu_i[this] += pmu[j[j > -1], :].sum(0)

        # find the unique node aggregation
        clusters, revert = pair_m.unique(dim = 0, return_inverse = True)
        pmu_u = torch.zeros(len(clusters), pmu.size(1)).to(device = "cuda")
        for this, j in zip(range(len(clusters)), clusters):
            get = j[j > -1]
            pmu_u[this] += pmu[get, :].sum(0)

        output[int(i)] = {}
        output[int(i)]["clusters"] = clusters
        output[int(i)]["unique_sum"] = pmu_u
        output[int(i)]["reverse_clusters"] = revert
        output[int(i)]["node_sum"] = pmu_i
    return output

def edge_aggregation():

    output = {}
    for num_nodes in range(4, 20):
        output[num_nodes] = {}
        for num_feat in range(1, 20):
            output[num_nodes][num_feat] = {}
            for cls in range(1, 10):
                p_ = 0.4 # probability that nodes are always connected with the same class, i.e. more equal clusters
                nodes = torch.tensor([[int(1 + 10*random()*(t+1)) for _ in range(num_feat)] for t in range(num_nodes)], device = "cuda")
                edges_t = torch.tensor([[(random() > p_)*int(random()*cls)] for _ in range(num_nodes**2)], device = "cuda")
                edge_i = torch.tensor([t for t in range(num_nodes) for _ in range(num_nodes)])
                edge_j = torch.tensor([t for _ in range(num_nodes) for t in range(num_nodes)])
                edge_index = torch.cat([edge_i.view(1, -1), edge_j.view(1, -1)], dim = 0).to(device = "cuda")

                x, y = [], []
                for _ in range(100):
                    t1 = time()
                    _ = py_aggregation(edge_index, edges_t, nodes, True)
                    t2 = time() - t1
                    x.append(t2)

                    t1 = time()
                    _ = graph.edge(edge_index, edges_t, nodes, True)
                    t2 = time() - t1
                    y.append(t2)

                t_py, t_cu = statistics.mean(x), statistics.mean(y)
                st_py, st_cu = statistics.stdev(x), statistics.stdev(y)
                output[num_nodes][num_feat][cls] = {}
                output[num_nodes][num_feat][cls]["py"] = [t_py, st_py]
                output[num_nodes][num_feat][cls]["cu"] = [t_cu, st_cu]
            PickleObject(output)


def plot_edge_aggregation():
    out = UnpickleObject()
    num_nodes_d = {}
    num_feats_d = {}

    perf_py = []
    perf_cu = []
    for k in range(1, 10):
        for num_nodes in out:
            num_nodes_d[num_nodes] = {}
            num_nodes_d[num_nodes]["pyc"] = []
            num_nodes_d[num_nodes]["py"] = []
            for num_feat in out[num_nodes]:
                if num_feat not in num_feats_d:
                    num_feats_d[num_feat] = {}
                    num_feats_d[num_feat]["pyc"] = []
                    num_feats_d[num_feat]["py"] = []

                for cls in out[num_nodes][num_feat]:
                    if cls != k: continue
                    if num_feat != 19: continue
                    num_nodes_d[num_nodes]["pyc"] += [out[num_nodes][num_feat][cls]["cu"][0]]
                    num_nodes_d[num_nodes]["py"] += [out[num_nodes][num_feat][cls]["py"][0]]
                    num_feats_d[num_feat]["pyc"] += [out[num_nodes][num_feat][cls]["cu"][0]]
                    num_feats_d[num_feat]["py"] += [out[num_nodes][num_feat][cls]["py"][0]]

        tl1 = TLine()
        tl1.Title = "PyC (node-features-" + str(k) + ")"
        tl1.xData = sum([[i]*len(num_nodes_d[i]["pyc"]) for i in num_nodes_d], [])
        tl1.yData = sum([num_nodes_d[i]["pyc"] for i in num_nodes_d], [])

        tl2 = TLine()
        tl2.Title = "PyTorch (node-features-" + str(k) + ")"
        tl2.xData = sum([[i]*len(num_nodes_d[i]["py"]) for i in num_nodes_d], [])
        tl2.yData = sum([num_nodes_d[i]["py"] for i in num_nodes_d], [])

        tls = TLine()
        tls.Title = "PyTorch vs CUDA (PyC)"
        tls.xTitle = "Number of Nodes"
        tls.yTitle = "Time (s)"
        tls.Filename = "cls-" + str(k) + "_num_feats-19"
        tls.Lines = [tl1, tl2]
        tls.Markers = ["x", "-"]
        tls.SaveFigure()








if __name__ == "__main__":
#    edge_aggregation()
    plot_edge_aggregation()
