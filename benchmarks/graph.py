from AnalysisG.Plotting import TLine, TH2F
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
            for cls in range(2, 10):
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
            PickleObject(output, "edge")


def plot_edge_aggregation():
    out = UnpickleObject("edge")
    lines = {}
    for num_nodes in out:
        xdata = []
        ydata = []
        weight = []
        for num_feat in out[num_nodes]:
            for cls in out[num_nodes][num_feat]:
                data = out[num_nodes][num_feat][cls]
                cu = data["cu"][0]
                py = data["py"][0]
                weight.append(py/cu)
                xdata.append(num_feat)
                ydata.append(cls)

                if cls not in lines:
                    lines[cls] = {}
                    lines[cls]["weight"] = []
                    lines[cls]["xData"] = []
                    lines[cls]["lines"] = []
                lines[cls]["weight"].append(py/cu)
                lines[cls]["xData"].append(num_feat)


        for cls in lines:
            tl1 = TLine()
            tl1.Title = "N-" + str(num_nodes)
            tl1.xData = lines[cls]["xData"]
            tl1.yData = lines[cls]["weight"]
            lines[cls]["lines"].append(tl1)
            lines[cls]["xData"] = []
            lines[cls]["weight"] = []


        x = TH2F()
        x.xData = xdata
        x.yData = ydata
        x.Weight = weight
        title = "Time Ratio between PyTorch Reference Implementation \n and Native CUDA for "
        title += str(num_nodes) + " Nodes"
        x.Title = title
        x.yTitle = "Number of Output Classifications"
        x.xTitle = "Number of Input Features"
        x.Filename = "Nodes-" + str(num_nodes)
        x.Style = "ROOT"
        x.xMin = 0
        x.yMin = 0
        x.SaveFigure()

    for cls in lines:
        tls = TLine()
        tls.Title = "Ratio plot of Computational Time between PyTorch and CUDA (pyc) \n for Different Number of Input Nodes (higher is better)"
        tls.yTitle = "PyTorch Time (s) /CUDA Time (s)"
        tls.Filename = "Performance-Per-Nodes-output_"+str(cls)
        tls.Lines = lines[cls]["lines"]
        tls.SaveFigure()



if __name__ == "__main__":
    edge_aggregation()
    plot_edge_aggregation()
