import torch
from random import random
torch.ops.load_library("../build/pyc/interface/libgraph_cuda.so")
device = "cuda"
torch.set_printoptions(threshold=1000000, linewidth = 120)


def test_edge_feature_aggregation():
    def aggregation(edge_index, pred, pmu):
        cls_ = torch.max(pred)
        output = {}
        p = pmu.size(0)
        for i in torch.arange(cls_+1):
            sel = int(i.item())
            pair_m = torch.ones(p, p, dtype = torch.long)*-1

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
            pmu_u = torch.zeros(len(clusters), pmu.size(1))
            for this, j in zip(range(len(clusters)), clusters):
                get = j[j > -1]
                pmu_u[this] += pmu[get, :].sum(0)

            output[int(i)] = {}
            output[int(i)]["pairs"] = pair_m
            output[int(i)]["clusters"] = clusters
            output[int(i)]["unique_sum"] = pmu_u
            output[int(i)]["reverse_clusters"] = revert
            output[int(i)]["node_sum"] = pmu_i
        return output

    # create a node feature of length 10
    n = 10
    n_nodes = 12
    nodes = torch.tensor([[int(1 + 10*random()*(t+1)) for _ in range(n)] for t in range(n_nodes)])

    # create some fake truth connections 
    n_cls = 2 # classifications
    p_ = 0.95 # probability that nodes are always connected with the same class, i.e. more equal clusters
    edges_t = torch.tensor([[(random() > p_)*int(random()*n_cls)] for _ in range(n_nodes**2)])

    # create the edge index
    edge_i = torch.tensor([t for t in range(n_nodes) for _ in range(n_nodes)])
    edge_j = torch.tensor([t for _ in range(n_nodes) for t in range(n_nodes)])
    edge_index = torch.cat([edge_i.view(1, -1), edge_j.view(1, -1)], dim = 0)

    # manually compute the truth clusters
    dic = {cls : {n : nodes[n].clone() for n in range(n_nodes)} for cls in range(n_cls)}
    for i, src, dst in zip(edges_t, edge_i, edge_j):
        if src == dst: continue
        dic[i.item()][src.item()] += nodes[dst]

    res = {}
    for cls in dic:
        res[cls] = []
        for node in dic[cls]: res[cls].append(dic[cls][node].view(1, -1))
        res[cls] = torch.cat(res[cls], dim = 0)

    out = aggregation(edge_index, edges_t, nodes)
    for i in out: assert (res[i] != out[i]["node_sum"]).view(-1).sum(-1) == 0
    edge_index = edge_index.to(device = "cuda")

    t = torch.zeros((n_nodes, n_nodes, n_cls), device = "cuda")
    t[edge_index[0], edge_index[1], edges_t.view(-1)] = 1
    t = t.view(-1, n_cls)

    nodes = nodes.to(device = "cuda")
    x = torch.ops.graph_cuda.graph_edge_aggregation(edge_index, t, nodes)
    assert not (x["cls::1::node-sum"] != out[1]["node_sum"].to(device = "cuda")).view(-1).sum(-1)

    node_inc = x["cls::1::node-indices"]
    node_int = out[1]["pairs"]

    xm = (node_int > -1).sum(-1).view(-1).max()
    node_int = node_int[:, :xm].to(device = "cuda")
    assert not (node_inc != node_int).view(-1).sum(-1)

    node_t = torch.rand((n_nodes, n_cls), device = "cuda").softmax(-1)
    x = torch.ops.graph_cuda.graph_node_aggregation(edge_index, node_t, nodes)

def test_edge_aggregation_nodupl():
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # create a node feature of length 10
    n_nodes = 1000
    dim = 100

    nodes = torch.rand((n_nodes, dim), device = dev)

    # create some fake truth connections 
    n_cls = 10 # classifications
    edges_t = torch.rand((n_nodes**2, n_cls), device = dev).softmax(-1)

    # create the edge index
    edge_i = torch.tensor([t for t in range(n_nodes) for _ in range(n_nodes)])
    edge_j = torch.tensor([t for _ in range(n_nodes) for t in range(n_nodes)])
    edge_index = torch.cat([edge_i.view(1, -1), edge_j.view(1, -1)], dim = 0).to(device = dev)

    edge_index = torch.cat([edge_index, edge_index], -1)
    edges_t = torch.cat([edges_t, edges_t], 0)
    x = torch.ops.graph_cuda.graph_edge_aggregation(edge_index, edges_t, nodes)
    clust = x["cls::1::node-indices"]
    clust_l, node_l = clust.tolist(), nodes.tolist()
    non_dupl = [[sum([node_l[k][f] for k in set(i) if k != -1]) for f in range(dim)] for i in clust_l]
    x = torch.ops.graph_cuda.graph_unique_aggregation(clust, nodes)["node-sum"]
    x = x.to(device = "cpu")
    assert not (torch.abs(x - torch.tensor(non_dupl)) > 10e-4).view(-1).sum(-1)

if __name__ == "__main__":
    test_edge_feature_aggregation()
    test_edge_aggregation_nodupl()

