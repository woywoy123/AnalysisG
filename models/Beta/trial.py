import torch_geometric
from AnalysisG.IO import UnpickleObject
from torch_geometric.utils import to_dense_adj
from torch_geometric.data import Batch, Data

import torch
from GRNN import RecursiveGraphNeuralNetwork
torch.set_printoptions(precision=3, sci_mode = False)
torch.set_printoptions(profile = "full", linewidth = 20000)

x = UnpickleObject("data/GraphTruthJet")
data = Batch().from_data_list(x[:1]).to(device = "cuda:0")
inpt = [
    "edge_index", "batch",
    "G_met", "G_phi", "G_n_jets", "G_n_lep",
    "N_pT", "N_eta", "N_phi", "N_energy", "N_is_lep", "N_is_b"
]

test = (getattr(data, i) for i in inpt)
inpt = {i : getattr(data, i) for i in inpt}
model = RecursiveGraphNeuralNetwork()
model = torch.jit.script(model).to(device = "cuda:0")
model.train()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.CrossEntropyLoss()
for i in range(100000):
    print("....")
    optimizer.zero_grad()
    model(**inpt)
    edge = model.O_top_edge
    edge_t = data.E_T_top_edge.to(dtype = torch.long).view(-1)
    loss = loss_fn(edge, edge_t)
    pred = edge.max(-1)[1]
    loss.backward()
    optimizer.step()
    #if i%100 != 0:continue
    train_acc = ((pred == edge_t).view(-1).sum(-1))/pred.size(0)
    print("Accuracy = {}, Loss = {}, iter = {}".format(train_acc, loss, model.iter))
    #print(to_dense_adj(inpt["edge_index"], edge_attr = edge_t)[0])
    #print(to_dense_adj(inpt["edge_index"], edge_attr = pred)[0])
