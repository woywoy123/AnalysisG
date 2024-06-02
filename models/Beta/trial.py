
import torch_geometric
from AnalysisG.IO import UnpickleObject
from torch_geometric.utils import to_dense_adj
from torch_geometric.data import Batch, Data

import torch
torch.set_printoptions(precision=3, sci_mode = False)
torch.set_printoptions(profile = "full", linewidth = 20000)

from GRNN import RecursiveGraphNeuralNetwork
x = UnpickleObject("data/GraphTruthJet")
data = Batch().from_data_list(x[:1]).to(device = "cuda:0")
inpt = [
    "edge_index", "batch",
    "G_met", "G_phi", "G_n_jets", "G_n_lep",
    "N_pT", "N_eta", "N_phi", "N_energy", "N_is_lep", "N_is_b"
]

inpt = {i : getattr(data, i) for i in inpt}
from pyc.interface import pyc_cuda
from torch.onnx import register_custom_op_symbolic
#from torch.onnx import symbolic_helper
#@symbolic_helper.parse_args("t")
#def symbolic(t):
#    return (t)
#t = torch.cat([data.N_pT, data.N_eta, data.N_phi, data.N_energy], -1)
#register_custom_op_symbolic("pyc_cuda::transform_combined_PxPyPzE", pyc_cuda.combined.transform.PxPyPzE, 14)

model = RecursiveGraphNeuralNetwork().to(device = "cuda:0")
cmodel = torch.jit.script(model) #, example_inputs = tuple(inpt.values()))
model = torch.compile(cmodel)
x = model(**inpt)
#torch.onnx.export(
#        cmodel, tuple(inpt.values()),
#        "test.onnx",
#        input_names = [
#            "edge_index", "batch",
#            "G_met", "G_phi", "G_n_jets", "G_n_lep",
#            "N_pT", "N_eta", "N_phi", "N_energy", "N_is_lep", "N_is_b"
#        ],
#        operator_export_type = torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH
#)
#import onnx
#model = onnx.load("test.onnx")
#onnx.checker.check_model(model)
#import onnxruntime as ort
#ses = ort.InferenceSession("test.onnx")
#o = ses.run(None, inpt)
#print(o)
#exit()
#
#model = torch.jit.script(model, example_inputs = inpt)
#model = model.to(device = "cuda:0")
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.CrossEntropyLoss()
for i in range(100000):
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
    print("Accuracy = {}, Loss = {}, iter = {}".format(train_acc, loss, model._cls))
    #print(to_dense_adj(inpt["edge_index"], edge_attr = edge_t)[0])
    #print(to_dense_adj(inpt["edge_index"], edge_attr = pred)[0])
