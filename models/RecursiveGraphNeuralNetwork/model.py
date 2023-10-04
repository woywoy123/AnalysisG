import torch
from torch_geometric.nn import MessagePassing, LayerNorm
from torch_geometric.nn.models import MLP

# custom pyc extension functions
import pyc




class RecursiveGraphNeuralNetwork(MessagePassing):

    def __init__(self):
        super().__init__(aggr = None, flow = "target_to_source")
        end = 32

        self.O_top_edge = None
        self.L_top_edge = "CEL"
        self._inpt = MLP([ef, end])
        self._edge = MLP([end, 2])


    def message(self, Pmc_i, Pmc_j):
        pass


    def forward(self, i, edge_index, batch, N_pT, N_eta, N_phi, N_energy):

        pass








