import torch
from torch.nn import Sequential as Seq, ReLU, Tanh, Sigmoid
import torch.nn.functional as F
from torch import nn

from torch_scatter import scatter
import torch_geometric
from torch_geometric.nn import MessagePassing, GCNConv, Linear

import LorentzVector as LV

class GraphNN(nn.Module):
    
    def __init__(self, inputs = 1):
        super(GraphNN, self).__init__()
        self.layers = nn.Sequential(
                nn.Linear(1, 64), 
                nn.ReLU(), 
                nn.Linear(64, 32), 
                nn.ReLU(), 
                nn.Linear(32, 1)
        )
        
        self.L_Signal = "CEL"
        self.C_Signal = True
        self.O_Signal = 0

    def forward(self, G_Signal, edge_index):
        self.O_Signal = self.layers(G_Signal.view(-1, 1))

        return self.O_Signal


class NodeConv(MessagePassing):

    def __init__(self, in_channels, out_channels):
        super(EdgeConv, self).__init__(aggr = "mean") 
        self.mlp = Seq(
                Linear(2 * in_channels, 2), 
                ReLU(), 
                Linear(2, out_channels)
        )
    
        self.L_x = "CEL"
        self.C_x = True
        self.O_x = 0
    
    def forward(self, N_x, N_Sig, edge_index):
        x = torch.cat((N_x, N_Sig), dim = 1)
        self.O_x = self.propagate(edge_index, x = x)        
        return self.O_x

    def message(self, x_i, x_j):
        tmp = torch.cat([x_i, x_j - x_i], dim = 1)
        return self.mlp(tmp)

    def update(self, aggr_out):
        return F.normalize(aggr_out)


class EdgeConv(MessagePassing):

    def __init__(self, in_channels, out_channels):
        super(EdgeConv, self).__init__(aggr = "add") 
        self.mlp = Seq(
                Linear(in_channels, 256),
                Linear(256, out_channels)
        )
    
        self.L_x = "MSEL"
        self.N_x = False
        self.C_x = False

        self.O_x = 0
    
    def forward(self, E_x, edge_index):
        self.__edge = E_x
        self.O_x = self.message()
        return self.O_x

    def message(self):
        tmp = self.__edge.view(-1, 1)
        return self.mlp(tmp)

    def update(self, aggr_out):
        return F.normalize(aggr_out)


class CombinedConv(MessagePassing):

    def __init__(self):
        super(CombinedConv, self).__init__(aggr = "add")
        self.mlp_EdgeFeatures = Seq(
                Linear(3, 256), 
                Sigmoid(),
                Linear(256, 2)
        )

        self.mlp_NodeFeatures = Seq(
                Linear(5*2, 256), 
                Sigmoid(),
                Linear(256, 2)
        )

        self.mlp_GraphSignal = Seq(
                Linear(3, 256), 
                Sigmoid(),
                Linear(256, 2)
        )
        
        # Predict the Topology of the event 
        self.O_Topology = 0
        self.L_Topology = "CEL"
        self.C_Topology = True
        self.N_Topology = True
        
        # Predict which particles in the event are resonance related
        self.O_NodeSignal = 0
        self.L_NodeSignal = "CEL"
        self.C_NodeSignal = True
        
        # Is the event a signal event i.e. originate from Z'
        self.O_GraphSignal = 0
        self.L_GraphSignal = "CEL"
        self.C_GraphSignal = True
        
        # For fun predict true pile-up
        self.O_GraphMuActual = 0
        self.L_GraphMuActual = "MSEL"
        self.C_GraphMuActual = False

        # For fun predict MissingET
        self.O_GraphEt = 0
        self.L_GraphEt = "MSEL"
        self.C_GraphEt = False

        # For fun predict MissingPhi
        self.O_GraphPhi = 0
        self.L_GraphPhi = "MSEL"
        self.C_GraphPhi = False
        
        self.Device = ""
    
    def forward(self, edge_index, 
            E_dr, E_mass, E_signal, 
            N_eta, N_pt, N_phi, N_energy, N_signal, 
            G_mu, G_m_phi, G_m_et, G_signal):
    
        
        self.O_Topology = self.mlp_EdgeFeatures(torch.cat([E_dr, E_mass, E_signal], dim = 1))
        self.O_NodeSignal = self.propagate(edge_index, x = torch.cat([N_eta, N_pt, N_phi, N_energy, N_signal], dim = 1))
        

        self.O_GraphSignal = self.mlp_GraphSignal(torch.cat([G_mu, G_m_phi, G_signal], dim = 0))
        
        p = LV.ToPxPyPzE(N_pt, N_eta, N_phi, N_energy, "cuda")
        





        print(Graph_Encode)
        #print(torch.cat([G_mu, G_m_phi, G_signal], dim = 0))








        return self.O_Topology, self.O_NodeSignal, self.O_GraphSignal, self.O_GraphMuActual, self.O_GraphEt, self.O_GraphPhi



    def message(self, x_i, x_j):
        tmp = torch.cat([x_i, x_j-x_i], dim = 1)
        return self.mlp_NodeFeatures(tmp)
        


