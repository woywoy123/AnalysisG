import torch
from torch.nn import Sequential as Seq, ReLU, Tanh, Sigmoid
import torch.nn.functional as F
from torch import nn

from torch_scatter import scatter
import torch_geometric
from torch_geometric.nn import MessagePassing, GCNConv, Linear

import LorentzVector as LV

class GraphNN(nn.Module):
    
    def __init__(self):
        super(GraphNN, self).__init__()
        self.layers = Seq(
                Linear(1, 64), 
                ReLU(), 
                Linear(64, 32), 
                ReLU(), 
                Linear(32, 2)
        )
        
        self.L_Signal = "CEL"
        self.C_Signal = True
        self.O_Signal = None

    def forward(self, G_Signal, edge_index):
        self.O_Signal = self.layers(G_Signal)
        return self.O_Signal


class NodeConv(MessagePassing):

    def __init__(self, in_channels, out_channels):
        super(NodeConv, self).__init__(aggr = "mean") 
        self.mlp = Seq(
                Linear(2 * in_channels, 256), 
                ReLU(), 
                Linear(256, out_channels)
        )
    
        self.L_x = "CEL"
        self.C_x = True
        self.O_x = None
    
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
                Linear(1, 256),
                Sigmoid(),
                Linear(256, 2)
        )

        self.mlp_NodeFeatures = Seq(
                Linear(1*2, 256), 
                Sigmoid(),
                Linear(256, 2)
        )

        self.mlp_GraphSignal = Seq(
                Linear(1, 256), 
                Sigmoid(),
                Linear(256, 2)
        )
    
        self.mlp_GraphMuActual = Seq(
                Linear(1, 256*4), 
                Linear(256*4, 1)
        )

        self.mlp_GraphMissingET = Seq(
                Linear(1, 256), 
                Linear(256, 1)
        )

        self.mlp_GraphMissingPhi = Seq(
                Linear(1, 4096), 
                Linear(4096, 1)
        )

        
        # Predict the Topology of the event 
        self.O_Topology = None
        self.L_Topology = "CEL"
        self.C_Topology = True
        self.N_Topology = True
        
        # Predict which particles in the event are resonance related
        self.O_NodeSignal = None
        self.L_NodeSignal = "CEL"
        self.C_NodeSignal = True
        
        # Is the event a signal event i.e. originate from Z'
        self.O_GraphSignal = None
        self.L_GraphSignal = "CEL"
        self.C_GraphSignal = True
        
        # For fun predict true pile-up
        self.O_GraphMuActual = None
        self.L_GraphMuActual = "MSEL"
        self.C_GraphMuActual = False

        # For fun predict MissingET
        self.O_GraphEt = None
        self.L_GraphEt = "MSEL"
        self.C_GraphEt = False

        # For fun predict MissingPhi
        self.O_GraphPhi = None
        self.L_GraphPhi = "MSEL"
        self.C_GraphPhi = False
        
        self.Device = ""
    
    def forward(self, edge_index, i,
            E_dr, E_mass, E_signal, 
            N_eta, N_pt, N_phi, N_energy, N_signal, 
            G_mu, G_m_phi, G_m_et, G_signal):
        
        batch_len = i.shape[0]
        
        self.O_Topology = self.mlp_EdgeFeatures(torch.cat([E_signal], dim = 1))

        self.O_NodeSignal = self.propagate(edge_index, x = torch.cat([N_signal], dim = 1))

        self.O_GraphSignal = self.mlp_GraphSignal(torch.cat([G_signal], dim = 1))
        self.O_GraphMuActual = self.mlp_GraphMuActual(G_mu) 
        
        # Calculate the net ET and Phi from nodes 
        Gr_T_MissingPhi = N_phi.view(batch_len, -1).sum(dim = 1, keepdim = True)
        Gr_T_MissingET = N_energy.view(batch_len, -1).sum(dim =1, keepdim = True)
        
        self.O_GraphPhi = self.mlp_GraphMissingPhi(Gr_T_MissingPhi)
        self.O_GraphEt = self.mlp_GraphMissingET(G_m_et)

        return self.O_Topology, self.O_NodeSignal, self.O_GraphSignal, self.O_GraphMuActual, self.O_GraphEt, self.O_GraphPhi



    def message(self, x_i, x_j):
        tmp = torch.cat([x_i, x_j-x_i], dim = 1)
        return self.mlp_NodeFeatures(tmp)
        

class MassGraphNeuralNetwork(MessagePassing):

    def __init__(self):
        super(MassGraphNeuralNetwork, self).__init__(aggr = None, flow = "target_to_source")
        self._mass_mlp = Seq(
                Linear(5, 256),
                Linear(256, 256), 
                Linear(256, 256)
        )
        
        self._edge_mass = Seq(
                Linear(256, 256), 
                ReLU(),
                Linear(256, 128), 
                ReLU(), 
                Linear(128, 2)
        )
        
        self._edge_mlp = Seq(
                Linear(4, 4), 
                Linear(4, 2)
        )


        self.O_Index = None
        self.L_Index = "CEL"
        self.C_Index = True

    def forward(self, edge_index, N_eta, N_energy, N_pT, N_phi):
        Pmu = torch.cat([N_pT, N_eta, N_phi, N_energy], dim = 1)
        Pmu_cart = LV.TensorToPxPyPzE(Pmu)
        TMP = self.propagate(edge_index, Pmu = Pmu_cart)
        self.O_Index = TMP
        return self.O_Index

    def message(self, index, Pmu_i, Pmu_j):
        Pmu = Pmu_i + Pmu_j
        Mass = LV.MassFromPxPyPzE(Pmu)
        return self._mass_mlp(torch.cat([Pmu, Mass], dim = 1)), Pmu_j
    
    def aggregate(self, message, index, Pmu):
        e_mlp = message[0]
        Pmu_inc = message[1]
        
        mass_mlp = self._edge_mass(e_mlp)
        mass_bool = mass_mlp.max(1)[1]
        for i in range(Pmu.shape[0]):
            swi = mass_bool[i == index].view(-1, 1)
            P_inc = torch.cumsum(Pmu_inc[i == index]*swi, dim = 0)
            if i == 0:
                Psum = P_inc
                continue
            Psum = torch.cat([Psum, P_inc], dim = 0)

        mass = LV.MassFromPxPyPzE(Psum)
        print(mass[mass != 0].unique()) 
        
        mass = self._mass_mlp(torch.cat([Psum, mass], dim = 1))
        mass = self._edge_mass(mass)
        
        return self._edge_mlp(torch.cat([mass, mass_mlp], dim = 1))

