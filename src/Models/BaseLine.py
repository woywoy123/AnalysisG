import torch
from torch.nn import Sequential as Seq, Linear, ReLU
from torch_geometric.nn import MessagePassing

# This model is a replica of the Paper: "Semi-supervised Classification with Graph Convolutional Networks" ~ https://arxiv.org/abs/1609.02907
class BaseLineModel(MessagePassing):

    def __init__(self, in_channels, out_channels):
        super().__init__(aggr = "max")
       
        self._mlp = Seq(Linear(2*in_channels, out_channels), 
                        ReLU(), 
                        Linear(out_channels, out_channels))

        self.O_Index = None
        self.L_Index = "CEL"
        self.C_Index = True
        #self.N_Index = True

        self.Device = ""

    
    def forward(self, edge_index, N_Index):
        self.O_Index = self.propagate(edge_index, x = N_Index) 
        return self.O_Index
    
    def message(self, x_i, x_j):
        tmp = torch.cat([x_i, x_j - x_i], dim = 1)
        return self._mlp(tmp)


import LorentzVector as LV
from torch_geometric.utils import *
class BaseLineModelAdvanced(MessagePassing):

    def __init__(self):
        super().__init__(aggr = "max")   

        self.O_Topo = None
        self.L_Topo = "CEL"
        self.C_Topo = True
        self.N_Topo = False

        self._mlp = Seq(Linear(11, 256), 
                        ReLU(),
                        Linear(256, 1024),
                        ReLU(),
                        Linear(1024, 12))

        self._mlp_dr = Seq(Linear(1, 256), 
                        ReLU(),
                        Linear(256, 1024),
                        ReLU(),
                        Linear(1024, 12))


        self._mlp_m = Seq(Linear(1, 1024), 
                          ReLU(),
                          Linear(1024, 1024),
                          ReLU(),
                          Linear(1024, 2))

        self._mlp_edge = Seq(Linear(12 + 12 + 2, 256),
                            ReLU(),
                            Linear(256, 2))

        self._iter = torch.tensor(0, dtype = torch.int)
        self._zero = torch.tensor(0, dtype = torch.int)
        self._lim = torch.tensor(2, dtype = torch.int)
        self._edge_index = None

    def forward(self, edge_index, N_pT, N_eta, N_phi, N_energy):
        self._iter.to(N_pT.device)
        self._zero.to(N_pT.device)
        self._lim.to(N_pT.device)

        if self._iter == self._zero:
            self._P_mu = LV.TensorToPxPyPzE(torch.cat([N_pT, N_eta, N_phi, N_energy], dim = 1))
            self._edge_index = edge_index
        
        if self._iter > self._lim or edge_index[0].shape[0] == self._zero:
            self._iter = self._zero
            return self.O_Topo

        prop = self.propagate(edge_index, x = self._P_mu)
        mass = self._mlp_m(LV.MassFromPxPyPzE(self._P_mu[edge_index[0]] + self._P_mu[edge_index[1]]))
        dr = self.deltaR(edge_index, N_eta, N_phi)

        if self._iter == self._zero:
            self.O_Topo = self._mlp_edge(torch.cat([prop[edge_index[0]], mass, dr], dim = 1))
        else: 
            self.O_Topo[edge_index[0]] += self._mlp_edge(torch.cat([prop[edge_index[0]], mass, dr], dim = 1))
        
        self._iter = torch.add(self._iter, torch.tensor(1))
        pred = self.O_Topo[edge_index[0]].max(dim = 1)[1]
        edges = torch.cat([edge_index[0][pred == 1].view(1, -1), edge_index[1][pred == 1].view(1, -1)], dim = 0)
        return self.forward(edges, N_pT, N_eta, N_phi, N_energy) 
        
    def message(self, x_i, x_j):
        edge_vector = x_i + x_j
        edge_mass = LV.MassFromPxPyPzE(edge_vector)
        score = self._mlp_m(edge_mass)
        return self._mlp(torch.cat([x_i, edge_vector, edge_mass, score], dim = 1))

    def deltaR(self, edge_index, eta, phi):
        d_eta = torch.pow(eta[edge_index[0]] - eta[edge_index[1]], 2)
        d_phi = torch.pow(phi[edge_index[0]] - phi[edge_index[1]], 2)
        return self._mlp_dr(torch.pow(d_eta + d_phi, 0.5))

class BaseLineModelEvent(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.O_Topo = None
        self.L_Topo = "CEL"
        self.C_Topo = True
        self._BaseLineAd = BaseLineModelAdvanced()
        self._mlp_Topo = Seq(Linear(2 + 1 + 4 + 4, 256),
                            ReLU(), 
                            Linear(256, 256), 
                            ReLU(), 
                            Linear(256, 2))

        self.O_mu_actual = None 
        self.L_mu_actual = "MSEL"
        self.C_mu_actual = False
        self._mlp_mu = Seq(Linear(6, 256), 
                        ReLU(),
                        Linear(256, 1024),
                        ReLU(),
                        Linear(1024, 1))

        self.O_nTops = None 
        self.L_nTops = "CEL"
        self.C_nTops = True
        self._mlp_nTops = Seq(Linear(4, 256),
                            ReLU(), 
                            Linear(256, 256), 
                            ReLU(), 
                            Linear(256, 5))


        self.O_Index = None 
        self.L_Index = "CEL"
        self.C_Index = True
        self._mlp_mNodeTops = Seq(Linear(1+4, 256),
                                ReLU(), 
                                Linear(256, 256), 
                                ReLU(), 
                                Linear(256, 4))



    def forward(self, edge_index, i,  N_pT, N_eta, N_phi, N_energy,
                                  G_mu, G_met, G_met_phi, G_pileup, G_nTruthJets):
       
        batch_len = i.shape[0]
        P_mu = LV.TensorToPxPyPzE(torch.cat([N_pT, N_eta, N_phi, N_energy], dim = 1))
        
        # Return the output of the baseline GNN that tries to reco the event topology 
        Topo_pred = self._BaseLineAd(edge_index, N_pT, N_eta, N_phi, N_energy)
        e_i_active = Topo_pred.max(dim = 1)[1]
        e_s = edge_index[0][e_i_active == 1].view(1, -1)
        e_r = edge_index[1][e_i_active == 1].view(1, -1)
        
        # Use the source nodes and destination nodes predicted by above GNN to rebuild tops
        scatPx_ = torch.zeros(1, N_pT.shape[0], device = i.device, dtype = P_mu.dtype)
        scatPy_ = torch.zeros(1, N_pT.shape[0], device = i.device, dtype = P_mu.dtype)
        scatPz_ = torch.zeros(1, N_pT.shape[0], device = i.device, dtype = P_mu.dtype)
        scatE_ = torch.zeros(1, N_pT.shape[0], device = i.device, dtype = P_mu.dtype)

        scatPx_ = scatPx_.scatter_(1, e_s, P_mu[e_r, 0].view(1, -1), reduce= "add").view(-1, 1)
        scatPy_ = scatPy_.scatter_(1, e_s, P_mu[e_r, 1].view(1, -1), reduce= "add").view(-1, 1)
        scatPz_ = scatPz_.scatter_(1, e_s, P_mu[e_r, 2].view(1, -1), reduce= "add").view(-1, 1)
        scatE_ = scatE_.scatter_(1, e_s, P_mu[e_r, 3].view(1, -1), reduce= "add").view(-1, 1)
        P_mu_p = torch.cat([scatPx_, scatPy_, scatPz_, scatE_], dim = 1)
        t_m = LV.MassFromPxPyPzE(P_mu_p)
        
        # Make sure to batch these tops! 
        t_ = t_m.view(batch_len, -1)
         
        nTops = torch.zeros((batch_len, 1), device = t_m.device)
        for j in range(t_.shape[0]):
            nTops[j][0] = torch.unique(t_[j][t_[j] > 0]).shape[0]
        
        # Here we calculate the missing PT (MeV) in the event 
        P_mu_b = P_mu.view(batch_len, -1)
        P_mu_b = P_mu.view(batch_len, -1, 4).sum(dim = 1)
        
        # Can only do ET = -PT this if we assume massless particles - neutrinos 
        calc = LV.TensorToPtEtaPhiE(P_mu_b)
        MET_Meas = -calc[:, 0].view(-1, 1)
        MET_Phi = -calc[:, 2].view(-1, 1)
        
        # ===== Output pileup stuff and nTop predictions
        self.O_mu_actual = self._mlp_mu(torch.cat([nTops, G_mu, G_pileup, MET_Meas - G_met, MET_Phi - G_met_phi, G_nTruthJets], dim = 1))
        self.O_nTops = self._mlp_nTops(torch.cat([nTops, MET_Meas, MET_Phi, G_nTruthJets], dim = 1))

        # ===== Output the Topology prediction 
        self.O_Topo = self._mlp_Topo(torch.cat([Topo_pred, e_i_active.view(-1, 1), P_mu_p[edge_index[0]].view(-1, 4), P_mu_p[edge_index[1]].view(-1, 4)], dim = 1))

        # ===== Now calculate the node classification 
        self.O_Index = self._mlp_mNodeTops(torch.cat([t_m, P_mu_p - P_mu], dim = 1))
        return self.O_Topo, self.O_mu_actual, self.O_nTops, self.O_Index




