import torch
from torch_geometric.nn import MessagePassing
from torch.nn import Sequential as Seq, Linear, ReLU, Sigmoid, Tanh
import torch.nn.functional as F
import PyC.NuSol.CUDA as NuC
import PyC.Transform.CUDA as TC
import PyC.Physics.CUDA.Cartesian as PCC
from torch_geometric.utils import to_dense_adj, dense_to_sparse

torch.set_printoptions(4, profile = "full", linewidth = 100000)

class Recursion(MessagePassing):

    def __init__(self):
        super().__init__(aggr = None, flow = "target_to_source")
    
        self.O_edge = None 
        self.L_edge = "CEL"

        self.O_res = None 
        self.L_res = "CEL"

        self.O_signal = None 
        self.L_signal = "CEL"
        
        end = 64
        self._isTopMass = Seq(
                        Linear(1, end), 
                        ReLU(), Linear(end, 256), 
                        Sigmoid(), Linear(256, 128), 
                        ReLU(), Linear(128, 2), Sigmoid()
        )

        self._it = 0

    def _PrefEdge(self, Px_i, Px_j, Py_i, Py_j, Pz_i, Pz_j, E_i, E_j):
        mass = PCC.M(Px_i + Px_j, Py_i + Py_j, Pz_i + Pz_j, E_i + E_j)
        return mass, self._isTopMass(mass)


    def _RecursivePathMass(self, edge_index, Px, Py, Pz, E):
        src, dst = edge_index
        mass, mass_mlp = self._PrefEdge(Px[src], Px[dst], Py[src], Py[dst], Pz[src], Pz[dst], E[src], E[dst]) 
        _, mass_ = mass_mlp.max(-1)
        
        cnt = self._adjCount[src, dst]
        sel = torch.bernoulli(mass_mlp[:, 1]*cnt)
        self._adjPx[src, dst] += Px[dst] * sel
        self._adjPy[src, dst] += Py[dst] * sel
        self._adjPz[src, dst] += Pz[dst] * sel
        self._adjE[src, dst]  +=  E[dst] * sel
        self._adjCount[src, dst] -= sel
        
        # Add the diagonal self loops
        tmp = torch.diag(torch.diag(self._adjCount) == 0)
        self._adjPx += tmp*Px
        self._adjPy += tmp*Py
        self._adjPz += tmp*Pz
        self._adjE += tmp*E
        self._adjCount += tmp
        
        Px_ = self._adjPx.sum(-1)
        Py_ = self._adjPy.sum(-1)
        Pz_ = self._adjPz.sum(-1)
        E_  = self._adjE.sum(-1)

        src_, dst_ = dense_to_sparse(self._adjCount)[0]
        print(src_)
        print(dst_)
        
        





    def forward(self, i, edge_index, N_pT, N_eta, N_phi, N_energy, G_met, G_met_phi):
        pt, eta, phi, E = N_pT/1000, N_eta, N_phi, (N_energy/1000).view(-1)
        PxPyPz = TC.PxPyPz(pt, eta, phi)
        Px, Py, Pz = PxPyPz[:, 0], PxPyPz[:, 1], PxPyPz[:, 2]

        self._device = N_pT.device
        self._adjMLP = to_dense_adj(edge_index).zero_()[0]
        self._adjMLP_ = to_dense_adj(edge_index).zero_()[0]
        self._adjCount = to_dense_adj(edge_index)[0] - torch.diag(torch.ones_like(E))
        self._adjPx = to_dense_adj(edge_index).zero_()[0]
        self._adjPy = to_dense_adj(edge_index).zero_()[0]
        self._adjPz = to_dense_adj(edge_index).zero_()[0]
        self._adjE = to_dense_adj(edge_index).zero_()[0]
        self._RecursivePathMass(edge_index, Px, Py, Pz, E)
        
        
        


        exit()

    def message(self, edge_index, Px_i, Px_j, Py_i, Py_j, Pz_i, Pz_j):
        pass

    def aggregate(self, message):

        pass
