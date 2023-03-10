import torch
from torch_geometric.nn import MessagePassing
from torch.nn import Sequential as Seq, Linear, ReLU, Sigmoid, Tanh, Softmax
import torch.nn.functional as F
import PyC.NuSol.CUDA as NuC
import PyC.Transform.CUDA as TC
import PyC.Physics.CUDA.Cartesian as PCC
from torch_geometric.utils import to_dense_adj, dense_to_sparse, softmax, sort_edge_index

torch.set_printoptions(4, profile = "full", linewidth = 100000)
torch.autograd.set_detect_anomaly(True)


class Recursion(MessagePassing):
    def __init__(self):
        super().__init__()
    
        self.O_edge = None 
        self.L_edge = "CEL"
        self.C_edge = True

        self.O_edge_res = None 
        self.L_edge_res = "CEL"

        self.O_signal = None 
        self.L_signal = "CEL"
        
        end = 256
        self._Mass = Seq(
                        Linear(1, end),
                        Tanh(),
                        Linear(end, 2), 
                        ReLU(), 
                        Linear(2, end),
                        Tanh(), 
                        Linear(end, 2), 
                        Sigmoid()
        )

        self._TopoSig = Seq(
                            Linear(2, end), 
                            Tanh(), 
                            Linear(end, 2), 
                            ReLU(), 
                            Linear(2, end), 
                            Tanh(), 
                            Linear(end, 2)
        )

        self._Sig = Seq(
                        Linear(4, end), 
                        Tanh(), 
                        Linear(end, 2), 
                        ReLU(), 
                        Linear(2, end), 
                        Tanh(), 
                        Linear(end, 2)
        )
    
    def _RecursivePath(self, edge_index, Px, Py, Pz, E, Px_, Py_, Pz_, E_):
        src, dst = edge_index
        Px_ij, Py_ij, Pz_ij, E_ij = Px_[src] + Px[dst], Py_[src] + Py[dst], Pz_[src] + Pz[dst], E_[src] + E[dst]
        M_i, M_j, M_ij = PCC.M(Px_[src], Py_[src], Pz_[src], E_[src])/1000, PCC.M(Px[dst], Py[dst], Pz[dst], E[dst])/1000, PCC.M(Px_ij, Py_ij, Pz_ij, E_ij)/1000
        gamma = self._Mass(M_ij) #torch.cat([M_i, M_ij, M_i - M_j, M_i + M_j], -1))       
        p0 = to_dense_adj(edge_index, edge_attr = gamma[:, 0])[0]*self._SelMap
        p1 = to_dense_adj(edge_index, edge_attr = gamma[:, 1])[0]*self._SelMap
        
        self._w0 += gamma[:, 0]
        self._w1 += gamma[:, 1]

        msk = (self._SelMap.sum(-1, keepdim = True) > 0).view(-1)
        if msk.sum(-1) == 0:
            return 
        
        next_node = torch.unique(edge_index[0]).view(-1, 1).to(dtype = edge_index[0].dtype)
        if self._iter != 0:
            next_node = next_node.fill_(-1)
            next_node[msk] = torch.multinomial(p1[msk], num_samples = 1)

        Px_[msk] += torch.gather(Px, 0, next_node[msk].view(-1))
        Py_[msk] += torch.gather(Py, 0, next_node[msk].view(-1))
        Pz_[msk] += torch.gather(Pz, 0, next_node[msk].view(-1))
        E_[msk] += torch.gather(E, 0, next_node[msk].view(-1))
        
        # Record the running Kinematics
        self._PmC[0, msk, self._iter] = Px_[msk]
        self._PmC[1, msk, self._iter] = Py_[msk]
        self._PmC[2, msk, self._iter] = Pz_[msk]
        self._PmC[3, msk, self._iter] = E_[msk]

        # Update the selection map and path
        _SelMap = self._SelMap.clone()
        _SelMap[msk] *= self._SelMap[msk].scatter_(1, next_node[msk], torch.zeros_like(next_node[msk]))
        self._SelMap = self._SelMap*_SelMap
        self._PMap[:, self._iter] = next_node.view(-1)
        
        # Record the weights of this path
        self._w0Path[msk, self._iter] += torch.gather(p0[msk], 1, next_node[msk]).view(-1)
        self._w1Path[msk, self._iter] += torch.gather(p1[msk], 1, next_node[msk]).view(-1)

        self._iter += 1
        return self._RecursivePath(edge_index, Px, Py, Pz, E, Px_, Py_, Pz_, E_)

    def forward(self, i, num_nodes, batch, edge_index, N_pT, N_eta, N_phi, N_energy, G_met, G_met_phi, E_T_edge):
        pt, eta, phi, E = N_pT, N_eta, N_phi, N_energy
        PxPyPz = TC.PxPyPz(pt, eta, phi)
        Px, Py, Pz, E = PxPyPz[:, 0], PxPyPz[:, 1], PxPyPz[:, 2], E.view(-1)

        self._device = N_pT.device
        self._iter = 0
        
        # Define the maximum length tensor
        _x = torch.zeros((num_nodes, batch.unique(return_counts = True)[1].max()), device = self._device)

        # Track the overall paths traversed
        self._PMap = _x.clone()
        self._PMap[:, self._iter] = torch.unique(edge_index[0]) 

        # Tracks which nodes have not been selected yet
        self._SelMap = to_dense_adj(edge_index)[0].to(dtype = edge_index[0].dtype)

        # Store MLP prediction weights of path selected 
        self._w0Path = _x.clone() 
        self._w1Path = _x.clone() 
        self._w0 = edge_index[0].clone().zero_().to(dtype = _x.dtype)
        self._w1 = edge_index[0].clone().zero_().to(dtype = _x.dtype)

        # Store the Kinematics after each node selection 
        self._PmC = _x.clone().view(1, _x.size()[0], _x.size()[1])
        self._PmC = torch.cat([self._PmC, self._PmC, self._PmC, self._PmC], 0)
        self._PmC[0, :, self._iter] = Px
        self._PmC[1, :, self._iter] = Py
        self._PmC[2, :, self._iter] = Pz
        self._PmC[3, :, self._iter] = E
        
        self._TRU = E_T_edge

        # Start the Recursion
        self._RecursivePath(edge_index, Px, Py, Py, E, Px.clone(), Py.clone(), Pz.clone(), E.clone())

        # Construct the Edge MLP 
        msk = self._PMap > -1
        edge_ = torch.cat([edge_index[0].view(1, -1), self._PMap[msk].view(1, -1)], 0)
        _, self._w0Path = sort_edge_index(edge_, edge_attr = self._w0Path[msk])
        _, self._w1Path = sort_edge_index(edge_, edge_attr = self._w1Path[msk])
            
        # Sum the MLP layers
        self._w0Path, self._w1Path = self._w0 + self._w0Path, self._w1 + self._w1Path
        self.O_edge = torch.cat([self._w0Path.view(-1, 1), self._w1Path.view(-1, 1)], -1)
        
        self.O_edge_res = self._TopoSig(self.O_edge)
        
        _x = torch.zeros((len(i), 4), device = self._device)
        _x[batch[edge_index[0]]] += torch.cat([self.O_edge, self.O_edge_res], -1)
        self.O_signal = self._Sig(_x)

