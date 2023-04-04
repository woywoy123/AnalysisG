import torch
from torch_geometric.nn import MessagePassing
from torch.nn import Sequential as Seq, Linear, ReLU, Sigmoid, Tanh, Softmax 
import torch.nn.functional as F
import PyC.NuSol.CUDA as NuC
import PyC.Transform.CUDA as TC
import PyC.Physics.CUDA.Cartesian as PCC
from torch_geometric.utils import to_dense_adj, sort_edge_index, dense_to_sparse, softmax, scatter, degree
from time import sleep
from torch_geometric.nn.resolver import aggregation_resolver as aggr_resolver


torch.set_printoptions(4, profile = "full", linewidth = 100000)
torch.autograd.set_detect_anomaly(True)



class Recursion(MessagePassing):
    def __init__(self):
        super().__init__(aggr = None, flow = "source_to_target")
    
        self.O_edge = None 
        self.L_edge = "CEL"
        self.C_edge = True

        end = 64

        self._MassMLP = Seq(
                Linear(1, end),
                Tanh(),
                Linear(end, end), 
        )

        self._RecurMLP = Seq(
                Linear(end*2 + 2, end), 
                Tanh(), ReLU(), Tanh(), 
                Linear(end, end), 
                Tanh(), 
                Linear(end, 2)
        )

        self.aggr_module_add = aggr_resolver("add")

    def LinkPrediction(self, pred):
        x0 = dense_to_sparse(pred[0].matmul(pred[0].t()))[1].view(-1, 1)
        x1 = dense_to_sparse(pred[1].matmul(pred[1].t()))[1].view(-1, 1)
        return torch.cat([x0, x1], -1)

    def message(self, edge_index, _pmc_i, _pmc_j, Pmc_i, Pmc_j):
        src, dst = edge_index
        nodes = torch.cat([src, dst], -1).max()+1
        
        _msk = self._msk.clone()
        pmc_i = self._MassMLP(PCC.Mass(_pmc_i))
        pmc_i_ = self._MassMLP(PCC.Mass(_pmc_i + Pmc_j))
        mlp_i = self._RecurMLP(torch.cat([pmc_i, pmc_i_, self.O_edge[_msk]], -1))
        
        pmc_j = self._MassMLP(PCC.Mass(_pmc_j))
        pmc_j_ = self._MassMLP(PCC.Mass(_pmc_j + Pmc_i))
        mlp_j = self._RecurMLP(torch.cat([pmc_j, pmc_j_, self.O_edge[_msk]], -1))

        mlp = self._RecurMLP(torch.cat([pmc_i_, pmc_j_, self.O_edge[_msk] + mlp_j + mlp_i], -1))

        msk = mlp.max(-1)[1]
        
        msk_ = to_dense_adj(edge_index)
        msk_ += to_dense_adj(edge_index, edge_attr = msk)[0]
        msk_ += to_dense_adj(edge_index, edge_attr = msk)[0].t()
        msk = (dense_to_sparse(msk_)[1]  > 1)

        #sel = F.softmax(mlp, -1)
        #num = sel.max(-1)[1].sum(-1)
        #sel = sel[:, 1]
        #
        #msk = self._T[self._msk].view(-1)
        #if sel.sum(-1) > 0:
        #    msk = torch.multinomial(sel, num_samples = num if num != 0 else 1)
        #    idx = edge_index[:, msk]
        #    msk = to_dense_adj(edge_index)[0]
        #    msk_ = to_dense_adj(idx, max_num_nodes = nodes)[0]
        #    msk += msk_ + msk_.t()
        #    msk = (dense_to_sparse(msk)[1]  != 1)
        #else:
        #    msk = (sel > 0)
        
        msk = msk.view(-1, 1)
        return mlp, (Pmc_j)*msk, msk.view(-1) == 1, src

    def aggregate(self, message, index, Pmc, _pmc, mlp__):
        (mlp, pmc_j, msk, src), dst = message, index
        
        eij_ = torch.cat([src.view(1, -1), dst.view(1, -1)], 0)[:, msk]
        self._Path = eij_ if self._Path == None else torch.cat([self._Path, eij_], -1) 

        pmc_ = self._MassMLP(PCC.Mass(_pmc[src]))
        _msk = self._msk.clone() 
        _indx = index.unique()
        _pmc[_indx] += self.aggr_module_add(pmc_j, dst)[_indx]

        m_ij = self._MassMLP(PCC.Mass(_pmc))
        
        #self._MassMatrix.append(to_dense_adj(self._Path, max_num_nodes = len(Pmc))*m_ij)
        mlp = self.aggr_module_add(mlp, index)#/degree(index)[_indx].view(-1, 1)
        mlp_ = self._RecurMLP(torch.cat([pmc_, m_ij[dst], self.O_edge[_msk] + mlp[src]], -1))
        mlp_ = self._RecurMLP(torch.cat([pmc_, m_ij[dst], mlp_], -1))
        mlp__[_msk] = mlp_
        return _pmc, msk, mlp__


    def forward(self, i, num_nodes, batch, edge_index, N_pT, N_eta, N_phi, N_energy, G_met, G_met_phi, E_T_edge):
        pt, eta, phi, E = N_pT/1000, N_eta, N_phi, N_energy/1000
        Pmc = torch.cat([TC.PxPyPz(pt, eta, phi), E.view(-1, 1)], -1)
        src, dst = edge_index
        
        self._T = E_T_edge.clone()
        self._Path = None
        
        pmc_i = self._MassMLP(PCC.Mass(Pmc[src]))
        pmc_j = self._MassMLP(PCC.Mass(Pmc[dst] + Pmc[src]))
        self.O_edge = self._RecurMLP(torch.cat([pmc_i, pmc_j, torch.zeros_like(edge_index).t()], -1))
        
        self._msk = src != dst
        self._MassMatrix = []
    
        pmc_ = Pmc.clone()
        while True:
            pmc_i = self._MassMLP(PCC.Mass(pmc_[src]))
            pmc_, msk, mlp_ = self.propagate(edge_index[:, self._msk], _pmc = pmc_,  Pmc = Pmc, mlp__ = torch.zeros_like(self.O_edge))
            msk_ = self._msk.clone()
            self._msk[msk_] *= msk == False
            pmc_j = self._MassMLP(PCC.Mass(pmc_[src]))
            self.O_edge = self._RecurMLP(torch.cat([pmc_i, pmc_j, mlp_ + self.O_edge], -1))

            if (msk_ != self._msk).sum(-1) == 0 or self._msk.sum(-1) == 0:
                break
        
        #x0 = to_dense_adj(edge_index, edge_attr = self.O_edge[:, 0][src])
        #x1 = to_dense_adj(edge_index, edge_attr = self.O_edge[:, 1][src])
        #self.O_edge = torch.cat([x0, x1], 0)
        #self.O_edge = self.LinkPrediction(self.O_edge)

        print(to_dense_adj(edge_index, edge_attr = self.O_edge.max(-1)[1])[0]) #* self.O_edge.max(-1)[0])[0])




    #def _RecursivePath(self, edge_index, Pmc, Pmc_, msk_):
    #    src, dst = edge_index[:, msk_]
    #    _mlp = self._EdgeAggregation(Pmc[src], Pmc_[dst])
    #    _mlp = self._Reduce(torch.cat([Pmc_[src], _mlp, Pmc[dst]], -1))
    #    sel = _mlp.max(-1)[1]

    #    _prob = to_dense_adj(edge_index[:, msk_], None, softmax(_mlp[:, 1], src)*sel, len(Pmc_))[0]
    #    msk = _prob.sum(-1) > 0
    #    if msk.sum(-1) == 0:
    #        return 

    #    aggr_node = _prob[msk].max(-1)[1]
    #    #aggr_node = torch.multinomial(_prob[msk], num_samples = 1).view(-1)
    #    
    #    Pmc_[msk] += Pmc[aggr_node]
    #    self._Path = torch.cat([ self._Path, torch.cat([ self._Nodes[msk].view(1, -1), aggr_node.view(1, -1) ], 0) ], -1)
    #    self._PathMass = torch.cat([ self._PathMass, PCC.Mass(Pmc_[msk]).view(-1) ], -1)
    #    self._K[msk_] += _mlp*(1/ (self._iter+1))

    #    _ss = to_dense_adj(edge_index, edge_attr = msk_)[0].to(dtype = aggr_node.dtype)
    #    _s = torch.zeros_like(_ss)
    #    _ss[msk] -= _s[msk].scatter_(1, aggr_node.view(-1, 1), torch.ones_like(aggr_node).view(-1, 1))

    #    self._iter += 1 
    #    msk_ = (_ss[edge_index[0], edge_index[1]] == 1)
    #    if msk_.sum(-1) == 0:
    #        return 
    #    return self._RecursivePath(edge_index, Pmc, Pmc_, msk_)

    #def forward(self, i, num_nodes, batch, edge_index, N_pT, N_eta, N_phi, N_energy, G_met, G_met_phi, E_T_edge):
    #    pt, eta, phi, E = N_pT/1000, N_eta, N_phi, N_energy/1000
    #    Pmc = torch.cat([TC.PxPyPz(pt, eta, phi), E.view(-1, 1)], -1)
    #    
    #    src, dst = edge_index
    #    msk, self._iter = src == dst, 0

    #    self._K = 

    #    self._Path = edge_index[:, msk]
    #    self._Nodes = src[msk].view(-1, 1)
    #    self._PathMass = PCC.Mass(Pmc).view(-1)

    #    self._RecursivePath(edge_index, Pmc, Pmc.clone(), src != dst)
    #    
    #    edge_, mass = sort_edge_index(self._Path, edge_attr = self._PathMass)
    #    
    #    mpl = self._Mass(mass.view(-1, 1))
    #    node_mass = to_dense_adj(edge_, None, mpl * mpl.max(-1)[1].view(-1, 1), len(Pmc))[0].view(-1, 2)

    #    self.O_edge = self._K #+ node_mass #+ node_mass[src] * node_mass[dst]
    #    print(to_dense_adj(edge_index, edge_attr = self.O_edge.max(-1)[1])[0])















class OldRecursion(MessagePassing):
    def __init__(self):
        super().__init__()
    
        self.O_edge = None 
        self.L_edge = "CEL"
        self.C_edge = True

        end = 32
        self._Mass = Seq(
                Linear(1, end, bias = False),
                Tanh(), ReLU(), 
                Linear(end, 2)
        )
        self._Reduce = Seq(
                Linear(4 + 4 + 4, end, bias = False),
                Tanh(), ReLU(), 
                Linear(end, 2), 
        )

    def _EdgeAggregation(self, Pmc, Pmc_):
        M_i, M_j = PCC.Mass(Pmc), PCC.Mass(Pmc_)
        Pmc_ = Pmc_ + Pmc_
        M_ij = PCC.Mass(Pmc_)
        _mpl = self._Mass(M_ij)
        return _mpl

    def _RecursivePath(self, edge_index, Pmc, Pmc_):
        src, dst = edge_index
        Node = Pmc_[dst]*(src != dst) if self._it != 0 else Pmc_*0
        mass_mlp = self._EdgeAggregation(Pmc[src], Node.view(-1, 1))
        mmlp0, mmlp1, sel = mass_mlp[:, 0], mass_mlp[:, 1], mass_mlp.max(-1)[1].view(-1, 1)
        
        _edges = self._remaining_edges == 1
        self._G[_edges] += self._Reduce(torch.cat([self._G[_edges], Pmc_[src], Pmc[dst], mass_mlp], -1)) + mass_mlp
        if len(self._G[:, 1][_edges]) == 0:
            return 
        
        _prob = to_dense_adj(edge_index, edge_attr = softmax(self._G[:, 1][_edges], src)*sel.view(-1), max_num_nodes = len(Pmc))[0]
        msk = (_prob.sum(-1) > 0).view(-1)
        if msk.sum(-1) == 0:
            return 
    
        aggr_node = torch.multinomial(_prob[msk], num_samples = 1)
        self._Path[msk, self._it] = aggr_node.view(-1)
        self._PathMass[msk, self._it] = PCC.Mass(Pmc_)
        self._it += 1

        # Update adjacency matrix
        _s = torch.zeros_like(_prob).to(dtype = aggr_node.dtype)
        _s[msk] = _s[msk].scatter_(1, aggr_node, torch.ones_like(aggr_node))
        _s = _s + to_dense_adj(self._edge_index, edge_attr = self._remaining_edges)[0]
        self._remaining_edges = dense_to_sparse(_s)[1]
        edge_index = self._edge_index[:, self._remaining_edges == 1]
        
        Pmc_[msk] += Pmc[aggr_node.view(-1)]
        return self._RecursivePath(edge_index, Pmc, Pmc_)
    
    def forward(self, i, num_nodes, batch, edge_index, N_pT, N_eta, N_phi, N_energy, G_met, G_met_phi, E_T_edge):
        pt, eta, phi, E = N_pT/1000, N_eta, N_phi, N_energy/1000
        Pmc = torch.cat([TC.PxPyPz(pt, eta, phi), E.view(-1, 1)], -1)
        
        self._it = 0
        self._edge_index = edge_index
        self._remaining_edges = torch.ones_like(edge_index[0]) 
        self.E_T_edge = E_T_edge.view(-1, 1)
            
        src, dst = edge_index
        self._G = self._EdgeAggregation(Pmc[src], Pmc[dst]*(src != dst).view(-1, 1))
        self._Path = to_dense_adj(edge_index)[0].fill_(-1).to(dtype = edge_index[0].dtype)
        self._PathMass = to_dense_adj(edge_index)[0].fill_(-1).to(dtype = edge_index[0].dtype)

        self._RecursivePath(edge_index, Pmc, Pmc.clone())
        
        edge_, attr = dense_to_sparse(self._Path * to_dense_adj(edge_index)[0])
        attr = attr.to(dtype = edge_index[0].dtype)
        msk = attr > 0
        mass = self._Mass(PCC.Mass(Pmc[edge_[0]][msk] + Pmc[attr[msk]]))
        _edge, mlp = sort_edge_index(torch.cat([edge_[0][msk].view(1, -1), attr[msk].view(1, -1)], 0), edge_attr = mass)
         
        self.O_edge = F.softmax(self._G, -1)*self._G
        print(to_dense_adj(edge_index, edge_attr = self.O_edge.max(-1)[1])[0])
        #print(to_dense_adj(edge_index, edge_attr = E_T_edge.view(-1))[0])
        #print("---")
        # Add path to sort.
