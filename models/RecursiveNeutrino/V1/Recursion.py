import torch
from torch_geometric.nn import MessagePassing
from torch.nn import Sequential as Seq, Linear, ReLU, Sigmoid, Tanh, Softmax 
import torch.nn.functional as F
import PyC.NuSol.CUDA as NuC
import PyC.Transform.CUDA as TC
import PyC.Physics.CUDA.Cartesian as PCC
from torch_geometric.utils import to_dense_adj, sort_edge_index, dense_to_sparse, softmax, scatter
from torch_geometric.nn.resolver import aggregation_resolver as aggr_resolver


torch.set_printoptions(4, profile = "full", linewidth = 100000)
torch.autograd.set_detect_anomaly(True)



class Recursion(MessagePassing):
    def __init__(self):
        super().__init__(aggr = None, flow = "source_to_target")
    
        self.O_edge = None 
        self.L_edge = "CEL"
        self.C_edge = True

        end = 32

        self._MassMLP = Seq(
                Linear(1, end),
                Linear(end, 2)
        )

        self.aggr_module_max = aggr_resolver("max")
        self.aggr_module_add = aggr_resolver("add")

    def message(self, edge_index, Pmc_i, Pmc_j):
        tmp = PCC.Mass(self._Pmc[edge_index[0]] + Pmc_i)
        #tmp = torch.cat([tmp, PCC.Mass(self._Pmc[edge_index[0]])], -1)
        mlp = self._MassMLP(tmp)
        softmax(mlp[:, 1], edge_index[0])
        return self._T[self._msk].view(-1), mlp, self._Pmc[edge_index[1]], edge_index[0]

    def aggregate(self, message, index, pmc):
        prob, mlp, pmc_, src = message
        
        prob_ = self.aggr_module_max(prob.view(-1, 1), src)
        edge_sel = (prob_.view(-1)[src] == prob).view(-1)
        src_, dst_, mlp_ = src[edge_sel], index[edge_sel], mlp[edge_sel]
        mlp_ = self.aggr_module_add(mlp_, src_)
        msk, pmc_ = self._msk.clone(), pmc_[edge_sel].clone()
        
        # Very weird bug. Inconsistent four vector...
        pmc = scatter(pmc_, src_, 0, dim_size = len(pmc)) + pmc.clone()

        print(pmc)
        self.O_edge[msk] = mlp
        self.O_edge[msk] += mlp_[src] + mlp_[index]

        edge_index_ = torch.cat([src_.view(1, -1), dst_.view(1, -1)], 0)
        self._Path = torch.cat([self._Path, edge_index_], -1)

        self._msk[msk] = self._msk[msk] * (edge_sel == False) * (mlp.max(-1)[1] == 1)
      
        return pmc
 
    def forward(self, i, num_nodes, batch, edge_index, N_pT, N_eta, N_phi, N_energy, G_met, G_met_phi, E_T_edge):
        pt, eta, phi, E = N_pT/1000, N_eta, N_phi, N_energy/1000
        Pmc = torch.cat([TC.PxPyPz(pt, eta, phi), E.view(-1, 1)], -1)
        src, dst = edge_index
        self._msk = src != dst
        
        self._Pmc = Pmc
        self._T = E_T_edge

        print(to_dense_adj(edge_index, edge_attr = E_T_edge.view(-1))[0])
        tmp = PCC.Mass(Pmc[src])
        #tmp = torch.cat([tmp, PCC.Mass(Pmc[src])], -1)
        tmp = self._MassMLP(tmp)
        
        self.O_edge = torch.zeros_like(tmp)
        self.O_edge[self._msk == False] = tmp[self._msk == False]
        self._Path = edge_index[:, self._msk == False]
        
        pmc_ = Pmc
        pmc_ = self.propagate(edge_index[:, self._msk], Pmc = pmc_.clone(), pmc = pmc_.clone())
        pmc_ = self.propagate(edge_index[:, self._msk], Pmc = pmc_.clone(), pmc = pmc_.clone())
        
        print(PCC.Mass(pmc_))




        print(to_dense_adj(edge_index, edge_attr = self.O_edge.max(-1)[1])[0])




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
