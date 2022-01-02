from itertools import combinations
import torch
import sys
sys.path.append("../")
from Functions.IO.IO import UnpickleObject, PickleObject
import time 
from skhep.math.vectors import LorentzVector
from torch.nn import Sequential as Seq, ReLU, Tanh
from torch_geometric.nn import MessagePassing, Linear
import torch.nn.functional as F
from torch_scatter import scatter
from numba import njit, prange
import numba
import numpy as np
from sklearn.cluster import AgglomerativeClustering

class PathNet(MessagePassing):
    def __init__(self, PCut = 0.5, complex = 64, path = 64, hidden = 64):
        super(PathNet, self).__init__("add")
        self.mlp_mass_complex = Seq(Linear(-1, hidden), Tanh(), Linear(hidden, 1)) # DO NOT CHANGE OUTPUT DIM 
        self.mlp_comp = Seq(Linear(-1, hidden), Tanh(), Linear(hidden, complex))
        self.mlp_path = Seq(Linear(-1, hidden), ReLU(), Linear(hidden, path))
        self.mlp = Seq(Linear(path + complex, path+complex), ReLU(), Linear(path+complex, 20))
        self.PCut = PCut 

    def forward(self, data):
        
        def PathMass(Lor, com):
            outp = []
            for i in com:
                v = LorentzVector()
                for k in i:
                    v += Lor[k]
                outp.append(v.mass/1000)
            return outp

        @njit(cache = True, parallel = True)
        def MakeMatrix(out, path_m, path):
            n = path.shape[0]
            for x in prange(n):
                m = path_m[x]
                tmp = path[x]
                pth = []
                for k in range(len(tmp)):
                    if tmp[k] != -1:
                        pth.append(tmp[k])

                m = m/len(pth) 
                for j in range(len(pth)-1):
                    out[x, pth[j], pth[j+1]] = m
                    out[x, pth[j+1], pth[j]] = m
               
                e_s = pth[0]
                e_e = pth[len(pth)-1]
                if out[x, e_s, e_e] == 0:
                    out[x, e_s, e_e] = m
                    out[x, e_e, e_s] = m

        edge_index = data.edge_index
        e, pt, eta, phi = data.e, data.pt, data.eta, data.phi
        unique = np.unique(edge_index.tolist())
        l = len(unique)
        
        Lor = []
        for i in unique:
            v = LorentzVector()
            v.setptetaphie(pt[i], eta[i], phi[i], e[i])
            Lor.append(v)
  
        path = []
        path_m = []
        for i in range(1,l):
            p = list(combinations(unique, r = i+1))
            path_m += PathMass(Lor, p)
            p = [list(k) for k in p]
            for k in p:
                k += [k[0]]
                k += [-1]*(l - len(k)+1)
            path += p

        # 1. ==== Assign the path mass to the adjacency matrix adj_p 
        adj_p = np.zeros((len(path_m), l, l), dtype = float)
        path_m = np.array(path_m, dtype = float)
        path = np.array(path, dtype = int)
        MakeMatrix(adj_p, path_m, path)
        block = torch.tensor(adj_p, device = edge_index.device).float()
        
        # 2. ==== Project the sum of the topological mass states across complexity 
        e_jxe_i = torch.sum(block, dim = 0)
        e_i = self.mlp_path(e_jxe_i)

        # 3. ==== Project along the complexity and learn the complexity
        comp_ej = torch.sum(block, dim = 1)
        comp_ej = self.mlp_mass_complex(comp_ej)
        adj_p[adj_p > 0] = 1     
        adj_p = torch.tensor(adj_p, device = edge_index.device).float()
        adj_p = adj_p * comp_ej[:, None]
        adj_p = adj_p.sum(dim = 0)
        adj_p = adj_p.add(e_jxe_i)
        adj_p_ei = self.mlp_comp(adj_p)

        return self.propagate(edge_index = edge_index, x = torch.cat([e], dim = 1), edge_attr = torch.cat([e_i, adj_p_ei], dim = 1))

    def message(self, edge_index, x_i, x_j, edge_attr):
        return self.mlp(torch.cat([edge_attr[edge_index[1].t()]], dim = 1))
    
    def update(self, aggr_out): 
        aggr_out = F.normalize(aggr_out)
        adj = torch.sigmoid(aggr_out.matmul(aggr_out.t()))
        l = adj.shape[0]

        ones = torch.zeros(adj.shape, device = adj.device)
        for i in range(l):
            ones[l - i-1] = torch.tensor([[1]*(l-i) + [0]*i])

        adj[adj <= self.PCut] = 0
        adj_ = adj.matmul(ones)
        adj_i = adj_.sum(dim = 0)
        adj_j = adj_.sum(dim = 1)
        
        adj_sum = adj_i + adj_j.flip(dims = [0])
        step = (adj_sum.max() - adj_sum.min())/len(adj_sum)


        c_ = np.array(torch.round(adj_sum).tolist()).reshape(-1, 1)
        clu = list(AgglomerativeClustering(n_clusters = None, distance_threshold = float(step)).fit(c_).labels_)

        self.n_cluster = len(list(set(clu)))
        self.Adj_M = adj
        self.cluster = clu
        return aggr_out


torch.set_printoptions(edgeitems = 20)
torch.set_printoptions(profile = "full")


data = UnpickleObject("Nodes_12.pkl")
data = data[0].Data
Model = PathNet()
Model.to(device = "cuda")
OP = torch.optim.Adam(Model.parameters(), lr = 1e-3, weight_decay= 1e-3)

Model.train()
for i in range(1000):
    inpt, trgt = data, data.y.t()[0]
    OP.zero_grad()

    pred = Model(data)
    _, x = pred.max(1)
    Loss = torch.nn.CrossEntropyLoss()
    L = Loss(pred, trgt)
    
    print(L)
    print(x, trgt)
    print(Model.n_cluster, Model.cluster)
    L.backward()
    OP.step()























# Reading 
t_s = time.time()
event = UnpickleObject("Nodes_12.pkl")
event = event[0].Data
edge_index = event.edge_index
e, pt, eta, phi = event.e, event.pt, event.eta, event.phi
t_e = time.time()
print(t_e - t_s)

# Combinations 
t_s = time.time()
unique = torch.unique(edge_index)
tmp = []
for i in range(1,len(unique)):
    p = torch.tensor(list(combinations(unique, r = i+1)), dtype = torch.int, device = unique.device)
    tmp += p
t_e = time.time()
print(t_e - t_s)

# Make Lorentz vector 
t_s = time.time()

Lor = []
for i in unique:
    v = LorentzVector()
    v.setptetaphie(pt[i], eta[i], phi[i], e[i])
    Lor.append(v)

path_m = []
for i in tmp:
    v = LorentzVector()
    for k in i:
        v+=Lor[k]
    path_m.append(v.mass/1000)

t_e = time.time()
print(t_e - t_s)



























#edge_index_T = edge_index.t()
#print(edge_index_T)
#print("")
#for sg, p in zip(path, score):
#    for k in sg.unfold(0, 2, 1):
#        sc[torch.where((k == edge_index_T).all(dim = 1))[0]] += p
#        print("---> ", torch.nonzero((k == edge_index_T).sum(dim = 1) == k.size(0)), k)
#
#        print(sc)
#        
#
#
#
#
#    #print(sg.unfold(0, 2, 1)) #, sc.t(), edge_index.t())
#    
#    print("++++")




#string = "0123456789"
#x = len(string)**2
#print(x)
#
#comb =  []
#for i in range(x):
#    temp = i+1
#    k = ""
#    for j in range(len(string)):
#        if temp & 1 == 1:
#            k += string[j]
#        temp = temp >> 1
#    comb.append(k)
#    k =""
#
#
#print(comb)





#import torch 
#
#v = torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
#x = list(torch.combinations(v, r = 4))
#
#print(len(x))
#
#
#
#for i in x:
#    print(i)


#v = torch.Tensor([[1, 2, 3], [1, 4, 5]])
#print(v.t())
#c = torch.matmul(v, v.t())
#print(c)



