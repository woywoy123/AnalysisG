from itertools import combinations
import torch
import sys
sys.path.append("../")
from Functions.IO.IO import UnpickleObject, PickleObject
import time 
from skhep.math.vectors import LorentzVector
from torch.nn import Sequential as Seq, Linear, ReLU, Tanh
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from torch_scatter import scatter
from numba import njit, prange
import numba
import numpy as np

class PathNet(MessagePassing):
    def __init__(self):
        super(PathNet, self).__init__("add")
        self.mlp_path = Seq(Linear(12, 256), Tanh(), Linear(256, 12))
        #self.mlp_mass = Seq(Linear(1, 64), Tanh(), Linear(64, 12))
        self.mlp = Seq(Linear(12, 128), ReLU(), Linear(128, 4))
    
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
                for k in tmp:
                    if k != -1:
                        pth.append(k)
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
        
        t_s = time.time()
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
                k += [-1]*(l - len(k))
            path += p
        t_e = time.time()
        print(t_e - t_s)
        
        adj_p = np.zeros((len(path_m), l, l), dtype = float)
        path_m = np.array(path_m, dtype = float)
        path = np.array(path, dtype = int)
        MakeMatrix(adj_p, path_m, path)
        block = torch.tensor(adj_p, device = edge_index.device).float()
        path = torch.tensor(path, device = edge_index.device).float() 
       
        #complex = torch.sum(block, dim = 2).sum(dim = 1).reshape((-1, 1))
        #ed_p = torch.matmul(path.t(), self.mlp_mass(complex))
            
        #print(ed_p)
        e_jxe_i = torch.sum(block, dim = 0)#.flatten()[1:].view(l-1, l+1)[:, :-1].reshape(l, l-1)
        tmp_p = self.mlp_path(e_jxe_i) #.reshape((-1, 1)))
        print(tmp_p)
        
        e_ij_p = torch.zeros((edge_index.shape[1], tmp_p.shape[1]), device = edge_index.device)
        #e_ij_p = torch.zeros((edge_index.shape[1], tmp_p.shape[2] + ed_p.shape[1]), device = edge_index.device)
        for p in range(edge_index.shape[1]):
            e_i = edge_index[0][p]
            e_j = edge_index[1][p]
            
            e_ij_p[p]= tmp_p[e_i]#[e_j]#, ed_p[e_i]])
            #e_ij_p[p]= tmp_p[e_j]#[e_i]#, ed_p[e_j]])
        
        return self.propagate(edge_index = edge_index, x = torch.cat([e, pt, eta, phi], dim = 1), edge_attr = e_ij_p)

    def message(self, edge_index, x_i, x_j, edge_attr):
        return self.mlp(torch.cat([edge_attr], dim = 0))
    
    def update(self, aggr_out):
        return F.normalize(aggr_out)


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



