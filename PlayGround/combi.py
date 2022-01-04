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
from numba import njit, prange, cuda, float32, int32
import numba
import numpy as np
import math
import torch_geometric
from torch_geometric.loader import DataLoader
from sklearn.cluster import AgglomerativeClustering

class PathNet(MessagePassing):
    def __init__(self, PCut = 0.5, complex = 64, path = 64, hidden = 20):
        super(PathNet, self).__init__("add")
        self.mlp_mass_complex = Seq(Linear(-1, hidden), Tanh(), Linear(hidden, 1)) # DO NOT CHANGE OUTPUT DIM 
        self.mlp_comp = Seq(Linear(-1, hidden), Tanh(), Linear(hidden, complex))
        self.mlp_path = Seq(Linear(-1, hidden), ReLU(), Linear(hidden, path))
        self.mlp = Seq(Linear(path + complex, path+complex), ReLU(), Linear(path+complex, hidden))
        self.PCut = PCut 
        self.N_Nodes = -1

    def forward(self, data):

        @njit(cache = True, parallel = True)
        def CreateLorentz(Lor_xyz, e, pt, eta, phi):
            
            for i in prange(Lor_xyz.shape[0]):
                Lor_xyz[i][0] = pt[i][0]*np.cos(phi[i][0])
                Lor_xyz[i][1] = pt[i][0]*np.sin(phi[i][0])
                Lor_xyz[i][2] = e[i][0]*np.tanh(eta[i][0])
                Lor_xyz[i][3] = e[i][0]
        
        @njit(cache = True, parallel = True)
        def CalcPathMass(Lor_xyz, comb, path_m, adj_p):
            n = path_m.shape[0] 
            b = 0 
            inc = 0             
            for i in prange(n):
                
                if b == len(comb):
                    b = 0
                    inc += comb.shape[1]-1

                tmp = comb[b] + inc
                l = []
                for k in np.unique(tmp):
                    if k != -1 + inc:
                       l.append(k) 
                p = np.array(l) 
                v = np.zeros(4, dtype = np.float64) 
                for j in Lor_xyz[p]:
                    v[0] = v[0] + j[0]
                    v[1] = v[1] + j[1]
                    v[2] = v[2] + j[2]
                    v[3] = v[3] + j[3]
                path_m[i] = np.exp(0.5*np.log(v[3]*v[3] - v[2]*v[2] - v[1]*v[1] - v[0]*v[0])) / 1000
        
                for j in prange(len(l)-1):
                    adj_p[i, p[j] - inc, p[j+1] - inc] = path_m[i]
                    adj_p[i, p[j+1] - inc, p[j] - inc] = path_m[i]
                b += 1



        def Performance(e, pt, eta, phi, event):
            if type(event) == torch_geometric.data.data.Data:
                unique = torch.unique(event.edge_index).tolist()
                n_event = 1
            else:
                event = event.to_data_list()
                unique = torch.unique(event[0].edge_index).tolist()
                n_event = len(event)
            
            if self.N_Nodes == -1:
                self.N_Nodes  = len(unique)

            e = np.array(e.tolist())
            pt = np.array(pt.tolist())
            eta = np.array(eta.tolist())
            phi = np.array(phi.tolist())
        
            Lor_xyz = np.zeros((len(e), 4))
            CreateLorentz(Lor_xyz, e, pt, eta, phi) 
            
            tmp = [] 
            l = len(unique)
            for i in range(1, l):
                p = list(combinations(unique, r = i+1))
                p = [list(k) for k in p]
                for k in p:
                    k += [k[0]]
                    k += [-1]*(self.N_Nodes - len(k)+1)
                tmp += p
            
            path = np.array(tmp)
            p_m = np.zeros(path.shape[0]*n_event, dtype = np.float32)
            adj_p = np.zeros((path.shape[0]*n_event, self.N_Nodes, self.N_Nodes), dtype = float)
            CalcPathMass(Lor_xyz, path, p_m, adj_p)
            return adj_p, p_m, n_event, path.shape[0]

        # 1. ==== Assign the path mass to the adjacency matrix adj_p 
        edge_index = data.edge_index
        adj_p, path_m, n_event, pth_l = Performance(data.e, data.pt, data.eta, data.phi, data)
        block = torch.tensor(adj_p, device = edge_index.device).float()
        
        e_jxe_i_tmp = []
        comp_ej_tmp = []
        for i in range(n_event):
            blk = block[i*pth_l:pth_l*(i+1)]

            # 2. ==== Project the sum of the topological mass states across complexity 
            e_jxe_i_tmp.append(torch.sum(blk, dim = 0))
       
            # 3. ==== Project along the complexity (the number of nodes included) and learn the complexity
            comp_ej_tmp.append(torch.sum(blk, dim = 1))
        
        # 2.1 ==== Learn the mass projection with dim (n*events * e_i) x e_j
        e_i = self.mlp_path(torch.cat(e_jxe_i_tmp, dim = 0))
       
        # 3.1 ==== Learn the different masses associated with path complexity (i.e. the number of nodes being included)
        comp_ej = self.mlp_mass_complex(torch.cat(comp_ej_tmp, dim = 0))
        
        # 3.2 ==== Set any connection with a mass to 1 and multiply (NOT DOT PRODUCT!!) with the learned mass value (-1 -> 1) to the matrix (n_events * e_i) x e_j
        adj_p[adj_p > 0] = 1 
        adj_p = torch.tensor(adj_p, device = edge_index.device).float()
        adj_p = adj_p * comp_ej[:, None]
        
        # 4. ==== Split into n_events and project the learned mass values along the complexity axis to get a matrix e_i x e_j and sum the mass projection (non learned) to the learned mass complexity
        adj_p_tmp = []
        for i in range(n_event):
            adj_p_tmp.append(adj_p[i*pth_l:pth_l*(i+1)].sum(dim = 0))
            adj_p_tmp[i].add(e_jxe_i_tmp[i])
       
        # 4.1 ==== Learn this sum after concatinating the list to a matrix (n_events * e_i) x e_j
        adj_p_ei = self.mlp_comp(torch.cat(adj_p_tmp, dim = 0))

        return self.propagate(edge_index = edge_index, x = torch.cat([data.e], dim = 1), edge_attr = torch.cat([e_i, adj_p_ei], dim = 1))

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
#torch.set_printoptions(profile = "full")


data = UnpickleObject("Nodes_12.pkl")
data2 = UnpickleObject("Nodes_10.pkl")
data = data.Data
data2 = data2.Data
#tmp = []
#for i in range(len(data)):
#    tmp.append(data[i].Data)
#    if i == 0:
#        break
#
#for data in DataLoader(tmp, batch_size = len(tmp)):
#    break



#data = data[0].Data
Model = PathNet()
Model.to(device = "cuda")
OP = torch.optim.Adam(Model.parameters(), lr = 1e-3, weight_decay= 1e-3)

Model.train()
for i in range(1000):
    inpt, trgt = data, data.y.t()[0]
    OP.zero_grad()
        
    t_s = time.time()
    pred = Model(data)
    _, x = pred.max(1)
    t_e = time.time()
    print(t_e - t_s)
    Loss = torch.nn.CrossEntropyLoss()
    L = Loss(pred, trgt)
   
    inpt, trgt = data2, data2.y.t()[0]
    OP.zero_grad()
        
    t_s = time.time()
    pred = Model(data2)
    _, x = pred.max(1)
    t_e = time.time()
    print(t_e - t_s)
    Loss = torch.nn.CrossEntropyLoss()
    L = Loss(pred, trgt)
   










    print(L)
    print(x, trgt)
    L.backward()
    OP.step()
    

import cmath
import torch_geometric
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj

@cuda.jit
def CalcPathMassCUDA(Lor_xyz, comb, path_m, v_size):
    x, y = cuda.threadIdx.x, cuda.threadIdx.y
    x_i, y_i = cuda.grid(2)
    
    v = cuda.shared.array(shape = (10, 4), dtype = float32)

    k = comb[x_i][y_i]
    if k != -1 and y_i > 0:
        v[0][0] += Lor_xyz[k][0]
        v[0][1] += Lor_xyz[k][1]
        v[0][2] += Lor_xyz[k][2]
        v[0][3] += Lor_xyz[k][3]
    
    cuda.syncthreads()
    path_m[x_i] = math.sqrt(abs(v[0][3]*v[0][3] - v[0][0]*v[0][0] - v[0][1]*v[0][1] - v[0][2]*v[0][2])) / 1000






@njit(cache = True, parallel = True)
def CreateLorentz(Lor_xyz, e, pt, eta, phi):
    
    for i in prange(Lor_xyz.shape[0]):
        Lor_xyz[i][0] = pt[i][0]*np.cos(phi[i][0])
        Lor_xyz[i][1] = pt[i][0]*np.sin(phi[i][0])
        Lor_xyz[i][2] = e[i][0]*np.tanh(eta[i][0])
        Lor_xyz[i][3] = e[i][0]

@njit(cache = True, parallel = True)
def CalcPathMass(Lor_xyz, comb, path_m, adj_p):
    n = comb.shape[0] 
    for i in prange(n):
        tmp = comb[i]
        l = []
        for k in np.unique(tmp):
            if k != -1:
               l.append(k) 
        p = np.array(l) 
        v = np.zeros(4, dtype = np.float64) 
        for j in Lor_xyz[p]:
            v[0] = v[0] + j[0]
            v[1] = v[1] + j[1]
            v[2] = v[2] + j[2]
            v[3] = v[3] + j[3]
        path_m[i] = np.exp(0.5*np.log(v[3]*v[3] - v[2]*v[2] - v[1]*v[1] - v[0]*v[0])) / 1000

        for j in prange(len(l)-1):
            adj_p[i, l[j], l[j+1]] = path_m[i]
            adj_p[i, l[j+1], l[j]] = path_m[i]

def Performance(e, pt, eta, phi, event):
    if type(event) == torch_geometric.data.data.Data:
        unique = torch.unique(event.edge_index).tolist()
    else:
        event = event.to_data_list()
        unique = torch.unique(event[0].edge_index).tolist()
    
    e = np.array(e.tolist())
    pt = np.array(pt.tolist())
    eta = np.array(eta.tolist())
    phi = np.array(phi.tolist())

    Lor_xyz = np.zeros((len(e), 4))
    CreateLorentz(Lor_xyz, e, pt, eta, phi) 
    
    tmp = [] 
    l = len(unique)
    for i in range(1, l):
        p = list(combinations(unique, r = i+1))
        p = [list(k) for k in p]
        for k in p:
            k += [k[0]]
            k += [-1]*(l - len(k)+1)
        tmp += p
    path = np.array(tmp)
    p_m = np.zeros(path.shape[0], dtype = np.float32)
    adj_p = np.zeros((path.shape[0], l, l), dtype = float)
    CalcPathMass(Lor_xyz, path, p_m, adj_p)
    return adj_p, p_m, path
    






event = UnpickleObject("Nodes_12.pkl")
#PickleObject(event[0].Data, "Nodes_12.pkl")
#event = UnpickleObject("Nodes_12.pkl")

tmp = []
for i in range(len(event)):
    tmp.append(event[i].Data)

for event in DataLoader(tmp, batch_size = len(tmp)):
    break
print(event)

t_s = time.time()
edge_index = event.edge_index
e, pt, eta, phi = event.e, event.pt, event.eta, event.phi

Performance(e, pt, eta, phi, event)

t_e = time.time()
print(t_e - t_s)


exit()


# Combinations 
t_s = time.time()
unique = torch.unique(edge_index)



exit()
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



