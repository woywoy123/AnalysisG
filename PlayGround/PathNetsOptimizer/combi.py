from itertools import combinations
import torch
import sys
sys.path.append("../../")
from Functions.IO.IO import UnpickleObject, PickleObject
import time 

import PathNetOptimizer_cpp 




t_s = time.time()
x = PathNetOptimizer_cpp.PathCombination(torch.tensor(12, device = "cuda"), 12)
t_e = time.time()
print(float(t_e - t_e))
#
#
#
#
#
#exit()
print(x)


event = UnpickleObject("Nodes_12.pkl")
event = event[0].Data

t_s = time.time()
edge_index = event.edge_index
e, pt, eta, phi = event.e, event.pt, event.eta, event.phi

e_i = edge_index[0]
e_j = edge_index[1]

unique = torch.unique(edge_index)





matrix = torch.zeros(unique.shape[0], unique.shape[0])
matrix[edge_index[0], edge_index[1]] = 1






print(matrix)

numbers = torch.tensor([[i+1] for i in range(len(unique))])





p = []
for i in range(1, 12):
    f = list(combinations(unique.tolist(), r = i+1))
    p += [list(i) for i in f]






def Combination(index, bits, number, l):
    if index == 0:
        if bits == 0:
            #out = [int(i) for i in str(bin(number))[2:]]
            #k = [0]*(l.Len - len(out))
            #k+= out
            l.com.append(number)
            return 
        return 
    if index -1 >= bits:
        Combination(index-1, bits, number, l)
    if bits > 0:
        Combination(index -1, bits -1, number | ( 1 << (index -1)), l)


class Get:
    def __init__(self):
        self.com = []
        self.Len = -1

g = Get()
g.Len = 12

t_s = time.time()
for i in range(1, 12):
    Combination(12, i+1, 0, g)
t_e = time.time()
print(t_e - t_s)





k = []
for i in g.com:
    mask = 2**torch.arange(12)
    print(mask)
    proj = torch.tensor(i)
    print(proj)
    proj = proj.unsqueeze(-1) 
    proj = proj.bitwise_and(mask) 
    print(proj)
    proj = proj.ne(0)
    print(proj)
    proj = proj.byte()#.reshape(12, 1).to(dtype = torch.float)
    print(proj)

    exit()



    comb = matrix.matmul(proj)
    
    comb[comb == proj.sum(dim =0)] = 0 
    comb[comb > 0] = 1
    
    comb = comb*numbers

    x = [int(k[0]-1) for k in comb.tolist()]
    x = list(filter(lambda a: a != -1, x)) 
    k.append(x)

t_e = time.time()
print(t_e - t_s)



for i in p:
        
    Found = False
    for j in k:
        
        if j == i:
            Found = True 
            break

    if Found == False:
        print(i)
    else:
        print("->", i, j)
    




#for i, j in zip(k, p):
#
#    print(i, j)
#













exit()






project = torch.ones(unique.shape[0], 1)
project = matrix.matmul(project)
project[project > 0] = 1
out = project*numbers
print([int(i[0]) for i in out.tolist()])





# Combinations 
t_s = time.time()
unique = torch.unique(edge_index)
tmp = []
for i in range(1,len(unique)):
    p = torch.tensor(list(combinations(unique, r = i+1)), dtype = torch.int, device = unique.device)
    tmp += p
t_e = time.time()
print(t_e - t_s)



exit()


