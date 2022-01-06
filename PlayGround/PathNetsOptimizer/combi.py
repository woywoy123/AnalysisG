from itertools import combinations
import torch
import sys
sys.path.append("../../")
from Functions.IO.IO import UnpickleObject, PickleObject
import time 

import PathNetOptimizer_cpp 



event = UnpickleObject("Nodes_12.pkl")
event = event[0].Data

t_s = time.time()
edge_index = event.edge_index
e, pt, eta, phi = event.e, event.pt, event.eta, event.phi

e_i = edge_index[0]
e_j = edge_index[1]

unique = torch.unique(edge_index)
matrix = torch.zeros(unique.shape[0], unique.shape[0], device = "cuda")
matrix[edge_index[0], edge_index[1]] = 1
print(matrix)

t_s = time.time()
x = PathNetOptimizer_cpp.PathCombination(matrix, 12)
t_e = time.time()
print(round(float(t_e - t_e), 10))

for i in x:
    print(i)
    print(e)
    print(e[i.to(torch.bool)].sum(0))
    exit()
numbers = torch.tensor([[i+1] for i in range(len(unique))], device = "cuda" )


p = []
t_s = time.time()
for i in range(1, 12):
    f = list(combinations(unique.tolist(), r = i+1))
    p += [list(i) for i in f]

k = []
for i in x:
    i = i*numbers

    t = [int(k[0]-1) for k in i.tolist()]
    t = list(filter(lambda a: a != -1, t)) 
    k.append(t)
        

t_e = time.time()
print(t_e - t_s)



for i in p:
    Found = False
    for j in k:
        
        if j == i:
            Found = True 
            break

    if Found == False:
        print("->", i)
    




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


