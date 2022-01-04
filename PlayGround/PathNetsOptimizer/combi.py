from itertools import combinations
import torch
import sys
sys.path.append("../../")
from Functions.IO.IO import UnpickleObject, PickleObject
import time 

event = UnpickleObject("Nodes_12.pkl")
event = event[0].Data

t_s = time.time()
edge_index = event.edge_index
e, pt, eta, phi = event.e, event.pt, event.eta, event.phi






















t_e = time.time()
print(t_e - t_s)
exit()


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


