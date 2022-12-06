import torch
from torch.distributions.categorical import Categorical
from torch.distributions.bernoulli import Bernoulli
from PathNetOptimizerCUDA import ToPxPyPzE, ToDeltaR, AggregateIncomingEdges

N = 50
edge_index = torch.tensor([[i for i in range(0, N) for j in range(0, N)], [j for i in range(0,N) for j in range(0, N)]], device = "cuda")
Pmu = torch.tensor([[i, i, i, i] for i in range(1, N+1)], device = "cuda", dtype = torch.float64)
Pmu1 = torch.tensor([[i+1, i+1, i+3, i+4] for i in range(1, N+1)], device = "cuda", dtype = torch.float64)


x = Bernoulli(torch.tensor([0.2], device = "cuda"))
for j in range(10): 
    t = x.sample(edge_index[1].shape).to(dtype = torch.int32).view(-1, 1)
    x = AggregateIncomingEdges(Pmu1[edge_index[1]], edge_index[0].view(-1, 1), t, False)
    print(x)
    
    print(t, edge_index[0].view(-1, 1), Pmu1[edge_index[1]])
    exit()
