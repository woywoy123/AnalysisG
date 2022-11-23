import torch
from torch.distributions.categorical import Categorical
from torch.distributions.bernoulli import Bernoulli
from PathNetOptimizerCUDA import ToPxPyPzE, ToDeltaR, AggregateIncomingEdges

N = 50
edge_index = torch.tensor([[i for i in range(0, N) for j in range(0, N)], [j for i in range(0,N) for j in range(0, N)]], device = "cuda")
Pmu = torch.tensor([[i, i, i, i] for i in range(1, N+1)], device = "cuda")
Pmu1 = torch.tensor([[i+1, i+1, i+3, i+4] for i in range(1, N+1)], device = "cuda")


print(edge_index)

x = Bernoulli(torch.tensor([1.0], device = "cuda"))
for j in range(10): 
    t = torch.cat([x.sample() for i in range(len(edge_index[1]))], dim = 0).to(dtype = torch.int32).view(-1, 1)
    print(AggregateIncomingEdges(Pmu1[edge_index[1]], edge_index[0].view(-1, 1), t, False))

