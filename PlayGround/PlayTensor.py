import torch 



t = [1, 0, 0, 0]
z = [0, 2, 0, 0]
u = [0, 0, 3, 1]
p = [0, 0, 0, 4]

x = torch.tensor([[t, z, u, p], [t, z, u, p], [t, z, u, p], [t, z, u, p], [t, z, u, p]])

print(torch.sum(x, dim = 0))  
print(torch.sum(x, dim = 1)) 
print(torch.sum(x, dim = 2).sum(dim = 1))
