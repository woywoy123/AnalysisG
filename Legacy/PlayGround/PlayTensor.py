import torch 





y = [[1. for i in range(10)] for j in range(10)]
for i in range(10):
    y[i][i] = 0.
y = torch.tensor(y)

dyn = [[i == j for i in range(12)] for j in range(10)]
#for i in range(10):
#    dyn[i][i] = 1.
dyn = torch.tensor(dyn, dtype = torch.float)

print(dyn)
exit()

print(y.shape)


x = y[:, :].matmul(dyn)
for i in range(1, len(x)-1):
    print(x[i])
    
    if i == 3:
        break



exit()

t = [1, 0, 0, 0]
z = [0, 2, 0, 0]
u = [0, 0, 3, 1]
p = [0, 0, 0, 4]

c = torch.tensor([[1], [2], [3], [4], [1]])



x = torch.tensor([[t, z, u, p], [z, z, u, p], [t, z, u, p], [t, z, u, p], [t, z, u, p]])


#print(x * c[:,None])



print(torch.index_select(x, 2, torch.LongTensor([0, 2, 1, 3])))





#print(torch.sum(x, dim = 0))  
#print(torch.sum(x, dim = 1)) 
#print(torch.sum(x, dim = 2).sum(dim = 1))
