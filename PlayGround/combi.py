
string = "0123456789"
x = len(string)**2

for i in range(x):
    temp = i+1
    k = ""
    for j in range(len(string)):
        if temp & 1 == 1:
            k += string[j]
        temp = temp >> 1
    if (len(k) >= 2):
        #print(k)
        pass
    k =""

import torch 

v = torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
x = list(torch.combinations(v, r = 3))
#print(x)


v = torch.Tensor([[1, 2, 3], [1, 4, 5]])
print(v.t())
c = torch.matmul(v, v.t())
print(c)



