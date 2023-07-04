import torch
import random
from time import time
torch.set_printoptions(4, profile="full", linewidth=100000)
torch.ops.load_library("../build/operators/libop_cuda.so")

tmp = 3
x = torch.tensor(
        [[[k for i in range(tmp)] for k in range(8)] for t in range(1)], 
        device = "cuda", dtype = torch.float64)

y = torch.tensor(
        [[[i for i in range(5)] for k in range(tmp)] for t in range(1)], 
        device = "cuda", dtype = torch.float64)


print("mul")
t1 = time()
x_ = x.matmul(y)
tm = time() - t1

t1 = time()
x_cu = torch.ops.op_cuda.mul(x, y)
tcu = time() - t1

print("v1")
print(x)

print("v2")
print(y)


print("")

print("___")

print(x_)
print("")
print(x_cu)

print(x_.size())
print(x_cu.size())

x = (x_ != x_cu).sum(-1).sum(-1).sum(-1)
print(x)
assert x == 0



print(tm/tcu)
