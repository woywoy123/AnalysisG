import torch
import PyC.Operators.Tensors as T
import PyC.Operators.CUDA as C
from Checks import * 
from time import time 

device = "cuda"
Matrix = [[i*j for i in range(100)] for j in range(10)]
T_matrix = torch.tensor(Matrix, device = device, dtype = torch.float64)

t1 = time()
exp = T.Dot(T_matrix, T_matrix)
diff1 = time() - t1 

t1 = time()
Exp = C.Dot(T_matrix, T_matrix)
diff2 = time() - t1

AssertEquivalenceRecursive(Exp.tolist(), exp.tolist())
print("--- Testing Performance Between C++ and CUDA of DOT ---")
print("Speed Factor (> 1 is better): ", diff1 / diff2)

Matrix = [[i for i in range(3)] for j in range(1000)]
T1_matrix = torch.tensor(Matrix, device = device, dtype = torch.float64)

Matrix = [[j*i + 1 for i in range(3)] for j in range(1000)]
T2_matrix = torch.tensor(Matrix, device = device, dtype = torch.float64)

t1 = time()
exp = T.CosTheta(T1_matrix, T2_matrix)
diff1 = time() - t1 

t1 = time()
Exp = C.CosTheta(T1_matrix, T2_matrix)
diff2 = time() - t1

print(AssertEquivalenceRecursive(Exp.tolist(), exp.tolist()))
print("--- Testing Performance Between C++ and CUDA of CosTheta ---")
print("Speed Factor (> 1 is better): ", diff1 / diff2)

import math
Matrix = [[math.pi/(j+1) for i in range(1)] for j in range(10000)]
T1_matrix = torch.tensor(Matrix, device = device, dtype = torch.float64)

t1 = time()
exp = T.Rx(T1_matrix)
diff1 = time() - t1 

t1 = time()
Exp = C.Rx(T1_matrix)
diff2 = time() - t1

print(AssertEquivalenceRecursive(Exp.tolist(), exp.tolist()))
print("--- Testing Performance Between C++ and CUDA of Rx ---")
print("Speed Factor (> 1 is better): ", diff1 / diff2)

t1 = time()
exp = T.Ry(T1_matrix)
diff1 = time() - t1 

t1 = time()
Exp = C.Ry(T1_matrix)
diff2 = time() - t1

print(AssertEquivalenceRecursive(Exp.tolist(), exp.tolist()))
print("--- Testing Performance Between C++ and CUDA of Ry ---")
print("Speed Factor (> 1 is better): ", diff1 / diff2)

t1 = time()
exp = T.Rz(T1_matrix)
diff1 = time() - t1 

t1 = time()
Exp = C.Rz(T1_matrix)
diff2 = time() - t1

print(AssertEquivalenceRecursive(Exp.tolist(), exp.tolist()))
print("--- Testing Performance Between C++ and CUDA of Rz ---")
print("Speed Factor (> 1 is better): ", diff1 / diff2)


