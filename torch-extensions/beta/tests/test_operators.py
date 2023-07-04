import torch
import random
from time import time
torch.set_printoptions(4, profile="full", linewidth=100000)
torch.ops.load_library("../build/operators/libop_cuda.so")

def _makeMatrix(l, m, n, tmp):
    x = torch.tensor(
            [[[random.random() for i in range(tmp)] for k in range(n)] for t in range(l)], 
            device = "cuda", dtype = torch.float64)

    y = torch.tensor(
            [[[random.random() for i in range(m)] for k in range(tmp)] for t in range(l)], 
            device = "cuda", dtype = torch.float64)
    return x, y

def _AttestEqual(truth, custom):
    x = (truth - custom).sum(-1).sum(-1).sum(-1)
    assert x < 10e-10

def test_matrix_multi():
    # Equally sized 
    x, y = _makeMatrix(10, 10, 10, 10)

    t1 = time()
    x_ = x.matmul(y)
    tm = time() - t1
 
    t1 = time()
    x_cu = torch.ops.op_cuda.mul(x, y)
    tcu = time() - t1
    print(tm/tcu)
    _AttestEqual(x_, x_cu)

    # M > N
    x, y = _makeMatrix(10, 17, 10, 10)

    t1 = time()
    x_ = x.matmul(y)
    tm = time() - t1

    t1 = time()
    x_cu = torch.ops.op_cuda.mul(x, y)
    tcu = time() - t1
    print(tm/tcu)
    _AttestEqual(x_, x_cu)

    # M < N
    x, y = _makeMatrix(10, 10, 17, 10)

    t1 = time()
    x_ = x.matmul(y)
    tm = time() - t1

    t1 = time()
    x_cu = torch.ops.op_cuda.mul(x, y)
    tcu = time() - t1
    print(tm/tcu)
    _AttestEqual(x_, x_cu)
    
    # intermediate n, M < N
    x, y = _makeMatrix(10, 15, 18, 5)

    t1 = time()
    x_ = x.matmul(y)
    tm = time() - t1

    t1 = time()
    x_cu = torch.ops.op_cuda.mul(x, y)
    tcu = time() - t1
    print(tm/tcu)
    _AttestEqual(x_, x_cu)

    # intermediate n, M > N
    x, y = _makeMatrix(10, 18, 15, 5)

    t1 = time()
    x_ = x.matmul(y)
    tm = time() - t1

    t1 = time()
    x_cu = torch.ops.op_cuda.mul(x, y)
    tcu = time() - t1
    print(tm/tcu)
    _AttestEqual(x_, x_cu)

    # intermediate n, M = N
    x, y = _makeMatrix(10, 15, 15, 5)

    t1 = time()
    x_ = x.matmul(y)
    tm = time() - t1

    t1 = time()
    x_cu = torch.ops.op_cuda.mul(x, y)
    tcu = time() - t1
    print(tm/tcu)
    _AttestEqual(x_, x_cu)

if __name__ == "__main__":
    pass
    test_matrix_multi()

