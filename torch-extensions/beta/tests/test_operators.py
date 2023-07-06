from time import time
import random
import pyext
import torch
torch.set_printoptions(4, profile="full", linewidth=100000)

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

def _compareMulti(l, m, n, tmp):
    x, y = _makeMatrix(l, m, n, tmp)
    t1 = time()
    x_ = x.matmul(y)
    tm = time() - t1

    t1 = time()
    x_cu = pyext.Operators.Mul(x, y)
    t_cu = time() - t1
    print(tm/t_cu)
    _AttestEqual(x_, x_cu)

def test_matrix_multi():
    # Equally sized 
    _compareMulti(10, 10, 10, 10)

    # M > N
    _compareMulti(10, 17, 10, 10)

    # M < N
    _compareMulti(10, 10, 17, 10)

    # intermediate n, M < N
    _compareMulti(10, 15, 18, 5)

    # intermediate n, M > N
    _compareMulti(10, 18, 15, 5)

    # intermediate n, M = N
    _compareMulti(10, 15, 15, 5)

def test_costheta():
    x = torch.tensor([[random.random() for i in range(1000)] for _ in range(100)], device = "cuda", dtype = torch.float64)
    y = torch.tensor([[random.random() for i in range(1000)] for _ in range(100)], device = "cuda", dtype = torch.float64)

    t1 = time()
    cu = pyext.Operators.CosTheta(x, y)
    t_cu = time() - t1

    t1 = time()
    cu = pyext.Operators.CosTheta(x, y)
    t_cu = time() - t1

    t1 = time()
    xy = (x*y).sum(-1, keepdim = True)
    x2 = (x*x).sum(-1, keepdim = True)
    y2 = (y*y).sum(-1, keepdim = True)
    t = xy/(torch.sqrt(x2*y2))
    t_m = time() - t1
    _AttestEqual(t, cu)

    print(t_m/t_cu)

def test_sintheta():
    x = torch.tensor([[random.random() for i in range(1000)] for _ in range(100)], device = "cuda", dtype = torch.float64)
    y = torch.tensor([[random.random() for i in range(1000)] for _ in range(100)], device = "cuda", dtype = torch.float64)

    t1 = time()
    cu = pyext.Operators.SinTheta(x, y)
    t_cu = time() - t1

    t1 = time()
    cu = pyext.Operators.SinTheta(x, y)
    t_cu = time() - t1

    t1 = time()
    xy2 = torch.pow((x*y).sum(-1, keepdim = True), 2)
    x2 = (x*x).sum(-1, keepdim = True)
    y2 = (y*y).sum(-1, keepdim = True)
    x2y2 = x2*y2

    t = torch.sqrt(1 - xy2/x2y2)
    t_m = time() - t1
    _AttestEqual(t, cu)

    print(t_m/t_cu)

def test_rx():
    x = torch.tensor([[random.random()] for i in range(10)], device = "cuda", dtype = torch.float64)
    t1 = time()
    x_ = pyext.Operators.Rx(x)
    t_cu = time() - t1

    z = torch.zeros_like(x)
    o = torch.ones_like(x)
    c = torch.cos(x)
    s = torch.sin(x)
    tru = torch.cat([o, z, z, z, c, -s, z, s, c], dim = -1).view(-1, 3, 3)
    _AttestEqual(tru, x_)

def test_ry():
    x = torch.tensor([[random.random()] for i in range(10)], device = "cuda", dtype = torch.float64)
    t1 = time()
    x_ = pyext.Operators.Ry(x)
    t_cu = time() - t1

    z = torch.zeros_like(x)
    o = torch.ones_like(x)
    c = torch.cos(x)
    s = torch.sin(x)
    tru = torch.cat([c, z, s, z, o, z, -s, z, c], dim = -1).view(-1, 3, 3)

    _AttestEqual(tru, x_)

def test_rz():
    x = torch.tensor([[random.random()] for i in range(10)], device = "cuda", dtype = torch.float64)
    t1 = time()
    x_ = pyext.Operators.Rz(x)
    t_cu = time() - t1

    z = torch.zeros_like(x)
    o = torch.ones_like(x)
    c = torch.cos(x)
    s = torch.sin(x)
    tru = torch.cat([c, -s, z, s, c, z, z, z, o], dim = -1).view(-1, 3, 3)
    _AttestEqual(tru, x_)

def test_cofactor():
    x = torch.tensor([[[(i+1) + (k+1) for i in range(3)] for k in range(3)] for i in range(2)], device = "cuda", dtype = torch.float64)
    t1 = time()
    x_ = pyext.Operators.CoFactors(x)
    t_cu = time() - t1

    x = torch.tensor([[-1, 2, -1, 2, -4, 2, -1, 2, -1] for i in range(2)], device = "cuda", dtype = torch.float64).view(-1, 3, 3)
    _AttestEqual(x, x_)

def test_det():
    x = torch.tensor([[[random.random()*10 for i in range(3)] for k in range(3)] for i in range(100)], device = "cuda", dtype = torch.float64)
    t1 = time()
    x_ = pyext.Operators.Determinant(x)
    t_cu = time() - t1

    t1 = time()
    x_t = torch.det(x)
    t_t = time() - t1
    print(t_t/t_cu)
    _AttestEqual(x_t.view(-1, 1), x_)

def test_inverse():
    x = torch.tensor([[[random.random()*10 for i in range(3)] for k in range(3)] for i in range(10000)], device = "cuda", dtype = torch.float64)

    det = pyext.Operators.Determinant(x).view(-1)
    t1 = time()
    x_ = pyext.Operators.Inverse(x)
    t_cu = time() - t1

    t1 = time()
    x_t = torch.inverse(x)
    t_t = time() - t1
    print(t_t/t_cu)
    _AttestEqual(x_t[det != 0], x_[det != 0])

def test_equivalence():
    def _compareMUL(l, m, n, tmp):
        x, y = _makeMatrix(l, m, n, tmp)
        t1 = time()
        x_cu = pyext.Operators.Mul(x, y)
        t_cu = time() - t1

        x, y = x.to(device = "cpu"), y.to(device = "cpu")
        t1 = time()
        x_ = pyext.Operators.Mul(x, y)
        tm = time() - t1
        print(tm/t_cu)

        _AttestEqual(x_, x_cu)
    def _Rot(inpt): 
        x = torch.tensor([[random.random() for i in range(1)] for _ in range(100)], device = "cuda", dtype = torch.float64)

        t1 = time()
        cu = inpt(x)
        t_cu = time() - t1

        x = x.to(device = "cpu")
        t1 = time()
        t = inpt(x)
        t_m = time() - t1
        _AttestEqual(t, cu.to(device = "cpu"))
        print(t_m/t_cu)


    # CHECK MUL
    _compareMulti(10, 10, 10, 10)
    _compareMulti(10, 17, 10, 10)
    _compareMulti(10, 10, 17, 10)
    _compareMulti(10, 15, 18, 5)
    _compareMulti(10, 18, 15, 5)
    _compareMulti(10, 15, 15, 5)

    # CHECK COS/SIN THETA
    x = torch.tensor([[random.random() for i in range(1000)] for _ in range(100)], device = "cuda", dtype = torch.float64)
    y = torch.tensor([[random.random() for i in range(1000)] for _ in range(100)], device = "cuda", dtype = torch.float64)

    t1 = time()
    cu = pyext.Operators.CosTheta(x, y)
    t_cu = time() - t1

    x, y = x.to(device = "cpu"), y.to(device = "cpu")
    t1 = time()
    t = pyext.Operators.CosTheta(x, y)
    t_m = time() - t1
    _AttestEqual(t, cu.to(device = "cpu"))
    print(t_m/t_cu)

    x = torch.tensor([[random.random() for i in range(1000)] for _ in range(100)], device = "cuda", dtype = torch.float64)
    y = torch.tensor([[random.random() for i in range(1000)] for _ in range(100)], device = "cuda", dtype = torch.float64)

    t1 = time()
    cu = pyext.Operators.SinTheta(x, y)
    t_cu = time() - t1

    x, y = x.to(device = "cpu"), y.to(device = "cpu")
    t1 = time()
    t = pyext.Operators.SinTheta(x, y)
    t_m = time() - t1
    _AttestEqual(t, cu.to(device = "cpu"))
    print(t_m/t_cu)

    _Rot(pyext.Operators.Rx)
    _Rot(pyext.Operators.Ry)
    _Rot(pyext.Operators.Rz)


if __name__ == "__main__":
    test_matrix_multi()
    test_costheta()
    test_sintheta()
    test_rx()
    test_ry()
    test_rz()
    test_cofactor()
    test_det()
    test_inverse()
    test_equivalence()

