from common import *
import random
import torch
import math
import time

torch.ops.load_library("../build/pyc/interface/libcupyc.so")
device = "cuda"
torch.set_printoptions(threshold=1000000)

def _makematrix(rows, cols):
    tmp = [[random.random() for i in range(cols)] for k in range(rows)]
    return torch.tensor(tmp, dtype = torch.float64, device = device)

def _testdot(_x, _y, tolerance = 10**-10):
    y1 = _makematrix(_x, _y)
    y2 = _makematrix(_x, _y)

    t1 = time.time()
    dotc = torch.ops.cupyc.operators_dot(y1, y2)
    cut = time.time() - t1

    t1 = time.time()
    dott = (y1*y2).sum(-1)
    ttt = time.time() - t1

    state = AttestEqual(dott, dotc, tolerance)
    print(_x, _y, "REFERNCE/CUDA: ", ttt/cut)
    if state: return state
    print(dott)
    print(dotc)
    exit()

def _testcostheta(_x, _y):
    x = torch.tensor([[random.random() + (i + 1 ) for i in range(_x)] for _ in range(_y)], device = device, dtype = torch.float64)
    y = torch.tensor([[random.random() + (_x - i) for i in range(_x)] for _ in range(_y)], device = device, dtype = torch.float64)

    t1 = time.time()
    cu = torch.ops.cupyc.operators_costheta(x, y)
    cut = time.time() - t1

    t1 = time.time()
    xy = (x*y).sum(-1, keepdim = True)
    x2 = (x*x).sum(-1, keepdim = True)
    y2 = (y*y).sum(-1, keepdim = True)
    t = xy/(torch.sqrt(x2*y2))
    ttt = time.time() - t1

    assert AttestEqual(t, cu)
    print(_x, _y, "REFERNCE/CUDA: ", ttt/cut)
    return ttt/cut

def _testsintheta(_x, _y):
    x = torch.tensor([[random.random() + (i + 1 ) for i in range(_x)] for _ in range(_y)], device = device, dtype = torch.float64)
    y = torch.tensor([[random.random() + (_x - i) for i in range(_x)] for _ in range(_y)], device = device, dtype = torch.float64)

    t1 = time.time()
    cu = torch.ops.cupyc.operators_sintheta(x, y)
    cut = time.time() - t1

    t1 = time.time()
    xy = (x*y).sum(-1, keepdim = True)
    x2 = (x*x).sum(-1, keepdim = True)
    y2 = (y*y).sum(-1, keepdim = True)
    t = torch.sqrt(1 - (xy/(torch.sqrt(x2*y2))).pow(2))
    ttt = time.time() - t1

    assert AttestEqual(t, cu)
    print(_x, _y, "REFERNCE/CUDA: ", ttt/cut)
    return ttt/cut

def _test_rotation(dx):
    x = torch.tensor([[random.random()] for i in range(dx)], device = device, dtype = torch.float64)

    rx_ = torch.ops.cupyc.operators_rx(x)
    ry_ = torch.ops.cupyc.operators_ry(x)
    rz_ = torch.ops.cupyc.operators_rz(x)

    z = torch.zeros_like(x)
    o = torch.ones_like(x)
    c = torch.cos(x)
    s = torch.sin(x)
    tru = torch.cat([o, z, z, z, c, -s, z, s, c], dim = -1).view(-1, 3, 3)
    assert AttestEqual(tru, rx_)

    z = torch.zeros_like(x)
    o = torch.ones_like(x)
    c = torch.cos(x)
    s = torch.sin(x)
    tru = torch.cat([c, z, s, z, o, z, -s, z, c], dim = -1).view(-1, 3, 3)
    assert AttestEqual(tru, ry_)

    z = torch.zeros_like(x)
    o = torch.ones_like(x)
    c = torch.cos(x)
    s = torch.sin(x)
    tru = torch.cat([c, -s, z, s, c, z, z, z, o], dim = -1).view(-1, 3, 3)
    assert AttestEqual(tru, rz_)

def _test_cofactor(dx = 10000):
    x = torch.tensor([[[(i+1) + (k+1) for i in range(3)] for k in range(3)] for i in range(dx)], device = device, dtype = torch.float64)
    x_ = torch.ops.cupyc.operators_cofactors(x)
    x = torch.tensor([[-1, 2, -1, 2, -4, 2, -1, 2, -1] for i in range(dx)], device = device, dtype = torch.float64).view(-1, 3, 3)
    assert AttestEqual(x, x_)

def _test_det(dx):
    x = torch.tensor([[[random.random()*10 for i in range(3)] for k in range(3)] for i in range(dx)], device = device, dtype = torch.float64)

    t1 = time.time()
    x_t = torch.det(x)
    ttt = time.time() - t1

    t1 = time.time()
    x_ = torch.ops.cupyc.operators_determinant(x)
    cut = time.time() - t1
    assert AttestEqual(x_t.view(-1, 1), x_)

    print(dx, "REFERNCE/CUDA: ", ttt/cut)
    return ttt/cut

def _test_inverse(dx = 1):
    x = torch.tensor([[[random.random()*10 for i in range(3)] for k in range(3)] for i in range(dx)], device = device, dtype = torch.float64)

    t1 = time.time()
    inv, det = torch.ops.cupyc.operators_inverse(x)
    cut = time.time() - t1

    t1 = time.time()
    x_t = torch.inverse(x)
    ttt = time.time() - t1

    det = det.view(-1) > 0
    assert compare(x_t[det].reshape(-1), inv[det].reshape(-1), 10**-3)

    print(dx, "REFERNCE/CUDA: ", ttt/cut)
    return ttt/cut

def _test_eig(dx = 1):
    if dx == 1:
        hr = torch.tensor([[1, -1, 1]], device = device, dtype = torch.float64)
        hi = torch.tensor([[0,  0, 0]], device = device, dtype = torch.float64)

        x = [[[3, 2, 6], [2, 2, 5], [-2, -1, -4]]]
        x = torch.tensor(x, device = device, dtype = torch.float64)
        crl, cig = torch.ops.cupyc.operators_eigenvalue(x)
        assert compare(crl, hr)
        assert compare(cig, hi)
        eg, _ = torch.linalg.eig(x)
        assert compare(crl, eg.real)
        assert compare(cig, eg.imag)
        return 0

    x = torch.tensor([[[
            random.random()*dx**random.random() for i in range(3)] for k in range(3)
                       ] for i in range(dx)], device = device, dtype = torch.float64)

    t1 = time.time()
    crl, cig = torch.ops.cupyc.operators_eigenvalue(x)
    cut = time.time() - t1

    x = x.clone()
    t1 = time.time()
    eg, _ = torch.linalg.eig(x)
    ttt = time.time() - t1

    crl, cig = crl.sort(-1)[0], cig.sort(-1)[0]
    rl, ig = eg.real.sort(-1)[0], eg.imag.sort(-1)[0]
    assert compare(crl, rl, 10**-3)
    assert compare(cig, ig, 10**-3)
    print(dx, "REFERNCE/CUDA: ", ttt/cut)
    return ttt/cut

def _test_cross(dx):
    def _r(v, k): return v*random.random() + k

    hr = torch.tensor([
            [
                [
                    [_r(1, k), _r(-1, k), _r(1, k)],
                    [_r(1, k),  _r(2, k), _r(3, k)],
                    [_r(3, k),  _r(1, k), _r(2, k)]
                ]
            ] for k in range(dx)
    ], device = device, dtype = torch.float64)

    hi = torch.tensor([
            [
                [_r(1, k), _r(2, k), _r(3, k)],
                [_r(2, k), _r(3, k), _r(4, k)],
                [_r(0, k), _r(0, k), _r(0, k)]
            ]  for k in range(dx)
    ], device = device, dtype = torch.float64)
    hrx = hr.view(-1, 3, 1, 3)
    hix = hi.view(-1, 1, 3, 3)

    t1 = time.time()
    truth = torch.linalg.cross(hrx, hix)
    ttt = time.time() - t1

    t1 = time.time()
    cu = torch.ops.cupyc.operators_cross(hr, hi)
    cut = time.time() - t1

    assert compare(truth, cu, 10**-3)
    print(dx, "REFERNCE/CUDA: ", ttt/cut)
    return ttt/cut


def test_operators():
#    testx = [_testdot(i, j) for i in range(48, 4096, 48) for j in range(48, 1024, 48)]
    testx = [_testcostheta(i, j) for i in range(1, 1024, 1) for j in range(48, 1024, 24)]
    testx = [_testsintheta(i, j) for i in range(3, 1024, 1) for j in range(48, 1024, 24)]
    testx = [_test_rotation(i) for i in range(100, 100000, 100)]
    testx = _test_cofactor(1000)
    testx = [_test_det(i) for i in range(1000, 1000000, 1000)]
    testx = [_test_inverse(i) for i in range(1000, 1000000, 1000)]
    testx = [_test_eig(i) for i in range(1, 1000000, 1000)]
    testx = [_test_cross(i) for i in range(1, 1000000, 1000)]

if __name__ == "__main__":
    test_operators()





