from common import *
import numpy
import random
import torch
import math
import time
from pyc import pyc
cupyc = pyc()

device = "cuda"
torch.set_printoptions(threshold=1000000, linewidth = 1000000)

def _makematrix(rows, cols):
    tmp = [[[random.random() for _ in range(cols)] for _ in range(cols)] for _ in range(rows)]
    return torch.tensor(tmp, dtype = torch.float64, device = device), numpy.array(tmp)

def _testdot(_k, _x, tolerance = 10**-10):
    y1, v1 = _makematrix(_k, _x)
    y2, v2 = _makematrix(_k, _x)

    xc = numpy.array([v1[i].dot(v2[i]) for i in range(_k)])
    dotc = cupyc.cupyc_operators_dot(y1, y2)
    state = AttestEqual(torch.tensor(xc), dotc.to(device = "cpu"), tolerance)
    print("->", _k, _x)
    if state: return state
    print(">>>>>>>>")
    exit()

def _testcostheta(_x, _y):
    x = torch.tensor([[random.random() + (i + 1 ) for i in range(_x)] for _ in range(_y)], device = device, dtype = torch.float64)
    y = torch.tensor([[random.random() + (_x - i) for i in range(_x)] for _ in range(_y)], device = device, dtype = torch.float64)

    t1 = time.time()
    cu = cupyc.cupyc_operators_costheta(x, y)
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
    cu = cupyc.cupyc_operators_sintheta(x, y)
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

    rx_ = cupyc.cupyc_operators_rx(x)
    ry_ = cupyc.cupyc_operators_ry(x)
    rz_ = cupyc.cupyc_operators_rz(x)

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
    x_ = cupyc.cupyc_operators_cofactors(x)
    x = torch.tensor([[-1, 2, -1, 2, -4, 2, -1, 2, -1] for i in range(dx)], device = device, dtype = torch.float64).view(-1, 3, 3)
    assert AttestEqual(x, x_)

def _test_det(dx):
    x = torch.tensor([[[random.random()*10 for i in range(3)] for k in range(3)] for i in range(dx)], device = device, dtype = torch.float64)

    t1 = time.time()
    x_t = torch.det(x)
    ttt = time.time() - t1

    t1 = time.time()
    x_ = cupyc.cupyc_operators_determinant(x)
    cut = time.time() - t1
    assert AttestEqual(x_t.view(-1, 1), x_)

    print(dx, "REFERNCE/CUDA: ", ttt/cut)
    return ttt/cut

def _test_inverse(dx = 1):
    x = torch.tensor([[[random.random()*10 for i in range(3)] for k in range(3)] for i in range(dx)], device = device, dtype = torch.float64)

    t1 = time.time()
    inv, det = cupyc.cupyc_operators_inverse(x)
    cut = time.time() - t1

    t1 = time.time()
    x_t = torch.inverse(x)
    ttt = time.time() - t1

    det = det.view(-1) > 0
    assert compare(x_t[det].reshape(-1), inv[det].reshape(-1), 10**-3)

    print(dx, "REFERNCE/CUDA: ", ttt/cut)
    return ttt/cut

def _test_eig(dx):
    x = torch.randn((dx, 3, 3), device = device, dtype = torch.float64)
    x = (x.transpose(-1, -2) + x)

    t1 = time.time()
    crl, cig = cupyc.cupyc_operators_eigenvalue(x)
    cut = time.time() - t1

    t1 = time.time()
    eg, _ = torch.linalg.eig(x)
    ttt = time.time() - t1

    crl, cig = crl.sort(-1)[0], cig.sort(-1)[0]
    rl, ig = eg.real.sort(-1)[0], eg.imag.sort(-1)[0]

    assert compare(crl, rl, 10**-5)
    assert compare(cig, ig, 10**-5)
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
    cu = cupyc.cupyc_operators_cross(hr, hi)
    cut = time.time() - t1

    assert compare(truth, cu, 10**-3)
    print(dx, "REFERNCE/CUDA: ", ttt/cut)
    return ttt/cut


def test_operators():
    testx = [_testdot(i, k) for i in range(1, 4096, 100) for k in range(2, 4, 1)]
    testx = [_test_cross(i) for i in range(1, 1000000, 1000)]
    testx = [_testcostheta(i, j) for i in range(1, 1024, 1) for j in range(48, 1024, 24)]
    testx = [_testsintheta(i, j) for i in range(3, 1024, 1) for j in range(48, 1024, 24)]
    testx = [_test_rotation(i) for i in range(100, 100000, 100)]
    testx = _test_cofactor(1000)
    testx = [_test_det(i) for i in range(1000, 1000000, 1000)]
    testx = [_test_inverse(i) for i in range(1000, 1000000, 1000)]
    testx = [_test_eig(i) for i in range(1, 1000000, 1000)]

if __name__ == "__main__":
    test_operators()





