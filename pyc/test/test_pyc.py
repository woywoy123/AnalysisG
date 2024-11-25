from common import *
from nusol import *
import random
import torch
import math
import time

torch.ops.load_library("../build/pyc/interface/libcupyc.so")
device = "cuda"


class pyc:

    def __init__(self):
        self.transform_separate_pt            = torch.ops.cupyc.transform_separate_pt
        self.transform_combined_pt            = torch.ops.cupyc.transform_combined_pt

        self.transform_separate_phi           = torch.ops.cupyc.transform_separate_phi
        self.transform_combined_phi           = torch.ops.cupyc.transform_combined_phi

        self.transform_separate_eta           = torch.ops.cupyc.transform_separate_eta
        self.transform_combined_eta           = torch.ops.cupyc.transform_combined_eta

        self.transform_separate_ptetaphi      = torch.ops.cupyc.transform_separate_ptetaphi
        self.transform_combined_ptetaphi      = torch.ops.cupyc.transform_combined_ptetaphi

        self.transform_separate_ptetaphie     = torch.ops.cupyc.transform_separate_ptetaphie
        self.transform_combined_ptetaphie     = torch.ops.cupyc.transform_combined_ptetaphie

        self.transform_separate_px            = torch.ops.cupyc.transform_separate_px
        self.transform_combined_px            = torch.ops.cupyc.transform_combined_px

        self.transform_separate_py            = torch.ops.cupyc.transform_separate_py
        self.transform_combined_py            = torch.ops.cupyc.transform_combined_py

        self.transform_separate_pz            = torch.ops.cupyc.transform_separate_pz
        self.transform_combined_pz            = torch.ops.cupyc.transform_combined_pz

        self.transform_separate_pxpypz        = torch.ops.cupyc.transform_separate_pxpypz
        self.transform_combined_pxpypz        = torch.ops.cupyc.transform_combined_pxpypz

        self.transform_separate_pxpypze       = torch.ops.cupyc.transform_separate_pxpypze
        self.transform_combined_pxpypze       = torch.ops.cupyc.transform_combined_pxpypze

        self.physics_cartesian_separate_p2    = torch.ops.cupyc.physics_cartesian_separate_p2
        self.physics_cartesian_combined_p2    = torch.ops.cupyc.physics_cartesian_combined_p2

        self.physics_polar_separate_p2        = torch.ops.cupyc.physics_polar_separate_p2
        self.physics_polar_combined_p2        = torch.ops.cupyc.physics_polar_combined_p2

        self.physics_cartesian_separate_p     = torch.ops.cupyc.physics_cartesian_separate_p
        self.physics_cartesian_combined_p     = torch.ops.cupyc.physics_cartesian_combined_p

        self.physics_polar_separate_p         = torch.ops.cupyc.physics_polar_separate_p
        self.physics_polar_combined_p         = torch.ops.cupyc.physics_polar_combined_p

        self.physics_cartesian_separate_beta2 = torch.ops.cupyc.physics_cartesian_separate_beta2
        self.physics_cartesian_combined_beta2 = torch.ops.cupyc.physics_cartesian_combined_beta2

        self.physics_polar_separate_beta2     = torch.ops.cupyc.physics_polar_separate_beta2
        self.physics_polar_combined_beta2     = torch.ops.cupyc.physics_polar_combined_beta2

        self.physics_cartesian_separate_beta  = torch.ops.cupyc.physics_cartesian_separate_beta
        self.physics_cartesian_combined_beta  = torch.ops.cupyc.physics_cartesian_combined_beta

        self.physics_polar_separate_beta      = torch.ops.cupyc.physics_polar_separate_beta
        self.physics_polar_combined_beta      = torch.ops.cupyc.physics_polar_combined_beta

        self.physics_cartesian_separate_m2    = torch.ops.cupyc.physics_cartesian_separate_m2
        self.physics_cartesian_combined_m2    = torch.ops.cupyc.physics_cartesian_combined_m2

        self.physics_polar_separate_m2        = torch.ops.cupyc.physics_polar_separate_m2
        self.physics_polar_combined_m2        = torch.ops.cupyc.physics_polar_combined_m2

        self.physics_cartesian_separate_m     = torch.ops.cupyc.physics_cartesian_separate_m
        self.physics_cartesian_combined_m     = torch.ops.cupyc.physics_cartesian_combined_m

        self.physics_polar_separate_m         = torch.ops.cupyc.physics_polar_separate_m
        self.physics_polar_combined_m         = torch.ops.cupyc.physics_polar_combined_m

        self.physics_cartesian_separate_mt2    = torch.ops.cupyc.physics_cartesian_separate_mt2
        self.physics_cartesian_combined_mt2    = torch.ops.cupyc.physics_cartesian_combined_mt2

        self.physics_polar_separate_mt2        = torch.ops.cupyc.physics_polar_separate_mt2
        self.physics_polar_combined_mt2        = torch.ops.cupyc.physics_polar_combined_mt2

        self.physics_cartesian_separate_mt     = torch.ops.cupyc.physics_cartesian_separate_mt
        self.physics_cartesian_combined_mt     = torch.ops.cupyc.physics_cartesian_combined_mt

        self.physics_polar_separate_mt         = torch.ops.cupyc.physics_polar_separate_mt
        self.physics_polar_combined_mt         = torch.ops.cupyc.physics_polar_combined_mt

        self.physics_cartesian_separate_theta  = torch.ops.cupyc.physics_cartesian_separate_theta
        self.physics_cartesian_combined_theta  = torch.ops.cupyc.physics_cartesian_combined_theta

        self.physics_polar_separate_theta      = torch.ops.cupyc.physics_polar_separate_theta
        self.physics_polar_combined_theta      = torch.ops.cupyc.physics_polar_combined_theta

        self.physics_cartesian_separate_deltaR  = torch.ops.cupyc.physics_cartesian_separate_deltaR
        self.physics_cartesian_combined_deltaR  = torch.ops.cupyc.physics_cartesian_combined_deltaR

        self.physics_polar_separate_deltaR      = torch.ops.cupyc.physics_polar_separate_deltaR
        self.physics_polar_combined_deltaR      = torch.ops.cupyc.physics_polar_combined_deltaR

def test_transform():
    test_case = [random.random() for i in range(4)]
    p1 = create_vector_cartesian(*test_case)
    d1_cu = create_tensor_cpu_1d(test_case).to(device = device)

    assert rounder(pyc().transform_separate_pt(d1_cu[:, 0], d1_cu[:, 1]), p1.pt)
    assert rounder(pyc().transform_combined_pt(d1_cu), p1.pt)
    assert rounder(pyc().transform_separate_phi(d1_cu[:, 0], d1_cu[:, 1]), p1.phi)
    assert rounder(pyc().transform_combined_phi(d1_cu), p1.phi)
    assert rounder(pyc().transform_separate_eta(d1_cu[:, 0], d1_cu[:, 1], d1_cu[:, 2]), p1.eta)
    assert rounder(pyc().transform_combined_eta(d1_cu), p1.eta)
    assert rounder_l(pyc().transform_separate_ptetaphi(d1_cu[:, 0], d1_cu[:, 1], d1_cu[:, 2]), [p1.pt, p1.eta, p1.phi])
    assert rounder_l(pyc().transform_combined_ptetaphi(d1_cu), [p1.pt, p1.eta, p1.phi])
    assert rounder_l(pyc().transform_separate_ptetaphie(d1_cu[:, 0], d1_cu[:, 1], d1_cu[:, 2], d1_cu[:, 3]), [p1.pt, p1.eta, p1.phi])
    assert rounder_l(pyc().transform_combined_ptetaphie(d1_cu), [p1.pt, p1.eta, p1.phi])

    p1 = create_vector_polar(*test_case)
    assert rounder(pyc().transform_separate_px(d1_cu[:, 0], d1_cu[:, 2]), p1.px)
    assert rounder(pyc().transform_combined_px(d1_cu), p1.px)
    assert rounder(pyc().transform_separate_py(d1_cu[:, 0], d1_cu[:, 2]), p1.py)
    assert rounder(pyc().transform_combined_py(d1_cu), p1.py)
    assert rounder(pyc().transform_separate_pz(d1_cu[:, 0], d1_cu[:, 1]), p1.pz)
    assert rounder(pyc().transform_combined_pz(d1_cu), p1.pz)
    assert rounder_l(pyc().transform_separate_pxpypz(d1_cu[:, 0], d1_cu[:, 1], d1_cu[:, 2]), [p1.px, p1.py, p1.pz])
    assert rounder_l(pyc().transform_combined_pxpypz(d1_cu), [p1.px, p1.py, p1.pz])
    assert rounder_l(pyc().transform_separate_pxpypze(d1_cu[:, 0], d1_cu[:, 1], d1_cu[:, 2], d1_cu[:, 3]), [p1.px, p1.py, p1.pz])
    assert rounder_l(pyc().transform_combined_pxpypze(d1_cu), [p1.px, p1.py, p1.pz])

def checkthis(data, cu, cufx, fx, cart = True):
    def lst(tmp): return [fx(p) for p in tmp]
    dx = "physics" + "_" + ("cartesian" if cart else "polar") + "_"
    sep = dx + "separate" + "_" + fx.__name__[1:]
    cmp = dx + "combined" + "_" + fx.__name__[1:]
    test = lst(data)
    print(sep)
    assert rounder_l(getattr(pyc(), sep)(*cufx(cu)), test)
    print(cmp)
    assert rounder_l(getattr(pyc(), cmp)(cu), test)

def test_physics():
    nums = 1000
    test_ct, test_pt, test_c, test_p = [], [], [], []
    while len(test_ct) < nums:
        tmp = [abs(random.random())*(i+1)*1000000 for i in range(4)]
        p1 = create_vector_cartesian(*tmp)
        if math.isnan(p1.Mt): continue
        ck = [p1.pt, p1.eta, p1.phi, p1.e]
        p2 = create_vector_polar(*ck)
        test_ct.append(tmp)
        test_pt.append(ck)
        test_c.append(p1)
        test_p.append(p2)
        if (len(test_ct) / nums)%10: continue
        print(len(test_ct))
    print("Done generating data")

    dcc = torch.tensor(test_ct, device = device, dtype = torch.float64)
    dcp = torch.tensor(test_pt, device = device, dtype = torch.float64)

    def _p2(p): return p.p2
    def _cup2(dcu): return [dcu[:, 0], dcu[:, 1], dcu[:, 2]]
    checkthis(test_c, dcc, _cup2, _p2)
    checkthis(test_p, dcp, _cup2, _p2, False)

    def _p(p): return p.p
    def _cup(dcu): return [dcu[:, 0], dcu[:, 1], dcu[:, 2]]
    checkthis(test_c, dcc, _cup, _p)
    checkthis(test_p, dcp, _cup, _p, False)

    def _beta2(p): return p.beta**2
    def _cubeta2(dcu): return [dcu[:, 0], dcu[:, 1], dcu[:, 2], dcu[:, 3]]
    checkthis(test_c, dcc, _cubeta2, _beta2)
    checkthis(test_p, dcp, _cubeta2, _beta2, False)

    def _beta(p): return p.beta
    def _cubeta(dcu): return [dcu[:, 0], dcu[:, 1], dcu[:, 2], dcu[:, 3]]
    checkthis(test_c, dcc, _cubeta, _beta)
    checkthis(test_p, dcp, _cubeta, _beta, False)

    def _m2(p): return p.M2
    def _cum2(dcu): return [dcu[:, 0], dcu[:, 1], dcu[:, 2], dcu[:, 3]]
    checkthis(test_c, dcc, _cum2, _m2)
    checkthis(test_p, dcp, _cum2, _m2, False)

    def _m(p): return p.M
    def _cum(dcu): return [dcu[:, 0], dcu[:, 1], dcu[:, 2], dcu[:, 3]]
    checkthis(test_c, dcc, _cum, _m)
    checkthis(test_p, dcp, _cum, _m, False)

    def _mt2(p): return p.Mt**2
    def _cumt2(dcu): return [dcu[:, 2], dcu[:, 3]]
    checkthis(test_c, dcc, _cumt2, _mt2)
    def _cumt2(dcu): return [dcu[:, 0], dcu[:, 1], dcu[:, 3]]
    checkthis(test_p, dcp, _cumt2, _mt2, False)

    def _mt(p): return p.Mt
    def _cumt(dcu): return [dcu[:, 2], dcu[:, 3]]
    checkthis(test_c, dcc, _cumt, _mt)
    def _cumt(dcu): return [dcu[:, 0], dcu[:, 1], dcu[:, 3]]
    checkthis(test_p, dcp, _cumt, _mt, False)

    def _theta(p): return p.theta
    def _cutheta(dcu): return [dcu[:, k] for k in range(3)]
    checkthis(test_c, dcc, _cutheta, _theta)
    def _cutheta(dcu): return [dcu[:, k] for k in range(3)]
    checkthis(test_p, dcp, _cutheta, _theta, False)

    indexing = torch.tensor([[i, j] for i in range(len(test_c)) for j in range(len(test_c))], device = device)
    test_c = [[i,j] for i in test_c for j in test_c]
    test_p = [[i,j] for i in test_p for j in test_p]

    dcc = [dcc[indexing[:,0]], dcc[indexing[:,1]]]
    dcp = [dcp[indexing[:,0]], dcp[indexing[:,1]]]

    def _deltaR(p):
        dphi = p[0].phi - p[1].phi
        if dphi > math.pi: dphi -= 2.0*math.pi
        elif dphi <= -math.pi: dphi += 2.0*math.pi
        return ((p[0].eta - p[1].eta)**2 + dphi**2)**0.5


    delta = 0
    x = [_deltaR(p) for p in test_p]
    y = pyc().physics_polar_combined_deltaR(dcp[0], dcp[1]).view(-1).tolist()
    for i in range(len(x)): delta += abs(x[i] - y[i])/float(len(x))
    if delta > 10**-10: print("Failed")
    else: delta = 0

    gc = sum([[dcp[0][:,k].view(-1, 1), dcp[1][:,k].view(-1, 1)] for k in range(1, 3)], [])
    y = pyc().physics_polar_separate_deltaR(*gc).view(-1).tolist()
    for i in range(len(x)): delta += abs(x[i] - y[i])/float(len(x))
    if delta > 10**-10: print("Failed")
    else: delta = 0

    y = pyc().physics_cartesian_combined_deltaR(dcc[0], dcc[1]).view(-1).tolist()
    for i in range(len(x)): delta += abs(x[i] - y[i])/float(len(x))
    if delta > 10**-10: print("Failed")
    else: delta = 0

    gc = sum([[dcc[0][:,k].view(-1, 1), dcc[1][:,k].view(-1, 1)] for k in range(0, 3)], [])
    y = pyc().physics_cartesian_separate_deltaR(*gc).view(-1).tolist()
    for i in range(len(x)): delta += abs(x[i] - y[i])/float(len(x))
    if delta > 10**-10: print("Failed")
    else: delta = 0


def test_operators():
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


#    testx = [_testdot(i, j) for i in range(48, 4096, 48) for j in range(48, 1024, 48)]
#    testx = [_testcostheta(i, j) for i in range(1, 1024, 1) for j in range(48, 1024, 24)]
#    testx = [_testsintheta(i, j) for i in range(3, 1024, 1) for j in range(48, 1024, 24)]
#    testx = [_test_rotation(i) for i in range(100, 100000, 100)]
#    testx = _test_cofactor(1000)
#    testx = [_test_det(i) for i in range(1000, 1000000, 1000)]
#    testx = [_test_inverse(i) for i in range(1000, 1000000, 1000)]
#    testx = [_test_eig(i) for i in range(1, 1000000, 1000)]
    testx = [_test_cross(i) for i in range(1, 1000000, 1000)]


def test_nusol_base_matrix():
    mW = 80.385*1000
    mT = 172.0*1000
    mN = 0
    masses = torch.tensor([[mT, mW, mN]], dtype = torch.float64, device = device)

    x = loadSingle()
    ita = iter(x)
    inpt = {"bq" : [], "lep" : [], "pred" : []}
    for i in range(len(x)):
        x_ = next(ita)
        lep, bquark = x_[1], x_[2]
        nu = NuSol(bquark.vec, lep.vec)
        lep_    = pyc().transform_combined_pxpypze(lep.ten)
        bquark_ = pyc().transform_combined_pxpypze(bquark.ten)
        inpt["bq"] += [bquark_.clone()]
        inpt["lep"] += [lep_.clone()]

        truth = torch.tensor(nu.H, device = device, dtype = torch.float64).view(-1, 3, 3)
        pred = torch.ops.cupyc.nusol_base_basematrix(bquark_, lep_, masses)["H"]
        inpt["pred"].append(pred)
        assert compare(truth, pred, 10**-1)
        print(i)

    inpt["bq"]  = torch.cat(inpt["bq"], dim = 0)
    inpt["lep"] = torch.cat(inpt["lep"], dim = 0)
    masses = torch.cat([masses]*len(x), 0)
    preds = torch.cat(inpt["pred"], dim = 0)
    pred = torch.ops.cupyc.nusol_base_basematrix(inpt["bq"], inpt["lep"], masses)["H"]
    assert compare(preds, pred, 10**-5)


def test_nusol_nu_cuda():
    x = loadSingle()
    ita = iter(x)
    S2 = torch.tensor([[[100, 9], [50, 100]]], dtype = torch.float64, device = device)

    mW = 80.385*1000
    mT = 172.0*1000
    mN = 0
    masses = torch.tensor([[mT, mW, mN]], dtype = torch.float64, device = device)

    for i in range(len(x)):
        x_ = next(ita)
        ev, lep, bquark = x_[0], x_[1], x_[2]
        ev_     = ev.ten

        if i == 0: continue
        print("_______", i, "________")
        lep_    = torch.ops.cupyc.transform_combined_pxpypze(lep.ten)
        bquark_ = torch.ops.cupyc.transform_combined_pxpypze(bquark.ten)
        nu      = SingleNu(bquark.vec, lep.vec, ev.vec)
        truth = torch.tensor(nu.M, device = device).view(-1, 3, 3)

        stress = 1
        bquark = torch.cat([bquark_]*stress, 0)
        lep    = torch.cat([lep_   ]*stress, 0)
        ev     = torch.cat([ev_    ]*stress, 0)
        mass   = torch.cat([masses ]*stress, 0)
        S      = torch.cat([S2     ]*stress, 0)
        pred = torch.ops.cupyc.nusol_nu(bquark, lep, ev, mass, S, 10e-10)

        assert compare(truth, pred["M"], 10**-1)
        #print(nu.nu)
        print(pred["chi2"])
        print(nu.chi2)
        exit()
        print(pred["nu"])
        print("_____")
        #print(pred["r"])
        #print(pred["lines"])
        #print(pred["A"])
        time.sleep(0.2)
        exit()

if __name__ == "__main__":
#    test_transform()
#    test_physics()
#    test_operators()
#    test_nusol_base_matrix()
    test_nusol_nu_cuda()












