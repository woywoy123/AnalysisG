from common import *
from nusol import *
import random
import torch
import math
import time

torch.ops.load_library("../build/pyc/interface/libcupyc.so")
device = "cuda"
torch.set_printoptions(threshold=1000000)

def checkthis(data, cu, cufx, fx, cart = True):
    def lst(tmp): return [fx(p) for p in tmp]
    dx = "physics" + "_" + ("cartesian" if cart else "polar") + "_"
    sep = dx + "separate" + "_" + fx.__name__[1:]
    cmp = dx + "combined" + "_" + fx.__name__[1:]
    test = lst(data)
    print(sep)
    assert rounder_l(getattr(torch.ops.cupyc, sep)(*cufx(cu)), test)
    print(cmp)
    assert rounder_l(getattr(torch.ops.cupyc, cmp)(cu), test)

def test_physics():
    nums = 100
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
    y = torch.ops.cupyc.physics_polar_combined_deltaR(dcp[0], dcp[1]).view(-1).tolist()
    for i in range(len(x)): delta += abs(x[i] - y[i])/float(len(x))
    if delta > 10**-10: print("Failed")
    else: delta = 0

    gc = sum([[dcp[0][:,k].view(-1, 1), dcp[1][:,k].view(-1, 1)] for k in range(1, 3)], [])
    y = torch.ops.cupyc.physics_polar_separate_deltaR(*gc).view(-1).tolist()
    for i in range(len(x)): delta += abs(x[i] - y[i])/float(len(x))
    if delta > 10**-10: print("Failed")
    else: delta = 0

    y = torch.ops.cupyc.physics_cartesian_combined_deltaR(dcc[0], dcc[1]).view(-1).tolist()
    for i in range(len(x)): delta += abs(x[i] - y[i])/float(len(x))
    if delta > 10**-10: print("Failed")
    else: delta = 0

    gc = sum([[dcc[0][:,k].view(-1, 1), dcc[1][:,k].view(-1, 1)] for k in range(0, 3)], [])
    y = torch.ops.cupyc.physics_cartesian_separate_deltaR(*gc).view(-1).tolist()
    for i in range(len(x)): delta += abs(x[i] - y[i])/float(len(x))
    if delta > 10**-10: print("Failed")
    else: delta = 0


if __name__ == "__main__":
    test_physics()

