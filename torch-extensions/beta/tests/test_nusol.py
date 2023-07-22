from nusol import NuSol, costheta, intersections_ellipses, UnitCircle, Derivative
import pyext
import torch
import numpy as np
import math
from common import *
mW = 80.385*1000
mT = 172.0*1000
mN = 0
masses = torch.tensor([[mW, mT, mN]], dtype = torch.float64, device = "cuda")
S2 = torch.tensor([[100, 9], [50, 100]], dtype = torch.float64, device = "cuda")
from time import sleep, time

def test_nusol_base():
    x = loadSingle()
    ita = iter(x)
    for i in range(len(x)):
        x_ = next(ita)
        lep, bquark = x_[1], x_[2]
        nu = NuSol(bquark.vec, lep.vec)
        lep_ = pyext.Transform.PxPyPzE(lep.ten)
        bquark_ = pyext.Transform.PxPyPzE(bquark.ten)
        truth = torch.tensor(nu.BaseMatrix).to(device = "cuda")
        pred = pyext.NuSol.BaseMatrix(bquark_, lep_, masses)
        assert (truth - pred).sum(-1).sum(-1) < 10e-3

def test_nusol_nu():
    x = loadSingle()
    ita = iter(x)

    lst = []
    inpt = []

    for i in range(len(x)):
        x_ = next(ita)
        ev, lep, bquark = x_[0], x_[1], x_[2]
        nu = NuSol(bquark.vec, lep.vec, ev.vec)
        lst.append(torch.tensor(nu.M).view(-1, 3, 3))

        lep_ = pyext.Transform.PxPyPzE(lep.ten)
        bquark_ = pyext.Transform.PxPyPzE(bquark.ten)
        inpt.append([bquark_, lep_, ev.ten])

    bquark_ = torch.cat([i[0] for i in inpt], 0)
    lep_    = torch.cat([i[1] for i in inpt], 0)
    ev_     = torch.cat([i[2] for i in inpt], 0)

    t1 = time()
    pred = pyext.NuSol.Nu(bquark_, lep_, ev_, masses, S2).to(device = "cpu")
    print("n-Events: " + str(len(x)) + " @ ", time() - t1)
    truth = torch.cat(lst, 0)

    x = abs((pred - truth).sum(-1).sum(-1)) > 10e-1
    assert x.sum(-1) == 0

def test_intersection():
    x = loadSingle()
    ita = iter(x)
    M_container = []
    U_container = []
    res = []

    # Numerical differences in C++ and Python appear below this number
    precision = 10e-10
    for i in range(len(x)):
        x_ = next(ita)
        ev, lep, b = x_[0], x_[1], x_[2]
        nu = NuSol(b.vec, lep.vec, ev.vec)

        unit = UnitCircle()
        points, diag = intersections_ellipses(nu.M, unit, precision)

        unit = torch.tensor(unit, device = "cuda", dtype = torch.float64)
        lep_ = pyext.Transform.PxPyPzE(lep.ten)
        bquark_ = pyext.Transform.PxPyPzE(b.ten)
        M = pyext.NuSol.Nu(bquark_, lep_, ev.ten, masses, S2)
        M_container.append(M)
        U_container.append(unit.view(-1, 3, 3))
        pts, dig = pyext.Intersection(M, unit.view(-1, 3, 3), precision)
        res.append(pts)

        points_t = torch.tensor(np.array(points), device ="cuda", dtype = torch.float64)
        points_cu = pts.view(-1, 3, 3)
        if points_cu.sum(-1).sum(-1).sum(-1) == 0 and points_t.sum(-1).sum(-1) == 0: continue
        lim = False

        points_t = points_t.view(-1, 2, 3)
        points_cu = points_cu.view(-1, 3, 3)
        for pairs_t in points_t:
            for pairs in points_cu:
                pairs = pairs[pairs.sum(-1) != 0]
                if len(pairs) != 2: continue
                sol = pairs - pairs_t
                lim = abs(sol.sum(-1).sum(-1).sum(-1)) < 1e-10
                if not lim: continue
                break
            if lim: break

        if not lim:
            print("----->>> Failure <<< ----")
            print("@ -> ", i)
            print(points_t)
            print(points_cu)
            print("-------------------------")
        assert lim

    m = torch.cat(M_container, dim = 0)
    u = torch.cat(U_container, dim = 0)
    res_t = torch.cat(res, dim = 0)
    res_cu, b = pyext.NuSol.Intersection(m, u, precision)
    x = res_cu[res_cu != res_t]
    assert len(x) == 0

    # test speed
    multi = 100
    nEvents = sum([m.size(0) for k in range(multi)])
    m = torch.cat([m for k in range(multi)], dim = 0).clone()
    u = torch.cat([u for k in range(multi)], dim = 0).clone()
    t1 = time()
    pyext.NuSol.Intersection(m, u, precision)
    t_ = time() - t1
    print("-> Timer: ", t_/nEvents, "per event @ ", nEvents, "events")

if __name__ == "__main__":
    test_nusol_base()
    test_nusol_nu()
    test_intersection()

