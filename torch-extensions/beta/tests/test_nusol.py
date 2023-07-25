from time import sleep, time
import pyext
import torch
import numpy as np
import math
from common import *
from nusol import (
        NuSol, intersections_ellipses, 
        UnitCircle, SingleNu, DoubleNu
)

mW = 80.385*1000
mT = 172.0*1000
mN = 0
masses = torch.tensor([[mW, mT, mN]], dtype = torch.float64, device = "cuda")
S2 = torch.tensor([[100, 9], [50, 100]], dtype = torch.float64, device = "cuda")

# Numerical differences in C++ and Python appear below this number
precision = 10e-10

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

    truth = torch.cat(lst, 0)

    bquark_ = torch.cat([i[0] for i in inpt], 0)
    lep_    = torch.cat([i[1] for i in inpt], 0)
    ev_     = torch.cat([i[2] for i in inpt], 0)

    t1 = time()
    pred = pyext.NuSol.Nu(bquark_, lep_, ev_, masses, S2, -1)
    t1 = time() - t1
    pred = pred.to(device = "cpu")
    print("n-Events: " + str(len(x)) + " @ ", t1)

    x = abs((pred - truth).sum(-1).sum(-1)) > 10e-1
    assert x.sum(-1) == 0

def test_intersection():
    x = loadSingle()
    ita = iter(x)
    M_container = []
    U_container = []
    res = []

    for i in range(len(x)):
        x_ = next(ita)
        ev, lep, b = x_[0], x_[1], x_[2]
        nu = NuSol(b.vec, lep.vec, ev.vec)

        unit = UnitCircle()
        points, diag = intersections_ellipses(nu.M, unit, precision)

        unit = torch.tensor(unit, device = "cuda", dtype = torch.float64)
        lep_ = pyext.Transform.PxPyPzE(lep.ten)
        bquark_ = pyext.Transform.PxPyPzE(b.ten)
        M = pyext.NuSol.Nu(bquark_, lep_, ev.ten, masses, S2, -1)
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

def test_nu():

    x = loadSingle()
    ita = iter(x)
    compare = []
    inpts = {"b_cu" : [], "lep_cu" : [], "ev_cu" : [], 
             "m_cu" : [], "s2" : []}
    for i in range(len(x)):
        x_ = next(ita)
        ev, lep, b = x_[0], x_[1], x_[2]
        nu = SingleNu(b.vec, lep.vec, ev.vec)
        nu_t = torch.tensor(np.array(nu.nu), device = "cpu", dtype = torch.float64)

        lep_cu = pyext.Transform.PxPyPzE(lep.ten)
        b_cu   = pyext.Transform.PxPyPzE(b.ten)
        inpts["b_cu"].append(b_cu)
        inpts["lep_cu"].append(lep_cu)
        inpts["m_cu"].append(masses)
        inpts["ev_cu"].append(ev.ten)
        inpts["s2"].append(S2.view(-1, 2, 2))
        _cu  = pyext.NuSol.Nu(b_cu, lep_cu, ev.ten, masses, S2, precision)
        nu_cu = _cu[0].to(device = "cpu").view(-1, 3)
        compare.append(nu_cu)
        if len(nu_cu) == 0: assert len(nu_t) == 0
        for pr_t in nu_t:
            lim = False
            for pr_cu in nu_cu:
                s = pr_t - pr_cu
                s = s.sum(-1).sum(-1)
                if s > 10e-4: continue
                lim = True
                break

            if not lim:
                print("----->>> Failure <<< ----")
                print("@ -> ", i)
                print(nu_t)
                print(nu_cu)
                print("-------------------------")
            assert lim

    b_cu = torch.cat(inpts["b_cu"], dim = 0)
    lep_cu = torch.cat(inpts["lep_cu"], dim = 0)
    ev_cu = torch.cat(inpts["ev_cu"], dim = 0)
    mass_cu = torch.cat(inpts["m_cu"], dim = 0)
    s2_cu = torch.cat(inpts["s2"], dim = 0)
    _cu = pyext.NuSol.Nu(b_cu, lep_cu, ev_cu, masses, S2, precision)[0].to(device = "cpu")
    _cu = _cu[_cu.sum(-1) != 0]

    t = torch.cat(compare, dim = 0)
    assert len(t[t - _cu > 1e-8]) == 0

    _cu = pyext.NuSol.Nu(b_cu, lep_cu, ev_cu, mass_cu, s2_cu, precision)[0].to(device = "cpu")
    _cu = _cu[_cu.sum(-1) != 0]
    assert len(t[t - _cu > 1e-8]) == 0

def test_nunu():

    x = loadDouble()
    ita = iter(x)

    compare = {"nu1" : [], "nu2" : []}
    inpts = {
                "b1_cu" : [], "b2_cu" : [],
                "l1_cu" : [], "l2_cu" : [],
                "ev_px" : [], "ev_py" : []
    }

    for i in range(len(x)):
        x_ = next(ita)
        ev, l1, l2, b1, b2 = x_
        metx, mety = ev.ten[:, 0], ev.ten[:, 1]
        _b1 = pyext.Transform.PxPyPzE(b1.ten)
        _l1 = pyext.Transform.PxPyPzE(l1.ten)
        _b2 = pyext.Transform.PxPyPzE(b2.ten)
        _l2 = pyext.Transform.PxPyPzE(l2.ten)
        nunu_cu = pyext.NuSol.NuNu(_b1, _b2, _l1, _l2, metx, mety, masses, precision)
        c_nu, c_nu_ = nunu_cu[0].to(device = "cpu"), nunu_cu[1].to(device = "cpu")
        compare["nu1"].append(c_nu.view(-1, 3))
        compare["nu2"].append(c_nu_.view(-1, 3))

        inpts["b1_cu"].append(_b1)
        inpts["b2_cu"].append(_b2)

        inpts["l1_cu"].append(_l1)
        inpts["l2_cu"].append(_l2)

        inpts["ev_px"].append(metx)
        inpts["ev_py"].append(mety)
        try: sols = DoubleNu((b1.vec, b2.vec), (l1.vec, l2.vec), ev).nunu_s
        except np.linalg.LinAlgError: sols = []

        t_nu  = torch.tensor(np.array([i for i, _ in sols]), dtype = torch.float64)
        t_nu_ = torch.tensor(np.array([j for _, j in sols]), dtype = torch.float64)
        if len(t_nu) != 0: assert len(c_nu[0]) > 0

        for t_nui, t_nuj in zip(t_nu, t_nu_):
            lim = False
            sum_i_, sum_j_ = 1000, 1000
            for c_nui, c_nuj in zip(c_nu[0], c_nu_[0]):
                sum_i = abs((t_nui - c_nui)/t_nui).sum(-1)
                sum_j = abs((t_nuj - c_nuj)/t_nuj).sum(-1)
                if sum_i_ > sum_i and sum_j_ > sum_j: 
                    sum_i_ = sum_i
                    sum_j_ = sum_j
                if sum_i > 1 or sum_j > 1: continue
                lim = True
                break

            if not lim:
                print("----->>> Failure <<< ----")
                print("@ -> ", i)
                print(t_nui)
                print(t_nuj)
                print("-----")
                print(c_nu[0])
                print(c_nu_[0])
                print("diff")
                print(sum_i_, sum_j_)
                print("-------------------------")
            assert lim

    b1_cu    = torch.cat(inpts["b1_cu"], dim = 0)
    l1_cu    = torch.cat(inpts["l1_cu"], dim = 0)
    b2_cu    = torch.cat(inpts["b2_cu"], dim = 0)
    l2_cu    = torch.cat(inpts["l2_cu"], dim = 0)
    px_cu    = torch.cat(inpts["ev_px"], dim = 0)
    py_cu    = torch.cat(inpts["ev_py"], dim = 0)

    nu1_t = torch.cat(compare["nu1"], dim = 0)
    nu2_t = torch.cat(compare["nu2"], dim = 0)

    _cu = pyext.NuSol.NuNu(b1_cu, b2_cu, l1_cu, l2_cu, px_cu, py_cu, masses, precision)
    nu1, nu2 = _cu[0].to(device = "cpu"), _cu[1].to(device = "cpu")
    msk = nu1.sum(-1) != 0
    nu1 = nu1[msk]
    nu2 = nu2[msk]
    msk1 = (nu1_t - nu1).sum(-1) != 0
    msk2 = (nu2_t - nu2).sum(-1) != 0
    assert msk1.sum(-1) == 0
    assert msk2.sum(-1) == 0

    # test speed
    multi = 100
    nEvents = sum([b1_cu.size(0) for k in range(multi)])
    b1_cu = torch.cat([b1_cu for k in range(multi)], dim = 0).clone()
    l1_cu = torch.cat([l1_cu for k in range(multi)], dim = 0).clone()

    b2_cu = torch.cat([b2_cu for k in range(multi)], dim = 0).clone()
    l2_cu = torch.cat([l2_cu for k in range(multi)], dim = 0).clone()

    px_cu = torch.cat([px_cu for k in range(multi)], dim = 0).clone()
    py_cu = torch.cat([py_cu for k in range(multi)], dim = 0).clone()

    t_ = time()
    _cu = pyext.NuSol.NuNu(b1_cu, b2_cu, l1_cu, l2_cu, px_cu, py_cu, masses, precision)
    t_ = time() - t_
    print("-> Timer: ", t_/nEvents, "per event @ ", nEvents, "events", "Time", t_)




if __name__ == "__main__":
    test_nusol_base()
    test_nusol_nu()
    test_intersection()
    test_nu()
    test_nunu()

