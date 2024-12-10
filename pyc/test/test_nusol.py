from common import *
from nusol import *
from pyc import pyc
import random
import torch
import math
import time

device = "cuda"
torch.set_printoptions(threshold=1000000, linewidth = 1000, precision = 5, sci_mode = True)
pyc = pyc()

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

        print("_______", i, "_______")


        nu = NuSol(bquark.vec, lep.vec)
        lep_    = pyc.cuda_transform_combined_pxpypze(lep.ten)
        bquark_ = pyc.cuda_transform_combined_pxpypze(bquark.ten)
        inpt["bq"] += [bquark_.clone()]
        inpt["lep"] += [lep_.clone()]

        truth = torch.tensor(nu.H, device = device, dtype = torch.float64).view(-1, 3, 3)
        pred = pyc.cuda_nusol_base_basematrix(bquark_, lep_, masses)["H"]
        inpt["pred"].append(pred)
        assert compare(truth, pred, 10**-5)

    inpt["bq"]  = torch.cat(inpt["bq"], dim = 0)
    inpt["lep"] = torch.cat(inpt["lep"], dim = 0)
    masses = torch.cat([masses]*len(x), 0)
    preds = torch.cat(inpt["pred"], dim = 0)
    pred = pyc.cuda_nusol_base_basematrix(inpt["bq"], inpt["lep"], masses)["H"]
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

#        if i != 14 and i != 19: continue
        print("_______", i, "________")
        lep_    = pyc.cuda_transform_combined_pxpypze(lep.ten)
        bquark_ = pyc.cuda_transform_combined_pxpypze(bquark.ten)
        nu      = SingleNu(bquark.vec, lep.vec, ev.vec)

        MT   = torch.tensor(nu._M, device = device).view(-1, 3, 3)
        XT   = torch.tensor(nu._X, device = device).view(-1, 3, 3)
        pred = pyc.cuda_nusol_nu(bquark_, lep_, ev_, masses, S2, 10e-10)

        assert compare(MT, pred["M"], 10**-5)
        assert compare(XT, pred["X"], 10**-5)
        nu_   = pred["nu"].view(-1, 3)
        chi2_ = pred["distances"]

        nu_   = nu_[(chi2_ > 0).view(-1)].clone()
        chi2_ = chi2_[chi2_ > 0].clone()

        nut = torch.tensor(nu.nu.tolist()  , device = "cpu", dtype = torch.float64)
        ch2 = torch.tensor(nu.chi2.tolist(), device = "cpu", dtype = torch.float64)
        srt = chi2_.sort()[1]

        if nut.size(0) == 0 and nu_.size(0) == 0: continue
        nu_ = nu_[srt].to(device = "cpu")
        chi2_ = chi2_[srt].to(device = "cpu")
        for j in range(nut.size(0)):
            found = False
            err = 100000
            for k in range(nu_.size(0)):
                lm = abs(nut[j] - nu_[k]) / abs(nut[j])
                if not (lm < 1**-7).sum(-1): continue
                if err < lm.sum(-1): continue
                found = True
                err = lm.sum(-1)

            if not found:
                print("____< truth >____")
                print(nut)
                print("____< pred >____")
                print(nu_)
                exit()

def test_nusol_nunu_cuda():
    def cat(x, key): return torch.cat(x[key], 0)

    import numpy
    x = loadDouble()
    ita = iter(x)
    mT = 172.0*1000
    mW = 80.385*1000
    mN = 0
    masses = torch.tensor([[mT, mW, mN]], dtype = torch.float64, device = device)

    data = {"mass" : [], "b1": [], "b2" : [], "l1" : [], "l2" : [], "met_xy" : [], "nu1": [], "nu2" : []}
    for i in range(len(x)):
        x_ = next(ita)
        ev, l1, l2, b1, b2 = x_
        met_xy = ev.ten

        print("_______", i, "________")
        b1_ = pyc.cuda_transform_combined_pxpypze(b1.ten)
        b2_ = pyc.cuda_transform_combined_pxpypze(b2.ten)
        l1_ = pyc.cuda_transform_combined_pxpypze(l1.ten)
        l2_ = pyc.cuda_transform_combined_pxpypze(l2.ten)


        pred = pyc.cuda_nusol_nunu(b1_, b2_, l1_, l2_, met_xy, masses, 1e-10)
        nu1c, nu2c = pred["nu1"].to(device = "cpu").view(-1, 3), pred["nu2"].to(device = "cpu").view(-1, 3)

        data["mass"].append(masses)
        data["b1"].append(b1_)
        data["l1"].append(l1_)
        data["b2"].append(b2_)
        data["l2"].append(l2_)
        data["met_xy"].append(met_xy)
        data["nu1"].append(pred["nu1"])
        data["nu2"].append(pred["nu2"])

        try: nu  = DoubleNu((b1.vec, b2.vec), (l1.vec, l2.vec), ev)
        except numpy.linalg.LinAlgError: continue
        nu1T, nu2T = nu.nunu_s
        nu1T, nu2T = [torch.tensor(f.tolist(), dtype = torch.float64) for f in [nu1T, nu2T]]

        if nu1T.size(0) == 0 and nu1c.size(0) == 0: continue
        for j in range(nu1T.size(0)):
            chi2 = None
            for k in range(nu1c.size(0)):
                lm1  = ((abs(nu1T[j] - nu1c[k])/nu1T[j])**2).sum(-1)
                lm2  = ((abs(nu2T[j] - nu2c[k])/nu2T[j])**2).sum(-1)
                chi_ = ((lm1**2 + lm2**2).sum(-1))**0.5
                if chi2 is not None and chi2 < chi_: continue
                chi2 = chi_
            print(chi2)
            if not (chi2 < 10):exit()

#        predx = pyc.cuda_nusol_nunu(
#                cat(data, "b1"), cat(data, "b2"),
#                cat(data, "l1"), cat(data, "l2"),
#                cat(data, "met_xy"), cat(data, "mass"),
#                1e-10
#        )
#        print(predx["nu1"].size())
#        print((predx["nu1"] - cat(data, "nu1")).view(-1).sum(-1))
#

def test_nusol_combinatorial_cuda():
    mW = 80.385*1000
    mT = 172.0*1000
    t_pm = 0.95
    w_pm = 0.95
    step = 20
    null = 1e-10
    gev = False

    batch = []
    metxy = []
    pid = []
    src = []
    dst = []
    particles = []

    tx = 0
    nunu_ = loadDouble()
    for i in iter(nunu_):
        ev, l1, l2, b1, b2 = i
        e_i         = len(particles)
        particles  += [l1.ten, l2.ten, b1.ten, b2.ten]
        pid        += [[1, 0], [1, 0], [0, 1], [0, 1]]
        src        += [k for k in range(e_i, len(particles)) for _ in range(e_i, len(particles))]
        dst        += [k for _ in range(e_i, len(particles)) for k in range(e_i, len(particles))]
        metxy      += [ev.ten]
        batch      += [tx]*4
        tx         += 1
        if tx == 3: break

    #nu_   = loadSingle()
    #for i in iter(nu_):
    #    e_i         = len(particles)
    #    particles  += [i[1].ten, i[2].ten, i[2].ten]
    #    pid        += [[1, 0]  , [0, 1]  , [0, 1]  ]
    #    src        += [k for k in range(e_i, len(particles)) for _ in range(e_i, len(particles))]
    #    dst        += [k for _ in range(e_i, len(particles)) for k in range(e_i, len(particles))]
    #    metxy      += [i[0].ten]
    #    batch      += [tx]*3
    #    tx         += 1
    #    if tx == 6: break

    print(batch)
    pid        = torch.tensor(pid, device = "cuda")
    batch      = torch.tensor(batch, device = "cuda")
    edge_index = torch.tensor([src, dst], device = "cuda")
    particles  = pyc.cuda_transform_combined_pxpypze(torch.cat(particles, 0).to(device = "cuda"))
    metxy      = torch.cat(metxy, 0).to(device = "cuda")

    cmb = pyc.cuda_nusol_combinatorial(edge_index, batch, particles, pid, metxy, mT, mW, t_pm, w_pm, 20, 1e-10, True)
    print(cmb["distances"])
    print(cmb["nu1"])
    print(torch.cat([cmb["l1"], cmb["l2"], cmb["b1"], cmb["b2"]], -1))

if __name__ == "__main__":
    #test_nusol_base_matrix()
    #test_nusol_nu_cuda()
    test_nusol_nunu_cuda()
    #test_nusol_combinatorial_cuda()





