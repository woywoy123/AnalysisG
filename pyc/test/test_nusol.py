from common import *
from nusol import *
import random
import torch
import math
import time

torch.ops.load_library("../build/pyc/interface/libcupyc.so")
device = "cuda"
torch.set_printoptions(threshold=1000000)

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
        lep_    = torch.ops.cupyc.transform_combined_pxpypze(lep.ten)
        bquark_ = torch.ops.cupyc.transform_combined_pxpypze(bquark.ten)
        inpt["bq"] += [bquark_.clone()]
        inpt["lep"] += [lep_.clone()]

        truth = torch.tensor(nu.H, device = device, dtype = torch.float64).view(-1, 3, 3)
        pred = torch.ops.cupyc.nusol_base_basematrix(bquark_, lep_, masses)["H"]
        inpt["pred"].append(pred)
        assert compare(truth, pred, 10**-5)

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
            for k in range(nu_.size(0)):
                lm = (abs(nut[j] - nu_[k]) / abs(nut[j])) < 10**-3
                if not lm.sum(-1): continue
                found = True
            if not found:
                print("____< truth >____")
                print(nut)
                print("____< pred >____")
                print(nu_)
                exit()

def test_nusol_nunu_cuda():
    x = loadDouble()
    ita = iter(x)
    mW = 80.385*1000
    mT = 172.0*1000
    mN = 0
    masses = torch.tensor([[mT, mW, mN]], dtype = torch.float64, device = device)

    for i in range(len(x)):
        x_ = next(ita)
        ev, l1, l2, b1, b2 = x_
        met_xy = ev.ten

        print("_______", i, "________")
        b1_ = torch.ops.cupyc.transform_combined_pxpypze(b1.ten)
        b2_ = torch.ops.cupyc.transform_combined_pxpypze(b2.ten)
        l1_ = torch.ops.cupyc.transform_combined_pxpypze(l1.ten)
        l2_ = torch.ops.cupyc.transform_combined_pxpypze(l2.ten)
        try: nu  = DoubleNu((b1.vec, b2.vec), (l1.vec, l2.vec), ev)
        except: continue

        stress = 1
        b1_  = torch.cat([b1_]*stress, 0)
        b2_  = torch.cat([b2_]*stress, 0)
        l1_  = torch.cat([l1_]*stress, 0)
        l2_  = torch.cat([l2_]*stress, 0)
        met  = torch.cat([met_xy]*stress, 0)
        mass = torch.cat([masses]*stress, 0)
        pred = torch.ops.cupyc.nusol_nunu(b1_, b2_, l1_, l2_, met, mass, 10e-10)

        nu1T, nu2T = nu.nunu_s
        nu1T, nu2T = [torch.tensor(f.tolist(), dtype = torch.float64) for f in [nu1T, nu2T]]
        nu1c, nu2c = pred["nu1"].to(device = "cpu").view(-1, 3), pred["nu2"].to(device = "cpu").view(-1, 3)
        if nu1T.size(0) == 0 and nu1c.size(0) == 0: continue
        for j in range(nu1T.size(0)):
            found = False
            for k in range(nu1c.size(0)):
                lm1  = (abs(nu1T[j] - nu1c[k]) / abs(nu1T[j]))
                lm2  = (abs(nu2T[j] - nu2c[k]) / abs(nu2T[j]))
                msk  = (lm1 < 10**-3)*(lm2 < 10**-3)
                if not msk.view(-1).sum(-1): continue
                found = True
                break

            if not found:
                print("____< truth >____")
                print(nu1T)
                print(nu2T)
                print("____< pred >____")
                print(nu1c)
                print(nu2c)
                print(pred["distances"])
                print(pred["passed"])
                exit()



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
    particles  = torch.ops.cupyc.transform_combined_pxpypze(torch.cat(particles, 0).to(device = "cuda"))
    metxy      = torch.cat(metxy, 0).to(device = "cuda")

    cmb = torch.ops.cupyc.nusol_combinatorial(edge_index, batch, particles, pid, metxy, mT, mW, t_pm, w_pm, 20, 1e-10, True)
    print(cmb["distances"])
    print(cmb["nu1"])
    print(torch.cat([cmb["l1"], cmb["l2"], cmb["b1"], cmb["b2"]], -1))

if __name__ == "__main__":
    test_nusol_base_matrix()
    test_nusol_nu_cuda()
    test_nusol_nunu_cuda()
    test_nusol_combinatorial_cuda()





