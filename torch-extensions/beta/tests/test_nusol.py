import pyext
import torch
from nusol import NuSol, costheta
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
        lst.append(torch.tensor(nu.X).view(-1, 3, 3))

        lep_ = pyext.Transform.PxPyPzE(lep.ten)
        bquark_ = pyext.Transform.PxPyPzE(bquark.ten)
        inpt.append([bquark_, lep_, ev.ten])

    bquark_ = torch.cat([i[0] for i in inpt], 0)
    lep_    = torch.cat([i[1] for i in inpt], 0)
    ev_     = torch.cat([i[2] for i in inpt], 0)

    t1 = time()
    pred = pyext.NuSol.Nu(bquark_, lep_, ev_, masses, S2).to(device = "cpu")
    print(time() - t1)
    print("n-Events: " + str(len(x)))
    truth = torch.cat(lst, 0)

    x = abs((pred - truth).sum(-1).sum(-1)) > 10e-1
    assert x.sum(-1) == 0





if __name__ == "__main__":
    #test_nusol_base()
    test_nusol_nu()

