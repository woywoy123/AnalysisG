from AnalysisTopGNN.IO import UnpickleObject
from math import sin, cos, sinh, sqrt
import torch
import Floats
import Tensors

def Px(pt, phi):
    return pt*cos(phi)

def Py(pt, phi):
    return pt*sin(phi)

def Pz(pt, eta):
    return pt*sinh(eta)

def PxPyPzE(pt, phi, eta, e):
    return [Px(pt, phi), Py(pt, phi), Pz(pt, eta), e]

def Beta(p):
    return sqrt(Px(p.pt, p.phi)**2 + Py(p.pt, p.phi)**2 + Pz(p.pt, p.eta)**2)/p.e

def CosTheta(v1, v2):
    mg1 = sum([i**2 for i in v1])
    mg2 = sum([i**2 for i in v2])
    d = sum([i*j for i, j in zip(v1, v2)]) 
    return d / (sqrt(mg1)*sqrt(mg2))


def TestFloats(particles):
    print("---- testing Floats ----")
    l = []
    lT = []
    for i in particles:
        l.append(PxPyPzE(i.pt, i.phi, i.eta, i.e)) 
        lT.append(Floats.ToPxPyPzE(i.pt, i.eta, i.phi, i.e, "cpu"))
    for i, j in zip(l, lT):
        px_p, px_t = int(i[0]), int(j[0][0])
        py_p, py_t = int(i[1]), int(j[0][1])       
        pz_p, pz_t = int(i[2]), int(j[0][2])
        e_p, e_t = int(i[3]), int(j[0][3])

        assert px_p == px_t
        assert py_p == py_t
        assert pz_p == pz_t
        assert e_p == e_t
    
    for i in particles:
        b_p = Beta(i)
        b_tb = Floats.BetaPolar(i.pt, i.eta, i.phi, i.e, "cpu")
        b_tc = Floats.BetaCartesian(Px(i.pt, i.phi), Py(i.pt, i.phi), Pz(i.pt, i.eta), i.e, "cpu")
        print(b_p, b_tb, b_tc)

def TestTensors(particles):
    print("---- testing Tensor ----")
    l = []
    lT = []
    l2 = []
    for i in particles:
        l.append(PxPyPzE(i.pt, i.phi, i.eta, i.e)) 
        lT.append(Tensors.ToPxPyPzE(torch.tensor([[i.pt, i.eta, i.phi, i.e]])))
        l2.append([i.pt, i.eta, i.phi, i.e])


    for i, j in zip(l, lT):
        px_p, px_t = int(i[0]), int(j[0][0])
        py_p, py_t = int(i[1]), int(j[0][1])       
        pz_p, pz_t = int(i[2]), int(j[0][2])
        e_p, e_t = int(i[3]), int(j[0][3])

        assert px_p == px_t
        assert py_p == py_t
        assert pz_p == pz_t
        assert e_p == e_t
    
    lt = Tensors.BetaPolar(torch.tensor(l2))
    for i, j in zip(particles, lt):
        b_p = Beta(i)
        b_tb = Tensors.BetaPolar(torch.tensor([[i.pt, i.eta, i.phi, i.e]]))
        b_t = Floats.BetaPolar(i.pt, i.eta, i.phi, i.e, "cpu")
        b_tc = Tensors.BetaCartesian(torch.tensor([[Px(i.pt, i.phi), Py(i.pt, i.phi), Pz(i.pt, i.eta), i.e]]))
        print(b_p, b_tb, b_tc, j)

def TestCosTheta():
    print("----- Test CosTheta -----")
    assert float(Floats.CosThetaCartesian(1, 2, 1, 2, 1, 2, 1, 2, "cpu")[0]) == CosTheta([1, 1, 1, 1], [2, 2, 2, 2])
    v1 = torch.tensor([[i**2 for i in range(4)] for j in range(4)])
    v2 = torch.tensor([[i for i in range(4)] for j in range(4)])
    print(Tensors.CosThetaCartesian(v1, v2))
    print(CosTheta([i**2 for i in range(4)], [i for i in range(4)]))

def TestMass(particles):
    for i in particles:
        
        M2PF = Floats.Mass2Polar(i.pt, i.eta, i.phi, i.e, "cpu")
        M2PT = Tensors.Mass2Polar(torch.tensor([[i.pt, i.eta, i.phi, i.e]]))
        
        MPF = Floats.MassPolar(i.pt, i.eta, i.phi, i.e, "cpu")
        MPT = Tensors.MassPolar(torch.tensor([[i.pt, i.eta, i.phi, i.e]]))
        
        print(M2PF, M2PT, MPF, MPT)


ev = UnpickleObject("TMP")
singlelepton = [i for i in ev.TopChildren if i.Parent[0].DecayLeptonically()]
b = singlelepton[0]
nu = singlelepton[1]
muon = singlelepton[2]

TestFloats(singlelepton)
TestTensors(singlelepton)
TestCosTheta()
TestMass(singlelepton)
