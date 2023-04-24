from AnalysisG.Events import Event
from AnalysisG import Analysis
import torch

from neutrino_reconstruction.NeutrinoSolutionDeconstruct import *
import PyC.Transform.Floats as F
import PyC.Transform.Tensors as T

import PyC.Physics.Tensors.Cartesian as TC
import PyC.Physics.Tensors.Polar as TP
import PyC.Operators.Tensors as TO
import PyC.NuSol.Tensors as NuT

is_cuda = True
try: 
    import PyC.Transform.CUDA as C
    import PyC.Physics.CUDA.Cartesian as CC
    import PyC.Physics.CUDA.Polar as CP
    import PyC.Operators.CUDA as CO
    import PyC.NuSol.CUDA as NuC
except: is_cuda = False


import math
from time import time
from check import *
Files = {
            "./samples/sample1" : ["smpl1.root"], 
            "./samples/sample2" : ["smpl1.root", "smpl2.root", "smpl3.root"]
}

def _MakeSample():
    Ana = Analysis()
    Ana.InputSample("", Files) 
    Ana.Event = Event 
    Ana.EventCache = True 
    Ana.Launch
    return Ana

def test_transformation():
    Ana = _MakeSample()

    _cart = []
    _polar = []
    for i in Ana:
        top = i.Tops
        if len(top) != 4: continue

        for t in top:
            assert AssertEquivalence(t.px, F.Px(t.pt, t.phi))
            assert AssertEquivalence(t.py, F.Py(t.pt, t.phi))
            assert AssertEquivalence(t.pz, F.Pz(t.pt, t.eta))
    
            assert AssertEquivalence(t.pt,   F.PT(t.px, t.py))
            assert AssertEquivalence(t.eta, F.Eta(t.px, t.py, t.pz))
            assert AssertEquivalence(t.phi, F.Phi(t.px, t.py))
            
            assert AssertEquivalenceRecursive([t.pt, t.eta, t.phi], F.PtEtaPhi(t.px, t.py, t.pz))

            t_px, t_py, t_pz = MakeTensor(t.px), MakeTensor(t.py), MakeTensor(t.pz)
            t_pt, t_eta, t_phi = MakeTensor(t.pt), MakeTensor(t.eta), MakeTensor(t.phi)
 
            assert AssertEquivalence(t.pt,   T.PT(t_px, t_py))
            assert AssertEquivalence(t.eta, T.Eta(t_px, t_py, t_pz))
            assert AssertEquivalence(t.phi, T.Phi(t_px, t_py))
            
            lc = [t.px, t.py,  t.pz]
            assert AssertEquivalenceList(lc, F.PxPyPz(t.pt, t.eta, t.phi))
            assert AssertEquivalenceList(lc, T.PxPyPz(t_pt, t_eta, t_phi)[0])
 
            lp = [t.pt, t.eta, t.phi] 
            assert AssertEquivalenceList(lp, F.PtEtaPhi(t.px, t.py, t.pz))
            assert AssertEquivalenceList(lp, T.PtEtaPhi(t_px, t_py, t_pz)[0])

            for c in t.Children:
                _polar.append([c.pt, c.eta, c.phi])
                _cart.append([c.px, c.py, c.pz])
 
            if is_cuda == False: continue
            c_pt, c_eta, c_phi =  MakeTensor(t.pt,  "cuda"), MakeTensor(t.eta, "cuda"), MakeTensor(t.phi, "cuda")
            c_px, c_py, c_pz =  MakeTensor(t.px,  "cuda"), MakeTensor(t.py, "cuda"), MakeTensor(t.pz, "cuda")
            
            assert AssertEquivalence(t.px, C.Px(c_pt, c_phi))
            assert AssertEquivalence(t.py, C.Py(c_pt, c_phi))
            assert AssertEquivalence(t.pz, C.Pz(c_pt, c_eta))
            
            assert AssertEquivalence(t.pt,   C.PT(c_px, c_py)[0])
            assert AssertEquivalence(t.eta, C.Eta(c_px, c_py, c_pz)[0])
            assert AssertEquivalence(t.phi, C.Phi(c_px, c_py)[0])

            assert AssertEquivalenceList(lc, C.PxPyPz(c_pt, c_eta, c_phi)[0])
            assert AssertEquivalenceList(lp, C.PtEtaPhi(c_px, c_py, c_pz)[0])
    
    if is_cuda == False: return 
    # ===== Test the performance difference ===== #
    CPU = torch.tensor(_polar, device = "cpu")
    CUDA = torch.tensor(_cart, device = "cuda")
    c_pt, c_eta, c_phi = CUDA[:, 0].clone(), CUDA[:, 1].clone(), CUDA[:, 2].clone()
    
    t1 = time()
    cpu_r = T.PxPyPz(CPU[:, 0], CPU[:, 1], CPU[:, 2])
    diff1 = time() - t1
    
    t1 = time()
    cuda_r = C.PxPyPz(c_pt, c_eta, c_phi)
    diff2 = time() - t1
    AssertEquivalenceRecursive(cpu_r, cuda_r)
    print("--- Testing Performance Between C++ and CUDA of PxPyPz ---")
    print("Speed Factor (> 1 is better): ", diff1 / diff2)
    
    CPU = torch.tensor(_cart, device = "cpu")
    CUDA = torch.tensor(_cart, device = "cuda")
    c_px, c_py, c_pz = CUDA[:, 0].clone(), CUDA[:, 1].clone(), CUDA[:, 2].clone()
    
    t1 = time()
    cpu_r = T.PtEtaPhi(CPU[:, 0], CPU[:, 1], CPU[:, 2])
    diff1 = time() - t1
    
    t1 = time()
    cuda_r = C.PtEtaPhi(c_px, c_py, c_pz)
    diff2 = time() - t1
    AssertEquivalenceRecursive(cpu_r, cuda_r)
    print("--- Testing Performance Between C++ and CUDA of PtEtaPhi ---")
    print("Speed Factor (> 1 is better): ", diff1 / diff2)

def test_physics():
    Ana = _MakeSample()
    _x, _y, _z, _e = [], [], [], []
    _x1, _y1, _z1, _e1 = [], [], [], []
    
    pt, eta, phi, e = [], [], [], []
    pt1, eta1, phi1, e1 = [], [], [], []
    for event in Ana:
        top = event.Tops[0]
        top2 = event.Tops[1]
        tx, ty, tz, te = ParticleToTorch(top, "cuda" if is_cuda else "cpu")
        top_r = ParticleToVector(top)
        theta_r = top_r.theta
        theta_p = TC.Theta(tx, ty, tz)
        assert AssertEquivalenceRecursive([[theta_r]], theta_p) 
        
        p_r = math.sqrt(top.px**2 + top.py**2 + top.pz**2)
        p_p = TC.P(tx, ty, tz)
        assert AssertEquivalenceRecursive([[p_r]], p_p) 
    
        beta_r = top_r.beta
        beta_p = TC.Beta(tx, ty, tz, te)
        assert AssertEquivalenceRecursive([[beta_r]], beta_p) 
    
        m_r = top_r.M
        m_p = TC.M(tx, ty, tz, te)
        assert AssertEquivalenceRecursive([[m_r]], m_p) 
    
        mt_r = top_r.Mt
        mt_p = TC.Mt(tz, te)
        assert AssertEquivalenceRecursive([[mt_r]], mt_p)   
        
        for k in range(100):
            _x.append([top.px]), _y.append([top.py])
            _z.append([top.pz]), _e.append([top.e])
    
            _x1.append([top2.px]), _y1.append([top2.py])
            _z1.append([top2.pz]), _e1.append([top2.e])
    
            pt.append([top.pt]), eta.append([top.eta])
            phi.append([top.phi]), e.append([top.e])
    
            pt1.append([top2.pt]), eta1.append([top2.eta])
            phi1.append([top2.phi]), e1.append([top2.e])
     
    if is_cuda == False: return   

    print(" ======= Cartesian stuff ======= ")
    p_x = torch.tensor(_x, device = "cuda")
    p_y = torch.tensor(_y, device = "cuda")
    p_z = torch.tensor(_z, device = "cuda")
    p_e = torch.tensor(_e, device = "cuda")
    
    p_x1 = torch.tensor(_x1, device = "cuda")
    p_y1 = torch.tensor(_y1, device = "cuda")
    p_z1 = torch.tensor(_z1, device = "cuda")
    p_e1 = torch.tensor(_e1, device = "cuda")
   
    PerformanceInpt(TC.P, CC.P, p_x, p_y, p_z)
    PerformanceInpt(TC.P2, CC.P2, p_x, p_y, p_z)
    print("")
    PerformanceInpt(TC.Beta, CC.Beta, p_x, p_y, p_z, p_e)
    PerformanceInpt(TC.Beta2, CC.Beta2, p_x, p_y, p_z, p_e)
    print("") 
    PerformanceInpt(TC.M, CC.M, p_x, p_y, p_z, p_e)
    PerformanceInpt(TC.M2, CC.M2, p_x, p_y, p_z, p_e)
    print("")
    PerformanceInpt(TC.Mt, CC.Mt, p_z, p_e)
    PerformanceInpt(TC.Mt2, CC.Mt2, p_z, p_e)
    print("")
    PerformanceInpt(TC.Theta, CC.Theta, p_x, p_y, p_z)
    PerformanceInpt(TC.DeltaR, CC.DeltaR, p_x, p_x1, p_y, p_y1, p_z, p_z1)
    print("")
    
    print(" ======= Polar stuff ======= ")
    _pt = torch.tensor(pt, device = "cuda")
    _eta = torch.tensor(eta, device = "cuda")
    _phi = torch.tensor(phi, device = "cuda")
    _e = torch.tensor(e, device = "cuda")
    
    _pt2 = torch.tensor(pt1, device = "cuda")
    _eta2 = torch.tensor(eta1, device = "cuda")
    _phi2 = torch.tensor(phi1, device = "cuda")
    _e2 = torch.tensor(e1, device = "cuda")
   
    PerformanceInpt(TP.P, CP.P, _pt, _eta, _phi)
    PerformanceInpt(TP.P2, CP.P2, _pt, _eta, _phi)
    print("")
    PerformanceInpt(TP.Beta, CP.Beta, _pt, _eta, _phi, _e)
    PerformanceInpt(TP.Beta2, CP.Beta2, _pt, _eta, _phi, _e)
    print("") 
    PerformanceInpt(TP.M, CP.M, _pt, _eta, _phi, _e)
    PerformanceInpt(TP.M2, CP.M2, _pt, _eta, _phi, _e)
    print("")
    PerformanceInpt(TP.Mt, CP.Mt, _pt, _eta, _phi)
    PerformanceInpt(TP.Mt2, CP.Mt2, _pt, _eta, _phi)
    print("")
    PerformanceInpt(TP.Theta, CP.Theta, _pt, _eta, _phi)
    PerformanceInpt(TP.DeltaR, CP.DeltaR, _eta, _eta2, _phi, _phi2)

def test_operators():
    if is_cuda == False: return 
    device = "cuda"
    Matrix = [[i*j for i in range(100)] for j in range(10)]
    T_matrix = torch.tensor(Matrix, device = device, dtype = torch.float64)
    
    t1 = time()
    exp = TO.Dot(T_matrix, T_matrix)
    diff1 = time() - t1 
    
    t1 = time()
    Exp = CO.Dot(T_matrix, T_matrix)
    diff2 = time() - t1
    
    assert AssertEquivalenceRecursive(Exp.tolist(), exp.tolist())
    print("--- Testing Performance Between C++ and CUDA of DOT ---")
    print("Speed Factor (> 1 is better): ", diff1 / diff2)
    
    Matrix = [[i for i in range(3)] for j in range(100000)]
    T1_matrix = torch.tensor(Matrix, device = device, dtype = torch.float64)
    
    Matrix = [[j*i + 1 for i in range(3)] for j in range(100000)]
    T2_matrix = torch.tensor(Matrix, device = device, dtype = torch.float64)
    
    t1 = time()
    exp = TO.CosTheta(T1_matrix, T2_matrix)
    diff1 = time() - t1 
    
    t1 = time()
    Exp = CO.CosTheta(T1_matrix, T2_matrix)
    diff2 = time() - t1
    print(diff1, diff2)
    
    assert AssertEquivalenceRecursive(Exp.tolist(), exp.tolist())
    print("--- Testing Performance Between C++ and CUDA of CosTheta ---")
    print("Speed Factor (> 1 is better): ", diff1 / diff2)
    
    import math
    Matrix = [[math.pi/(j+1) for i in range(1)] for j in range(100000)]
    T1_matrix = torch.tensor(Matrix, device = device, dtype = torch.float64)
    
    t1 = time()
    exp = TO.Rx(T1_matrix)
    diff1 = time() - t1 
    
    t1 = time()
    Exp = CO.Rx(T1_matrix)
    diff2 = time() - t1
    
    print(diff1, diff2)
    assert AssertEquivalenceRecursive(Exp.tolist(), exp.tolist())
    print("--- Testing Performance Between C++ and CUDA of Rx ---")
    print("Speed Factor (> 1 is better): ", diff1 / diff2)
    
    t1 = time()
    exp = TO.Ry(T1_matrix)
    diff1 = time() - t1 
    
    t1 = time()
    Exp = CO.Ry(T1_matrix)
    diff2 = time() - t1
    
    assert AssertEquivalenceRecursive(Exp.tolist(), exp.tolist())
    print("--- Testing Performance Between C++ and CUDA of Ry ---")
    print("Speed Factor (> 1 is better): ", diff1 / diff2)
    
    t1 = time()
    exp = TO.Rz(T1_matrix)
    diff1 = time() - t1 
    
    t1 = time()
    Exp = CO.Rz(T1_matrix)
    diff2 = time() - t1
    
    assert AssertEquivalenceRecursive(Exp.tolist(), exp.tolist())
    print("--- Testing Performance Between C++ and CUDA of Rz ---")
    print("Speed Factor (> 1 is better): ", diff1 / diff2)
    
    T1_matrix = torch.tensor([[i for i in range(3)] for j in range(1)], device = "cuda", dtype = torch.float64)
    x = CO.Rz(T1_matrix)
    y = torch.tensor([[[k for i in range(1)] for k in range(3)] for t in range(1)], device = "cuda", dtype = torch.float64)
    
    t1 = time()
    c = torch.matmul(x, y)
    diff1 = time() - t1 
    
    t1 = time()
    l = CO.Mul(x, y)
    diff2 = time() - t1
    
    assert AssertEquivalenceRecursive(c.tolist(), l.tolist())
    print("--- Testing Performance Between C++ and CUDA of Mul ---")
    print("Speed Factor (> 1 is better): ", diff1 / diff2)
    
    y = torch.tensor([[[k/(1+t) for i in range(3)] for k in range(3)] for t in range(100)], device = "cuda", dtype = torch.float64)
    x = torch.tensor([[[i*k for i in range(3)] for k in range(3)] for t in range(100)], device = "cuda", dtype = torch.float64)
    diff = [[], []]
    for t in range(10000):
        t1 = time()
        c = torch.matmul(x, y)
        t2 = time()
        diff1 = t2 - t1 
        diff[0].append(diff1)
        
        t1 = time()
        l = CO.Mul(x, y)
        t2 = time()
        diff2 = t2 - t1
        diff[1].append(diff2)
    
    assert AssertEquivalenceRecursive(c.tolist(), l.tolist())
    print("--- Testing Performance Between C++ and CUDA of MatMul ---")
    print("Speed Factor (> 1 is better): ", (sum(diff[0])) / sum(diff[1]))
    
    m = torch.tensor([[[1/(i+1), 4, 7/(i+1)], [3, i, 5], [-1/(i+1), 9, 1]] for i in range(10000)], dtype = torch.float64, device = "cuda")
    diff = [[], []]
    for t in range(10000):
        t1 = time()
        c = torch.inverse(m)
        t2 = time()
        diff1 = t2 - t1 
        diff[0].append(diff1)
        
        t1 = time()
        l = CO.Inv(m)
        t2 = time()
        diff2 = t2 - t1
        diff[1].append(diff2)

    assert AssertEquivalenceRecursive(c.tolist(), l.tolist())
    print("--- Testing Performance Between C++ and CUDA of Inverse ---")
    print("Speed Factor (> 1 is better): ", (sum(diff[0])) / sum(diff[1]))




if __name__ == "__main__":
    #test_transformation()
    #test_physics()
    #test_operators()

    pass
