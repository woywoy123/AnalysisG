import torch
import PyC.Vectors.Floats as F
import PyC.Vectors.Tensors as T
import PyC.Vectors.CUDA as C
from AnalysisTopGNN import Analysis
from AnalysisTopGNN.Events import Event
from time import time
from Checks import *

Ana = Analysis()
Ana.InputSample("bsm4t-1000", "/home/tnom6927/Downloads/samples/tttt/DAOD_TOPQ1.21955717._000003.root")
Ana.Event = Event 
Ana.EventStop = 100
Ana.EventCache = True 
Ana.DumpPickle = True 
Ana.Launch()

_tmp = []
for ev in Ana:
    event = ev.Trees["nominal"]
    top = event.Tops[0]

    AssertEquivalence(top._px, F.Px(top.pt, top.phi))
    AssertEquivalence(top._py, F.Py(top.pt, top.phi))
    AssertEquivalence(top._pz, F.Pz(top.pt, top.eta))
   
    t_pt = MakeTensor(top.pt)
    t_eta = MakeTensor(top.eta)
    t_phi = MakeTensor(top.phi)

    c_pt = MakeTensor(top.pt, "cuda")
    c_eta = MakeTensor(top.eta, "cuda")
    c_phi = MakeTensor(top.phi, "cuda")

    AssertEquivalence(top._px, T.Px(t_pt, t_phi))
    AssertEquivalence(top._py, T.Py(t_pt, t_phi))
    AssertEquivalence(top._pz, T.Pz(t_pt, t_eta))

    AssertEquivalence(top._px, C.Px(c_pt, c_phi))
    AssertEquivalence(top._py, C.Py(c_pt, c_phi))
    AssertEquivalence(top._pz, C.Pz(c_pt, c_eta))

    l = [top._px, top._py, top._pz]
    AssertEquivalenceList(l, F.PxPyPz(top.pt, top.eta, top.phi))
    AssertEquivalenceList(l, T.PxPyPz(t_pt, t_eta, t_phi)[0])
    AssertEquivalenceList(l, C.PxPyPz(c_pt, c_eta, c_phi)[0])
   
    for c in event.TopChildren:
        _tmp.append([c.pt, c.eta, c.phi])


# ===== Test the performance difference ===== #
CPU = torch.tensor(_tmp, device = "cpu")
CUDA = torch.tensor(_tmp, device = "cuda")
c_pt = CUDA[:, 0].clone()
c_eta = CUDA[:, 1].clone()
c_phi = CUDA[:, 2].clone()

t1 = time()
cpu_r = T.PxPyPz(CPU[:, 0], CPU[:, 1], CPU[:, 2])
diff1 = time() - t1

t1 = time()
cuda_r = C.PxPyPz(c_pt, c_eta, c_phi)
diff2 = time() - t1

AssertEquivalenceRecursive(cpu_r, cuda_r)

print("--- Testing Performance Between C++ and CUDA of PxPyPz ---")
print("Speed Factor (> 1 is better): ", diff1 / diff2)

