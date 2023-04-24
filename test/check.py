import torch
import vector
from time import time
import numpy as np 
mW = 80.385*1000 # MeV : W Boson Mass
mN = 0           # GeV : Neutrino Mass

def MakeTensor(inpt, device = "cpu"):
    return torch.tensor([inpt], device = device)

def AssertEquivalence(truth, pred, threshold = 0.0001):
    diff = abs(truth - pred)
    if truth == 0: truth += 1
    diff = abs((diff/truth))*100
    if diff < threshold: return True 
    if truth < 1e-12 and pred < 1e-12: return True # Getting to machine precision difference 
    print("-> ", diff, truth, pred)
    return False

def AssertEquivalenceList(truth, pred, threshold = 0.0001):
    for i, j in zip(truth, pred):
        if AssertEquivalence(i, j): continue
        return False
    return True

def AssertEquivalenceRecursive(truth, pred, threshold = 0.001):
    try: return AssertEquivalence(float(truth), float(pred), threshold)
    except:
        for i, j in zip(truth, pred):
            if AssertEquivalenceRecursive(i, j, threshold): continue
            return False 
        return True 

def ParticleToTorch(part, device = "cuda"):
    tx = torch.tensor([[part.px]], device = device)
    ty = torch.tensor([[part.py]], device = device)
    tz = torch.tensor([[part.pz]], device = device)
    te = torch.tensor([[part.e]], device = device) 
    return tx, ty, tz, te

def ParticleToVector(part):
    return vector.obj(pt = part.pt, eta = part.eta, phi = part.phi, E = part.e)

def PerformanceInpt(varT, varC, v1 = None, v2 = None, v3 = None, v4 = None, v5 = None, v6 = None):

    if v5 != None and v6 != None:
        t1 = time()
        rest = varT(v1, v2, v3, v4, v5, v6)
        diff1 = time() - t1

        t1 = time()
        resc = varC(v1, v2, v3, v4, v5, v6)
        diff2 = time() - t1
     
    elif v3 != None and v4 != None:
        t1 = time()
        rest = varT(v1, v2, v3, v4)
        diff1 = time() - t1

        t1 = time()
        resc = varC(v1, v2, v3, v4)
        diff2 = time() - t1
 
    elif v3 != None and v4 is None: 
        t1 = time()
        rest = varT(v1, v2, v3)
        diff1 = time() - t1

        t1 = time()
        resc = varC(v1, v2, v3)
        diff2 = time() - t1

    elif v3 is None and v4 is None: 
        t1 = time()
        rest = varT(v1, v2)
        diff1 = time() - t1

        t1 = time()
        resc = varC(v1, v2)
        diff2 = time() - t1
    
    print("--- Testing Performance Between C++ and CUDA of " + varT.__name__ + " ---")
    print("Speed Factor (> 1 is better): ", diff1 / diff2)
    assert AssertEquivalenceRecursive(resc, rest)
