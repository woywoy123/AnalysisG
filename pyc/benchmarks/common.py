import sys
sys.path.append("../build/pyc/")
from core.plotting import TLine, TH1F

import random
import vector
import torch
import time
import math

def makematrix(rows, cols, device): return torch.rand((rows, cols), device = device, dtype = torch.float64)
def create_vector_cartesian(px, py, pz, e): return vector.obj(px = px, py = py, pz = pz, E = e)
def create_vector_polar(pt, eta, phi, e): return vector.obj(pt = pt, eta = eta, phi = phi, E = e)

def create_particle(nums = 100, device = "cpu"):
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

    dcc = torch.tensor(test_ct, device = device, dtype = torch.float64)
    dcp = torch.tensor(test_pt, device = device, dtype = torch.float64)
    return {"cartesian" : dcc, "polar" : dcp}

def performance(cpu, tcuda, cuda):
    def error(lst1, lst2):
        m1 = sum(lst1)/len(lst1)
        m2 = sum(lst2)/len(lst2)
        sig1 = sum([(m1 - x)**2 for x in lst1])/len(lst1)
        sig2 = sum([(m2 - x)**2 for x in lst2])/len(lst2)
        err = ((m1/m2)**2)*(sig1/m1 + sig2/m2)
        return (m1/m2), math.sqrt(err)
    data1 = error(cpu, tcuda)
    data2 = error(cpu, cuda)
    data3 = error(tcuda, cuda)
    out = {}
    out |= {"cpu/cuda(t)"     : [data1[0]], "sig(cpu/cuda(t))"     : [data1[1]]}
    out |= {"cpu/cuda(k)"     : [data2[0]], "sig(cpu/cuda(k))"     : [data2[1]]}
    out |= {"cuda(t)/cuda(k)" : [data3[0]], "sig(cuda(t)/cuda(k))" : [data3[1]]}
    return out

def merge(data1, data2):
    if not len(data1): return data2
    data1["cpu/cuda(t)"]          += data2["cpu/cuda(t)"    ]
    data1["sig(cpu/cuda(t))"]     += data2["sig(cpu/cuda(t))"]
    data1["cpu/cuda(k)"]          += data2["cpu/cuda(k)"    ]
    data1["sig(cpu/cuda(k))"]     += data2["sig(cpu/cuda(k))"]
    data1["cuda(t)/cuda(k)"]      += data2["cuda(t)/cuda(k)"]
    data1["sig(cuda(t)/cuda(k))"] += data2["sig(cuda(t)/cuda(k))"]
    data1["dx"]                   += data2["dx"]
    return data1


def Line(title, xdata = None, ydata = None, err = None, col = None, linst = None):
    tl = TLine()
    tl.Title = title
    if xdata is not None: tl.xData = xdata
    if ydata is not None: tl.yData = ydata
    if err is not None: tl.yDataUp   = err
    if err is not None: tl.yDataDown = err
    if col is not None: tl.Color = col
    if linst is not None: tl.LineStyle = linst
    if xdata is None: tl.xTitle = "Length of Tensor (Arb.)"
    if xdata is None: tl.yTitle = "Ratio (See Legend) - Higher is Better"
    tl.xMin = 0
    tl.yMin = 0
    tl.Style = "ATLAS"
    tl.ErrorBars = True
    return [tl] if xdata is not None else tl

def MakeLines(data):
    lins = []
    lins += Line("Tensor (CPU)/Tensor (CUDA)"   , data["dx"], data["cpu/cuda(t)"], data["sig(cpu/cuda(t))"], "red", "--")
    lins += Line("Tensor (CPU)/Kernel (CUDA)"   , data["dx"], data["cpu/cuda(k)"], data["sig(cpu/cuda(k))"], "blue", "-.")
    lins += Line("Tensor (CUDA) / Kernel (CUDA)", data["dx"], data["cuda(t)/cuda(k)"], data["sig(cuda(t)/cuda(k))"], "green", ":")
    return lins

def repeat(fx, data, iters, dx):
    cpu, tcuda, cuda = [], [], []
    for _ in range(iters):
        c, tc, cu = fx(data)
        cpu.append(c)
        tcuda.append(tc)
        cuda.append(cu)
    return {"dx" : [dx]} | performance(cpu, tcuda, cuda)
