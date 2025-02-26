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

    dcc = torch.tensor(test_ct, device = device, dtype = torch.float64)
    dcp = torch.tensor(test_pt, device = device, dtype = torch.float64)
    return {"cartesian" : dcc, "polar" : dcp}

def performance(cpu, tcuda, cuda):
    def error(lst1, lst2):
        m1 = sum(lst1)/len(lst1)
        m2 = sum(lst2)/len(lst2)
        sig1 = sum([(m1 - x)**2 for x in lst1])/len(lst1)
        sig2 = sum([(m2 - x)**2 for x in lst2])/len(lst2)
        err = ((m1/m2)**2)*(sig1/(m1**2) + sig2/(m2**2))
        return (m1/m2), math.sqrt(err)

    def single_error(lst):
        m1 = sum(lst)/len(lst)
        sig1 = sum([(m1 - x)**2 for x in lst])/len(lst)
        return m1, math.sqrt(sig1)

    datac1 = single_error(cpu)
    datac2 = single_error(tcuda)
    datac3 = single_error(cuda)

    data1  = error(cpu, tcuda)
    data2  = error(cpu, cuda)
    data3  = error(tcuda, cuda)
    out = {}
    out |= {"cpu"     : [datac1[0]], "sig(cpu)"             : [datac1[1]]}
    out |= {"cuda(t)" : [datac2[0]], "sig(cuda(t))"         : [datac2[1]]}
    out |= {"cuda(k)" : [datac3[0]], "sig(cuda(k))"         : [datac3[1]]}

    out |= {"cpu/cuda(t)"     : [data1[0]], "sig(cpu/cuda(t))"     : [data1[1]]}
    out |= {"cpu/cuda(k)"     : [data2[0]], "sig(cpu/cuda(k))"     : [data2[1]]}
    out |= {"cuda(t)/cuda(k)" : [data3[0]], "sig(cuda(t)/cuda(k))" : [data3[1]]}
    return out


def merge(data1, data2):
    if not len(data1): return data2
    data1["cpu"]                  += data2["cpu"]
    data1["cuda(t)"]              += data2["cuda(t)"]
    data1["cuda(k)"]              += data2["cuda(k)"]
    data1["sig(cpu)"]             += data2["sig(cpu)"]
    data1["sig(cuda(t))"]         += data2["sig(cuda(t))"]
    data1["sig(cuda(k))"]         += data2["sig(cuda(k))"]

    data1["cpu/cuda(t)"]          += data2["cpu/cuda(t)"    ]
    data1["cpu/cuda(k)"]          += data2["cpu/cuda(k)"    ]
    data1["cuda(t)/cuda(k)"]      += data2["cuda(t)/cuda(k)"]
    data1["sig(cpu/cuda(t))"]     += data2["sig(cpu/cuda(t))"]
    data1["sig(cpu/cuda(k))"]     += data2["sig(cpu/cuda(k))"]
    data1["sig(cuda(t)/cuda(k))"] += data2["sig(cuda(t)/cuda(k))"]
    data1["dx"]                   += data2["dx"]
    return data1

def Line(title, xdata = None, ydata = None, err = None, col = None, linst = None, restrict = 2):
    tl = TLine()
    tl.Title = title
    tl.LineWidth = 0.5
    tl.ErrorBars = False
    if xdata is not None: tl.xData = [xdata[i] / 1000 for i in range(len(xdata))]
    if ydata is not None: tl.yData = [ydata[i] for i in range(len(xdata))]
    if err is not None: tl.yDataUp   = [err[i]*(1/math.sqrt(10000-1)) for i in range(len(xdata))]
    if err is not None: tl.yDataDown = [err[i]*(1/math.sqrt(10000-1)) for i in range(len(xdata))]
    if col is not None: tl.Color = col
    if linst is not None: tl.LineStyle = linst
    if xdata is None: tl.xTitle = "Length of Tensor - (Units of 1000)"
    if xdata is None: tl.yTitle = "CUDA Performance Ratio (Tensor / Kernel) - Higher is Better"
    return [tl] if xdata is not None else tl

def MakeLines(data, dev, col = "red"):
    lins = []
#    lins += Line("Tensor - (" + dev + ")" , data["dx"], data["cpu/cuda(t)"], data["sig(cpu/cuda(t))"], col, ":", 10)
#    lins += Line("Kernel - (" + dev + ")" , data["dx"], data["cpu/cuda(k)"], data["sig(cpu/cuda(k))"], col, "-", 10)
    if   dev == "a100": dv = "A100"
    elif dev == "a30" : dv = "A30 "
    elif dev == "h100": dv = "H100"
    elif dev == "v100": dv = "V100"
    else: dv = dev
    lins += Line(dv, data["dx"], data["cuda(t)/cuda(k)"], data["sig(cuda(t)/cuda(k))"], col, ":", 10)
    return lins

def repeat(fx, data, iters, dx):
    cpu, tcuda, cuda = [], [], []
    for _ in range(iters):
        c, tc, cu = fx(data)
        cpu.append(c)
        tcuda.append(tc)
        cuda.append(cu)
    return {"dx" : [dx]} | performance(cpu, tcuda, cuda)
