import matplotlib
matplotlib.use("Agg")
from AnalysisG.IO import PickleObject, UnpickleObject
from AnalysisG.Plotting import TLine
from random import random
from time import time
import statistics
import vector
import math
import torch
import pyc

device = "cuda"
nums = 100

def create_vector_polar(number):
    lx = []
    for _ in range(number): lx.append(vector.obj(pt = random()*1000, eta = random()*4, phi = 2*math.pi*random(), e = random()*1000))
    return lx

def create_vector_cartesian(number):
    lx = []
    for _ in range(number):
        v = vector.obj(px = random()*1000, py = random()*1000, pz = random()*1000, e = random()*1000)
        lx.append(v)
    return lx

def p2_loop_p(ten):
    lst = []
    for _ in range(1000):
        t1 = time()
        _ = pyc.Physics.Polar.P2(ten)
        lst.append(time() - t1)
    return lst

def p_loop_p(ten):
    lst = []
    for _ in range(1000):
        t1 = time()
        _ = pyc.Physics.Polar.P(ten)
        lst.append(time() - t1)
    return lst

def beta2_loop_p(ten):
    lst = []
    for _ in range(1000):
        t1 = time()
        _ = pyc.Physics.Polar.Beta2(ten)
        lst.append(time() - t1)
    return lst

def beta_loop_p(ten):
    lst = []
    for _ in range(1000):
        t1 = time()
        _ = pyc.Physics.Polar.Beta(ten)
        lst.append(time() - t1)
    return lst


def M2_loop_p(ten):
    lst = []
    for _ in range(1000):
        t1 = time()
        _ = pyc.Physics.Polar.M2(ten)
        lst.append(time() - t1)
    return lst

def M_loop_p(ten):
    lst = []
    for _ in range(1000):
        t1 = time()
        _ = pyc.Physics.Polar.M(ten)
        lst.append(time() - t1)
    return lst

def theta_loop_p(ten):
    lst = []
    for _ in range(1000):
        t1 = time()
        _ = pyc.Physics.Polar.Theta(ten)
        lst.append(time() - t1)
    return lst

def deltaR_loop_p(t1_, t2_):
    prm = torch.randperm(t1_.size()[0])
    _t1, _t2 = t1_[prm], t2_[prm]
    lst1, lst2 = [], []
    for _ in range(1000):
        t1 = time()
        _ = pyc.Physics.Polar.DeltaR(_t1, t1_)
        lst1.append(time() - t1)

        t1 = time()
        _ = pyc.Physics.Polar.DeltaR(_t2, t2_)
        lst2.append(time() - t1)
    return lst1, lst2



def p2_loop_c(ten):
    lst = []
    for _ in range(1000):
        t1 = time()
        _ = pyc.Physics.Cartesian.P2(ten)
        lst.append(time() - t1)
    return lst

def p_loop_c(ten):
    lst = []
    for _ in range(1000):
        t1 = time()
        _ = pyc.Physics.Cartesian.P(ten)
        lst.append(time() - t1)
    return lst

def beta2_loop_c(ten):
    lst = []
    for _ in range(1000):
        t1 = time()
        _ = pyc.Physics.Cartesian.Beta2(ten)
        lst.append(time() - t1)
    return lst

def beta_loop_c(ten):
    lst = []
    for _ in range(1000):
        t1 = time()
        _ = pyc.Physics.Cartesian.Beta(ten)
        lst.append(time() - t1)
    return lst


def M2_loop_c(ten):
    lst = []
    for _ in range(1000):
        t1 = time()
        _ = pyc.Physics.Cartesian.M2(ten)
        lst.append(time() - t1)
    return lst

def M_loop_c(ten):
    lst = []
    for _ in range(1000):
        t1 = time()
        _ = pyc.Physics.Cartesian.M(ten)
        lst.append(time() - t1)
    return lst

def theta_loop_c(ten):
    lst = []
    for _ in range(1000):
        t1 = time()
        _ = pyc.Physics.Cartesian.Theta(ten)
        lst.append(time() - t1)
    return lst

def deltaR_loop_c(t1_, t2_):
    prm = torch.randperm(t1_.size()[0])
    _t1, _t2 = t1_[prm], t2_[prm]
    lst1, lst2 = [], []
    for _ in range(1000):
        t1 = time()
        _ = pyc.Physics.Cartesian.DeltaR(_t1, t1_)
        lst1.append(time() - t1)

        t1 = time()
        _ = pyc.Physics.Cartesian.DeltaR(_t2, t2_)
        lst2.append(time() - t1)
    return lst1, lst2

def physics_polar():

    data = {}

    step_ = 100
    t_cpu, t_cu = None, None
    for i in range(nums):
        lx = create_vector_polar(step_)
        _cpu = torch.tensor([[v.pt, v.eta, v.phi, v.e] for v in lx], device = "cpu", dtype = torch.float64)
        _cu = _cpu.to(device = device)
        if t_cpu is None:
            t_cpu = _cpu
            t_cu = _cu
        else:
            t_cpu = torch.cat([t_cpu, _cpu], 0)
            t_cu = torch.cat([t_cu, _cu], 0)

        lex = t_cpu.size()[0]
        data[lex] = {}

        cpu, cu = p2_loop_p(t_cpu), p2_loop_p(t_cu)
        data[lex]["p2"] = statistics.mean(cpu) / statistics.mean(cu)

        cpu, cu = p_loop_p(t_cpu), p2_loop_p(t_cu)
        data[lex]["p"] = statistics.mean(cpu) / statistics.mean(cu)

        cpu, cu = beta2_loop_p(t_cpu), p2_loop_p(t_cu)
        data[lex]["beta2"] = statistics.mean(cpu) / statistics.mean(cu)

        cpu, cu = beta_loop_p(t_cpu), p2_loop_p(t_cu)
        data[lex]["beta"] = statistics.mean(cpu) / statistics.mean(cu)

        cpu, cu = M2_loop_p(t_cpu), p2_loop_p(t_cu)
        data[lex]["m2"] = statistics.mean(cpu) / statistics.mean(cu)

        cpu, cu = M_loop_p(t_cpu), p2_loop_p(t_cu)
        data[lex]["m"] = statistics.mean(cpu) / statistics.mean(cu)

        cpu, cu = theta_loop_p(t_cpu), p2_loop_p(t_cu)
        data[lex]["theta"] = statistics.mean(cpu) / statistics.mean(cu)

        cpu, cu = deltaR_loop_p(t_cpu, t_cu)
        data[lex]["delR"] = statistics.mean(cpu) / statistics.mean(cu)


        print("-> ", i)
        if i % 100 != 0: continue
        PickleObject(data, "polar-physics")

    PickleObject(data, "polar-physics")

def plot_polar():

    x = UnpickleObject("polar-physics")
    tline = TLine()
    title = "Computational Time Ratio Between Torch (CPU) and CUDA Native for Various\n"
    title += "Physics Quantities as a Function of number of Particles (higher is better)"
    tline.Title = title

    t1 = TLine()
    t1.Title = "Polar $\\rightarrow P^2$"
    t1.xData = list(x)
    t1.yData = [x[t]["p2"] for t in list(x)]

    t2 = TLine()
    t2.Title = "Polar $\\rightarrow P$"
    t2.xData = list(x)
    t2.yData = [x[t]["p"] for t in list(x)]

    t3 = TLine()
    t3.Title = "Polar $\\rightarrow \\beta^2$"
    t3.xData = list(x)
    t3.yData = [x[t]["beta2"] for t in list(x)]

    t4 = TLine()
    t4.Title = "Polar $\\rightarrow \\beta$"
    t4.xData = list(x)
    t4.yData = [x[t]["beta"] for t in list(x)]

    t5 = TLine()
    t5.Title = "Polar $\\rightarrow mass^2$"
    t5.xData = list(x)
    t5.yData = [x[t]["m2"] for t in list(x)]

    t6 = TLine()
    t6.Title = "Polar $\\rightarrow mass$"
    t6.xData = list(x)
    t6.yData = [x[t]["m"] for t in list(x)]

    t7 = TLine()
    t7.Title = "Polar $\\rightarrow \\theta$"
    t7.xData = list(x)
    t7.yData = [x[t]["theta"] for t in list(x)]

    t8 = TLine()
    t8.Title = "Polar $\\rightarrow \\Delta R$"
    t8.xData = list(x)
    t8.yData = [x[t]["delR"] for t in list(x)]

    tline.Lines = [t1, t2, t3, t4, t5, t6, t7, t8]
    tline.LineWidth = 0.2
    tline.Colors = ["r", "r", "b", "b", "y", "y", "m", "k"]
    tline.Markers = ["x", "*", "x", "*", "x", "*", "x", "x"]
    tline.xTitle = "Number of Particles"
    tline.yTitle = "Ratio Time of Computation (Torch-CPU / Torch-CUDA)"
    tline.yMin = 0
    tline.xStep = 100
    tline.Filename = "Polar-To-Physics"
    tline.SaveFigure()



def physics_cartesian():

    data = {}

    step_ = 100
    t_cpu, t_cu = None, None
    for i in range(nums):
        lx = create_vector_cartesian(step_)
        _cpu = torch.tensor([[v.px, v.py, v.pz, v.e] for v in lx], device = "cpu", dtype = torch.float64)
        _cu = _cpu.to(device = device)
        if t_cpu is None:
            t_cpu = _cpu
            t_cu = _cu
        else:
            t_cpu = torch.cat([t_cpu, _cpu], 0)
            t_cu = torch.cat([t_cu, _cu], 0)

        lex = t_cpu.size()[0]
        data[lex] = {}

        cpu, cu = p2_loop_c(t_cpu), p2_loop_c(t_cu)
        data[lex]["p2"] = statistics.mean(cpu) / statistics.mean(cu)

        cpu, cu = p_loop_c(t_cpu), p2_loop_c(t_cu)
        data[lex]["p"] = statistics.mean(cpu) / statistics.mean(cu)

        cpu, cu = beta2_loop_c(t_cpu), p2_loop_c(t_cu)
        data[lex]["beta2"] = statistics.mean(cpu) / statistics.mean(cu)

        cpu, cu = beta_loop_c(t_cpu), p2_loop_c(t_cu)
        data[lex]["beta"] = statistics.mean(cpu) / statistics.mean(cu)

        cpu, cu = M2_loop_c(t_cpu), p2_loop_c(t_cu)
        data[lex]["m2"] = statistics.mean(cpu) / statistics.mean(cu)

        cpu, cu = M_loop_c(t_cpu), p2_loop_c(t_cu)
        data[lex]["m"] = statistics.mean(cpu) / statistics.mean(cu)

        cpu, cu = theta_loop_c(t_cpu), p2_loop_c(t_cu)
        data[lex]["theta"] = statistics.mean(cpu) / statistics.mean(cu)

        cpu, cu = deltaR_loop_c(t_cpu, t_cu)
        data[lex]["delR"] = statistics.mean(cpu) / statistics.mean(cu)

        print("-> ", i)
        if i % 100 != 0: continue
        PickleObject(data, "cartesian-physics")

    PickleObject(data, "cartesian-physics")

def plot_cartesian():

    x = UnpickleObject("cartesian-physics")
    tline = TLine()
    title = "Computational Time Ratio Between Torch (CPU) and CUDA Native for Various\n"
    title += "Physics Quantities as a Function of number of Particles (higher is better)"
    tline.Title = title

    t1 = TLine()
    t1.Title = "Cartesian $\\rightarrow P^2$"
    t1.xData = list(x)
    t1.yData = [x[t]["p2"] for t in list(x)]

    t2 = TLine()
    t2.Title = "Cartesian $\\rightarrow P$"
    t2.xData = list(x)
    t2.yData = [x[t]["p"] for t in list(x)]

    t3 = TLine()
    t3.Title = "Cartesian $\\rightarrow \\beta^2$"
    t3.xData = list(x)
    t3.yData = [x[t]["beta2"] for t in list(x)]

    t4 = TLine()
    t4.Title = "Cartesian $\\rightarrow \\beta$"
    t4.xData = list(x)
    t4.yData = [x[t]["beta"] for t in list(x)]

    t5 = TLine()
    t5.Title = "Cartesian $\\rightarrow mass^2$"
    t5.xData = list(x)
    t5.yData = [x[t]["m2"] for t in list(x)]

    t6 = TLine()
    t6.Title = "Cartesian $\\rightarrow mass$"
    t6.xData = list(x)
    t6.yData = [x[t]["m"] for t in list(x)]

    t7 = TLine()
    t7.Title = "Cartesian $\\rightarrow \\theta$"
    t7.xData = list(x)
    t7.yData = [x[t]["theta"] for t in list(x)]

    t8 = TLine()
    t8.Title = "Cartesian $\\rightarrow \\Delta R$"
    t8.xData = list(x)
    t8.yData = [x[t]["delR"] for t in list(x)]

    tline.Lines = [t1, t2, t3, t4, t5, t6, t7, t8]
    tline.LineWidth = 0.2
    tline.Colors = ["r", "r", "b", "b", "y", "y", "m", "k"]
    tline.Markers = ["x", "*", "x", "*", "x", "*", "x", "x"]
    tline.xTitle = "Number of Particles"
    tline.yTitle = "Ratio Time of Computation (Torch-CPU / Torch-CUDA)"
    tline.yMin = 0
    tline.xStep = 100
    tline.Filename = "Cartesian-To-Physics"
    tline.SaveFigure()

