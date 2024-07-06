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

def transform_cartesian():

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
        cpu, cu = [], []
        for _ in range(1000):
            t1 = time()
            _ = pyc.Transform.Px(t_cpu)
            cpu.append(time() - t1)

            t1 = time()
            _ = pyc.Transform.Px(t_cu)
            cu.append(time() - t1)

        data[lex] = {}
        data[lex]["px"] = statistics.mean(cpu) / statistics.mean(cu)

        cpu, cu = [], []
        for _ in range(1000):
            t1 = time()
            _ = pyc.Transform.Py(t_cpu)
            cpu.append(time() - t1)

            t1 = time()
            _ = pyc.Transform.Py(t_cu)
            cu.append(time() - t1)

        data[lex]["py"] = statistics.mean(cpu) / statistics.mean(cu)

        cpu, cu = [], []
        for _ in range(1000):
            t1 = time()
            _ = pyc.Transform.Pz(t_cpu)
            cpu.append(time() - t1)

            t1 = time()
            _ = pyc.Transform.Pz(t_cu)
            cu.append(time() - t1)

        data[lex]["pz"] = statistics.mean(cpu) / statistics.mean(cu)

        cpu, cu = [], []
        for _ in range(1000):
            t1 = time()
            _ = pyc.Transform.PxPyPz(t_cpu)
            cpu.append(time() - t1)

            t1 = time()
            _ = pyc.Transform.PxPyPz(t_cu)
            cu.append(time() - t1)

        data[lex]["pxpypz"] = statistics.mean(cpu) / statistics.mean(cu)

        print("-> ", i)
        if i % 100 != 0: continue
        PickleObject(data, "cartesian-transform")

    PickleObject(data, "cartesian-transform")



def transform_polar():

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
        cpu, cu = [], []
        for _ in range(1000):
            t1 = time()
            _ = pyc.Transform.Pt(t_cpu)
            cpu.append(time() - t1)

            t1 = time()
            _ = pyc.Transform.Pt(t_cu)
            cu.append(time() - t1)

        data[lex] = {}
        data[lex]["pt"] = statistics.mean(cpu) / statistics.mean(cu)

        cpu, cu = [], []
        for _ in range(1000):
            t1 = time()
            _ = pyc.Transform.Eta(t_cpu)
            cpu.append(time() - t1)

            t1 = time()
            _ = pyc.Transform.Eta(t_cu)
            cu.append(time() - t1)

        data[lex]["eta"] = statistics.mean(cpu) / statistics.mean(cu)

        cpu, cu = [], []
        for _ in range(1000):
            t1 = time()
            _ = pyc.Transform.Phi(t_cpu)
            cpu.append(time() - t1)

            t1 = time()
            _ = pyc.Transform.Phi(t_cu)
            cu.append(time() - t1)

        data[lex]["phi"] = statistics.mean(cpu) / statistics.mean(cu)

        cpu, cu = [], []
        for _ in range(1000):
            t1 = time()
            _ = pyc.Transform.PtEtaPhi(t_cpu)
            cpu.append(time() - t1)

            t1 = time()
            _ = pyc.Transform.PtEtaPhi(t_cu)
            cu.append(time() - t1)

        data[lex]["ptetaphi"] = statistics.mean(cpu) / statistics.mean(cu)

        print("-> ", i)
        if i % 100 != 0: continue
        PickleObject(data, "polar-transform")

    PickleObject(data, "polar-transform")

def plot_transform_cartesian():

    x = UnpickleObject("cartesian-transform")
    tline = TLine()
    title = "Computational Time Ratio Between Torch (CPU) and CUDA Native for Various\n"
    title += "Physics Operations as a Function of number of Particles (higher is better)"
    tline.Title = title

    tx = TLine()
    tx.Title = "Polar $\\rightarrow P_x$"
    tx.xData = list(x)
    tx.yData = [x[t]["px"] for t in list(x)]

    ty = TLine()
    ty.Title = "Polar $\\rightarrow P_y$"
    ty.xData = list(x)
    ty.yData = [x[t]["py"] for t in list(x)]

    tz = TLine()
    tz.Title = "Polar $\\rightarrow P_z$"
    tz.xData = list(x)
    tz.yData = [x[t]["pz"] for t in list(x)]

    tcom = TLine()
    tcom.Title = "Polar $\\rightarrow P_{x}P_{y}P_{z}$"
    tcom.xData = list(x)
    tcom.yData = [x[t]["pxpypz"] for t in list(x)]

    tline.Lines = [tx, ty, tz, tcom]
    tline.xTitle = "Number of Particles"
    tline.yTitle = "Ratio Time of Computation (Torch-CPU / Torch-CUDA)"
    tline.yMin = 0
    tline.xStep = 100
    tline.Filename = "PolarToCartesian"
    tline.SaveFigure()

def plot_transform_polar():

    x = UnpickleObject("polar-transform")
    tline = TLine()
    title = "Computational Time Ratio Between Torch (CPU) and CUDA Native for Various\n"
    title += "Physics Operations as a Function of number of Particles (higher is better)"
    tline.Title = title

    tx = TLine()
    tx.Title = "Cartesian $\\rightarrow P_T$"
    tx.xData = list(x)
    tx.yData = [x[t]["pt"] for t in list(x)]

    ty = TLine()
    ty.Title = "Cartesian $\\rightarrow \\eta$"
    ty.xData = list(x)
    ty.yData = [x[t]["eta"] for t in list(x)]

    tz = TLine()
    tz.Title = "Cartesian $\\rightarrow \\phi$"
    tz.xData = list(x)
    tz.yData = [x[t]["phi"] for t in list(x)]

    tcom = TLine()
    tcom.Title = "Cartesian $\\rightarrow P_{T},\\eta,\\phi$"
    tcom.xData = list(x)
    tcom.yData = [x[t]["ptetaphi"] for t in list(x)]

    tline.Lines = [tx, ty, tz, tcom]
    tline.xTitle = "Number of Particles"
    tline.yTitle = "Ratio Time of Computation (Torch-CPU / Torch-CUDA)"
    tline.yMin = 0
    tline.xStep = 100
    tline.Filename = "CartesianToPolar"
    tline.SaveFigure()

