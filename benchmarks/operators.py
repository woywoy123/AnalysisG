import matplotlib
matplotlib.use("Agg")
from AnalysisG.IO import PickleObject, UnpickleObject
from AnalysisG.Plotting import TLine, TH2F
from random import random
from time import time
import statistics
import torch
import pyc

device = "cuda"
nums = 10

def _makeMatrix(l, m, n, tmp):
    x = torch.tensor(
            [[[random() for i in range(tmp)] for k in range(n)] for t in range(l)],
            device = device, dtype = torch.float64)

    y = torch.tensor(
            [[[random() for i in range(m)] for k in range(tmp)] for t in range(l)],
            device = device, dtype = torch.float64)
    return x, y

def loop_mul(ten1, ten2):
    tch, pc = [], []
    t1 = time()
    _ = ten1.matmul(ten2)
    tch.append(time() - t1)

    t1 = time()
    _ = pyc.Operators.Mul(ten1, ten2)
    pc.append(time() - t1)
    return tch, pc

def matrix_multiplication():

    data = {}
    for n_s in range(10):
        n_matrix = 10*(n_s +1)

        for r_s in range(2):
            cols = 5*(r_s +1)

            for c_s in range(2):
                rows = 5*(c_s+1)
                tch, pc = [], []
                for _ in range(100):
                    cu_x = torch.rand(n_matrix, cols, rows, device = device)
                    cu_y = torch.rand(n_matrix, rows, cols, device = device)
                    tch_, pc_ = loop_mul(cu_x, cu_y)
                    tch += tch_
                    pc += pc_
                r = statistics.mean(tch) / statistics.mean(pc)

                if n_matrix not in data: data[n_matrix] = {}
                if cols not in data[n_matrix]: data[n_matrix][cols] = {}
                data[n_matrix][cols][rows] = r

            PickleObject(data)
            print(n_s, r_s, c_s)


def plot_multiplication():

    x = UnpickleObject()
    for n_s in x:
        tf = TH2F()
        tf.Title = "Matrix Multiplication (MatMul) Compared to pyc CUDA Implementation With " + str(n_s) + "-Matrices"
        tf.xTitle = "Number of Rows for each Matrix"
        tf.yTitle = "Number of Columns for each Matrix"
        tf.xData = [r for t in x[n_s] for r in x[n_s][t]]
        tf.yData = [t for t in x[n_s] for r in x[n_s][t]]
        tf.Weight = [x[n_s][t][r] for t in x[n_s] for r in x[n_s][t]]
        tf.Filename = "MatMul-"+str(n_s)
        tf.SaveFigure()

def loop_det(ten1):
    _ = torch.det(ten1)
    _ = pyc.Operators.Determinant(ten1)
    for _ in range(1000):
        tch, pc = [], []
        t1 = time()
        _ = torch.det(ten1)
        tch.append(time() - t1)

        t1 = time()
        _ = pyc.Operators.Determinant(ten1)
        pc.append(time() - t1)
    return tch, pc

def loop_inv(ten1):
    _ = torch.inverse(ten1)
    _ = pyc.Operators.Inverse(ten1)
    for _ in range(1000):
        tch, pc = [], []
        t1 = time()
        _ = torch.inverse(ten1)
        tch.append(time() - t1)

        t1 = time()
        _ = pyc.Operators.Inverse(ten1)
        pc.append(time() - t1)
    return tch, pc

def loop_cross(ten1):
    ten2 = torch.inverse(ten1)
    ten2 = pyc.Operators.Inverse(ten1)
    torch.linalg.cross(ten1, ten2)
    pyc.Operators.Cross(ten1, ten2)
    for _ in range(1000):
        tch, pc = [], []
        t1 = time()
        _ = torch.linalg.cross(ten1, ten2)
        tch.append(time() - t1)

        t1 = time()
        _ = pyc.Operators.Cross(ten1, ten2)
        pc.append(time() - t1)
    return tch, pc


def matrix():

    data = {}
    for n_s in range(100):
        n_matrix = 1000*(n_s+1)
        data[n_matrix] = {}
        cu = torch.rand(n_matrix, 3, 3, device = device)

        tch, pc = loop_det(cu)
        data[n_matrix]["det"] = statistics.mean(tch) / statistics.mean(pc)

        tch, pc = loop_inv(cu)
        data[n_matrix]["inv"] = statistics.mean(tch) / statistics.mean(pc)

        tch, pc = loop_cross(cu)
        data[n_matrix]["cross"] = statistics.mean(tch) / statistics.mean(pc)

        print(n_s)
        if n_s%100: continue
        PickleObject(data)
        print(n_s)
    PickleObject(data)

def plot_matrix():

    x = UnpickleObject()
    tline = TLine()
    title = "Computational Time Ratio Between Torch (CUDA) and CUDA Native for Various\n"
    title += "for various Matrix Operations as a Function of Matrix Length"
    tline.Title = title

    t1 = TLine()
    t1.Title = "Determinant (3x3)"
    t1.xData = list(x)
    t1.yData = [x[t]["det"] for t in list(x)]

    t2 = TLine()
    t2.Title = "Inverse (3x3)"
    t2.xData = list(x)
    t2.yData = [x[t]["inv"] for t in list(x)]

    t3 = TLine()
    t3.Title = "Cross Product (3x3)"
    t3.xData = list(x)
    t3.yData = [x[t]["cross"] for t in list(x)]

    tline.Lines = [t1, t2, t3]
    tline.LineWidth = 0.2
    tline.Colors = ["r", "b", "y"]
    tline.xTitle = "Number of Matrices"
    tline.yTitle = "Ratio Time of Computation (Torch-CUDA / pyc-CUDA)"
    tline.yMin = 0
    tline.xStep = 100
    tline.Filename = "Matrix"
    tline.SaveFigure()

if __name__ == "__main__":
#    matrix_multiplication()
#    plot_multiplication()
    matrix()
    plot_matrix()
