from atomics import *
import numpy as np

def GetZ2Coeffs(data):
    data._zdata = {}
    data._zdata["a"] = 1 / data.o ** 2 - 1
    data._zdata["b"] = 2 * data.w / data.o ** 2
    data._zdata["c"] = (data.l_b**2 - 1) / data.o ** 2
    data._zdata["d"] = 2 * data.lep.p
    data._zdata["e"] = data.lep.mass ** 2 - data.m_nu**2
    return data

def Z2(data, Sx = None, Sy = None): 
    if Sx is None: return GetZ2Coeffs(data)
    try: data = data._zdata
    except AttributeError: data = GetZ2Coeffs(data)._zdata
    p = iter([Sx**2, Sx*Sy, Sy**2, Sx, 1])
    return complex(sum([next(p) * data[k] for k in data]))

def geometry(alpha, beta, gamma):
    alpha, beta, gamma = np.acos(alpha), np.acos(beta), np.acos(gamma)
    a = np.cos(alpha) * np.cos(beta)
    b = np.sin(alpha) * np.sin(beta)
    return a + b * np.cos(gamma)

def angle(title, cos_a):
    try: print(title + "->", [np.acos(i) * 180 / np.pi for i in cos_a])
    except: print(title + "->", np.acos(cos_a) * 180 / np.pi)






