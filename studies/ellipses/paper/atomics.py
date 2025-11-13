import math
import cmath
import numpy as np

def get_numbers(arr):
    data = []
    for i in arr:
        if "0x" in i: data.append(i)
        try: f = float(i)
        except: continue
        data.append(f)
    return data

def string(obj, tl, mrg = 8): 
    d = str(getattr(obj, tl))
    l = "".join([" " for i in range(mrg - len(d))])
    return str(tl) + ": " + l + d

def string_clx(val):
    if not isinstance(val, complex): return str(val)
    d = str(val.real)
    if val.imag == 0: return d
    sign = " + " if val.imag > 0 else " - "
    return d + sign + str(abs(val.imag)) + "i"

def string_(tl, val, mrg = 8): 
    d = string_clx(val)
    l = "".join([" " for i in range(mrg - len(d))])
    return str(tl) + ": " + l + d

def costheta(v1, v2):
    v1_sq = v1.px**2 + v1.py**2 + v1.pz**2
    if v1_sq == 0: return 0
    v2_sq = v2.px**2 + v2.py**2 + v2.pz**2
    if v2_sq == 0: return 0
    v1v2 = v1.px*v2.px + v1.py*v2.py + v1.pz*v2.pz
    return v1v2/math.sqrt(v1_sq * v2_sq)

def rotation_z(phi):
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    
def rotation_y(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    
def rotation_x(psi):
    c, s = np.cos(psi), np.sin(psi)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

def nulls(dim_x, dim_y):
    return [[0 for i in range(dim_x)] for j in range(dim_y)]

def cosh(tau): return math.cosh(tau.real)
def sinh(tau): return math.sinh(tau.real)
def tanh(tau): return math.tanh(tau.real)
def atanh(tau): return math.atanh(tau.real) if abs(tau) < 1 else None

def M_nu(idx, idy, m_nu):
    nl = nulls(4, 4)
    for i in range(len(idx)): nl[idy[i]][idx[i]] = m_nu[i]
    return np.array(nl)


def circle():
    return np.array([
        [1, 0,  0], 
        [0, 1,  0],
        [0, 0, -1]
    ])

