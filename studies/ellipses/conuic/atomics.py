import numpy as np
import math

def string(obj, tl, mrg = 8): 
    d = str(getattr(obj, tl))
    l = "".join([" " for i in range(mrg - len(d))])
    return str(tl) + ": " + l + d

def signs(v1, v2, s): return v1 if s > 0 else v2

def costheta(v1, v2):
    v1_sq = v1.px**2 + v1.py**2 + v1.pz**2
    if v1_sq == 0: return 0
    v2_sq = v2.px**2 + v2.py**2 + v2.pz**2
    if v2_sq == 0: return 0
    v1v2 = v1.px*v2.px + v1.py*v2.py + v1.pz*v2.pz
    return v1v2/math.sqrt(v1_sq * v2_sq)

def angle(cs): return math.acos(cs) * 180 / np.pi


def rotation_z(phi):
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    
def rotation_y(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    
def rotation_x(psi):
    c, s = np.cos(psi), np.sin(psi)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _print(tlt, val = None, obj = None):
    if val is None: return print("====== " + tlt + " ======")
    if isinstance(val, list) and obj is not None: 
        val = sum([string(obj, i) + " \n" for i in val])
    if isinstance(val, dict):
        o = ""
        for i in val: o += string_(i, val[i]) + " \n"
        val = o

    print("-------- " + tlt + " ------")
    print(val)

def assertions(tlt, trgt, val, limit =  0.1, hard_limit = False):
    try: 
        if isinstance(trgt, np.ndarray): 
            assert sum(sum(abs(trgt - val) / abs(trgt + (trgt == 0))*100 > limit)) == 0
            print(tlt + "+>: diff:\n", abs(trgt - val))
            return True
        else: 
            if trgt == 0: assert abs(trgt - val) < limit
            else: assert (abs(trgt - val) / abs(trgt))*100 < limit
        print(tlt + "+>:\n", trgt, val, "diff:", abs(trgt - val))
        return True
    except AssertionError: print(tlt + "!>:\n", trgt, "\n", val, "\ndiff:\n", abs(trgt - val))
    if hard_limit: raise AssertionError

def observation(tlt, val):
    print("NOTE: " + tlt + " +>: ", val)


