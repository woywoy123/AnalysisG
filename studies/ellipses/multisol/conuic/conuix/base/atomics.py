import numpy as np
import math

def costheta(v1, v2):
    v1_sq = v1.px**2 + v1.py**2 + v1.pz**2
    if v1_sq == 0: return 0
    v2_sq = v2.px**2 + v2.py**2 + v2.pz**2
    if v2_sq == 0: return 0
    v1v2 = v1.px*v2.px + v1.py*v2.py + v1.pz*v2.pz
    return v1v2/np.sqrt(v1_sq * v2_sq)

def R(axis, angle):
    c, s = math.cos(angle), math.sin(angle)
    R = c * np.eye(3)
    for i in [-1, 0, 1]: R[(axis - i) % 3, (axis + i) % 3] = i * s + (1 - i * i)
    return R

def string(obj, tl, mrg = 8): 
    d = str(getattr(obj, tl))
    l = "".join([" " for i in range(mrg - len(d))])
    return str(tl) + ": " + l + d

def _print(tlt, val = None, obj = None):
    if val is None: return print("====== " + tlt + " ======")
    if isinstance(val, list) and obj is not None: val = sum([string(obj, i) + " \n" for i in val])
    if isinstance(val, dict): val = sum([string_(i, val[i]) + " \n" for i in val]) 
    print("-------- " + tlt + " ------")
    print(val)

def assertions(tlt, trgt, val, limit =  0.1, hard_limit = False, verb = False):
    try: 
        if isinstance(trgt, np.ndarray): 
            assert sum(sum(abs(trgt - val) / abs(trgt + (trgt == 0))*100 > limit)) == 0
            if not verb: return True 
            print(tlt + "+>: diff:\n", abs(trgt - val))
            return True
        else: 
            if trgt == 0: assert abs(trgt - val) < limit
            else: assert (abs(trgt - val) / abs(trgt))*100 < limit
        if not verb: return True
        print(tlt + "+>: ", trgt, val, "diff:", abs(trgt - val))
        return True
    except AssertionError: print(tlt + "!>:\n", trgt, "\n", val, "\ndiff:\n", abs(trgt - val))
    if hard_limit: raise AssertionError

