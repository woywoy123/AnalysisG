from constants import *
import numpy as np
import math

def signs(v1, v2, s): return v1 if s > 0 else v2

def x1(sx, sy, data, sign): return sx - (sx + omega(data, sign) * sy) / Omega(data, sign)**2 
def y1(sx, sy, data, sign): return sy - (sx + omega(data, sign) * sy) * omega(data, sign) / Omega(data, sign)**2 

def mW(data, m_nu, sx): return complex(m_nu**2 - data.m_mu**2 - 2 * data.p_mu * sx) ** 0.5
def mT(data, m_nu, sx, sy):
    a = m_nu**2 - data.m_mu**2 + data.m_b**2
    return complex(a + -2 * (data.p_mu * sx + data.p_b * (data.theta.cos * sx + data.theta.sin * sy)))**0.5

def cosphi(data, eps, tau, s1, s2):
    def Lambda(data, dt, s1 = +1): return np.sin(math.atan(omega(data, s1))) + dt * np.cos(math.atan(omega(data, s1)))
    def Sigma(data , dt, s1 = +1): return np.cos(math.atan(omega(data, s1))) - dt * np.sin(math.atan(omega(data, s1)))

    Op, Om, wp, wm = Omega(data, +1), Omega(data, -1), omega(data, +1), omega(data, -1)
    dp, dm = (delta(data, +1), delta(data, -1)) if s1 == s2 else (delta(data, -1), delta(data, +1))
    lp, lm, sp, sm = Lambda(data, dp, +1), Lambda(data, dm, -1), Sigma(data, dp, +1), Sigma(data, dm, -1)
    m_mu, b_mu, E_mu = data.m_mu, data.b_mu, data.e_mu
    
    a =  (Op * sp - Om * sm) * m_mu ** 2 + (dp * wp * Om * sm - dm * wm * Op * sp) * E_mu ** 2
    b =  (lp           - lm) * m_mu ** 2 + (dp * wp * lm      - dm * wm * lp     ) * E_mu ** 2
    return eps / (b_mu * np.tanh(tau)) * (a / b)

def cross(v1, v2): 
    i = v1[1] * v2[2] - v1[2] * v2[1]
    j = v1[0] * v2[2] - v1[2] * v2[0]
    k = v1[0] * v2[1] - v1[1] * v2[0]
    return abs(i - j + k) < 1e-2

def get_pt(matrix): return np.array([matrix[0][2], matrix[1][2], matrix[2][1]]).real


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

