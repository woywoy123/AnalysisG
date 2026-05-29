from conuix.base.atomics import *
import numpy as np

def RT(data): 
    b_xyz = data.bq.px, data.bq.py, data.bq.pz
    R_z = R(2, -data.lp.phi)
    R_y = R(1, 0.5 * math.pi - data.lp.theta)
    R_x = next(R(0, -math.atan2(z, y)) for x, y, z in (R_y.dot(R_z.dot(b_xyz)),))
    return R_z.T.dot(R_y.T.dot(R_x.T))

def omega(data, sign): return (1 / data.sin) * (sign * data.r - data.cos)
def Omega(data, sign): return (omega(data, sign) ** 2 + 1 - data.b_mu ** 2) ** 0.5 

def _sign(s, v1, v2): return v1 if s > 0 else v2
def _swO(s, data): return _sign(s, (data.wp, data.Op), (data.wm, data.Om))
def _sO(s, data): return _sign(s, data.Op, data.Om)
def _sw(s, data): return _sign(s, data.wp, data.wm)
def _sd(s, data): return _sign(s, data.dp, data.dm)

def zA(data, sign): return (1 - _sO(sign, data) ** 2)/ _sO(sign, data) ** 2 
def zB(data, sign): return  2 * _sw(sign, data) / _sO(sign, data) ** 2      
def zC(data, sign): return - (1 - data.b_mu**2) / _sO(sign, data) **2
def zD(data): return 2 * data.p_mu 
def zE(data): return data.m_mu ** 2

def MQ(data, sign):
    A, B, C = zA(data, sign), zB(data, sign), zC(data, sign)
    D, E = zD(data), zE(data)
    return np.array([
        [A    , B / 2,  0, D / 2], 
        [B / 2, C    ,  0,     0], 
        [0    ,     0, -1,     0], 
        [D / 2,     0,  0,     E]
    ])

def Sx0(data):      return - data.m_mu ** 2 / data.p_mu
def Sy0(data, sig): return - data.e_mu * _sw(sig, data) / data.b_mu

def Gamma(data, sig): return (data.wp + sig * data.wm) / _sO(sig, data) ** 2 
def delta(data, sig): 
    a = (1 - data.b_mu ** 2 - data.wp * data.wm)
    b = data.Op * data.Om
    c = data.wp + data.wm
    return (a + sig * b) / c

def phi_delta(data):
    g  = (1 - data.b_mu**2) ** 0.5
    wp, wm = data.wp, data.wm
    return 0.5 * np.asinh( - g * (wp + wm) / (g**2 - wp * wm) )

def xP(data, _br): return  (_br["w"] - data.dm * (_br["O"]**2 - 1)) /_br["N"] 
def xM(data, _br): return  (data.dp * (_br["O"]**2 - 1) - _br["w"]) /_br["N"]

def yP(data, _br): return  (_br["w"] * data.dm - (1 - data.b_mu ** 2))/_br["N"]
def yM(data, _br): return  (1 - data.b_mu ** 2 - _br["w"] * data.dp)  /_br["N"]

def eP(data, _br): return  (data.b_mu * (_br["w"] + data.dm)) / _br["N"]
def eM(data, _br): return -(data.b_mu * (_br["w"] + data.dp)) / _br["N"]

def mx_Pnu(data, _br, br):
    return np.array([
        [xP(data, _br), xM(data, _br),       1.0 / _br["O"], 0, - data.p_mu], 
        [yP(data, _br), yM(data, _br),   _br["w"]/ _br["O"], 0,           0], 
        [           0 ,            0 ,                  0  , 1,           0], 
        [eP(data, _br), eM(data, _br), data.b_mu / _br["O"], 0, - data.e_mu]
    ])







def N11(dn, sig): return zA(dn, sig) * dn.dm ** 2 + zB(dn, sig) * dn.dm + zC(dn, sig) 
def N22(dn, sig): return zA(dn, sig) * dn.dp ** 2 + zB(dn, sig) * dn.dp + zC(dn, sig) 
def Nxx(dn, sig): return - (zA(dn, sig) * dn.dp * dn.dm + zC(dn, sig) + 0.5 * zB(dn, sig) * (dn.dp + dn.dm))

def LambdaN(data, sig):
    lx = 1 / (data.dp - data.dm)
    n11, n22 = N11(data, sig), N22(data, sig)
    n12, n21 = Nxx(data, sig), Nxx(data, sig)
    
    L0 = -2 * data.p_mu * lx * np.array([data.dm, - data.dp])
    return {"N" : np.array([[n11, n12], [n21, n22]]) * lx ** 2, "L" : L0, "C": data.m_mu ** 2}

def L0(data, s1, s2):
    dt, wt = _sd(s1, data), _sw(s2, data)
    return (data.e_mu / data.b_mu) * (dt * wt + data.dp * data.dm)

def degen_nu2(data, s1):
    a = data.m_mu * ( data.wp - data.wm ) 
    b = (data.wp - data.wm) 
    c = (data.Op + s1 * data.Om) * data.b_mu
    return a ** 2 / (b ** 2 - c ** 2)

def lambdaK(data, s1):
    g = Gamma(data, +1) * Gamma(data, -1) * 0.5 
    s = _sign(data.dp + data.dm, +1.0, -1.0) 
    g = g * (- data.b_mu ** 2 + s * s1 *((1 + data.dp ** 2) * (1 + data.dm ** 2)) ** 0.5)
    return g, 1.0 if g > 0 else -1.0

def alphaK(data, v1, v2, s1, inv):
    eta = np.asinh( - (1 - data.dp * data.dm)/(data.dp + data.dm) ) 
    exp = np.exp(eta / 2)
    lx = (2 * np.cosh(eta))
    if not inv: 
        lb, s = lambdaK(data, s1)
        a, b = abs(lb / lx) ** 0.5 * ( exp ** s1), abs(lb / lx) ** 0.5 * s1 * (exp ** -s1)
        return a * v1 + b * v2, s

    lb, lk = lambdaK(data, +1), lambdaK(data, -1)
    a, b = abs(1 / (lx * lb[0])) ** 0.5 * ( exp ** s1), abs(1 / (lx * lk[0])) ** 0.5 * s1 * (exp ** -s1)
    return a * v1 + b * v2 

