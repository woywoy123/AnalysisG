import numpy as np
import cmath

def _sign(s, v1, v2):      return v1 if s > 0 else v2
def _sO(s, data):          return _sign(s, data.Op, data.Om)
def _sw(s, data):          return _sign(s, data.wp, data.wm)
def _swO(s, data):         return _sign(s, (data.wp, data.Op), (data.wm, data.Om))

def mw2(data, sx, m_nu): return m_nu**2 - data.m_mu ** 2 - 2 * data.p_mu * sx
def mt2(data, sx, sy, m_nu): 
    m = data.m_bq ** 2 - data.m_mu ** 2 + m_nu ** 2 
    a = 2 * (data.p_mu + data.p_bq * data.cos) 
    b = 2 * data.p_bq * data.sin
    return m - a * sx - b * sy

def Z2(data, sx, sy, m_nu, sg):
    S = [sx ** 2, sx * sy, sy ** 2, sx, 1, 1]
    L = _sign(sg, [data.Ap, data.Bp, data.Cp], [data.Am, data.Bm, data.Cm])
    L += [data.D, data.E, - m_nu **2]
    return sum([S[i] * L[i] for i in range(len(S))]), L

def LpS(data, sx, sy): return sx - data.dp * sy
def LmS(data, sx, sy): return sx - data.dm * sy
def G2(data, sx, sy):  return - data.Gp * data.Gm * LpS(data, sx, sy) * LmS(data, sx, sy)

def LpA(data, sx, sy): return sx - data.ap * sy
def LmA(data, sx, sy): return sx - data.am * sy
def K2(data, sx, sy):  return data.gp * data.gm * LpA(data, sx, sy) * LmA(data, sx, sy)

def x1(data, sx, sy, sg): return sx - (sx + _sw(sg, data) * sy) / _sO(sg, data) ** 2
def y1(data, sx, sy, sg): return sy - (sx + _sw(sg, data) * sy) * _sw(sg, data) / _sO(sg, data) ** 2

def BQ(data, sx, sy, m_nu2):
    c, s, bt = data.cos, data.sin, data.b_bq
    a = bt**2 * c * ( c * sx + s * sy )
    b = bt**2 * s * ( c * sx + s * sy )
    t = m_nu2 - data.m_mu ** 2 - 2 * data.p_mu * sx - (bt * (c * sx + s * sy)) ** 2 
    return np.array([
        [1 - (bt * c)**2 , - bt**2 * c * s, 0, a], 
        [-bt ** 2 * c * s, 1 - (bt * s)**2, 0, b], 
        [0               ,               0, 1, 0],
        [a               ,               b, 0, t]
    ])

def LQ(data, sx, sy, m_nu2):
    bt = data.b_mu
    t = m_nu2 - data.m_mu ** 2 - 2 * data.p_mu * sx - (bt * sx)**2
    return np.array([
        [1 - bt**2 , 0, 0, sx*bt**2], 
        [0         , 1, 0,        0], 
        [0         , 0, 1,        0],
        [sx * bt**2, 0, 0,        t]
    ])

def F_frame(data, nu):
    if nu is None: nu = data.nu
    mx = np.array([nu.px, nu.py, nu.pz])
    return np.array(data.R_T.T.dot(mx).tolist() + [nu.e])

def Fnu_ny(data, sx, sy, z, chi, sg): return y1(data, sx, sy, sg) + _sw(sg, data) / _sO(sg, data) * z * np.cos(chi)
def Fnu_nx(data, sx, sy, z, chi, sg): return x1(data, sx, sy, sg) - data.p_mu + (z / _sO(sg, data)) * np.cos(chi)
def Fnu_ne(data, sx, sy, z, chi, sg): return data.b_mu * (Fnu_nx(data, sx, sy, z, chi, sg) - sx) + data.Sx0
def Fnu_nz(data, z, chi):             return z * np.sin(chi)

def SxL(data, lp, lm): return (data.dp * lm - data.dm * lp)/(data.dp - data.dm)
def SyL(data, lp, lm): return (lm - lp)/(data.dp - data.dm)

def htilde(data, sx, sy, z, sg):
    w, O = _swO(sg, data)
    return np.array([
        [z / O    , 0, x1(data, sx, sy, sg) - data.p_mu],
        [z * w / O, 0,             y1(data, sx, sy, sg)], 
        [0        , z,                                0]
    ])

def pcl2_P(data, m_nu, s1):
    wp, wm = data.wp, data.wm
    dp, dm = data.dp, data.dm 
    d = dp if s1 > 0 else dm 
    a = (data.m_mu / m_nu) ** 2 - 1 
    return complex(a * (wp - wm) ** 2 + (data.b_mu * (data.Op + s1 * data.Om))**2)

def pl2_lambda(data, m_nu, s1, s2):
    Pp, Pm = pcl2_P(data, m_nu, +1) ** 0.5, pcl2_P(data, m_nu, -1) ** 0.5
    a = (2 * data.b_mu * _sO(s1, data)) ** 2 
    return ((1 / a) * (Pp + s1 * s2 * Pm) ** 2) ** s2

def pl2_mass(data, l, s1):
    O1, O2 = _sign(s1, [data.Op, data.Om], [data.Om, data.Op])
    r = (O1 ** 2 * l - O2 ** 2) * (l - 1) * data.b_mu ** 2  / (l * (data.wp - data.wm) ** 2)
    return data.m_mu ** 2 / ( 1 + r )

def m2nu_t(data, t):
    dw, pw = data.wp - data.wm, data.wp + data.wm
    dp, dm = data.dp, data.dm
    a = dw ** 2 + (1 + dp * dm) * pw * ( 0.5 * (dp + dm) * (np.sqrt(t) + 1 / np.sqrt(t)) - (pw + dp + dm))
    return data.m_mu ** 2 * dw ** 2 / a

def mt_nu(data, m_nu):
    pp, pm = complex(pcl2_P(data, m_nu, +1)) ** 0.5, complex(pcl2_P(data, m_nu, -1))**0.5
    t = ( (pp + pm) / (pp - pm) ) ** 2
    p = (data.wm ** 2 - data.dp * data.dm) / (data.wp ** 2 - data.dp * data.dm)
    return t, p ** 2 * (1 - p ** 2 * t) / (t - p ** 2) 


